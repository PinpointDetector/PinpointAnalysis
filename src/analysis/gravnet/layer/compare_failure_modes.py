"""
Failure-mode comparison: binary classifier vs energy regressor.

Loads both models, runs inference on the same val split, aligns events by
E_nu (rather than assuming index order), then quantifies whether the events
the classifier finds hard are the same events the regressor finds hard.

Usage:
    python compare_failure_modes.py

Output:
    figures saved to figures_path / "failure_mode_comparison/"
    summary printed to stdout
"""

# ── Config ───────────────────────────────────────────────────────────────────
CLASSIFIER_WEIGHTS_DIR      = "gravnet_binary_classifier_faser_final"
REGRESSOR_WEIGHTS_DIR       = "gravnet_regression_faser_huber1.0_nofaser_final"
REGRESSOR_PLUS_CLF_DIR      = "gravnet_regression_faser_huber1.0_binaryprob_nofaser_final"
REGRESSOR_TRUTH_DIR         = "gravnet_regression_faser_huber1.0_truth2_nofaser_final"
RUN    = 10000   # run 10000 is unseen by both models → clean comparison test set
GPU    = "cuda:0"

# If the regressor WEIGHTS_DIR contains "_binaryprob", it was trained with
# binary classifier softmax probs concatenated to each node. Set this to the
# path of the binary classifier best_model.pt to enable that augmentation;
# None is fine for all other regressor variants.
BINARY_CLASSIFIER_WEIGHTS = None   # e.g. get_weights_path() / "gravnet_binary_classifier_faser/best_model.pt"

N_BOOTSTRAP   = 2000   # resamples for 95% CI on Pearson r
HARD_QUANTILE = 0.75   # top-quartile threshold for "regressor-hard"
EASY_QUANTILE = 0.25   # bottom-quartile threshold for "classifier-hard"
# Metric used to define the classifier-hard set.
# "f1"  → bottom-quartile F1 (top-quartile 1-F1 error)
# "bce" → top-quartile per-event BCE loss
CLF_HARD_METRIC = "bce"

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import stats
import re as _re
import json as _json
import datetime as _dt
import matplotlib.patches as mpatches
from scipy.optimize import brentq as _brentq
from numpy.linalg import lstsq as _lstsq

from torch_geometric.loader import DataLoader

from analysis.gravnet.model import NeutrinoGravNetNodesFaser, NeutrinoGravNetRegressionFASER
from analysis.utils.utils import get_torch_path, get_weights_path, get_figures_path

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.2)
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":   "dejavuserif",
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "figure.dpi":         350,
})

C1     = "#353D4C"
C2     = "#E17883"
C3       = "#5691D9"
C_TRUTH  = "#A0785A"   # earthy brown — used exclusively for G_{truth} oracle lines
COLORS = ["#4477AA", "#EE6677"]
# Colormap: grey (low y) → dark blue → light blue (high y)
YCMAP  = mcolors.LinearSegmentedColormap.from_list(
    "ltblue_dkbluegrey", ["#87c9ea", "#3a7bbf", "#172332"]
)

# ── Paths ────────────────────────────────────────────────────────────────────
device        = torch.device(GPU if torch.cuda.is_available() else "cpu")
torch_path    = get_torch_path()
weights_path  = get_weights_path()
figures_path  = get_figures_path() / "failure_mode_comparison"
figures_path.mkdir(parents=True, exist_ok=True)

clf_weights  = weights_path / CLASSIFIER_WEIGHTS_DIR
reg_weights  = weights_path / REGRESSOR_WEIGHTS_DIR

print(f"Device     : {device}")
print(f"Classifier : {clf_weights}")
print(f"Regressor  : {reg_weights}")
print(f"Figures    : {figures_path}")

# ── Early cache check — skip dataset loading if all inference caches exist ───
print("\n── Cache check ──────────────────────────────────────────────────────")
_clf_caches = sorted(clf_weights.glob(f"inference_cache_run{RUN}_n*_ep*_vl*.npz"))
_reg_caches = sorted(reg_weights.glob(f"inference_cache_run{RUN}_n*_ep*_vl*.npz"))
print(f"  clf cache : {'FOUND — ' + _clf_caches[-1].name if _clf_caches else 'not found'}")
print(f"  reg cache : {'FOUND — ' + _reg_caches[-1].name if _reg_caches else 'not found'}")
SKIP_DATA_LOAD = bool(_clf_caches and _reg_caches)
if SKIP_DATA_LOAD:
    _m = _re.search(r"_n(\d+)_", _clf_caches[-1].name)
    N_VAL_EVENTS = int(_m.group(1)) if _m else None
    print(f"  → Both caches present. Skipping dataset load ({N_VAL_EVENTS} events).")
    print(f"    Recovering ground-truth arrays (Enu, inel) from clf cache...")
    _cc_early     = np.load(_clf_caches[-1])
    Enu_true_all  = _cc_early["clf_Enu"]
    inel_true_all = _cc_early["clf_inel"]
    Eroe_true_all = Enu_true_all * inel_true_all
    val_dataset   = None
    print(f"    Done. Enu range: {Enu_true_all.min():.3f}–{Enu_true_all.max():.3f} TeV")
    print(f"    Note: val_dataset=None — event-level cells (n_nodes, plus/truth augment) need a full run.")
else:
    N_VAL_EVENTS = None
    missing = []
    if not _clf_caches: missing.append("clf")
    if not _reg_caches: missing.append("reg")
    print(f"  → Missing cache(s): {', '.join(missing)}. Loading dataset and running inference.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def binned_median(x, y, n_bins=15):
    """Sort events by x, split into equal-count bins, return (bin_centre, median_y)."""
    valid  = np.isfinite(x) & np.isfinite(y)
    x, y   = x[valid], y[valid]
    order  = np.argsort(x)
    xs, ys = x[order], y[order]
    chunks = np.array_split(np.arange(len(xs)), n_bins)
    return (np.array([xs[c].mean()     for c in chunks if len(c)]),
            np.array([np.median(ys[c]) for c in chunks if len(c)]))


def per_event_f1_pEM(pred, true):
    """F1 score for the primary_EM_e class (label=1) for a single event."""
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    d  = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


def per_event_precision_recall_pEM(pred, true):
    """Precision and recall for the primary_EM class (label=1) for a single event."""
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall    = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return precision, recall


def bootstrap_pearsonr(x, y, n=N_BOOTSTRAP, rng=None):
    """Return (r, ci_lo, ci_hi) with 95% bootstrap CI."""
    if rng is None:
        rng = np.random.default_rng(0)
    r_obs  = np.corrcoef(x, y)[0, 1]
    boots  = np.empty(n)
    idx    = np.arange(len(x))
    for i in range(n):
        s       = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = np.corrcoef(x[s], y[s])[0, 1]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    return r_obs, ci_lo, ci_hi


def preds_to_physical(preds, norm_stats=None, beta_loss=False):
    """Convert raw model output tensor → (E_nu, E_lep, E_roe) in TeV [N×3 numpy]."""
    if norm_stats is not None:
        log_E_nu = preds[:, 0] * norm_stats["sigma_enu"] + norm_stats["mu_enu"]
    else:
        log_E_nu = preds[:, 0]
    E_nu = 10 ** log_E_nu
    if beta_loss:
        alpha, beta_p = preds[:, 1], preds[:, 2]
        y = alpha / (alpha + beta_p)
    else:
        if norm_stats is not None:
            logit_y = preds[:, 1] * norm_stats["sigma_logit"] + norm_stats["mu_logit"]
        else:
            logit_y = preds[:, 1]
        y = torch.sigmoid(logit_y)
    E_roe = y * E_nu
    E_lep = (1 - y) * E_nu
    return torch.stack([E_nu, E_lep, E_roe], dim=1).numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load dataset  (shared by both models — same split, same random_state)
# ══════════════════════════════════════════════════════════════════════════════
if not SKIP_DATA_LOAD:
    print("\n── Loading dataset ──────────────────────────────────────────────────")
    run_path    = torch_path / f"{RUN}/pointnetpp_faser_all_events"
    chunk_files = sorted(run_path.glob("nue_*.pt"))
    chunk_files = [f for f in chunk_files if "_particle_prob" not in f.stem]

    dataset = []
    for f in chunk_files:
        dataset.extend(torch.load(f, weights_only=False))
    print(f"Total events: {len(dataset)}")

    val_dataset   = dataset
    N_VAL_EVENTS  = len(val_dataset)
    print(f"Val events  : {N_VAL_EVENTS}")

    Enu_true_all  = np.array([float(d.E_nu)  for d in val_dataset])
    Eroe_true_all = np.array([float(d.E_roe) for d in val_dataset])
    inel_true_all = np.where(Enu_true_all > 0, Eroe_true_all / Enu_true_all, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Classifier inference  → per-event F1_pEM
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Classifier inference ─────────────────────────────────────────────")
clf_ckpt  = torch.load(clf_weights / "best_model.pt", map_location=device, weights_only=False)
clf_cfg   = clf_ckpt.get("model_config", {})
input_dim_clf = 2 if "_vertexdist" in CLASSIFIER_WEIGHTS_DIR else 1

clf_model = NeutrinoGravNetNodesFaser(
    input_dim=input_dim_clf, num_node_classes=2, faser_dim=5,
    n_gravstack=clf_cfg.get("n_gravstack", 3),
    out_channels=clf_cfg.get("out_channels", 16),
    n_feature_transform=clf_cfg.get("n_feature_transform", 16),
    k=clf_cfg.get("k", 12),
).to(device)
clf_model.load_state_dict(clf_ckpt["model_state_dict"])
clf_model.eval()
print(f"Classifier epoch {clf_ckpt['epoch'] + 1}")

clf_cache_key  = f"run{RUN}_n{N_VAL_EVENTS}_ep{clf_ckpt['epoch']}_vl{clf_ckpt.get('val_loss', 0):.6f}"
clf_cache_path = clf_weights / f"inference_cache_{clf_cache_key}.npz"

_cache_valid = (clf_cache_path.exists() and
                all(k in np.load(clf_cache_path) for k in
                    ("clf_acc", "clf_precision", "clf_recall",
                     "clf_n_true_pEM", "clf_n_pred_pEM",
                     "clf_tp", "clf_fp", "clf_fn", "clf_tn",
                     "clf_ce_per_event",
                     "clf_mean_prob_signal", "clf_mean_prob_bg",
                     "clf_bce_signal", "clf_bce_bg")))
if _cache_valid:
    print(f"Loading classifier cache: {clf_cache_path.name}")
    _cc           = np.load(clf_cache_path)
    clf_f1        = _cc["clf_f1"]
    clf_acc       = _cc["clf_acc"]
    clf_Enu       = _cc["clf_Enu"]
    clf_inel      = _cc["clf_inel"]
    clf_precision = _cc["clf_precision"]
    clf_recall    = _cc["clf_recall"]
    clf_n_true_pEM = _cc["clf_n_true_pEM"]
    clf_n_pred_pEM = _cc["clf_n_pred_pEM"]
    clf_tp         = _cc["clf_tp"]
    clf_fp         = _cc["clf_fp"]
    clf_fn         = _cc["clf_fn"]
    clf_tn         = _cc["clf_tn"]
    clf_ce_per_event     = _cc["clf_ce_per_event"]
    clf_mean_prob_signal = _cc["clf_mean_prob_signal"]
    clf_mean_prob_bg     = _cc["clf_mean_prob_bg"]
    clf_bce_signal       = _cc["clf_bce_signal"]
    clf_bce_bg           = _cc["clf_bce_bg"]
else:
    clf_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    clf_Enu          = []
    clf_f1           = []
    clf_acc          = []
    clf_inel         = []
    clf_precision    = []
    clf_recall       = []
    clf_n_true_pEM   = []
    clf_n_pred_pEM   = []
    clf_tp           = []
    clf_fp           = []
    clf_fn           = []
    clf_tn           = []
    clf_ce_per_event     = []   # per-event mean binary cross-entropy of softmax probs
    clf_mean_prob_signal = []   # per-event mean P(pEM) for true signal nodes
    clf_mean_prob_bg     = []   # per-event mean P(pEM) for true background nodes
    clf_bce_signal       = []   # per-event mean -log(p_pEM) for true signal nodes
    clf_bce_bg           = []   # per-event mean -log(1-p_pEM) for true background nodes

    _CE_EPS = 1e-7   # numerical stability for log

    with torch.no_grad():
        for batch in clf_loader:
            batch   = batch.to(device)
            out     = clf_model(batch.x, batch.pos, batch.batch, batch.x_faser)
            probs   = torch.softmax(out, dim=1).cpu()   # [N_nodes, 2]
            pred    = out.argmax(dim=1).cpu()
            true    = (batch.pdg_label == 2).long().cpu()
            b_cpu   = batch.batch.cpu()
            E_nu_b  = batch.E_nu.cpu()
            E_roe_b = batch.E_roe.cpu()
            for g in range(int(b_cpu.max().item()) + 1):
                m      = b_cpu == g
                p_g, t_g = pred[m].numpy(), true[m].numpy()
                f1     = per_event_f1_pEM(p_g, t_g)
                acc    = float((p_g == t_g).mean())
                prec, rec = per_event_precision_recall_pEM(p_g, t_g)
                enu_g  = float(E_nu_b[g])
                eroe_g = float(E_roe_b[g])
                clf_f1.append(f1)
                clf_acc.append(acc)
                clf_precision.append(prec)
                clf_recall.append(rec)
                clf_n_true_pEM.append(int((t_g == 1).sum()))
                clf_n_pred_pEM.append(int((p_g == 1).sum()))
                clf_tp.append(int(((p_g == 1) & (t_g == 1)).sum()))
                clf_fp.append(int(((p_g == 1) & (t_g == 0)).sum()))
                clf_fn.append(int(((p_g == 0) & (t_g == 1)).sum()))
                clf_tn.append(int(((p_g == 0) & (t_g == 0)).sum()))
                clf_Enu.append(enu_g)
                raw_inel = eroe_g / enu_g if enu_g > 0 else np.nan
                clf_inel.append(np.clip(raw_inel, 0.0, 1.0) if np.isfinite(raw_inel) else np.nan)
                # per-event mean cross-entropy: -[t*log(p_pEM) + (1-t)*log(p_bg)]
                p_pEM = probs[m, 1].numpy()
                ce_nodes = -(t_g * np.log(p_pEM + _CE_EPS)
                             + (1.0 - t_g) * np.log(1.0 - p_pEM + _CE_EPS))
                clf_ce_per_event.append(float(ce_nodes.mean()))
                clf_mean_prob_signal.append(float(p_pEM[t_g == 1].mean()) if (t_g == 1).any() else np.nan)
                clf_mean_prob_bg.append(float(p_pEM[t_g == 0].mean()) if (t_g == 0).any() else np.nan)
                clf_bce_signal.append(float(-np.log(p_pEM[t_g == 1] + _CE_EPS).mean()) if (t_g == 1).any() else np.nan)
                clf_bce_bg.append(float(-np.log(1.0 - p_pEM[t_g == 0] + _CE_EPS).mean()) if (t_g == 0).any() else np.nan)

    clf_f1           = np.array(clf_f1)
    clf_acc          = np.array(clf_acc)
    clf_precision    = np.array(clf_precision)
    clf_recall       = np.array(clf_recall)
    clf_n_true_pEM   = np.array(clf_n_true_pEM)
    clf_n_pred_pEM   = np.array(clf_n_pred_pEM)
    clf_tp           = np.array(clf_tp)
    clf_fp           = np.array(clf_fp)
    clf_fn           = np.array(clf_fn)
    clf_tn           = np.array(clf_tn)
    clf_Enu          = np.array(clf_Enu)
    clf_inel         = np.array(clf_inel)
    clf_ce_per_event     = np.array(clf_ce_per_event)
    clf_mean_prob_signal = np.array(clf_mean_prob_signal)
    clf_mean_prob_bg     = np.array(clf_mean_prob_bg)
    clf_bce_signal       = np.array(clf_bce_signal)
    clf_bce_bg           = np.array(clf_bce_bg)
    np.savez_compressed(clf_cache_path, clf_f1=clf_f1, clf_acc=clf_acc,
                        clf_Enu=clf_Enu, clf_inel=clf_inel,
                        clf_precision=clf_precision, clf_recall=clf_recall,
                        clf_n_true_pEM=clf_n_true_pEM, clf_n_pred_pEM=clf_n_pred_pEM,
                        clf_tp=clf_tp, clf_fp=clf_fp, clf_fn=clf_fn, clf_tn=clf_tn,
                        clf_ce_per_event=clf_ce_per_event,
                        clf_mean_prob_signal=clf_mean_prob_signal,
                        clf_mean_prob_bg=clf_mean_prob_bg,
                        clf_bce_signal=clf_bce_signal,
                        clf_bce_bg=clf_bce_bg)
    print(f"Saved classifier cache: {clf_cache_path.name}")

print(f"Classifier scored {len(clf_f1)} events  |  "
      f"F1_pEM {clf_f1.min():.3f}–{clf_f1.max():.3f}  |  "
      f"acc {clf_acc.min():.3f}–{clf_acc.max():.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Regressor inference  → per-event abserr_Enu, abserr_y
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Regressor inference ──────────────────────────────────────────────")
reg_ckpt   = torch.load(reg_weights / "best_model.pt", map_location=device, weights_only=False)
norm_stats = reg_ckpt.get("norm_stats", None)
beta_loss  = reg_ckpt.get("beta_loss", False)
pooling    = "sum" if "_sum_" in REGRESSOR_WEIGHTS_DIR else "mean"
use_faser  = "_nofaser" not in REGRESSOR_WEIGHTS_DIR

# Derive input_dim — mirror the notebook logic exactly
wd = REGRESSOR_WEIGHTS_DIR
if   "_truth4"     in wd: input_dim_reg = 5
elif "_truth3bprob" in wd: input_dim_reg = 4
elif "_truth3b"    in wd: input_dim_reg = 4
elif "_truth3"     in wd: input_dim_reg = 4
elif "_truth2"     in wd: input_dim_reg = 3
elif "_faserglobal" in wd: input_dim_reg = 6
elif "_vertexdist" in wd: input_dim_reg = 2
elif "_binaryprob" in wd: input_dim_reg = 3
elif "_embedding"  in wd: input_dim_reg = 57
elif "_prob"       in wd: input_dim_reg = 5
else:                      input_dim_reg = 1
print(f"Regressor input_dim={input_dim_reg}, pooling={pooling}, use_faser={use_faser}")

reg_model = NeutrinoGravNetRegressionFASER(
    input_dim=input_dim_reg,
    num_targets=2,
    faser_dim=5,
    pooling=pooling,
    beta_loss=beta_loss,
    use_faser=use_faser,
).to(device)
reg_model.load_state_dict(reg_ckpt["model_state_dict"])
reg_model.eval()
print(f"Regressor epoch {reg_ckpt['epoch'] + 1}")

# Augment val_dataset for _binaryprob models (only needed when reg cache is absent)
if val_dataset is not None:
    reg_val = list(val_dataset)
    if "_binaryprob" in wd:
        if BINARY_CLASSIFIER_WEIGHTS is None:
            raise ValueError("Set BINARY_CLASSIFIER_WEIGHTS for _binaryprob regressor.")
        print(f"Augmenting with binary classifier probs: {BINARY_CLASSIFIER_WEIGHTS}")
        _clf2 = NeutrinoGravNetNodesFaser(input_dim=1, num_node_classes=2, faser_dim=5)
        _ckpt2 = torch.load(BINARY_CLASSIFIER_WEIGHTS, map_location=device, weights_only=False)
        _clf2.load_state_dict(_ckpt2["model_state_dict"])
        _clf2.to(device).eval()
        with torch.no_grad():
            for data in reg_val:
                _bv  = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                _out = _clf2(data.x.to(device), data.pos.to(device), _bv,
                             data.x_faser.unsqueeze(0).to(device))
                _prob = torch.softmax(_out, dim=1).cpu()
                data.x = torch.cat([data.x, _prob], dim=1)
        del _clf2
        torch.cuda.empty_cache()
    reg_loader = DataLoader(reg_val, batch_size=32, shuffle=False)
else:
    reg_val    = None
    reg_loader = None

cache_key  = f"run{RUN}_n{N_VAL_EVENTS}_ep{reg_ckpt['epoch']}_vl{reg_ckpt['val_loss']:.6f}"
cache_path = reg_weights / f"inference_cache_{cache_key}.npz"

if cache_path.exists():
    print(f"Loading regressor cache: {cache_path.name}")
    _c = np.load(cache_path)
    preds_raw      = torch.from_numpy(_c["preds_raw"])
    targets_linear = _c["targets_linear"]
    preds_linear   = preds_to_physical(preds_raw, norm_stats, beta_loss=beta_loss)
else:
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data in reg_loader:
            data = data.to(device)
            raw  = reg_model(data.x, data.pos, data.batch, data.x_faser)
            tgt  = torch.stack([data.E_nu, data.E_lepton, data.E_roe], dim=1)
            all_preds.append(raw.cpu())
            all_targets.append(tgt.cpu())
    preds_raw      = torch.cat(all_preds)
    targets_linear = torch.cat(all_targets).numpy()
    preds_linear   = preds_to_physical(preds_raw, norm_stats, beta_loss=beta_loss)
    np.savez_compressed(cache_path, preds_raw=preds_raw.numpy(),
                        targets_linear=targets_linear)
    print(f"Saved regressor cache: {cache_path.name}")

reg_Enu_true = targets_linear[:, 0]
reg_Enu_pred = preds_linear[:,  0]
reg_Eroe_true = targets_linear[:, 2]
reg_Eroe_pred = preds_linear[:,  2]

abserr_Enu = np.abs((reg_Enu_pred  - reg_Enu_true)  / np.maximum(reg_Enu_true,  1e-9))
y_true     = reg_Eroe_true / np.maximum(reg_Enu_true, 1e-9)
y_pred     = reg_Eroe_pred / np.maximum(reg_Enu_pred, 1e-9)
abserr_y   = np.abs(y_pred - y_true)   # absolute error in y (avoids denominator blowup at low y)

print(f"Regressor scored {len(abserr_Enu)} events")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Align events by E_nu  (do NOT assume index order is identical)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Aligning events by E_nu ──────────────────────────────────────────")

# Both models iterate val_dataset in the same DataLoader order (shuffle=False),
# so indices should match. Verify by checking E_nu agreement element-wise.
max_rel_diff = np.max(np.abs(clf_Enu - reg_Enu_true) / np.maximum(reg_Enu_true, 1e-9))
print(f"Max |ΔE_nu|/E_nu between classifier and regressor passes: {max_rel_diff:.2e}")

if max_rel_diff > 1e-4:
    print("WARNING: E_nu mismatch > 0.01% — attempting to re-align by sorting on E_nu.")
    clf_order = np.argsort(clf_Enu)
    reg_order = np.argsort(reg_Enu_true)
    f1_aligned          = clf_f1[clf_order]
    acc_aligned         = clf_acc[clf_order]
    ce_aligned               = clf_ce_per_event[clf_order]
    mean_prob_signal_aligned = clf_mean_prob_signal[clf_order]
    mean_prob_bg_aligned     = clf_mean_prob_bg[clf_order]
    bce_signal_aligned       = clf_bce_signal[clf_order]
    bce_bg_aligned           = clf_bce_bg[clf_order]
    precision_aligned        = clf_precision[clf_order]
    recall_aligned      = clf_recall[clf_order]
    n_true_pEM_aligned  = clf_n_true_pEM[clf_order]
    n_pred_pEM_aligned  = clf_n_pred_pEM[clf_order]
    tp_aligned          = clf_tp[clf_order]
    fp_aligned          = clf_fp[clf_order]
    fn_aligned          = clf_fn[clf_order]
    tn_aligned          = clf_tn[clf_order]
    inel_aligned        = clf_inel[clf_order]
    abserr_Enu_al     = abserr_Enu[reg_order]
    abserr_y_al       = abserr_y[reg_order]
    print("Re-aligned by E_nu sort.")
else:
    print("Alignment verified — indices are consistent.")
    f1_aligned         = clf_f1
    acc_aligned        = clf_acc
    ce_aligned               = clf_ce_per_event
    mean_prob_signal_aligned = clf_mean_prob_signal
    mean_prob_bg_aligned     = clf_mean_prob_bg
    bce_signal_aligned       = clf_bce_signal
    bce_bg_aligned           = clf_bce_bg
    precision_aligned        = clf_precision
    recall_aligned     = clf_recall
    n_true_pEM_aligned = clf_n_true_pEM
    n_pred_pEM_aligned = clf_n_pred_pEM
    tp_aligned         = clf_tp
    fp_aligned         = clf_fp
    fn_aligned         = clf_fn
    tn_aligned         = clf_tn
    inel_aligned       = clf_inel
    abserr_Enu_al     = abserr_Enu
    abserr_y_al       = abserr_y

# Classifier error: (1 - F1_pEM); skip events with no primary_EM nodes (f1 undefined)
clf_err     = 1.0 - f1_aligned
clf_acc_err = 1.0 - acc_aligned          # accuracy-based error (for comparison)
valid   = (np.isfinite(clf_err) & np.isfinite(abserr_Enu_al)
           & np.isfinite(abserr_y_al) & np.isfinite(inel_aligned))
n_valid = valid.sum()
print(f"Valid events for analysis: {n_valid} / {len(valid)}")

ce      = clf_err[valid]
ce_acc  = clf_acc_err[valid]             # (1 - accuracy) per event
ae_E    = abserr_Enu_al[valid]
ae_y    = abserr_y_al[valid]
yv      = np.clip(inel_aligned[valid], 0.0, 1.0)  # clip float artefacts (E_roe slightly <0)
f1v     = f1_aligned[valid]
accv    = acc_aligned[valid]
cev            = ce_aligned[valid]              # per-event mean binary cross-entropy of softmax probs
prob_signal_v  = mean_prob_signal_aligned[valid]  # mean P(pEM) for true signal nodes
prob_bg_v      = mean_prob_bg_aligned[valid]      # mean P(pEM) for true background nodes
bce_signal_v   = bce_signal_aligned[valid]        # mean -log(p_pEM) for true signal nodes
bce_bg_v       = bce_bg_aligned[valid]            # mean -log(1-p_pEM) for true background nodes


# ══════════════════════════════════════════════════════════════════════════════
# 5. Failure-mode overlap statistics
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Failure-mode overlap statistics ──────────────────────────────────")

# Thresholds
if CLF_HARD_METRIC == "bce":
    _clf_metric     = cev
    _clf_metric_lbl = r"top-quartile $\overline{\mathcal{L}}_\mathrm{BCE}^{(e)}$"
    thresh_clf = np.quantile(cev, HARD_QUANTILE)
else:
    _clf_metric     = ce
    _clf_metric_lbl = "bottom-quartile $F_{1,\\mathrm{pEM}}$"
    thresh_clf = np.quantile(ce, 1 - EASY_QUANTILE)
thresh_E   = np.quantile(ae_E, HARD_QUANTILE)       # abserr_Enu > Q75 → regressor-hard (E_nu)
thresh_y   = np.quantile(ae_y, HARD_QUANTILE)       # abserr_y   > Q75 → regressor-hard (y)

clf_hard = _clf_metric >= thresh_clf
print(f"clf-hard definition: {_clf_metric_lbl}  (CLF_HARD_METRIC='{CLF_HARD_METRIC}')")
reg_hard_E = ae_E >= thresh_E
reg_hard_y = ae_y >= thresh_y
reg_hard   = reg_hard_E | reg_hard_y

n_clf  = clf_hard.sum()
n_reg  = reg_hard.sum()
n_both = (clf_hard & reg_hard).sum()
n      = n_valid

p_clf_hard  = n_clf  / n
p_reg_hard  = n_reg  / n
p_both_obs  = n_both / n
p_both_exp  = p_clf_hard * p_reg_hard   # expected under independence

print(f"Classifier-hard (worst {int(EASY_QUANTILE*100)}th pct F1): {n_clf} events ({100*p_clf_hard:.1f}%)")
print(f"Regressor-hard  (top {int(HARD_QUANTILE*100)}th pct |err|, either target): {n_reg} events ({100*p_reg_hard:.1f}%)")
print(f"Both hard — observed : {n_both} events ({100*p_both_obs:.1f}%)")
print(f"Both hard — expected by chance : {p_both_exp*n:.0f} events ({100*p_both_exp:.1f}%)")

# Pearson and Spearman correlations with bootstrap CIs
rng = np.random.default_rng(42)
r_E,  ci_E_lo,  ci_E_hi  = bootstrap_pearsonr(ce, ae_E, rng=rng)
r_y,  ci_y_lo,  ci_y_hi  = bootstrap_pearsonr(ce, ae_y, rng=rng)
r_EY, ci_EY_lo, ci_EY_hi = bootstrap_pearsonr(ae_E, ae_y, rng=rng)

sp_E = stats.spearmanr(ce, ae_E).statistic
sp_y = stats.spearmanr(ce, ae_y).statistic

# Accuracy-based equivalents (for comparison — checks whether U-shape matters)
r_accE,  _, _  = bootstrap_pearsonr(ce_acc, ae_E, rng=rng)
r_accy,  _, _  = bootstrap_pearsonr(ce_acc, ae_y, rng=rng)
sp_accE = stats.spearmanr(ce_acc, ae_E).statistic
sp_accy = stats.spearmanr(ce_acc, ae_y).statistic

print(f"\n── F1_pEM-based (pipeline-relevant metric) ──")
print(f"Pearson r(clf_err, abserr_Enu) = {r_E:+.3f}  95% CI [{ci_E_lo:+.3f}, {ci_E_hi:+.3f}]  Spearman={sp_E:+.3f}")
print(f"Pearson r(clf_err, abserr_y)   = {r_y:+.3f}  95% CI [{ci_y_lo:+.3f}, {ci_y_hi:+.3f}]  Spearman={sp_y:+.3f}")
print(f"Pearson r(abserr_Enu, abserr_y)= {r_EY:+.3f}  95% CI [{ci_EY_lo:+.3f}, {ci_EY_hi:+.3f}]")
print(f"\n── Accuracy-based (majority-class dominated, for comparison) ──")
print(f"Pearson r(acc_err, abserr_Enu) = {r_accE:+.3f}  Spearman={sp_accE:+.3f}")
print(f"Pearson r(acc_err, abserr_y)   = {r_accy:+.3f}  Spearman={sp_accy:+.3f}")
print(f"  → If these differ from F1-based, the metric choice matters (U-shape vs monotone)")

# Fisher exact tests — separate for each regression target
print(f"\n── Fisher tests: clf-hard vs each regression target ─────────────────")

def _fisher_row(clf_h, reg_h, label):
    ct_ = np.array([
        [(~clf_h & ~reg_h).sum(), (~clf_h &  reg_h).sum()],
        [( clf_h & ~reg_h).sum(), ( clf_h &  reg_h).sum()],
    ])
    or_, p_ = stats.fisher_exact(ct_)
    nb = (clf_h & reg_h).sum()
    exp = clf_h.mean() * reg_h.mean() * n
    print(f"  [{label}]  reg-hard={reg_h.sum()} ({100*reg_h.mean():.1f}%)"
          f"  |  both: obs={nb} ({100*nb/n:.1f}%)  exp={exp:.0f} ({100*reg_h.mean()*clf_h.mean():.1f}%)"
          f"  |  OR={or_:.2f}  p={p_:.4f}")
    return or_, p_, nb

_fisher_row(clf_hard, reg_hard_E, "|ε_Enu| Q75")
_fisher_row(clf_hard, reg_hard_y, "|ε_y|   Q75")

# Keep union Fisher for downstream partial-correlation code and JSON
ct = np.array([
    [(~clf_hard & ~reg_hard).sum(), (~clf_hard &  reg_hard).sum()],
    [( clf_hard & ~reg_hard).sum(), ( clf_hard &  reg_hard).sum()],
])
odds, p_fisher = stats.fisher_exact(ct)
n_both = (clf_hard & reg_hard).sum()
p_both_obs = n_both / n
p_both_exp = p_clf_hard * p_reg_hard
interpretation = (
    "INDEPENDENT — no significant association between classifier and regressor failure modes."
    if p_fisher >= 0.05 else
    ("OVERLAP" if odds > 1 else "COMPLEMENTARY")
)

# Overlap between the two regressor-hard subsets
print(f"\n── Overlap: reg-hard-Enu vs reg-hard-y (Venn components) ───────────")
n_E_only   = ( reg_hard_E & ~reg_hard_y).sum()
n_y_only   = (~reg_hard_E &  reg_hard_y).sum()
n_both_reg = ( reg_hard_E &  reg_hard_y).sum()
n_neither  = (~reg_hard_E & ~reg_hard_y).sum()
ct_reg = np.array([[n_neither, n_y_only], [n_E_only, n_both_reg]])
or_reg, p_reg = stats.fisher_exact(ct_reg)
print(f"  Enu-hard only   : {n_E_only}  ({100*n_E_only/n:.1f}%)")
print(f"  y-hard only     : {n_y_only}  ({100*n_y_only/n:.1f}%)")
print(f"  Both hard       : {n_both_reg}  ({100*n_both_reg/n:.1f}%)")
print(f"  Neither hard    : {n_neither}  ({100*n_neither/n:.1f}%)")
print(f"  Fisher Enu-hard × y-hard: OR={or_reg:.2f}  p={p_reg:.4f}")
print(f"  (OR>1 → events hard for Enu are disproportionately also hard for y)")
n_triple    = ( clf_hard &  reg_hard_E &  reg_hard_y).sum()
n_none      = (~clf_hard & ~reg_hard_E & ~reg_hard_y).sum()
print(f"\n  3-way Venn regions (clf / Enu / y):")
print(f"  clf ∩ Enu ∩ y (triple) : {n_triple}  ({100*n_triple/n:.1f}%)")
print(f"  Outside all three      : {n_none}   ({100*n_none/n:.1f}%)")
print(f"\nInterpretation (union): {interpretation}")

# ── Venn diagram — area-proportional circle positioning ──────────────────────
#   Circle radii are equal (all hard sets = 25%).
#   Centre-to-centre distances are chosen so the geometric intersection area of
#   each circle pair matches the observed pairwise overlap fraction.
_n_clf_only = int(( clf_hard & ~reg_hard_E & ~reg_hard_y).sum())
_n_Enu_only = int((~clf_hard &  reg_hard_E & ~reg_hard_y).sum())
_n_y_only   = int((~clf_hard & ~reg_hard_E &  reg_hard_y).sum())
_n_clf_Enu  = int(( clf_hard &  reg_hard_E & ~reg_hard_y).sum())
_n_clf_y    = int(( clf_hard & ~reg_hard_E &  reg_hard_y).sum())
_n_Enu_y    = int((~clf_hard &  reg_hard_E &  reg_hard_y).sum())
_n_tri      = int(n_triple)
_n_out      = int(n_none)
_n_set      = int(clf_hard.sum())   # = 1769

def _ifrac(dr):
    """Intersection area of two equal circles / (pi*R^2) as a function of d/R."""
    x = dr / 2
    if x >= 1.0: return 0.0
    if x <= 0.0: return 1.0
    return (2/np.pi)*np.arccos(x) - (dr/np.pi)*np.sqrt(1 - x**2)

def _solve_d(n_inter):
    """d/R such that geometric intersection = n_inter / n_set * pi*R^2."""
    f = n_inter / _n_set
    if f <= 0: return 2.0
    if f >= 1: return 0.0
    return _brentq(lambda x: _ifrac(x) - f, 1e-9, 2.0 - 1e-9)

# Pairwise distances (R=1 normalised); smaller d → circles closer → larger overlap
_dAB = _solve_d(_n_clf_Enu + _n_tri)   # clf–Enu  (anti-correlated, small overlap)
_dAC = _solve_d(_n_clf_y   + _n_tri)   # clf–y    (positive, large overlap → smallest d)
_dBC = _solve_d(_n_Enu_y   + _n_tri)   # Enu–y

# Triangle: A=clf at origin, B=Enu along +x, C=y below the AB line
_Ax, _Ay = 0.0, 0.0
_Bx, _By = _dAB, 0.0
_Cx = (_dAC**2 - _dBC**2 + _dAB**2) / (2 * _dAB)
_Cy = -np.sqrt(max(_dAC**2 - _Cx**2, 0.0))

# Centre on origin then scale to display radius R_disp
_mx = (_Ax + _Bx + _Cx) / 3;  _my = (_Ay + _By + _Cy) / 3
_Ax -= _mx;  _Ay -= _my
_Bx -= _mx;  _By -= _my
_Cx -= _mx;  _Cy -= _my
R_disp = 1.3
_Ax *= R_disp;  _Ay *= R_disp
_Bx *= R_disp;  _By *= R_disp
_Cx *= R_disp;  _Cy *= R_disp

_vA = np.array([_Ax, _Ay])
_vB = np.array([_Bx, _By])
_vC = np.array([_Cx, _Cy])

def _push(ctr, away, pull):
    """Shift point 'ctr' away from 'away' by pull * R_disp."""
    d = ctr - away;  nd = np.linalg.norm(d)
    return ctr + (d / nd if nd > 1e-9 else np.array([0., 1.])) * pull * R_disp

# Region label positions: centroid of owning circles, pushed away from excluded ones
_p_clf = _push(_vA,               (_vB + _vC) / 2, 0.48)
_p_Enu = _push(_vB,               (_vA + _vC) / 2, 0.48)
_p_y   = _push(_vC,               (_vA + _vB) / 2, 0.48)
_p_cE  = _push((_vA + _vB) / 2,  _vC,             0.28)
_p_cy  = _push((_vA + _vC) / 2,  _vB,             0.28)
_p_Ey  = _push((_vB + _vC) / 2,  _vA,             0.28)
_p_tri = (_vA + _vB + _vC) / 3

# Circle-label positions: outside the circles
_loff  = (R_disp + 0.55) / R_disp
_l_clf = _push(_vA, (_vB + _vC) / 2, _loff)
_l_Enu = _push(_vB, (_vA + _vC) / 2, _loff)

# ── Draw ─────────────────────────────────────────────────────────────────────
fig_v, ax_v = plt.subplots(figsize=(5.5, 4.8))
ax_v.set_aspect("equal")
ax_v.axis("off")

for (_cx, _cy, _col) in [(_Ax, _Ay, C1), (_Bx, _By, C2), (_Cx, _Cy, C3)]:
    ax_v.add_patch(mpatches.Circle((_cx, _cy), R_disp, alpha=0.07, color=_col, zorder=1))
    ax_v.add_patch(mpatches.Circle((_cx, _cy), R_disp, fill=False, edgecolor=_col, lw=0.6, zorder=2))

_l_y_new = np.array([_Cx - R_disp - 0.7, _Cy - 1.1])
for (_lp, _lab, _col, _ha) in [
    (_l_clf + np.array([-0.18, 0.0]), "clf-hard\n(low $F_{1,\\mathrm{pEM}}$)", C1, "center"),
    (_l_Enu,                          "$|\\varepsilon_{E_\\nu}|$-hard",         C2, "center"),
    (_l_y_new,                        "$|\\varepsilon_y|$-hard",                C3, "left"),
]:
    ax_v.text(_lp[0], _lp[1], _lab, ha=_ha, va="center",
              fontsize=9.5, color=_col, fontweight="bold")

def _vt(pos, count, pct, fs=9.5, fw="normal"):
    ax_v.text(pos[0], pos[1], f"{count}\n({pct:.1f}%)",
              ha="center", va="center", fontsize=fs, color="#1a1a2e", zorder=5,
              fontweight=fw)

_vt(_p_clf + np.array([ 0.10,  0.10]), _n_clf_only, 100*_n_clf_only/n)
_vt(_p_Enu + np.array([-0.10,  0.00]), _n_Enu_only, 100*_n_Enu_only/n)
_vt(_p_y   + np.array([ 0.12, -0.06]), _n_y_only,   100*_n_y_only/n)
_vt(_p_cE  + np.array([-0.15,  0.08]), _n_clf_Enu,  100*_n_clf_Enu/n, fs=9)
_vt(_p_cy,                            _n_clf_y,    100*_n_clf_y/n,   fs=9, fw="bold")
_vt(_p_Ey  + np.array([-0.03, -0.18]), _n_Enu_y,   100*_n_Enu_y/n,   fs=9)
_vt(_p_tri + np.array([ 0.17, -0.06]), _n_tri,      100*_n_tri/n,     fs=8.5)

# Dynamic axis limits that include all label positions
_all_x = [_Ax, _Bx, _Cx, _l_clf[0], _l_Enu[0], _l_y_new[0]]
_all_y = [_Ay, _By, _Cy, _l_clf[1], _l_Enu[1], _l_y_new[1], _Cy - R_disp]
_pad   = 0.55
_xl = (min(_all_x) - _pad, max(_all_x) + _pad)
_yl = (min(_all_y) - _pad, max(_all_y) + _pad + 0.3)
ax_v.set_xlim(*_xl)
ax_v.set_ylim(*_yl)

ax_v.text(_Cx + R_disp + 0.2, _Cy - 0.7,
          f"$\\varnothing$ = {_n_out}  ({100*_n_out/n:.1f}%)",
          ha="left", va="center", fontsize=9, color="#1a1a2e")
# ax_v.set_title(rf"Failure-mode overlap  ($n={n:,}$, each hard set $=25\%$)",
#                fontsize=10, pad=8, color=C1)

out_venn = figures_path / "fig1_venn.jpg"
fig_v.savefig(out_venn, dpi=350, bbox_inches="tight")
plt.close(fig_v)
print(f"\nSaved: {out_venn}")


# ══════════════════════════════════════════════════════════════════════════════
# 5a. y-distribution per Venn region — verifies whether clf∩y overlap
#     is driven by high-y events (as claimed in text).
# ══════════════════════════════════════════════════════════════════════════════
_venn_regions = {
    "clf only":        ( clf_hard & ~reg_hard_E & ~reg_hard_y),
    "Enu only":        (~clf_hard &  reg_hard_E & ~reg_hard_y),
    "y only":          (~clf_hard & ~reg_hard_E &  reg_hard_y),
    "clf ∩ Enu":       ( clf_hard &  reg_hard_E & ~reg_hard_y),
    "clf ∩ y":         ( clf_hard & ~reg_hard_E &  reg_hard_y),
    "Enu ∩ y":         (~clf_hard &  reg_hard_E &  reg_hard_y),
    "clf ∩ Enu ∩ y":   ( clf_hard &  reg_hard_E &  reg_hard_y),
    "none":            (~clf_hard & ~reg_hard_E & ~reg_hard_y),
}

print("\n── y-distribution per Venn region ───────────────────────────────────")
print(f"  {'Region':<20}  {'n':>5}  {'mean y':>7}  {'median y':>9}  {'Q25':>6}  {'Q75':>6}")
for label, mask in _venn_regions.items():
    ys = yv[mask]
    if len(ys) == 0:
        continue
    print(f"  {label:<20}  {len(ys):>5}  {ys.mean():>7.3f}  {np.median(ys):>9.3f}  "
          f"{np.percentile(ys,25):>6.3f}  {np.percentile(ys,75):>6.3f}")

# Boxplot of y per region (key regions only)
_key_sets = {
    "clf-hard":  clf_hard,
    "clf $\\cap$ $y$-hard": ( clf_hard & reg_hard_y),
    "y-hard":    reg_hard_y,
    "Enu-hard":  reg_hard_E,
    "none":      _venn_regions["none"],
}
_key_data = [yv[mask] for mask in _key_sets.values()]

fig_ybox, ax_ybox = plt.subplots(figsize=(7, 4), constrained_layout=True)
bp = ax_ybox.boxplot(_key_data, patch_artist=True, medianprops=dict(color="black", lw=2),
                     whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
                     flierprops=dict(marker=".", ms=3, alpha=0.3))
_box_colors = [C1, C3, C2, "purple", "#aaaaaa"]
for patch, color in zip(bp["boxes"], _box_colors):
    patch.set_facecolor(color); patch.set_alpha(0.55)
ax_ybox.set_xticks(range(1, len(_key_sets) + 1))
ax_ybox.set_xticklabels([f"{r}\n(n={int(mask.sum())})"
                          for r, mask in _key_sets.items()], fontsize=9)
ax_ybox.set_ylabel(r"True inelasticity $y$")
ax_ybox.set_ylim(0, 1)
ax_ybox.spines[["top", "right"]].set_visible(False)
ax_ybox.axhline(np.median(yv), color="grey", ls=":", lw=1.2, alpha=0.6,
                label=f"overall median $y$ = {np.median(yv):.2f}")
ax_ybox.legend(frameon=False, fontsize=9)
out_ybox = figures_path / "fig2_venn_y_distribution.jpg"
fig_ybox.savefig(out_ybox, dpi=350, bbox_inches="tight")
plt.close(fig_ybox)
print(f"\nSaved: {out_ybox}")


# ══════════════════════════════════════════════════════════════════════════════
# 5b. Partial correlations — controlling for y and log(E_nu)
#     Removes the shared kinematic confound: both clf and reg errors rise at
#     low y and low E_nu, so raw r may overstate the true relationship.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Partial correlations (controlling for y, log E_nu) ───────────────")


def partial_pearsonr(x, y, controls, rng=None, n_boot=N_BOOTSTRAP):
    """r(x, y | controls) via OLS residuals, with bootstrap 95% CI."""
    Z = np.column_stack([np.ones(len(x)), controls])
    rx = x - Z @ _lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ _lstsq(Z, y, rcond=None)[0]
    r_obs = np.corrcoef(rx, ry)[0, 1]
    if rng is None:
        rng = np.random.default_rng(0)
    idx  = np.arange(len(rx))
    boot = np.array([np.corrcoef(rx[s := rng.choice(idx, len(idx), replace=True)],
                                  ry[s])[0, 1] for _ in range(n_boot)])
    return r_obs, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

log_Enu_v  = np.log10(np.maximum(reg_Enu_true[valid], 1e-9))
controls   = np.column_stack([yv, log_Enu_v])

rng2 = np.random.default_rng(99)
pr_E,  pr_E_lo,  pr_E_hi  = partial_pearsonr(ce, ae_E, controls, rng=rng2)
pr_y,  pr_y_lo,  pr_y_hi  = partial_pearsonr(ce, ae_y, controls, rng=rng2)
pr_EY, pr_EY_lo, pr_EY_hi = partial_pearsonr(ae_E, ae_y, controls, rng=rng2)

print(f"Partial r(clf_err, abserr_Enu | y, logE) = {pr_E:+.3f}  95% CI [{pr_E_lo:+.3f}, {pr_E_hi:+.3f}]")
print(f"Partial r(clf_err, abserr_y   | y, logE) = {pr_y:+.3f}  95% CI [{pr_y_lo:+.3f}, {pr_y_hi:+.3f}]")
print(f"Partial r(abserr_Enu, abserr_y| y, logE) = {pr_EY:+.3f}  95% CI [{pr_EY_lo:+.3f}, {pr_EY_hi:+.3f}]")
print("  → Compare with raw Pearson above: residual correlation after kinematic confound removed")

# Isolate y vs logE: single-variable controls to identify which drives the sign flip
rng2b = np.random.default_rng(77)
pr_E_y_only,    _, _ = partial_pearsonr(ce, ae_E, yv.reshape(-1, 1),          rng=rng2b)
pr_y_y_only,    _, _ = partial_pearsonr(ce, ae_y, yv.reshape(-1, 1),          rng=rng2b)
pr_E_logE_only, _, _ = partial_pearsonr(ce, ae_E, log_Enu_v.reshape(-1, 1),   rng=rng2b)
pr_y_logE_only, _, _ = partial_pearsonr(ce, ae_y, log_Enu_v.reshape(-1, 1),   rng=rng2b)
print(f"\n  Single-control attribution:")
print(f"  r(clf_err, abserr_Enu | y only)    = {pr_E_y_only:+.3f}  (raw was {r_E:+.3f})")
print(f"  r(clf_err, abserr_y   | y only)    = {pr_y_y_only:+.3f}  (raw was {r_y:+.3f})")
print(f"  r(clf_err, abserr_Enu | logE only) = {pr_E_logE_only:+.3f}  (raw was {r_E:+.3f})")
print(f"  r(clf_err, abserr_y   | logE only) = {pr_y_logE_only:+.3f}  (raw was {r_y:+.3f})")
print(f"  → whichever flips the sign is the dominant confound")


# ══════════════════════════════════════════════════════════════════════════════
# 5c. Stratified Fisher test — low-y (y<0.3) vs high-y (y≥0.3)
#     The classifier is known to degrade at low y; this checks whether the
#     failure-mode independence holds separately in each regime.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stratified Fisher test (y<0.3 vs y≥0.3) ─────────────────────────")

Y_SPLIT = 0.3
strata = {
    f"low-y  (y < {Y_SPLIT})":  yv <  Y_SPLIT,
    f"high-y (y ≥ {Y_SPLIT})":  yv >= Y_SPLIT,
}

for label, mask in strata.items():
    n_s = mask.sum()
    if n_s < 10:
        print(f"  {label}: too few events ({n_s}), skipping")
        continue
    ch   = clf_hard[mask]
    rh   = reg_hard[mask]
    ct_s = np.array([
        [(~ch & ~rh).sum(), (~ch &  rh).sum()],
        [( ch & ~rh).sum(), ( ch &  rh).sum()],
    ])
    or_s, p_s = stats.fisher_exact(ct_s)
    r_s_E = np.corrcoef(ce[mask], ae_E[mask])[0, 1]
    r_s_y = np.corrcoef(ce[mask], ae_y[mask])[0, 1]
    print(f"  {label}  (n={n_s})")
    print(f"    Fisher OR={or_s:.2f}, p={p_s:.4f}  |  "
          f"r(clf,Enu)={r_s_E:+.3f}  r(clf,y)={r_s_y:+.3f}")
    print(f"    clf-hard={ch.sum()}  reg-hard={rh.sum()}  both={( ch &  rh).sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5d. Continuous analysis outside the worst tail
#     Fisher and quantile thresholds describe the tails. Here we ask whether
#     the correlation structure is uniform across the distribution or driven
#     by the extremes: compute correlations for events BELOW Q75 clf error
#     and BELOW Q75 reg error (the "easy" majority of events).
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Correlation structure outside the hard tail ───────────────────────")

tail_masks = {
    "all events":                       np.ones(n_valid, dtype=bool),
    "clf NOT hard (bottom 75% clf err)": ~clf_hard,
    "reg NOT hard (bottom 75% reg err)": ~reg_hard,
    "neither hard":                     (~clf_hard) & (~reg_hard),
}

for label, mask in tail_masks.items():
    n_m = mask.sum()
    if n_m < 10:
        continue
    r_E_m  = np.corrcoef(ce[mask], ae_E[mask])[0, 1]
    r_y_m  = np.corrcoef(ce[mask], ae_y[mask])[0, 1]
    r_EY_m = np.corrcoef(ae_E[mask], ae_y[mask])[0, 1]
    print(f"  [{label}]  n={n_m}")
    print(f"    r(clf,Enu)={r_E_m:+.3f}  r(clf,y)={r_y_m:+.3f}  r(Enu,y)={r_EY_m:+.3f}")

# Quantile sweep: how does r(clf_err, abserr_Enu) change as we exclude more of the tail?
print("\n  Quantile sweep — r(clf_err, abserr_Enu) excluding top-q% of clf_err:")
for q in [0.50, 0.60, 0.75, 0.80, 0.90, 1.00]:
    thr = np.quantile(ce, q)
    m   = ce <= thr
    r_q = np.corrcoef(ce[m], ae_E[m])[0, 1] if m.sum() > 2 else np.nan
    print(f"    q≤{q:.0%} (n={m.sum():5d}): r={r_q:+.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5e. N_nodes as mediator of shared difficulty
#     We know N_nodes is the strongest predictor of regressor error from the
#     feature-error correlation analysis. If controlling for N_nodes collapses
#     the partial r(clf_err, abserr_Enu | y, logE) = +0.20, the shared
#     difficulty is entirely explained by event size, not any deeper coupling.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── N_nodes as mediator of shared difficulty ─────────────────────────")

n_nodes_all = np.array([float(d.num_nodes) for d in val_dataset]) if val_dataset is not None else None
n_nodes_v   = n_nodes_all[valid] if n_nodes_all is not None else None

if n_nodes_v is None:
    print("  Skipped — val_dataset not loaded (cache run). Re-run without cache to compute.")
else:
    log_nn_v    = np.log1p(n_nodes_v)   # log(1 + N_nodes) — stabilises skewed count distribution

    controls_nn = np.column_stack([yv, log_Enu_v, log_nn_v])

    rng3 = np.random.default_rng(7)
    pr_E_nn,  pr_E_nn_lo,  pr_E_nn_hi  = partial_pearsonr(ce, ae_E, controls_nn, rng=rng3)
    pr_y_nn,  pr_y_nn_lo,  pr_y_nn_hi  = partial_pearsonr(ce, ae_y, controls_nn, rng=rng3)
    pr_EY_nn, pr_EY_nn_lo, pr_EY_nn_hi = partial_pearsonr(ae_E, ae_y, controls_nn, rng=rng3)

    r_clf_nn, _, _  = partial_pearsonr(ce,   log_nn_v, np.column_stack([yv, log_Enu_v]), rng=rng3)
    r_reg_nn, _, _  = partial_pearsonr(ae_E, log_nn_v, np.column_stack([yv, log_Enu_v]), rng=rng3)

    print(f"Partial r(clf_err,   log N_nodes | y, logE) = {r_clf_nn:+.3f}")
    print(f"Partial r(abserr_Enu, log N_nodes | y, logE) = {r_reg_nn:+.3f}")
    print()
    print(f"Partial r(clf_err, abserr_Enu | y, logE, logN) = {pr_E_nn:+.3f}  95% CI [{pr_E_nn_lo:+.3f}, {pr_E_nn_hi:+.3f}]")
    print(f"Partial r(clf_err, abserr_y   | y, logE, logN) = {pr_y_nn:+.3f}  95% CI [{pr_y_nn_lo:+.3f}, {pr_y_nn_hi:+.3f}]")
    print(f"Partial r(abserr_Enu, abserr_y| y, logE, logN) = {pr_EY_nn:+.3f}  95% CI [{pr_EY_nn_lo:+.3f}, {pr_EY_nn_hi:+.3f}]")
    print(f"  → Was +0.20 / -0.10 / +0.07 before adding N_nodes control")
    print(f"  → If these collapse toward 0, N_nodes is the shared difficulty factor")


# ══════════════════════════════════════════════════════════════════════════════
# 5f. G_{+cl} vs G_{bl} on classifier-hard events
#     The key question: on events where the classifier performs worst, does
#     G_{+cl} (which uses classifier probs as node features) do better or worse
#     than G_{bl} (baseline, no classifier input)?
#     If G_{+cl} is worse than G_{bl} on classifier-hard events → the bad probs
#     are actively hurting the regressor on the hardest events.
#     If G_{+cl} is better or equal → it is robust to classifier errors.
# ══════════════════════════════════════════════════════════════════════════════
if REGRESSOR_PLUS_CLF_DIR is not None:
    print("\n── G_{{+cl}} vs G_{{bl}} on classifier-hard events ──────────────────────")

    plus_weights = weights_path / REGRESSOR_PLUS_CLF_DIR
    plus_ckpt    = torch.load(plus_weights / "best_model.pt", map_location=device, weights_only=False)
    plus_norm    = plus_ckpt.get("norm_stats", None)
    plus_beta    = plus_ckpt.get("beta_loss", False)
    plus_pool    = "sum" if "_sum_" in REGRESSOR_PLUS_CLF_DIR else "mean"

    wd2 = REGRESSOR_PLUS_CLF_DIR
    if   "_truth4"      in wd2: idim2 = 5
    elif "_truth3bprob" in wd2: idim2 = 4
    elif "_truth3b"     in wd2: idim2 = 4
    elif "_truth3"      in wd2: idim2 = 4
    elif "_truth2"      in wd2: idim2 = 3
    elif "_binaryprob"  in wd2: idim2 = 3
    elif "_prob"        in wd2: idim2 = 5
    else:                        idim2 = 1

    plus_model = NeutrinoGravNetRegressionFASER(
        input_dim=idim2, num_targets=2, faser_dim=5,
        pooling=plus_pool, beta_loss=plus_beta, use_faser=False,
    ).to(device)
    plus_model.load_state_dict(plus_ckpt["model_state_dict"])
    plus_model.eval()
    print(f"G_{{+cl}} epoch {plus_ckpt['epoch'] + 1}, input_dim={idim2}")

    plus_cache_key  = f"run{RUN}_n{N_VAL_EVENTS}_ep{plus_ckpt['epoch']}_vl{plus_ckpt['val_loss']:.6f}"
    plus_cache_path = plus_weights / f"inference_cache_{plus_cache_key}.npz"

    if plus_cache_path.exists():
        print(f"Loading G_{{+cl}} cache: {plus_cache_path.name}")
        _pc = np.load(plus_cache_path)
        plus_preds_raw = torch.from_numpy(_pc["preds_raw"])
        plus_preds_lin = preds_to_physical(plus_preds_raw, plus_norm, beta_loss=plus_beta)
    else:
        if val_dataset is None:
            raise RuntimeError("val_dataset not loaded — re-run without cache to generate G_{+cl} inference.")
        # Augment node features with classifier softmax probs — batched for speed
        plus_val = list(val_dataset)
        if "_binaryprob" in wd2 or "_prob" in wd2:
            print("Augmenting node features with classifier probs (batched)...")
            _aug_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            _all_probs  = []
            with torch.no_grad():
                for _batch in _aug_loader:
                    _batch = _batch.to(device)
                    _out   = clf_model(_batch.x, _batch.pos, _batch.batch, _batch.x_faser)
                    _probs = torch.softmax(_out, dim=1).cpu()
                    _bc    = _batch.batch.cpu()
                    for _g in range(int(_bc.max().item()) + 1):
                        _all_probs.append(_probs[_bc == _g])
            for _data, _prob in zip(plus_val, _all_probs):
                _data.x = torch.cat([_data.x, _prob], dim=1)

        plus_loader   = DataLoader(plus_val, batch_size=32, shuffle=False)
        all_plus      = []
        with torch.no_grad():
            for data in plus_loader:
                data = data.to(device)
                raw  = plus_model(data.x, data.pos, data.batch, data.x_faser)
                all_plus.append(raw.cpu())
        plus_preds_raw = torch.cat(all_plus)
        plus_preds_lin = preds_to_physical(plus_preds_raw, plus_norm, beta_loss=plus_beta)
        np.savez_compressed(plus_cache_path, preds_raw=plus_preds_raw.numpy())
        print(f"Saved G_{{+cl}} cache: {plus_cache_path.name}")

    plus_Enu_pred     = plus_preds_lin[:, 0]
    plus_Eroe_pred    = plus_preds_lin[:, 2]
    plus_y_pred_arr   = plus_Eroe_pred / np.maximum(plus_Enu_pred, 1e-9)
    plus_abserr       = np.abs((plus_Enu_pred - reg_Enu_true) / np.maximum(reg_Enu_true, 1e-9))
    plus_abserr_y_arr = np.abs(plus_y_pred_arr - y_true)
    plus_abserr_v     = plus_abserr[valid]
    plus_abserr_y_v   = plus_abserr_y_arr[valid]

    # ── Load G_{truth}: oracle ceiling — truth node labels instead of clf probs ─
    truth_abserr_v   = None
    truth_abserr_y_v = None
    if REGRESSOR_TRUTH_DIR is not None:
        truth_weights    = weights_path / REGRESSOR_TRUTH_DIR
        truth_ckpt       = torch.load(truth_weights / "best_model.pt", map_location=device, weights_only=False)
        truth_norm       = truth_ckpt.get("norm_stats", None)
        truth_beta       = truth_ckpt.get("beta_loss", False)
        truth_pool       = "sum" if "_sum_" in REGRESSOR_TRUTH_DIR else "mean"
        truth_model      = NeutrinoGravNetRegressionFASER(
            input_dim=3, num_targets=2, faser_dim=5,
            pooling=truth_pool, beta_loss=truth_beta, use_faser=False,
        ).to(device)
        truth_model.load_state_dict(truth_ckpt["model_state_dict"])
        truth_model.eval()
        print(f"G_{{truth}} epoch {truth_ckpt['epoch'] + 1}, input_dim=3")

        truth_cache_key  = f"run{RUN}_n{N_VAL_EVENTS}_ep{truth_ckpt['epoch']}_vl{truth_ckpt['val_loss']:.6f}"
        truth_cache_path = truth_weights / f"inference_cache_{truth_cache_key}.npz"

        if truth_cache_path.exists():
            print(f"Loading G_{{truth}} cache: {truth_cache_path.name}")
            _tc             = np.load(truth_cache_path)
            truth_preds_raw = torch.from_numpy(_tc["preds_raw"])
            truth_preds_lin = preds_to_physical(truth_preds_raw, truth_norm, beta_loss=truth_beta)
        else:
            if val_dataset is None:
                raise RuntimeError("val_dataset not loaded — re-run without cache to generate G_{truth} inference.")
            print("Augmenting node features with binary truth labels (not_primary_EM / primary_EM_e)...")
            _remap     = torch.tensor([0, 0, 1, 0])
            truth_val  = [d.clone() for d in val_dataset]
            for d in truth_val:
                binary_label = _remap[d.pdg_label]
                y_oh         = F.one_hot(binary_label, num_classes=2).float()
                d.x          = torch.cat([d.x, y_oh], dim=1)
            truth_loader    = DataLoader(truth_val, batch_size=32, shuffle=False)
            all_truth       = []
            with torch.no_grad():
                for data in truth_loader:
                    data = data.to(device)
                    raw  = truth_model(data.x, data.pos, data.batch, data.x_faser)
                    all_truth.append(raw.cpu())
            truth_preds_raw = torch.cat(all_truth)
            truth_preds_lin = preds_to_physical(truth_preds_raw, truth_norm, beta_loss=truth_beta)
            np.savez_compressed(truth_cache_path, preds_raw=truth_preds_raw.numpy())
            print(f"Saved G_{{truth}} cache: {truth_cache_path.name}")

        truth_Enu_pred    = np.asarray(truth_preds_lin[:, 0])
        truth_Eroe_pred   = np.asarray(truth_preds_lin[:, 2])
        truth_y_pred_arr  = truth_Eroe_pred / np.maximum(truth_Enu_pred, 1e-9)
        truth_abserr      = np.abs((truth_Enu_pred - reg_Enu_true) / np.maximum(reg_Enu_true, 1e-9))
        truth_abserr_y_arr= np.abs(truth_y_pred_arr - y_true)
        truth_abserr_v    = truth_abserr[valid]
        truth_abserr_y_v  = truth_abserr_y_arr[valid]

    # Compare on all events and on classifier-hard / classifier-easy events
    for label, mask in [("all events",   np.ones(n_valid, dtype=bool)),
                         ("clf-hard",     clf_hard),
                         ("clf-easy",    ~clf_hard)]:
        n_m          = mask.sum()
        med_bl       = np.median(ae_E[mask])
        med_plus     = np.median(plus_abserr_v[mask])
        med_bl_y     = np.median(ae_y[mask])
        med_plus_y   = np.median(plus_abserr_y_v[mask])
        truth_str    = ""
        if truth_abserr_v is not None:
            truth_str = (f"  G_{{truth}}={np.median(truth_abserr_v[mask]):.4f}"
                         f"  |  y: G_{{truth}}={np.median(truth_abserr_y_v[mask]):.4f}")
        print(f"  [{label}]  n={n_m}"
              f"  Enu: G_bl={med_bl:.4f}  G_{{+cl}}={med_plus:.4f}{truth_str}"
              f"  |  y: G_bl={med_bl_y:.4f}  G_{{+cl}}={med_plus_y:.4f}")

    # All-quartile breakdown: Q1 (worst F1) through Q4 (best F1)
    q_edges = np.quantile(ce, [0.25, 0.50, 0.75])
    q_labels = ["Q4 (best  F1, ce≤25%)",
                "Q3 (mid-high F1, ce 25–50%)",
                "Q2 (mid-low  F1, ce 50–75%)",
                "Q1 (worst F1, ce≥75%)"]
    q_masks = [
        ce <  q_edges[0],
        (ce >= q_edges[0]) & (ce < q_edges[1]),
        (ce >= q_edges[1]) & (ce < q_edges[2]),
        ce >= q_edges[2],
    ]
    print(f"\n  All-quartile breakdown (n≈{n_valid//4} each):")
    for label, mask in zip(q_labels, q_masks):
        n_m          = mask.sum()
        med_bl       = np.median(ae_E[mask])
        med_plus     = np.median(plus_abserr_v[mask])
        med_bl_y     = np.median(ae_y[mask])
        med_plus_y   = np.median(plus_abserr_y_v[mask])
        mean_y_q     = yv[mask].mean()
        truth_str    = ""
        if truth_abserr_v is not None:
            truth_str = (f"  G_{{truth}}={np.median(truth_abserr_v[mask]):.4f}"
                         f"  |  y: G_{{truth}}={np.median(truth_abserr_y_v[mask]):.4f}")
        print(f"  [{label}]  n={n_m}  mean_y={mean_y_q:.2f}"
              f"  Enu: G_bl={med_bl:.4f}  G_{{+cl}}={med_plus:.4f}{truth_str}"
              f"  |  y: G_bl={med_bl_y:.4f}  G_{{+cl}}={med_plus_y:.4f}")

    N_C_BINS = 12

    def _bin_median_se(vals, idx, n_bins):
        """Median and IQR-based SE of the median for each bin."""
        med = np.full(n_bins, np.nan)
        se  = np.full(n_bins, np.nan)
        for k in range(n_bins):
            v = vals[idx == k]
            n = len(v)
            if n > 0:
                med[k] = np.median(v)
            if n > 1:
                iqr = np.percentile(v, 75) - np.percentile(v, 25)
                se[k] = 1.35 * iqr / (2.0 * np.sqrt(n))
        return med, se

    yc_edges    = np.percentile(yv, np.linspace(0, 100, N_C_BINS + 1))
    yc_edges[-1] += 1e-9
    yc_idx      = np.clip(np.digitize(yv, yc_edges) - 1, 0, N_C_BINS - 1)
    yc_centres  = 0.5 * (yc_edges[:-1] + yc_edges[1:])
    med_bl_y_yc,   se_bl_y_yc   = _bin_median_se(ae_y,             yc_idx, N_C_BINS)
    med_plus_y_yc, se_plus_y_yc = _bin_median_se(plus_abserr_y_v, yc_idx, N_C_BINS)
    med_bl_E_yc,   se_bl_E_yc   = _bin_median_se(ae_E,             yc_idx, N_C_BINS)
    med_plus_E_yc, se_plus_E_yc = _bin_median_se(plus_abserr_v,   yc_idx, N_C_BINS)
    med_truth_y_yc = se_truth_y_yc = med_truth_E_yc = se_truth_E_yc = None
    if truth_abserr_v is not None:
        med_truth_y_yc, se_truth_y_yc = _bin_median_se(truth_abserr_y_v, yc_idx, N_C_BINS)
        med_truth_E_yc, se_truth_E_yc = _bin_median_se(truth_abserr_v,   yc_idx, N_C_BINS)

    # Right column: bin by per-event mean cross-entropy (CE) rather than F1.
    # Equal-count bins (same n events per bin) so all medians are equally reliable;
    # geometric mean centres are correct for the log x-axis.
    cec_edges   = np.percentile(cev, np.linspace(0, 100, N_C_BINS + 1))
    cec_edges[-1] += 1e-9
    cec_idx     = np.clip(np.digitize(cev, cec_edges) - 1, 0, N_C_BINS - 1)
    _ce_lo      = max(cec_edges[0], 1e-6)
    _ce_hi      = cec_edges[-1]
    cec_centres = np.sqrt(np.maximum(cec_edges[:-1], 1e-6) * cec_edges[1:])   # geometric mean
    med_bl_y_cec,   se_bl_y_cec   = _bin_median_se(ae_y,             cec_idx, N_C_BINS)
    med_plus_y_cec, se_plus_y_cec = _bin_median_se(plus_abserr_y_v, cec_idx, N_C_BINS)
    med_bl_E_cec,   se_bl_E_cec   = _bin_median_se(ae_E,             cec_idx, N_C_BINS)
    med_plus_E_cec, se_plus_E_cec = _bin_median_se(plus_abserr_v,   cec_idx, N_C_BINS)
    med_truth_y_cec = se_truth_y_cec = med_truth_E_cec = se_truth_E_cec = None
    if truth_abserr_v is not None:
        med_truth_y_cec, se_truth_y_cec = _bin_median_se(truth_abserr_y_v, cec_idx, N_C_BINS)
        med_truth_E_cec, se_truth_E_cec = _bin_median_se(truth_abserr_v,   cec_idx, N_C_BINS)

    def _insert_crossovers(x, y_bl, y_plus):
        delta = y_plus - y_bl
        xi, yb, yp = list(x), list(y_bl), list(y_plus)
        offset = 0
        for i in range(len(x) - 1):
            if delta[i] * delta[i + 1] < 0:
                j = i + offset + 1
                x_cross = x[i] - delta[i] * (x[i + 1] - x[i]) / (delta[i + 1] - delta[i])
                t = (x_cross - x[i]) / (x[i + 1] - x[i])
                y_cross = float(y_bl[i] + t * (y_bl[i + 1] - y_bl[i]))
                xi.insert(j, x_cross)
                yb.insert(j, y_cross)
                yp.insert(j, y_cross)
                offset += 1
        return np.array(xi), np.array(yb), np.array(yp)

    C_GREEN = "#4caf50"
    C_GREY  = "#555555"

    fig10c, axc = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    fig10c.get_layout_engine().set(w_pad=0.05, h_pad=0.05)

    _panels_c = [
        # (ax, x_centres, y_bl, y_plus, y_truth, se_bl, se_plus, se_truth, scale, x_label, y_label, add_legend)
        (axc[0, 0], yc_centres,  med_bl_y_yc,  med_plus_y_yc,  med_truth_y_yc,  se_bl_y_yc,   se_plus_y_yc,  se_truth_y_yc,  1.0,
         r"True inelasticity $y$",         r"Median $|y_\mathrm{pred} - y_\mathrm{true}|$", False),
        (axc[0, 1], cec_centres, med_bl_y_cec, med_plus_y_cec, med_truth_y_cec, se_bl_y_cec,  se_plus_y_cec, se_truth_y_cec, 1.0,
         r"Mean cross-entropy per event",  r"Median $|y_\mathrm{pred} - y_\mathrm{true}|$", False),
        (axc[1, 0], yc_centres,  med_bl_E_yc,  med_plus_E_yc,  med_truth_E_yc,  se_bl_E_yc,   se_plus_E_yc,  se_truth_E_yc,  100.0,
         r"True inelasticity $y$",         r"Median $|\varepsilon_{E_\nu}|$ (%)",            True),
        (axc[1, 1], cec_centres, med_bl_E_cec, med_plus_E_cec, med_truth_E_cec, se_bl_E_cec,  se_plus_E_cec, se_truth_E_cec, 100.0,
         r"Mean cross-entropy per event",  r"Median $|\varepsilon_{E_\nu}|$ (%)",            False),
    ]

    _ce_x_range = (_ce_lo * 0.9, _ce_hi * 1.1)   # auto range for CE panels

    for ax, xc, yb_raw, yp_raw, yt_raw, se_bl_raw, se_plus_raw, se_truth_raw, scale, xlabel, ylabel, add_leg in _panels_c:
        yb    = scale * yb_raw
        yp    = scale * yp_raw
        yt    = (scale * yt_raw)       if yt_raw       is not None else None
        se_bl = scale * se_bl_raw
        se_pl = scale * se_plus_raw
        se_tr = (scale * se_truth_raw) if se_truth_raw is not None else None
        xi, ybi, ypi = _insert_crossovers(xc, yb, yp)
        ax.fill_between(xi, ybi, ypi, where=(ypi <= ybi), alpha=0.18, color=C_GREEN, lw=0)
        ax.fill_between(xi, ybi, ypi, where=(ypi >= ybi), alpha=0.15, color=C_GREY,  lw=0)
        ax.errorbar(xc, yb, yerr=se_bl, color=C1,       marker="o", ms=2.5, lw=1.2,
                    elinewidth=0.8, capsize=2, capthick=0.8, label=r"$\mathrm{G}_\text{bl}$")
        ax.errorbar(xc, yp, yerr=se_pl, color="#1565C0", marker="s", ms=2.5, lw=1.2, ls="--",
                    elinewidth=0.8, capsize=2, capthick=0.8, label=r"$\mathrm{G}_{+\text{cl}}$")
        if yt_raw is not None:
            ax.errorbar(xc, yt, yerr=se_tr, color=C_TRUTH, marker="^", ms=2.5, lw=1.2, ls=":",
                        elinewidth=0.8, capsize=2, capthick=0.8, label=r"$\mathrm{G}_\text{truth}$")
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(labelsize=13)
        # CE x-axis: log scale with plain decimal labels (range is narrow)
        if "entropy" in xlabel:
            ax.set_xscale("log")
            ax.set_xlim(*_ce_x_range)
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5], numticks=6))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        if add_leg:
            ax.legend(frameon=False, fontsize=17, loc="upper right")

    # Cap top-right y-axis so the last CE bin doesn't dominate — the crossover
    # structure in the main body of the distribution is the key result.
    _tr_top = np.nanmax(np.concatenate([med_bl_y_cec[:-1], med_plus_y_cec[:-1]])) * 1.5
    axc[0, 1].set_ylim(top=_tr_top)

    for row in range(2):
        for col in range(2):
            axc[row, col].yaxis.set_major_locator(mticker.MaxNLocator(4, prune="lower"))
            if col == 1:
                axc[row, col].set_ylabel("")
            if row == 0:
                axc[row, col].set_xlabel("")
                _fmt = mticker.ScalarFormatter(useMathText=True)
                _fmt.set_powerlimits((-2, -2))
                axc[row, col].yaxis.set_major_formatter(_fmt)
                axc[row, col].yaxis.get_offset_text().set_fontsize(12)

    out3 = figures_path / "fig3_combined_2x2.jpg"
    fig10c.savefig(out3, dpi=350, bbox_inches="tight")
    plt.close(fig10c)
    print(f"Saved: {out3}")

    def _panel_dict(x_centres, y_bl, y_plus, y_truth, x_name, y_name, y_unit):
        if y_truth is None:
            y_truth = np.full_like(y_bl, np.nan)
        delta_plus  = y_plus  - y_bl
        delta_truth = y_truth - y_bl
        return {
            "x_axis": x_name,
            "y_axis": y_name,
            "y_unit": y_unit,
            "note": "delta = model minus G_bl; negative delta means model is better than baseline",
            "bins": [
                {
                    "bin_centre":      round(float(x), 4),
                    "G_bl":            round(float(b), 5),
                    "G_+cl":           round(float(p), 5),
                    "G_truth":         round(float(t), 5),
                    "delta_+cl":       round(float(dp), 5),
                    "delta_truth":     round(float(dt), 5),
                    "G_+cl_better":    bool(dp < 0),
                    "G_truth_better":  bool(dt < 0),
                }
                for x, b, p, t, dp, dt in zip(x_centres, y_bl, y_plus, y_truth, delta_plus, delta_truth)
            ],
        }

    fig10c_data = {
        "description": (
            "Numerical results for fig3_combined_2x2. "
            "Each panel shows median error for G_bl (baseline GravNet), "
            "G_+cl (GravNet + classifier probs), and G_truth (GravNet + truth labels) "
            "binned by two variables. "
            f"N_bins={N_C_BINS} equal-count bins."
        ),
        "panels": {
            "top_left":     _panel_dict(yc_centres,  med_bl_y_yc,  med_plus_y_yc,  med_truth_y_yc,
                                        "true_inelasticity_y", "median_abs_y_error", "dimensionless"),
            "top_right":    _panel_dict(cec_centres, med_bl_y_cec, med_plus_y_cec, med_truth_y_cec,
                                        "mean_cross_entropy_per_event", "median_abs_y_error", "dimensionless"),
            "bottom_left":  _panel_dict(yc_centres,  100*med_bl_E_yc,  100*med_plus_E_yc,  100*med_truth_E_yc,
                                        "true_inelasticity_y", "median_rel_Enu_error", "percent"),
            "bottom_right": _panel_dict(cec_centres, 100*med_bl_E_cec, 100*med_plus_E_cec, 100*med_truth_E_cec,
                                        "mean_cross_entropy_per_event", "median_rel_Enu_error", "percent"),
        },
    }

    out3_json = figures_path / "fig3_combined_2x2_data.json"
    out3_json.write_text(_json.dumps(fig10c_data, indent=2))
    print(f"Saved: {out3_json}")
else:
    print("\n── G_{{+cl}} vs G_{{bl}} skipped — set REGRESSOR_PLUS_CLF_DIR to enable ──")


# ── Binning over y used by later figure sections ─────────────────────────────
N_BINS = 10
bin_edges   = np.linspace(yv.min(), yv.max(), N_BINS + 1)
bin_idx     = np.digitize(yv, bin_edges, right=False) - 1
bin_idx     = np.clip(bin_idx, 0, N_BINS - 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])


# ══════════════════════════════════════════════════════════════════════════════
# 6. Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("FAILURE MODE COMPARISON SUMMARY")
print("═" * 65)
print(f"  r(clf_err[F1],  abserr_Enu) = {r_E:+.3f}  [{ci_E_lo:+.3f}, {ci_E_hi:+.3f}]  (Spearman {sp_E:+.3f})")
print(f"  r(clf_err[F1],  abserr_y)   = {r_y:+.3f}  [{ci_y_lo:+.3f}, {ci_y_hi:+.3f}]  (Spearman {sp_y:+.3f})")
print(f"  r(clf_err[acc], abserr_Enu) = {r_accE:+.3f}  (Spearman {sp_accE:+.3f})")
print(f"  r(clf_err[acc], abserr_y)   = {r_accy:+.3f}  (Spearman {sp_accy:+.3f})")
print(f"  r(abserr_Enu, abserr_y)     = {r_EY:+.3f}  [{ci_EY_lo:+.3f}, {ci_EY_hi:+.3f}]")
print(f"  Observed overlap : {100*p_both_obs:.1f}%  |  Expected by chance : {100*p_both_exp:.1f}%")
print(f"  Fisher OR={odds:.2f}, p={p_fisher:.4f}")
print(f"\n  Verdict: {interpretation}")
print("═" * 65)

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})



# ══════════════════════════════════════════════════════════════════════════════
# 7. Save failure-mode stats to report_stats.json
# ══════════════════════════════════════════════════════════════════════════════

_fm_stats = {
    "meta": {
        "classifier_weights_dir": CLASSIFIER_WEIGHTS_DIR,
        "regressor_weights_dir":  REGRESSOR_WEIGHTS_DIR,
        "n_events":               int(n_valid),
        "timestamp":              _dt.datetime.now().isoformat(),
    },
    "pearson_r": {
        "clf_err_vs_abserr_Enu": {
            "r": float(r_E), "ci_lo": float(ci_E_lo), "ci_hi": float(ci_E_hi),
            "spearman": float(sp_E),
        },
        "clf_err_vs_abserr_y": {
            "r": float(r_y), "ci_lo": float(ci_y_lo), "ci_hi": float(ci_y_hi),
            "spearman": float(sp_y),
        },
        "abserr_Enu_vs_abserr_y": {
            "r": float(r_EY), "ci_lo": float(ci_EY_lo), "ci_hi": float(ci_EY_hi),
        },
    },
    "overlap": {
        "n_clf_hard":       int(n_clf),
        "n_reg_hard":       int(n_reg),
        "n_both_hard":      int(n_both),
        "p_both_observed":  float(p_both_obs),
        "p_both_expected":  float(p_both_exp),
        "fisher_odds_ratio": float(odds),
        "fisher_p_value":    float(p_fisher),
        "interpretation":    interpretation,
    },
}

_stats_file = weights_path / "report_stats.json"
_existing   = _json.loads(_stats_file.read_text()) if _stats_file.exists() else {}
_existing["failure_mode_comparison"] = _fm_stats
_stats_file.write_text(_json.dumps(_existing, indent=2))
print(f"\nSaved failure-mode stats → {_stats_file}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Precision and recall vs inelasticity y
# ══════════════════════════════════════════════════════════════════════════════
valid_pr  = np.isfinite(precision_aligned) & np.isfinite(recall_aligned) & np.isfinite(inel_aligned)
yv_pr     = np.clip(inel_aligned[valid_pr], 0.0, 1.0)
prec_v    = precision_aligned[valid_pr]
rec_v     = recall_aligned[valid_pr]

bin_edges_pr   = np.linspace(yv_pr.min(), yv_pr.max(), N_BINS + 1)
bin_idx_pr     = np.clip(np.digitize(yv_pr, bin_edges_pr, right=False) - 1, 0, N_BINS - 1)
bin_centres_pr = 0.5 * (bin_edges_pr[:-1] + bin_edges_pr[1:])
bw_pr          = (bin_edges_pr[1] - bin_edges_pr[0]) * 0.8

mean_prec     = np.array([prec_v[bin_idx_pr == k].mean() if (bin_idx_pr == k).sum() > 0 else np.nan for k in range(N_BINS)])
mean_rec      = np.array([rec_v[bin_idx_pr == k].mean()  if (bin_idx_pr == k).sum() > 0 else np.nan for k in range(N_BINS)])
bin_counts_pr = np.array([(bin_idx_pr == k).sum() for k in range(N_BINS)])

fig6, (ax6_top, ax6_cnt) = plt.subplots(
    2, 1, figsize=(6, 5), constrained_layout=True,
    gridspec_kw={"height_ratios": [4, 1]},
)

ax6_top.plot(bin_centres_pr, mean_prec, color=C2, marker="o", ms=6, lw=2,
             label="Precision")
ax6_top.plot(bin_centres_pr, mean_rec,  color=C3, marker="s", ms=6, lw=2,
             label="Recall")
ax6_top.set_ylabel("Primary EM metric")
ax6_top.set_ylim(0, 1.05)
ax6_top.set_xlim(0, 1)
ax6_top.tick_params(axis="x", labelbottom=False)
ax6_top.spines[["top", "right"]].set_visible(False)
ax6_top.legend(loc="lower left", frameon=False)
ax6_top.set_title(r"Precision and recall vs inelasticity $y$")

ax6_cnt.bar(bin_centres_pr, bin_counts_pr, width=bw_pr, color=C3, alpha=0.35, linewidth=0)
ax6_cnt.set_xlabel(r"True inelasticity $y$")
ax6_cnt.set_ylabel("N", fontsize=10)
ax6_cnt.spines[["top", "right"]].set_visible(False)
ax6_cnt.yaxis.set_major_locator(mticker.MaxNLocator(3, integer=True))

out6 = figures_path / "fig4_precision_recall_vs_y.jpg"
fig6.savefig(out6, dpi=350, bbox_inches="tight")
plt.close(fig6)
print(f"Saved: {out6}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Primary EM node count vs inelasticity y
# ══════════════════════════════════════════════════════════════════════════════
valid_nc   = np.isfinite(inel_aligned)
yv_nc      = np.clip(inel_aligned[valid_nc], 0.0, 1.0)
n_true_nc  = n_true_pEM_aligned[valid_nc].astype(float)
n_pred_nc  = n_pred_pEM_aligned[valid_nc].astype(float)

bin_edges_nc   = np.linspace(yv_nc.min(), yv_nc.max(), N_BINS + 1)
bin_idx_nc     = np.clip(np.digitize(yv_nc, bin_edges_nc, right=False) - 1, 0, N_BINS - 1)
bin_centres_nc = 0.5 * (bin_edges_nc[:-1] + bin_edges_nc[1:])
bw_nc          = (bin_edges_nc[1] - bin_edges_nc[0]) * 0.8

mean_n_true    = np.array([n_true_nc[bin_idx_nc == k].mean() if (bin_idx_nc == k).sum() > 0 else np.nan for k in range(N_BINS)])
mean_n_pred    = np.array([n_pred_nc[bin_idx_nc == k].mean() if (bin_idx_nc == k).sum() > 0 else np.nan for k in range(N_BINS)])
bin_counts_nc  = np.array([(bin_idx_nc == k).sum() for k in range(N_BINS)])

fig7, (ax7_top, ax7_cnt) = plt.subplots(
    2, 1, figsize=(6, 5), constrained_layout=True,
    gridspec_kw={"height_ratios": [4, 1]},
)

ax7_top.plot(bin_centres_nc, mean_n_true, color=C1, marker="o", ms=6, lw=2,
             label="True primary EM nodes")
ax7_top.plot(bin_centres_nc, mean_n_pred, color=C2, marker="s", ms=6, lw=2, ls="--",
             label="Predicted primary EM nodes")
ax7_top.set_ylabel("Mean node count per event")
ax7_top.set_xlim(0, 1)
ax7_top.set_ylim(bottom=0)
ax7_top.tick_params(axis="x", labelbottom=False)
ax7_top.spines[["top", "right"]].set_visible(False)
ax7_top.legend(loc="upper right", frameon=False)
ax7_top.set_title(r"Primary EM node count vs inelasticity $y$")

ax7_cnt.bar(bin_centres_nc, bin_counts_nc, width=bw_nc, color=C3, alpha=0.35, linewidth=0)
ax7_cnt.set_xlabel(r"True inelasticity $y$")
ax7_cnt.set_ylabel("N", fontsize=10)
ax7_cnt.spines[["top", "right"]].set_visible(False)
ax7_cnt.yaxis.set_major_locator(mticker.MaxNLocator(3, integer=True))

out7 = figures_path / "fig5_node_count_vs_y.jpg"
fig7.savefig(out7, dpi=350, bbox_inches="tight")
plt.close(fig7)
print(f"Saved: {out7}")


# ── Variable defs for confusion matrix components — used by section 10 ───────
valid_cm   = np.isfinite(inel_aligned)
yv_cm      = np.clip(inel_aligned[valid_cm], 0.0, 1.0)
tp_v       = tp_aligned[valid_cm].astype(float)
fp_v       = fp_aligned[valid_cm].astype(float)
fn_v       = fn_aligned[valid_cm].astype(float)
tn_v       = tn_aligned[valid_cm].astype(float)

bin_edges_cm   = np.linspace(yv_cm.min(), yv_cm.max(), N_BINS + 1)
bin_idx_cm     = np.clip(np.digitize(yv_cm, bin_edges_cm, right=False) - 1, 0, N_BINS - 1)
bin_centres_cm = 0.5 * (bin_edges_cm[:-1] + bin_edges_cm[1:])


# ══════════════════════════════════════════════════════════════════════════════
# 10. Classifier rates vs y: TPR, FPR, precision
# TPR = TP/(TP+FN),  FPR = FP/(FP+TN),  precision = TP/(TP+FP)
# ══════════════════════════════════════════════════════════════════════════════
_pos      = tp_v + fn_v
_neg      = fp_v + tn_v
_pred_pos = tp_v + fp_v

tpr_v  = np.divide(tp_v, _pos,      out=np.full_like(tp_v, np.nan), where=_pos > 0)
fpr_v  = np.divide(fp_v, _neg,      out=np.full_like(fp_v, np.nan), where=_neg > 0)
prec_v = np.divide(tp_v, _pred_pos, out=np.full_like(tp_v, np.nan), where=_pred_pos > 0)

def _mean_se(arr, idx, n_bins):
    ms, ses = [], []
    for k in range(n_bins):
        vals = arr[idx == k]; vals = vals[~np.isnan(vals)]
        n = len(vals)
        ms.append(vals.mean() if n > 0 else np.nan)
        ses.append(vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
    return np.array(ms), np.array(ses)

def _median_se(arr, idx, n_bins):
    ms, ses = [], []
    for k in range(n_bins):
        vals = arr[idx == k]; vals = vals[~np.isnan(vals)]
        n = len(vals)
        ms.append(np.median(vals) if n > 0 else np.nan)
        ses.append(1.2533 * vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
    return np.array(ms), np.array(ses)

mean_tpr,   se_tpr   = _mean_se(tpr_v,  bin_idx_cm, N_BINS)
mean_fpr,   se_fpr   = _mean_se(fpr_v,  bin_idx_cm, N_BINS)
mean_prec,  se_prec  = _mean_se(prec_v, bin_idx_cm, N_BINS)
med_f1_9b,  se_f1    = _median_se(f1v,  bin_idx,    N_BINS)
mean_acc_9b, se_acc  = _mean_se(accv,   bin_idx,    N_BINS)
mean_ce_9b,  se_ce   = _mean_se(cev,    bin_idx,    N_BINS)


_fig9b_data = {
    "description": (
        "Numerical results for fig6_7 (fig6_metrics_vs_y, fig7_rates_vs_y). "
        "Left panel: classifier metrics vs true inelasticity y. "
        "Right panel: rates vs true inelasticity y. "
        "TPR=recall=TP/(TP+FN), FPR=fall-out=FP/(FP+TN), precision=TP/(TP+FP). "
        "Values are per-bin means (or median for F1). SE = SEM for means, "
        "1.2533*sigma/sqrt(n) for median. NaN = empty bin."
    ),
    "left_panel": {
        "x_label": "true inelasticity y",
        "bin_centres":   [round(float(v), 4) for v in bin_centres],
        "median_F1_pEM": [None if np.isnan(v) else round(float(v), 4) for v in med_f1_9b],
        "se_F1_pEM":     [None if np.isnan(v) else round(float(v), 4) for v in se_f1],
        "mean_accuracy": [None if np.isnan(v) else round(float(v), 4) for v in mean_acc_9b],
        "se_accuracy":   [None if np.isnan(v) else round(float(v), 4) for v in se_acc],
        "mean_BCE_loss": [None if np.isnan(v) else round(float(v), 6) for v in mean_ce_9b],
        "se_BCE_loss":   [None if np.isnan(v) else round(float(v), 6) for v in se_ce],
    },
    "right_panel": {
        "x_label": "true inelasticity y",
        "bin_centres": [round(float(v), 4) for v in bin_centres_cm],
        "TPR_recall":  [None if np.isnan(v) else round(float(v), 4) for v in mean_tpr],
        "se_TPR":      [None if np.isnan(v) else round(float(v), 4) for v in se_tpr],
        "FPR_fallout": [None if np.isnan(v) else round(float(v), 4) for v in mean_fpr],
        "se_FPR":      [None if np.isnan(v) else round(float(v), 4) for v in se_fpr],
        "precision":   [None if np.isnan(v) else round(float(v), 4) for v in mean_prec],
        "se_precision":[None if np.isnan(v) else round(float(v), 4) for v in se_prec],
    },
}
out9b_json = figures_path / "fig6_7_data.json"
with open(out9b_json, "w") as _f:
    _json.dump(_fig9b_data, _f, indent=2)
print(f"Saved: {out9b_json}")


# ══════════════════════════════════════════════════════════════════════════════
# fig6 + fig7 — metrics vs y and rates vs y (separate figures)
# ══════════════════════════════════════════════════════════════════════════════

fig9b_a, ax9b_a = plt.subplots(figsize=(8, 4.5))
ax9b_a.plot(bin_centres, med_f1_9b,   color=C3, marker="o", ms=7, lw=1.8,
            label=r"median $F_{1,\mathrm{pEM}}$")
ax9b_a.plot(bin_centres, mean_acc_9b, color=C2, marker="s", ms=7, lw=1.8, ls="--",
            label="mean accuracy")
ax9b_a.set_xlabel(r"True inelasticity $y$")
ax9b_a.set_ylabel("Classifier metric")
ax9b_a.set_xlim(0, 1)
ax9b_a.set_ylim(0, 1.05)
ax9b_a.spines["top"].set_visible(False)
ax9b_a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'{v:g}'))

_ax9b_a_ce = ax9b_a.twinx()
_ax9b_a_ce.plot(bin_centres, mean_ce_9b, color="black", marker="D", ms=6, lw=1.6, ls="-.",
                label=r"mean $\overline{\mathcal{L}}_\mathrm{BCE}^{(e)}$")
_ax9b_a_ce.tick_params(axis="y", labelcolor="black")
_ax9b_a_ce.set_ylim(bottom=0)
_ax9b_a_ce.spines["top"].set_visible(False)
_ax9b_a_ce.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'{v:g}'))
_ax9b_a_ce.set_ylabel(r"mean $\overline{\mathcal{L}}_\mathrm{BCE}^{(e)}$", color="black")

_lines_a, _labels_a = ax9b_a.get_legend_handles_labels()
_lce_a,   _lce_al   = _ax9b_a_ce.get_legend_handles_labels()
ax9b_a.legend(_lines_a + _lce_a, _labels_a + _lce_al, loc="lower left", frameon=False)

fig9b_a.tight_layout()
out9b_a = figures_path / "fig6_metrics_vs_y.jpg"
fig9b_a.savefig(out9b_a, dpi=350, bbox_inches="tight")
print(f"Saved: {out9b_a}")
plt.close(fig9b_a)

fig9b_b, ax9b_b = plt.subplots(figsize=(8, 4.5))
ax9b_b.plot(bin_centres_cm, mean_tpr,  color=C3,      marker="o", ms=7, lw=1.8, label="TPR = recall")
ax9b_b.plot(bin_centres_cm, mean_fpr,  color=C2,      marker="s", ms=7, lw=1.8, label="FPR = fall-out")
ax9b_b.plot(bin_centres_cm, mean_prec, color="black", marker="D", ms=7, lw=1.8, ls="--", label="precision")
ax9b_b.set_xlabel(r"True inelasticity $y$")
ax9b_b.set_ylabel("Rate")
ax9b_b.set_xlim(0, 1)
ax9b_b.set_ylim(0, 1.05)
ax9b_b.spines[["top", "right"]].set_visible(False)
ax9b_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'{v:g}'))
ax9b_b.legend(frameon=False, loc="lower left")

fig9b_b.tight_layout()
out9b_b = figures_path / "fig7_rates_vs_y.jpg"
fig9b_b.savefig(out9b_b, dpi=350, bbox_inches="tight")
print(f"Saved: {out9b_b}")
plt.close(fig9b_b)


# ══════════════════════════════════════════════════════════════════════════════
# 11. fig8 + fig9 — mean P(pEM) vs y and BCE terms vs y
# Proves whether the BCE curve shape is driven by confidence collapse
# (both signal and background scores drop at high y) vs ambiguity at mid-y
# (scores converge toward 0.5).
# ══════════════════════════════════════════════════════════════════════════════
mean_prob_sig_9c, se_prob_sig_9c = _mean_se(prob_signal_v, bin_idx, N_BINS)
mean_prob_bg_9c,  se_prob_bg_9c  = _mean_se(prob_bg_v,     bin_idx, N_BINS)
mean_bce_sig_9c,  se_bce_sig_9c  = _mean_se(bce_signal_v,  bin_idx, N_BINS)
mean_bce_bg_9c,   se_bce_bg_9c   = _mean_se(bce_bg_v,      bin_idx, N_BINS)

fig9c_a, ax9c_l = plt.subplots(figsize=(8, 4.5))

# ── Left: mean P(pEM) per true class ─────────────────────────────────────────
ax9c_l.errorbar(bin_centres, mean_prob_sig_9c, yerr=se_prob_sig_9c,
                color=C3, marker="o", ms=7, lw=1.8, capsize=3,
                label=r"true signal (primary EM)")
ax9c_l.errorbar(bin_centres, mean_prob_bg_9c,  yerr=se_prob_bg_9c,
                color=C2, marker="s", ms=7, lw=1.8, ls="--", capsize=3,
                label=r"true background")
ax9c_l.axhline(0.5, color="grey", ls=":", lw=1.2, alpha=0.7, label="$P = 0.5$")
ax9c_l.set_xlabel(r"True inelasticity $y$")
ax9c_l.set_ylabel(r"Mean $P(\mathrm{primary\ EM})$ per event")
ax9c_l.set_xlim(0, 1)
ax9c_l.set_ylim(0, 1.05)
ax9c_l.spines[["top", "right"]].set_visible(False)
ax9c_l.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'{v:g}'))
ax9c_l.legend(frameon=False, loc="upper right")

fig9c_a.tight_layout()
out9c_a = figures_path / "fig8_mean_prob_vs_y.jpg"
fig9c_a.savefig(out9c_a, dpi=350, bbox_inches="tight")
print(f"Saved: {out9c_a}")
plt.close(fig9c_a)

fig9c_b, ax9c_r = plt.subplots(figsize=(8, 4.5))

# ── Right: BCE terms — signal (-log p) and background (-log(1-p)) ────────────
ax9c_r.errorbar(bin_centres, mean_bce_sig_9c, yerr=se_bce_sig_9c,
                color=C3, marker="o", ms=7, lw=1.8, capsize=3,
                label=r"signal: $-\log p_\mathrm{pEM}$")
ax9c_r.errorbar(bin_centres, mean_bce_bg_9c,  yerr=se_bce_bg_9c,
                color=C2, marker="s", ms=7, lw=1.8, ls="--", capsize=3,
                label=r"background: $-\log(1 - p_\mathrm{pEM})$")
ax9c_r.set_xlabel(r"True inelasticity $y$")
ax9c_r.set_ylabel(r"Mean BCE term per node")
ax9c_r.set_xlim(0, 1)
ax9c_r.set_ylim(bottom=0)
ax9c_r.spines[["top", "right"]].set_visible(False)
ax9c_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '' if v == 0 else f'{v:g}'))
ax9c_r.legend(frameon=False, loc="upper right")

fig9c_b.tight_layout()
out9c_b = figures_path / "fig9_bce_vs_y.jpg"
fig9c_b.savefig(out9c_b, dpi=350, bbox_inches="tight")
print(f"Saved: {out9c_b}")
plt.close(fig9c_b)


# ══════════════════════════════════════════════════════════════════════════════
# 12. fig10 — y vs E_nu scatter, coloured by F1_pEM
# ══════════════════════════════════════════════════════════════════════════════
log_Enu_all = np.log10(np.maximum(reg_Enu_true[valid], 1e-9))

fig11, ax11 = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
sc11 = ax11.scatter(
    log_Enu_all, yv, c=f1v,
    vmin=0, vmax=1, cmap="viridis",
    s=3, alpha=0.4, linewidths=0, rasterized=True,
)
cb11 = fig11.colorbar(sc11, ax=ax11, pad=0.02)
cb11.set_label(r"$F_{1,\mathrm{pEM}}$ (per event)", fontsize=10)
ax11.set_xlabel(r"$\log_{10}(E_\nu / \mathrm{TeV})$")
ax11.set_ylabel(r"True inelasticity $y$")
ax11.set_ylim(0, 1)
ax11.spines[["top", "right"]].set_visible(False)

out11 = figures_path / "fig10_y_vs_Enu_coloured_f1.jpg"
fig11.savefig(out11, dpi=350, bbox_inches="tight")
plt.close(fig11)
print(f"Saved: {out11}")
