"""
visualise_events.py
====================
Plots best / worst events for four ranking categories:

  general      – overall per-event node accuracy
  mu           – mu hallucinations  (other → mu false positives);
                 restricted to events that contain ≥1 true mu node
  primary_EM_e – per-event recall for primary_EM_e nodes (class 2);
                 restricted to events that contain ≥1 true primary_EM_e node
  secondary_e  – per-event recall for secondary_e nodes  (class 1);
                 restricted to events that contain ≥1 true secondary_e node

For every category K best + K worst events are plotted (3-panel scatter).
Figures are saved under figures_path / <WEIGHTS_DIR> / visualise_events / <category>/.

Usage
-----
    python visualise_events.py
    python visualise_events.py --weights-dir gravnet_nodes_faser_all_events_4class_0323_1003
    python visualise_events.py --weights-dir ... --run 10000 --k 6 --gpu cuda:0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repo src root

from analysis.gravnet.model import NeutrinoGravNetNodesFaser
from analysis.gravnet.create_data_particle_prob_4class import evaluate, get_str_from_run
from analysis.utils.utils import get_torch_path, get_weights_path, get_figures_path

# ── constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["other", "secondary_e", "primary_EM_e", "mu"]
COLORS      = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]

# ── ranking definitions ───────────────────────────────────────────────────────
# Each entry defines how to score events for best/worst selection.
#   key             : field name in the event dict
#   higher_is_better: True  → best = highest score, worst = lowest
#                     False → best = lowest score,  worst = highest
#   filter_key      : optional event dict field; skip event if field == 0
RANKINGS = [
    dict(
        name="general",
        title="Overall accuracy",
        key="acc",
        higher_is_better=True,
        filter_key=None,
    ),
    dict(
        name="mu",
        title="Mu hallucinations (other→mu)",
        key="mu_hallucinations",
        higher_is_better=False,   # worst = most hallucinations
        filter_key="n_true_mu",
    ),
    dict(
        name="primary_EM_e",
        title="Primary EM-e recall",
        key="prim_e_recall",
        higher_is_better=True,
        filter_key="n_true_prim_e",
    ),
    dict(
        name="secondary_e",
        title="Secondary-e recall",
        key="sec_e_recall",
        higher_is_better=True,
        filter_key="n_true_sec_e",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Building the per-event list
# ─────────────────────────────────────────────────────────────────────────────

def _per_class_recall(pred, true, cls):
    mask = true == cls
    if mask.sum() == 0:
        return np.nan
    return (pred[mask] == cls).mean()


def build_events(y_true_all, y_pred_all, y_prob_all, val_dataset):
    """Split concatenated arrays back into per-event dicts using node counts."""
    node_counts = [int(data.pos.shape[0]) for data in val_dataset]
    assert sum(node_counts) == len(y_true_all), (
        f"Node count mismatch: {sum(node_counts)} in dataset vs "
        f"{len(y_true_all)} in arrays — val split must match (random_state=42)."
    )
    events, offset = [], 0
    for data, n in zip(val_dataset, node_counts):
        true  = y_true_all[offset:offset + n]
        pred  = y_pred_all[offset:offset + n]
        probs = y_prob_all[offset:offset + n]
        pos   = data.pos.numpy()
        correct = pred == true

        events.append(dict(
            pos=pos, true=true, pred=pred, probs=probs,
            acc=float(correct.mean()),
            # mu
            mu_hallucinations=int(((pred == 3) & (true != 3)).sum()),
            n_true_mu=int((true == 3).sum()),
            mu_recall=_per_class_recall(pred, true, 3),
            # primary EM e (class 2)
            prim_e_recall=_per_class_recall(pred, true, 2),
            n_true_prim_e=int((true == 2).sum()),
            # secondary e (class 1)
            sec_e_recall=_per_class_recall(pred, true, 1),
            n_true_sec_e=int((true == 1).sum()),
        ))
        offset += n
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_event(ev, title="", save_path=None):
    pos, true, pred = ev["pos"], ev["true"], ev["pred"]
    correct = true == pred
    z, x = pos[:, 2], pos[:, 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for ax, labels, panel_title in zip(
        axes,
        [true, pred, None],
        ["True labels", "Predicted labels", "Misclassified nodes"],
    ):
        if labels is not None:
            for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
                mask = labels == cls_idx
                if mask.sum() > 0:
                    ax.scatter(z[mask], x[mask], c=color, s=3, alpha=0.6,
                               label=f"{name} ({mask.sum():,})", linewidths=0)
        else:
            ax.scatter(z[correct], x[correct], c="#cccccc", s=2, alpha=0.3,
                       label=f"Correct ({correct.sum():,})", linewidths=0)
            for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
                mask = (~correct) & (pred == cls_idx)
                if mask.sum() > 0:
                    ax.scatter(z[mask], x[mask], c=color, s=15, marker="x",
                               alpha=0.9, label=f"→{name} ({mask.sum():,})",
                               linewidths=0.8)

        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x (mm)")
        ax.set_title(panel_title)
        ax.legend(fontsize=7, markerscale=2, frameon=False)

    fig.suptitle(
        f"{title} | acc={ev['acc']:.3f} | mu_hall={ev['mu_hallucinations']} "
        f"| true_mu={ev['n_true_mu']} | prim_e_rec={ev['prim_e_recall']:.2f} "
        f"| sec_e_rec={ev['sec_e_recall']:.2f}",
        fontsize=9,
    )
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {save_path.name}")

    plt.show()
    plt.close(fig)


def plot_category(events, ranking, k, out_dir):
    key              = ranking["key"]
    higher_is_better = ranking["higher_is_better"]
    filter_key       = ranking["filter_key"]
    name             = ranking["name"]
    title            = ranking["title"]

    # filter events that have relevant nodes
    candidates = [
        (i, e) for i, e in enumerate(events)
        if filter_key is None or e[filter_key] > 0
    ]
    if not candidates:
        print(f"[{name}] No eligible events — skipping.")
        return

    # sort: descending if higher_is_better (best first), ascending otherwise
    sorted_cands = sorted(
        candidates,
        key=lambda ie: ie[1][key] if not np.isnan(ie[1][key]) else -np.inf,
        reverse=higher_is_better,
    )
    best_pairs  = sorted_cands[:k]
    worst_pairs = sorted_cands[-k:][::-1]   # worst = opposite end

    cat_dir = out_dir / name
    cat_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Category: {name}  ({title})")
    print(f"  Eligible events : {len(candidates)}")

    print(f"\n  --- BEST {k} ---")
    for rank, (idx, ev) in enumerate(best_pairs):
        score = ev[key]
        label = f"Best #{rank+1} | event {idx} | {key}={score:.3f}"
        print(f"  {label}")
        plot_event(ev, title=label,
                   save_path=cat_dir / f"best_{rank+1:02d}_ev{idx}.png")

    print(f"\n  --- WORST {k} ---")
    for rank, (idx, ev) in enumerate(worst_pairs):
        score = ev[key]
        label = f"Worst #{rank+1} | event {idx} | {key}={score:.3f}"
        print(f"  {label}")
        plot_event(ev, title=label,
                   save_path=cat_dir / f"worst_{rank+1:02d}_ev{idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir", default="gravnet_nodes_faser_all_events_4class_0323_1003")
    p.add_argument("--run",  type=int, default=10000)
    p.add_argument("--k",   type=int, default=6, help="best/worst events per category")
    p.add_argument("--gpu", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    device       = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    weights_path = get_weights_path() / args.weights_dir / "best_model.pt"
    torch_path   = get_torch_path()
    figures_path = get_figures_path() / "gravnet_nodes_4class" / args.weights_dir
    out_dir      = figures_path / "visualise_events"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device      : {device}")
    print(f"Weights     : {weights_path}")
    print(f"Output dir  : {out_dir}")

    # ── Load val dataset ──────────────────────────────────────────────────────
    run_str     = get_str_from_run(args.run)
    run_path    = torch_path / f"{args.run}/pointnetpp_faser_all_events"
    chunk_files = sorted(run_path.glob(f"{run_str}_*.pt"))
    chunk_files = [f for f in chunk_files if "_particle_prob" not in f.stem]

    dataset = []
    for cf in chunk_files:
        dataset.extend(torch.load(cf, weights_only=False))

    _, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Val events  : {len(val_dataset)}")

    # ── Load ckpt (CPU only for metadata) ────────────────────────────────────
    ckpt       = torch.load(weights_path, map_location="cpu", weights_only=False)
    _ep        = ckpt["epoch"]
    _vl        = ckpt.get("val_loss", 0.0)
    cache_path = figures_path / f"infer_cache_ep{_ep}_vl{_vl:.4f}.npz"

    # ── Get y_true / y_pred / y_prob ─────────────────────────────────────────
    if cache_path.exists():
        print(f"Loading evaluate cache: {cache_path.name}")
        _c = np.load(cache_path)
        y_true_all = _c["y_true"]
        y_pred_all = _c["y_pred"]
        y_prob_all = _c["y_prob"]
    else:
        print("Cache not found — running model inference...")
        cfg = ckpt.get("model_config", {})
        if not cfg:
            sd = ckpt["model_state_dict"]
            n_gravstack         = sum(1 for k in sd if k.startswith("fts.") and k.endswith(".0.weight"))
            n_feature_transform = sd["fts.0.0.weight"].shape[0]
            out_channels        = (sd["gns.0.lin.weight"].shape[0]
                                   if "gns.0.lin.weight" in sd
                                   else sd[[k for k in sd if "gns.0" in k][0]].shape[0])
            concat_features     = sd["node_classifier.0.weight"].shape[1]
            out_channels        = (concat_features - 8) // n_gravstack
            cfg = dict(n_gravstack=n_gravstack, out_channels=out_channels,
                       n_feature_transform=n_feature_transform, k=12)
            print(f"Inferred model config: {cfg}")

        model = NeutrinoGravNetNodesFaser(
            input_dim=1, num_node_classes=4, faser_dim=5,
            n_gravstack=cfg.get("n_gravstack", 3),
            out_channels=cfg.get("out_channels", 16),
            n_feature_transform=cfg.get("n_feature_transform", 16),
            k=cfg.get("k", 12),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        y_true_list, y_pred_list, y_prob_list = evaluate(
            model, val_loader, device, use_faser=True
        )
        y_true_all = np.concatenate(y_true_list)
        y_pred_all = np.concatenate(y_pred_list)
        y_prob_all = np.concatenate(y_prob_list)

    # ── Build per-event dicts ─────────────────────────────────────────────────
    events = build_events(y_true_all, y_pred_all, y_prob_all, val_dataset)
    print(f"Events      : {len(events)}")

    # ── Plot each category ────────────────────────────────────────────────────
    for ranking in RANKINGS:
        plot_category(events, ranking, k=args.k, out_dir=out_dir)

    print(f"\nDone. All figures saved under:\n  {out_dir}")


if __name__ == "__main__":
    main()
