#!/usr/bin/env python3
"""
Training script for NeutrinoGravNetRegressionFASER model.

Predicts log10(E_nu) and logit(y) where y = E_roe / E_nu (Bjorken inelasticity = hadronic energy fraction).
Note: this is 1 - lepton fraction (lepton fraction = E_lepton/E_nu).
E_roe and E_lepton are derived at inference as y·E_nu and (1-y)·E_nu, so energy
conservation E_lepton + E_roe = E_nu holds exactly by construction.

Usage:
    python -m analysis.gravnet.train_gravnet_regression_faser -r 10000 -b 8 --num-epochs 50 --pooling sum
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetNodesFaser, NeutrinoGravNetRegressionFASER
from analysis.utils.utils import get_torch_path, get_weights_path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_NAMES = ["E_nu", "E_lepton", "E_roe"]


def get_str_from_run(run: int) -> str:
    """Convert run number to string label."""
    if run % 4 == 0:
        return "nue"
    elif run % 4 == 1:
        return "num"
    elif run % 4 == 2:
        return "nut"
    elif run % 4 == 3:
        return "nun"


def compute_targets(data, norm_stats):
    """
    Compute reparametrised, standardised training targets from a batch.

    Both targets are standardised to zero mean / unit variance using training-set
    statistics so equal MSE weights are appropriate.

    Returns targets [batch_size, 2]:
        col 0: (log10(E_nu) - mu_enu) / sigma_enu
        col 1: (logit(y) - mu_logit) / sigma_logit   where y = E_roe / E_nu  (Bjorken inelasticity = 1 - lepton fraction)
    """
    E_nu = data.E_nu.clamp(min=1e-6)
    E_lepton = data.E_lepton.clamp(min=1e-6)
    E_roe = data.E_roe.clamp(min=1e-6)
    log_E_nu = torch.log10(E_nu)
    logit_y = torch.log(E_roe / E_lepton)   # logit of inelasticity y = E_roe/E_nu
    log_E_nu_std = (log_E_nu - norm_stats["mu_enu"]) / norm_stats["sigma_enu"]
    logit_y_std  = (logit_y  - norm_stats["mu_logit"]) / norm_stats["sigma_logit"]
    return torch.stack([log_E_nu_std, logit_y_std], dim=1)  # [batch_size, 2]


def preds_to_physical(preds, targets, norm_stats, beta_loss=False):
    """
    Convert standardised model outputs and targets to physical energies.

    Args:
        preds:      [N, 2] or [N, 3] — when beta_loss=False: (t1_std, t2_std);
                    when beta_loss=True: (t1_std, alpha, beta) with alpha/beta already softplus-activated
        targets:    [N, 2] — standardised (log10_E_nu, logit_y)
        norm_stats: dict with mu_enu, sigma_enu, mu_logit, sigma_logit
        beta_loss:  If True, recover y_pred as Beta distribution mean alpha/(alpha+beta)

    Returns:
        preds_linear, targets_linear: both [N, 3] — (E_nu, E_lepton, E_roe) in TeV
    """
    # Destandardise t1 (common to both modes)
    log_E_nu_pred = preds[:, 0] * norm_stats["sigma_enu"]   + norm_stats["mu_enu"]
    log_E_nu_true = targets[:, 0] * norm_stats["sigma_enu"]   + norm_stats["mu_enu"]
    logit_y_true  = targets[:, 1] * norm_stats["sigma_logit"] + norm_stats["mu_logit"]

    E_nu_pred = 10 ** log_E_nu_pred
    E_nu_true = 10 ** log_E_nu_true

    if beta_loss:
        alpha, beta_p = preds[:, 1], preds[:, 2]
        y_pred = alpha / (alpha + beta_p)              # E[Beta(α,β)] ∈ (0,1)
    else:
        logit_y_pred = preds[:, 1] * norm_stats["sigma_logit"] + norm_stats["mu_logit"]
        y_pred = torch.sigmoid(logit_y_pred)           # y = inelasticity = E_roe/E_nu

    y_true = torch.sigmoid(logit_y_true)
    E_roe_pred    = y_pred * E_nu_pred
    E_lepton_pred = (1 - y_pred) * E_nu_pred
    E_roe_true    = y_true * E_nu_true
    E_lepton_true = (1 - y_true) * E_nu_true

    preds_linear = torch.stack([E_nu_pred, E_lepton_pred, E_roe_pred], dim=1)
    targets_linear = torch.stack([E_nu_true, E_lepton_true, E_roe_true], dim=1)
    return preds_linear, targets_linear


def compute_loss(pred, target, loss_fn, huber_delta=1.0):
    if loss_fn == "huber":
        return F.huber_loss(pred, target, delta=huber_delta)
    return F.mse_loss(pred, target)


def compute_beta_loss(alpha, beta_param, y_true):
    """NLL of y_true under Beta(alpha, beta_param). y_true must be in (0,1)."""
    return -torch.distributions.Beta(alpha, beta_param).log_prob(y_true).mean()


def train_epoch(model, loader, optimizer, device, norm_stats, loss_fn="mse", huber_delta=1.0, accumulation_steps=1, beta_loss=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_t1 = 0
    total_loss_t2 = 0
    total_events = 0

    all_preds = []
    all_targets = []

    train_bar = tqdm(loader, desc="Training", disable=not sys.stdout.isatty())
    optimizer.zero_grad()
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)

        try:
            predictions = model(data.x, data.pos, data.batch, data.x_faser)
            targets = compute_targets(data, norm_stats)

            loss_t1 = compute_loss(predictions[:, 0], targets[:, 0], loss_fn, huber_delta)
            if beta_loss:
                y_true_frac = torch.sigmoid(
                    targets[:, 1] * norm_stats["sigma_logit"] + norm_stats["mu_logit"]
                ).clamp(1e-6, 1 - 1e-6)
                loss_t2 = compute_beta_loss(predictions[:, 1], predictions[:, 2], y_true_frac)
            else:
                loss_t2 = compute_loss(predictions[:, 1], targets[:, 1], loss_fn, huber_delta)
            loss = (loss_t1 + loss_t2) / accumulation_steps

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if grad_norm > 50.0:
                logger.warning(f"Large gradient norm: {grad_norm:.2f}")
            optimizer.step()
            optimizer.zero_grad()

        batch_size = predictions.size(0)
        total_loss_t1 += loss_t1.item() * batch_size
        total_loss_t2 += loss_t2.item() * batch_size
        total_loss += (loss_t1.item() + loss_t2.item()) * batch_size
        total_events += batch_size

        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())

        current_loss = total_loss / total_events
        train_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        if not sys.stdout.isatty() and (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx + 1}: Loss={current_loss:.4f}")

    # Handle trailing batches when len(loader) is not divisible by accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total_events
    avg_loss_t1 = total_loss_t1 / total_events
    avg_loss_t2 = total_loss_t2 / total_events
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if beta_loss:
        rmse = torch.sqrt(F.mse_loss(all_preds[:, 0], all_targets[:, 0]))
    elif loss_fn == "mse":
        rmse = torch.sqrt(torch.tensor(avg_loss))
    else:
        rmse = torch.sqrt(F.mse_loss(all_preds, all_targets))

    preds_linear, targets_linear = preds_to_physical(all_preds, all_targets, norm_stats, beta_loss=beta_loss)
    rel_err = torch.abs(preds_linear - targets_linear) / targets_linear.clamp(min=1e-6)
    mean_rel_err = rel_err.mean(dim=0)  # [3]
    std_rel_err = rel_err.std(dim=0)    # [3]

    return avg_loss, avg_loss_t1, avg_loss_t2, rmse.item(), mean_rel_err.cpu(), std_rel_err.cpu()


def validate_epoch(model, loader, device, norm_stats, loss_fn="mse", huber_delta=1.0, beta_loss=False):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_loss_t1 = 0
    total_loss_t2 = 0
    total_events = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation", disable=not sys.stdout.isatty())
        for data in val_bar:
            data = data.to(device)

            try:
                predictions = model(data.x, data.pos, data.batch, data.x_faser)
                targets = compute_targets(data, norm_stats)

                loss_t1 = compute_loss(predictions[:, 0], targets[:, 0], loss_fn, huber_delta)
                if beta_loss:
                    y_true_frac = torch.sigmoid(
                        targets[:, 1] * norm_stats["sigma_logit"] + norm_stats["mu_logit"]
                    ).clamp(1e-6, 1 - 1e-6)
                    loss_t2 = compute_beta_loss(predictions[:, 1], predictions[:, 2], y_true_frac)
                else:
                    loss_t2 = compute_loss(predictions[:, 1], targets[:, 1], loss_fn, huber_delta)

                batch_size = predictions.size(0)
                total_loss_t1 += loss_t1.item() * batch_size
                total_loss_t2 += loss_t2.item() * batch_size
                total_loss += (loss_t1.item() + loss_t2.item()) * batch_size
                total_events += batch_size

                all_preds.append(predictions)
                all_targets.append(targets)

                current_loss = total_loss / total_events
                val_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

            except RuntimeError:
                continue

    avg_loss = total_loss / total_events
    avg_loss_t1 = total_loss_t1 / total_events
    avg_loss_t2 = total_loss_t2 / total_events
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    if beta_loss:
        rmse = torch.sqrt(F.mse_loss(all_preds[:, 0], all_targets[:, 0]))
    elif loss_fn == "mse":
        rmse = torch.sqrt(torch.tensor(avg_loss))
    else:
        rmse = torch.sqrt(F.mse_loss(all_preds, all_targets))

    preds_linear, targets_linear = preds_to_physical(all_preds, all_targets, norm_stats, beta_loss=beta_loss)
    rel_err = torch.abs(preds_linear - targets_linear) / targets_linear.clamp(min=1e-6)
    mean_rel_err = rel_err.mean(dim=0)
    std_rel_err = rel_err.std(dim=0)

    return avg_loss, avg_loss_t1, avg_loss_t2, rmse.item(), mean_rel_err.cpu(), std_rel_err.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-d", "--data-type", type=str, default="all", choices=["all", "good"]
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        default=[10000],
        help="Run numbers to load (default: 10000 = nue only)",
    )
    parser.add_argument(
        "-c",
        "--chunks",
        nargs="*",
        type=int,
        required=False,
        help="Chunk numbers to load (if not provided, all chunks will be loaded)",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        default=None,
        help="Number of events to use from each chunk (default: all)",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Graph pooling method (default: sum)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Additional suffix to append to output directory name",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber"],
        help="Loss function (default: mse)",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Delta parameter for Huber loss (default: 1.0)",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size × steps (default: 1, no accumulation).",
    )
    parser.add_argument(
        "--bin-size",
        type=str,
        default="200um",
        help="Bin size of the input .pt files, e.g. '200um' or '100um' (default: 200um)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--truth-label",
        action="store_true",
        default=False,
        help="Augment node features with one-hot truth node labels (3-class: other/e/mu)",
    )
    parser.add_argument(
        "--truth-label-4",
        action="store_true",
        default=False,
        help="Augment node features with one-hot raw pdg_label (4-class: other/secondary_e/primary_EM_e/mu)",
    )
    parser.add_argument(
        "--truth-label-2",
        action="store_true",
        default=False,
        help="Augment node features with binary truth label: primary_EM_e vs everything else (remap {0→0,1→0,2→1,3→0})",
    )
    parser.add_argument(
        "--truth-label-3b",
        action="store_true",
        default=False,
        help="Augment node features with physically-correct 3-class label: (other+mu) / secondary_e / primary_EM_e (remap {0→0,1→1,2→2,3→0})",
    )
    parser.add_argument(
        "--faser-global",
        action="store_true",
        default=False,
        help="Broadcast x_faser features to every node before GNN (global variable injection).",
    )
    parser.add_argument(
        "--particle-prob",
        action="store_true",
        default=False,
        help="Load *_particle_prob.pt files (data.x augmented with 4-class node softmax probs).",
    )
    parser.add_argument(
        "--binary-prob-weights",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to binary classifier best_model.pt. Runs inference in-memory at startup, "
             "appending [P(background), P(primary_EM_e)] to each node's features.",
    )
    parser.add_argument(
        "--truth3b-prob-weights",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to truth3b (3-class) classifier best_model.pt. Runs inference in-memory, "
             "appending [P(other_mu), P(secondary_e), P(primary_EM_e)] to each node's features.",
    )
    parser.add_argument(
        "--vertex-dist",
        action="store_true",
        default=False,
        help="Append ground-truth distance from each node to the true neutrino interaction "
             "vertex as an additional node feature. Upper-bound study only — uses Geant4 truth.",
    )
    parser.add_argument(
        "--classifier-embedding-weights",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to any classifier best_model.pt. Appends the full pre-classifier node "
             "embedding [N, n_gravstack*out_channels+8] instead of softmax probs. "
             "num_node_classes is read from the checkpoint automatically.",
    )
    parser.add_argument(
        "--beta-loss",
        action="store_true",
        default=False,
        help="Replace MSE on t2=logit(y) with Beta NLL (outputs alpha, beta per event).",
    )
    parser.add_argument(
        "--no-faser-features",
        action="store_true",
        default=False,
        help="Disable FASER spectrometer MLP branch (ablation study).",
    )
    args = parser.parse_args()

    # Print all arguments
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Build output directory name from varying flags only.
    # Fixed choices (reparam+std normalisation, all-events, pooling) are omitted
    # since they no longer differentiate runs.
    parts = []
    if args.loss != "mse":
        loss_str = args.loss + (str(args.huber_delta) if args.loss == "huber" else "")
        parts.append(loss_str)
    if args.truth_label:
        parts.append("truth3")
    if args.truth_label_4:
        parts.append("truth4")
    if args.truth_label_2:
        parts.append("truth2")
    if args.truth_label_3b:
        parts.append("truth3b")
    if args.faser_global:
        parts.append("faserglobal")
    if args.particle_prob:
        parts.append("prob")
    if args.binary_prob_weights:
        parts.append("binaryprob")
    if args.truth3b_prob_weights:
        parts.append("truth3bprob")
    if args.vertex_dist:
        parts.append("vertexdist")
    if args.classifier_embedding_weights:
        parts.append("embedding")
    if args.beta_loss:
        parts.append("beta")
    if args.no_faser_features:
        parts.append("nofaser")
    if args.bin_size != "200um":
        parts.append(args.bin_size)
    if args.suffix:
        parts.append(args.suffix)

    dir_name = "gravnet_regression_faser"
    if parts:
        dir_name += "_" + "_".join(parts)

    # Setup paths
    torch_path = get_torch_path()
    if args.resume:
        weights_path = Path(args.resume).parent
    else:
        weights_path = get_weights_path() / dir_name
        if weights_path.exists():
            stamp = datetime.datetime.now().strftime("%m%d_%H%M")
            weights_path = weights_path.parent / f"{weights_path.name}_{stamp}"
            logger.warning(f"Output directory already exists — writing to {weights_path}")
    weights_path.mkdir(parents=True, exist_ok=True)

    # Add file handler to logger (append if resuming, overwrite if fresh run)
    log_file = weights_path / "training.log"
    file_handler = logging.FileHandler(log_file, mode="a" if args.resume else "w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")

    dataset = []
    bin_suffix = f"_{args.bin_size}" if args.bin_size != "200um" else ""
    for run in args.runs:
        run_str = get_str_from_run(run)
        run_path = torch_path / f"{run}/pointnetpp_faser_{args.data_type}_events{bin_suffix}"

        if args.chunks is None:
            chunk_files = sorted(run_path.glob(f"{run_str}_*.pt"))
            chunk_files = [f for f in chunk_files
                           if "_particle_prob" not in f.stem and "_binary_prob" not in f.stem]
            chunks_to_load = [int(f.stem.split("_")[-1]) for f in chunk_files]
        else:
            chunks_to_load = args.chunks

        logger.info(f"Loading {len(chunks_to_load)} chunks for run {run} ({run_str})")

        file_suffix = "_particle_prob" if args.particle_prob else ""
        for chunk in chunks_to_load:
            chunk_file = run_path / f"{run_str}_{chunk:03d}{file_suffix}.pt"
            if chunk_file.exists():
                chunk_data = torch.load(chunk_file, weights_only=False)
                if args.num_events is not None:
                    chunk_data = chunk_data[: args.num_events]
                dataset.extend(chunk_data)
                logger.info(f"  Loaded chunk {chunk}: {len(chunk_data)} events")
            else:
                logger.warning(f"  Chunk file not found: {chunk_file}")

    logger.info(f"Total loaded events: {len(dataset)}")

    # Optionally augment node features with ground-truth vertex distance
    if args.vertex_dist:
        logger.info("Augmenting node features with ground-truth vertex distance.")
        for data in dataset:
            # data.pos [N,3] and data.true_pos_centered [3] share the same
            # normalisation frame (centred at pos_mean, divided by 100 mm)
            vertex_dist = torch.norm(
                data.pos - data.true_pos_centered.unsqueeze(0), dim=1, keepdim=True
            )  # [N, 1]
            data.x = torch.cat([data.x, vertex_dist], dim=1)  # [N, 2]
        logger.info("Vertex distance augmentation complete.")

    # Optionally augment node features with one-hot truth labels
    if args.truth_label:
        logger.info("Augmenting node features with one-hot truth labels (3 classes: other/e/mu).")
        for data in dataset:
            y_oh = F.one_hot(data.y, num_classes=3).float()
            data.x = torch.cat([data.x, y_oh], dim=1)
    if args.truth_label_4:
        logger.info("Augmenting node features with one-hot raw pdg_label (4 classes).")
        for data in dataset:
            y_oh = F.one_hot(data.pdg_label, num_classes=4).float()
            data.x = torch.cat([data.x, y_oh], dim=1)
    if args.truth_label_2:
        logger.info("Augmenting node features with binary truth label (2 classes: not_primary_EM / primary_EM_e).")
        remap = torch.tensor([0, 0, 1, 0])
        for data in dataset:
            binary_label = remap[data.pdg_label]
            y_oh = F.one_hot(binary_label, num_classes=2).float()
            data.x = torch.cat([data.x, y_oh], dim=1)
    if args.truth_label_3b:
        logger.info("Augmenting node features with new 3-class truth label (other+mu / secondary_e / primary_EM_e).")
        remap = torch.tensor([0, 1, 2, 0])
        for data in dataset:
            new3_label = remap[data.pdg_label]
            y_oh = F.one_hot(new3_label, num_classes=3).float()
            data.x = torch.cat([data.x, y_oh], dim=1)
    if args.faser_global:
        logger.info("Augmenting node features with global FASER features (injected to each node).")
        for data in dataset:
            faser_expanded = data.x_faser.unsqueeze(0).expand(data.num_nodes, -1)
            data.x = torch.cat([data.x, faser_expanded], dim=1)
    if args.binary_prob_weights:
        logger.info(f"Running binary classifier inference in-memory (weights: {args.binary_prob_weights}).")
        ckpt = torch.load(args.binary_prob_weights, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        classifier = NeutrinoGravNetNodesFaser(
            input_dim=1, num_node_classes=ckpt.get("num_node_classes", 2), faser_dim=5,
            n_gravstack=cfg.get("n_gravstack", 3),
            out_channels=cfg.get("out_channels", 16),
            n_feature_transform=cfg.get("n_feature_transform", 16),
            k=cfg.get("k", 12),
        )
        classifier.load_state_dict(ckpt["model_state_dict"])
        classifier.to(device).eval()
        with torch.no_grad():
            for data in dataset:
                batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                out = classifier(
                    data.x.to(device),
                    data.pos.to(device),
                    batch_vec,
                    data.x_faser.unsqueeze(0).to(device),
                )
                prob = torch.softmax(out, dim=1).cpu()
                data.x = torch.cat([data.x, prob], dim=1)
        del classifier
        torch.cuda.empty_cache()
        logger.info("Binary classifier augmentation complete.")
    if args.truth3b_prob_weights:
        logger.info(f"Running truth3b classifier inference in-memory (weights: {args.truth3b_prob_weights}).")
        ckpt = torch.load(args.truth3b_prob_weights, map_location=device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        classifier = NeutrinoGravNetNodesFaser(
            input_dim=1, num_node_classes=ckpt.get("num_node_classes", 3), faser_dim=5,
            n_gravstack=cfg.get("n_gravstack", 3),
            out_channels=cfg.get("out_channels", 16),
            n_feature_transform=cfg.get("n_feature_transform", 16),
            k=cfg.get("k", 12),
        )
        classifier.load_state_dict(ckpt["model_state_dict"])
        classifier.to(device).eval()
        with torch.no_grad():
            for data in dataset:
                batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                out = classifier(
                    data.x.to(device),
                    data.pos.to(device),
                    batch_vec,
                    data.x_faser.unsqueeze(0).to(device),
                )
                prob = torch.softmax(out, dim=1).cpu()
                data.x = torch.cat([data.x, prob], dim=1)
        del classifier
        torch.cuda.empty_cache()
        logger.info("truth3b classifier augmentation complete.")
    if args.classifier_embedding_weights:
        logger.info(f"Running classifier embedding inference in-memory (weights: {args.classifier_embedding_weights}).")
        ckpt = torch.load(args.classifier_embedding_weights, map_location=device, weights_only=False)
        n_classes = ckpt["num_node_classes"]
        cfg = ckpt.get("model_config", {})
        classifier = NeutrinoGravNetNodesFaser(
            input_dim=1, num_node_classes=n_classes, faser_dim=5,
            n_gravstack=cfg.get("n_gravstack", 3),
            out_channels=cfg.get("out_channels", 16),
            n_feature_transform=cfg.get("n_feature_transform", 16),
            k=cfg.get("k", 12),
        )
        classifier.load_state_dict(ckpt["model_state_dict"])
        classifier.to(device).eval()
        emb_dim = None
        with torch.no_grad():
            for data in dataset:
                batch_vec = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
                _, embedding = classifier(
                    data.x.to(device),
                    data.pos.to(device),
                    batch_vec,
                    data.x_faser.unsqueeze(0).to(device),
                    return_embedding=True,
                )
                data.x = torch.cat([data.x, embedding.cpu()], dim=1)
                if emb_dim is None:
                    emb_dim = embedding.shape[1]
        del classifier
        torch.cuda.empty_cache()
        logger.info(f"Classifier embedding augmentation complete (embedding dim={emb_dim}).")

    # Split into train/val
    train_dataset, val_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")

    # Check data format
    sample = train_dataset[0]
    logger.info(f"Input features dim: {sample.x.shape[1]}")
    logger.info(f"Position dim: {sample.pos.shape[1]}")
    logger.info(f"FASER features dim: {sample.x_faser.shape[0]}")
    logger.info(f"Number of nodes: {sample.num_nodes}")
    logger.info(f"E_nu: {sample.E_nu:.4f} TeV")
    logger.info(f"E_lepton: {sample.E_lepton:.4f} TeV")
    logger.info(f"E_roe: {sample.E_roe:.4f} TeV")

    # Compute normalisation statistics from training set.
    # Standardising both targets to zero mean / unit variance removes the
    # ~31x scale difference between log10(E_nu) and logit(y), so equal MSE
    # weights are correct and neither target dominates the gradient.
    log_enu_vals = torch.stack(
        [torch.log10(d.E_nu.clamp(1e-6)) for d in train_dataset]
    )
    logit_y_vals = torch.stack(
        [torch.log(d.E_roe.clamp(1e-6) / d.E_lepton.clamp(1e-6))
         for d in train_dataset]
    )
    norm_stats = {
        "mu_enu":    log_enu_vals.mean().item(),
        "sigma_enu": log_enu_vals.std().item(),
        "mu_logit":  logit_y_vals.mean().item(),
        "sigma_logit": logit_y_vals.std().item(),
    }
    logger.info(
        f"Normalisation — log10(E_nu): mu={norm_stats['mu_enu']:.4f}, "
        f"sigma={norm_stats['sigma_enu']:.4f}"
    )
    logger.info(
        f"Normalisation — logit(y):    mu={norm_stats['mu_logit']:.4f}, "
        f"sigma={norm_stats['sigma_logit']:.4f}"
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model (2 outputs: log10 E_nu, logit y where y = E_roe/E_nu = Bjorken inelasticity)
    # beta_loss=True: 3 outputs (t1, alpha, beta) with softplus on alpha/beta
    model = NeutrinoGravNetRegressionFASER(
        input_dim=sample.x.shape[1],
        num_targets=2,
        faser_dim=sample.x_faser.shape[0],
        pooling=args.pooling,
        dropout=0.2,
        beta_loss=args.beta_loss,
        use_faser=not args.no_faser_features,
    ).to(device)

    logger.info(f"Model: {model}")
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )
    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0

    metrics = {
        "train_loss": [],
        "train_loss_t1": [],
        "train_loss_t2": [],
        "train_rmse": [],
        "val_loss": [],
        "val_loss_t1": [],
        "val_loss_t2": [],
        "val_rmse": [],
        "train_rel_err": [],   # [epochs, 3] — E_nu, E_lepton, E_roe in physical space
        "val_rel_err": [],
        "train_resolution": [],
        "val_resolution": [],
    }

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", ckpt["val_loss"])
        best_epoch = ckpt.get("best_epoch", ckpt["epoch"])
        metrics_file = weights_path / "training_metrics.npz"
        if metrics_file.exists():
            saved = np.load(metrics_file)
            for key in metrics:
                if key in saved:
                    metrics[key] = list(saved[key])
            logger.info(
                f"Restored {len(metrics['train_loss'])} epochs of metrics "
                f"from {metrics_file}"
            )
        logger.info(
            f"Resuming from epoch {start_epoch + 1}/{args.num_epochs} "
            f"(best val loss so far: {best_val_loss:.4f} at epoch {best_epoch + 1})"
        )

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_loss_t1, train_loss_t2, train_rmse, train_rel_err, train_resolution = train_epoch(
            model, train_loader, optimizer, device, norm_stats, args.loss, args.huber_delta,
            accumulation_steps=args.accumulation_steps, beta_loss=args.beta_loss,
        )

        val_loss, val_loss_t1, val_loss_t2, val_rmse, val_rel_err, val_resolution = validate_epoch(
            model, val_loader, device, norm_stats, args.loss, args.huber_delta,
            beta_loss=args.beta_loss,
        )

        scheduler.step(val_loss)

        metrics["train_loss"].append(train_loss)
        metrics["train_loss_t1"].append(train_loss_t1)
        metrics["train_loss_t2"].append(train_loss_t2)
        metrics["train_rmse"].append(train_rmse)
        metrics["val_loss"].append(val_loss)
        metrics["val_loss_t1"].append(val_loss_t1)
        metrics["val_loss_t2"].append(val_loss_t2)
        metrics["val_rmse"].append(val_rmse)
        metrics["train_rel_err"].append(train_rel_err.numpy())
        metrics["val_rel_err"].append(val_rel_err.numpy())
        metrics["train_resolution"].append(train_resolution.numpy())
        metrics["val_resolution"].append(val_resolution.numpy())

        rel_err_str = ", ".join(
            f"{TARGET_NAMES[i]}: {val_rel_err[i]:.3f}±{val_resolution[i]:.3f}"
            for i in range(3)
        )
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.4f} (t1={train_loss_t1:.4f}, t2={train_loss_t2:.4f}), RMSE: {train_rmse:.4f} - "
            f"Val Loss: {val_loss:.4f} (t1={val_loss_t1:.4f}, t2={val_loss_t2:.4f}), RMSE: {val_rmse:.4f} - "
            f"Val RelErr: {rel_err_str}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_loss_t1": val_loss_t1,
            "val_loss_t2": val_loss_t2,
            "val_rmse": val_rmse,
            "val_rel_err": val_rel_err.numpy(),
            "val_resolution": val_resolution.numpy(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "norm_stats": norm_stats,
            "parametrisation": "reparam_std",
            "loss_fn": args.loss,
            "huber_delta": args.huber_delta,
            "accumulation_steps": args.accumulation_steps,
            "bin_size": args.bin_size,
            "beta_loss": args.beta_loss,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint["best_val_loss"] = best_val_loss
            checkpoint["best_epoch"] = best_epoch
            torch.save(checkpoint, weights_path / "best_model.pt")
            logger.info(f"Saved best model at epoch {epoch + 1}")

        np.savez(
            weights_path / "training_metrics.npz",
            **{key: np.array(value) for key, value in metrics.items()},
        )

        torch.save(checkpoint, weights_path / "latest_checkpoint.pt")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, weights_path / f"checkpoint_epoch_{epoch + 1}.pt")

    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    logger.info(f"Training metrics saved to {weights_path / 'training_metrics.npz'}")


if __name__ == "__main__":
    main()
