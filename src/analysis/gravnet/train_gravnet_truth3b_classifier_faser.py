#!/usr/bin/env python3
"""
3-class node classifier (truth3b): other+muon / secondary_e / primary_EM_e.
Label mapping from data.pdg_label: 0(other)→0, 1(secondary_e)→1, 2(primary_EM_e)→2, 3(muon)→0.

Motivation: compared to the binary classifier, explicitly separating secondary_e from
other/muon forces richer latent representations that may sharpen the primary_EM_e boundary.

Usage:
    python -m analysis.gravnet.train_gravnet_truth3b_classifier_faser \
        -r 10000 -b 8 --num-epochs 75 -g cuda:0
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetNodesFaser
from analysis.utils.utils import get_torch_path, get_weights_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

NUM_NODE_CLASSES = 3
CLASS_NAMES = ["other_mu", "secondary_e", "primary_EM_e"]

# pdg_label 0(other)→0, 1(secondary_e)→1, 2(primary_EM_e)→2, 3(muon)→0
LABEL_REMAP = torch.tensor([0, 1, 2, 0])


def get_str_from_run(run: int) -> str:
    if run % 4 == 0:
        return "nue"
    elif run % 4 == 1:
        return "num"
    elif run % 4 == 2:
        return "nut"
    elif run % 4 == 3:
        return "nun"


def compute_class_weights(dataset):
    """Compute inverse-frequency class weights from label_3b."""
    counts = torch.zeros(NUM_NODE_CLASSES)
    for data in dataset:
        counts += torch.bincount(data.label_3b, minlength=NUM_NODE_CLASSES).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_NODE_CLASSES
    logger.info(f"Label counts : {counts.long().tolist()}  "
                f"(other_mu={counts[0]:.0f}, secondary_e={counts[1]:.0f}, primary_EM_e={counts[2]:.0f})")
    logger.info(f"Class weights: other_mu={weights[0]:.4f}, secondary_e={weights[1]:.4f}, primary_EM_e={weights[2]:.4f}")
    return weights


def train_epoch(model, loader, optimizer, device, class_weights, accumulation_steps=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    optimizer.zero_grad()
    for batch_idx, data in enumerate(tqdm(loader, desc="Training")):
        data = data.to(device)
        try:
            node_out = model(data.x, data.pos, data.batch, data.x_faser)
            loss = F.cross_entropy(node_out, data.label_3b, weight=class_weights)
            pred = node_out.argmax(dim=1)
            correct += (pred == data.label_3b).sum().item()
            total += data.label_3b.size(0)
        except RuntimeError as e:
            logger.warning(f"Skipping batch {batch_idx}: {e}")
            continue

        (loss / accumulation_steps).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * data.label_3b.size(0)

        if not sys.stdout.isatty() and (batch_idx + 1) % 100 == 0:
            logger.info(
                f"Batch {batch_idx + 1}/{len(loader)}: "
                f"Loss={total_loss/total:.4f}, Acc={correct/total:.4f}"
            )

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def validate_epoch(model, loader, device, class_weights):
    """Validate; returns loss, accuracy, multiclass AUC-ROC (OvR), per-class metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []   # shape [N, 3] after stacking — for multiclass AUC-ROC

    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            data = data.to(device)
            try:
                node_out = model(data.x, data.pos, data.batch, data.x_faser)
                loss = F.cross_entropy(node_out, data.label_3b, weight=class_weights)
                pred = node_out.argmax(dim=1)
                prob = torch.softmax(node_out, dim=1)  # [N, 3]

                correct += (pred == data.label_3b).sum().item()
                total += data.label_3b.size(0)
                total_loss += loss.item() * data.label_3b.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(data.label_3b.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
            except RuntimeError:
                continue

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    val_auc = 0.0
    per_class_precision = np.zeros(NUM_NODE_CLASSES)
    per_class_recall    = np.zeros(NUM_NODE_CLASSES)
    per_class_f1        = np.zeros(NUM_NODE_CLASSES)
    per_class_acc       = np.zeros(NUM_NODE_CLASSES)

    if len(all_preds) > 0:
        all_preds   = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs   = np.array(all_probs)  # [N, 3]
        for i in range(NUM_NODE_CLASSES):
            mask = all_targets == i
            per_class_acc[i] = (all_preds[mask] == i).mean() if mask.sum() > 0 else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, labels=[0, 1, 2], zero_division=0
        )
        per_class_precision = precision
        per_class_recall    = recall
        per_class_f1        = f1
        try:
            val_auc = roc_auc_score(all_targets, all_probs, multi_class="ovr")
        except ValueError:
            val_auc = 0.0

    return avg_loss, acc, val_auc, per_class_acc, per_class_precision, per_class_recall, per_class_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train truth3b GravNet node classifier (other+mu / secondary_e / primary_EM_e)"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-n", "--num-epochs", type=int, default=75)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-d", "--data-type", type=str, default="all",
        choices=["all", "good"],
    )
    parser.add_argument(
        "-r", "--runs", nargs="+", type=int, default=[10000],
        help="Run numbers to load (default: 10000)",
    )
    parser.add_argument(
        "-c", "--chunks", nargs="*", type=int, default=None,
        help="Chunk numbers to load (default: all chunks)",
    )
    parser.add_argument(
        "--num-events", type=int, default=None, metavar="N",
        help="Cap events per chunk (default: all)",
    )
    parser.add_argument(
        "--bin-size", type=str, default="200um",
        help="Bin size of input .pt files (default: 200um)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="CHECKPOINT",
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument("--suffix", type=str, default=None,
        help="Optional suffix appended to output directory name")
    parser.add_argument("--n-gravstack",         type=int, default=3)
    parser.add_argument("--out-channels",        type=int, default=16)
    parser.add_argument("--n-feature-transform", type=int, default=16)
    parser.add_argument("--k",                   type=int, default=12)
    parser.add_argument("--accumulation-steps",  type=int, default=1)
    parser.add_argument(
        "--vertex-dist",
        action="store_true",
        default=False,
        help="Append ground-truth distance from each node to the true neutrino interaction "
             "vertex as an additional node feature. Upper-bound study only — uses Geant4 truth.",
    )
    args = parser.parse_args()

    torch_path = get_torch_path()
    dir_name = "gravnet_truth3b_classifier_faser"
    if args.vertex_dist:
        dir_name += "_vertexdist"
    if args.suffix:
        dir_name += f"_{args.suffix}"

    if args.resume:
        weights_path = Path(args.resume).parent
    else:
        weights_path = get_weights_path() / dir_name
        if weights_path.exists():
            stamp = datetime.datetime.now().strftime("%m%d_%H%M")
            weights_path = weights_path.parent / f"{weights_path.name}_{stamp}"
            logger.warning(f"Output dir exists — writing to {weights_path}")
    weights_path.mkdir(parents=True, exist_ok=True)

    log_file = weights_path / "training.log"
    fh = logging.FileHandler(log_file, mode="a" if args.resume else "w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.info(f"Logging to {log_file}")

    logger.info("=" * 70)
    logger.info("Training configuration:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k}: {v}")
    logger.info("=" * 70)

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    dataset = []
    bin_suffix = f"_{args.bin_size}" if args.bin_size != "200um" else ""

    for run in args.runs:
        run_str  = get_str_from_run(run)
        run_path = torch_path / f"{run}/pointnetpp_faser_{args.data_type}_events{bin_suffix}"

        if args.chunks is None:
            chunk_files = sorted(run_path.glob(f"{run_str}_*.pt"))
            chunk_files = [f for f in chunk_files
                           if "_particle_prob" not in f.stem
                           and "_binary_prob" not in f.stem
                           and "_truth3b_prob" not in f.stem]
            chunks_to_load = [int(f.stem.split("_")[-1]) for f in chunk_files]
        else:
            chunks_to_load = args.chunks

        logger.info(f"Run {run} ({run_str}): {len(chunks_to_load)} chunks")
        for chunk in chunks_to_load:
            chunk_file = run_path / f"{run_str}_{chunk:03d}.pt"
            if chunk_file.exists():
                chunk_data = torch.load(chunk_file, weights_only=False)
                if args.num_events is not None:
                    chunk_data = chunk_data[: args.num_events]
                for d in chunk_data:
                    d.label_3b = LABEL_REMAP[d.pdg_label]
                    if args.vertex_dist:
                        vertex_dist = torch.norm(
                            d.pos - d.true_pos_centered.unsqueeze(0), dim=1, keepdim=True
                        )  # [N, 1]
                        d.x = torch.cat([d.x, vertex_dist], dim=1)  # [N, 2]
                dataset.extend(chunk_data)
                logger.info(f"  chunk {chunk}: {len(chunk_data)} events")
            else:
                logger.warning(f"  chunk file not found: {chunk_file}")

    logger.info(f"Total events loaded: {len(dataset)}")

    sample = dataset[0]
    if not hasattr(sample, "pdg_label") or sample.pdg_label is None:
        logger.error("data.pdg_label not found — check dataset version")
        return
    logger.info(f"Input features dim : {sample.x.shape[1]}")
    logger.info(f"Position dim       : {sample.pos.shape[1]}")
    logger.info(f"FASER features dim : {sample.x_faser.shape[0]}")
    logger.info(f"Nodes per event    : {sample.num_nodes}")
    prim_EM_frac = sum((d.label_3b == 2).sum().item() for d in dataset) / \
                   sum(d.label_3b.size(0) for d in dataset)
    logger.info(f"primary_EM_e fraction: {prim_EM_frac*100:.2f}%")

    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_dataset)}   Val: {len(val_dataset)}")

    class_weights = compute_class_weights(train_dataset).to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    model = NeutrinoGravNetNodesFaser(
        input_dim=sample.x.shape[1],
        num_node_classes=NUM_NODE_CLASSES,
        faser_dim=sample.x_faser.shape[0],
        n_gravstack=args.n_gravstack,
        out_channels=args.out_channels,
        n_feature_transform=args.n_feature_transform,
        k=args.k,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)

    start_epoch   = 0
    best_val_loss = float("inf")
    best_epoch    = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        best_epoch    = ckpt.get("epoch", 0)
        logger.info(f"Resumed from epoch {start_epoch} (val_loss={best_val_loss:.4f})")

    metrics = {k: [] for k in [
        "train_loss", "train_acc",
        "val_loss", "val_acc", "val_auc",
        "val_per_class_acc", "val_per_class_f1",
        "val_per_class_precision", "val_per_class_recall",
    ]}

    if args.resume:
        metrics_file = weights_path / "training_metrics.npz"
        if metrics_file.exists():
            existing = np.load(metrics_file, allow_pickle=True)
            for k in metrics:
                if k in existing:
                    metrics[k] = list(existing[k])
            logger.info(f"Prepended {len(metrics['train_loss'])} epochs of prior metrics")

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, class_weights,
            accumulation_steps=args.accumulation_steps,
        )
        val_loss, val_acc, val_auc, val_per_class_acc, val_precision, val_recall, val_f1 = validate_epoch(
            model, val_loader, device, class_weights,
        )
        scheduler.step(val_loss)

        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_auc"].append(val_auc)
        metrics["val_per_class_acc"].append(val_per_class_acc)
        metrics["val_per_class_f1"].append(val_f1)
        metrics["val_per_class_precision"].append(val_precision)
        metrics["val_per_class_recall"].append(val_recall)

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}  "
            f"val_auc={val_auc:.4f}"
        )
        for name, acc, prec, rec, f1 in zip(CLASS_NAMES, val_per_class_acc, val_precision, val_recall, val_f1):
            logger.info(f"  {name}: acc={acc*100:.1f}%  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

        ckpt = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss":             val_loss,
            "val_acc":              val_acc,
            "val_auc":              val_auc,
            "best_val_loss":        best_val_loss,
            "num_node_classes":     NUM_NODE_CLASSES,
            "class_names":          CLASS_NAMES,
            "model_config": {
                "n_gravstack":         args.n_gravstack,
                "out_channels":        args.out_channels,
                "n_feature_transform": args.n_feature_transform,
                "k":                   args.k,
            },
        }
        torch.save(ckpt, weights_path / "latest_checkpoint.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, weights_path / "best_model.pt")
            logger.info(f"  ↑ new best (epoch {epoch+1}, val_loss={val_loss:.4f}, "
                        f"val_auc={val_auc:.4f})")

        np.savez(
            weights_path / "training_metrics.npz",
            **{k: np.array(v) for k, v in metrics.items()},
        )

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
