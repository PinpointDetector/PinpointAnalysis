#!/usr/bin/env python3
"""
Training script for NeutrinoGravNetFASER model on calorimeter data with FASER spectrometer information.

Usage:
    python -m analysis.gravnet.train_gravnet_faser --data-dir data/processed --batch-size 8 --num-epochs 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetFASER
from analysis.utils.utils import get_torch_path, get_weights_path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_class_weights(dataset, num_classes):
    """Compute class weights for imbalanced dataset."""
    counts = torch.zeros(num_classes)
    for data in dataset:
        counts[data.y_graph] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    logger.info(f"Graph class distribution: {counts.numpy()}")
    logger.info(f"Graph class weights: {weights.numpy()}")
    return weights


def train_epoch(model, loader, optimizer, device, graph_class_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    weighted_correct = 0
    weighted_total = 0

    train_bar = tqdm(loader, desc="Training")
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            # Forward pass with FASER data
            out = model(data.x, data.pos, data.batch, data.x_faser)

            # Check for batch size mismatch
            num_graphs = data.y_graph.size(0)
            if out.size(0) != num_graphs:
                print(
                    f"\nSkipping batch {batch_idx}: output size {out.size(0)} != target size {num_graphs}"
                )
                continue

            # Graph loss
            graph_loss = F.cross_entropy(out, data.y_graph, weight=graph_class_weights)

            loss = graph_loss
        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if grad_norm > 5.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")

        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        # Compute accuracy
        pred = out.argmax(dim=1)
        correct += (pred == data.y_graph).sum().item()
        total += data.num_graphs

        # Compute weighted accuracy
        class_weights = graph_class_weights[data.y_graph]
        weighted_correct += (
            ((pred == data.y_graph).float() * class_weights).sum().item()
        )
        weighted_total += class_weights.sum().item()

        # Update progress bar
        current_loss = total_loss / total
        current_acc = 100 * correct / total
        current_weighted_acc = 100 * weighted_correct / weighted_total
        train_bar.set_postfix(
            {
                "Loss": f"{current_loss:.4f}",
                "Acc": f"{current_acc:.2f}%",
                "WAcc": f"{current_weighted_acc:.2f}%",
            }
        )

    return total_loss / total, correct / total, weighted_correct / weighted_total


def validate_epoch(model, loader, device, graph_class_weights):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    weighted_correct = 0
    weighted_total = 0

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation")
        for data in val_bar:
            data = data.to(device)

            try:
                # Forward pass with FASER data
                out = model(data.x, data.pos, data.batch, data.x_faser)

                # Check for batch size mismatch
                num_graphs = data.y_graph.size(0)
                if out.size(0) != num_graphs:
                    continue

                # Graph loss
                graph_loss = F.cross_entropy(
                    out, data.y_graph, weight=graph_class_weights
                )

                loss = graph_loss
                total_loss += loss.item() * data.num_graphs

                # Compute accuracy
                pred = out.argmax(dim=1)
                correct += (pred == data.y_graph).sum().item()
                total += data.num_graphs

                # Compute weighted accuracy
                class_weights = graph_class_weights[data.y_graph]
                weighted_correct += (
                    ((pred == data.y_graph).float() * class_weights).sum().item()
                )
                weighted_total += class_weights.sum().item()

                # Update progress bar
                current_loss = total_loss / total
                current_acc = 100 * correct / total
                current_weighted_acc = 100 * weighted_correct / weighted_total
                val_bar.set_postfix(
                    {
                        "Loss": f"{current_loss:.4f}",
                        "Acc": f"{current_acc:.2f}%",
                        "WAcc": f"{current_weighted_acc:.2f}%",
                    }
                )
            except RuntimeError as e:
                continue

    return total_loss / total, correct / total, weighted_correct / weighted_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-n", "--num-epochs", type=int, default=50)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "--use-particle-prob",
        action="store_true",
        default=False,
        help="Use particle probabilities predicted from previous network.",
    )
    parser.add_argument(
        "--use-truth-selection",
        action="store_true",
        default=False,
        help="Apply good event selection based on truth information (default: False).",
    )
    args = parser.parse_args()

    particle_suffix = ""
    if args.use_particle_prob:
        particle_suffix = "_particle_prob"

    event_suffix = ""
    if args.use_truth_selection:
        event_suffix = "_good_events"

    # Setup paths
    torch_path = get_torch_path()
    output_path = get_torch_path() / f"gravnet_faser{particle_suffix}{event_suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = get_weights_path() / f"gravnet_faser{particle_suffix}{event_suffix}"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data (from pointnetpp_faser directory with FASER features)
    logger.info("Loading datasets with FASER data...")

    if args.use_truth_selection:
        model_dir = "pointnetpp_faser_good_events"
    else:
        model_dir = "pointnetpp_faser_all_events"

    nue_dataset = torch.load(
        torch_path / f"10000/{model_dir}/nue{particle_suffix}.pt", weights_only=False
    )
    num_dataset = torch.load(
        torch_path / f"10001/{model_dir}/num{particle_suffix}.pt", weights_only=False
    )
    nun_dataset = torch.load(
        torch_path / f"10003/{model_dir}/nun{particle_suffix}.pt", weights_only=False
    )

    logger.info(
        f"Loaded {len(nue_dataset)} nue, {len(num_dataset)} num, {len(nun_dataset)} nun events"
    )
    num_events = 2000
    dataset = (
        nue_dataset[:num_events] + num_dataset[:num_events] + nun_dataset[:num_events]
    )

    # Split into train/val/test
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=0.2, random_state=42
    )
    train_dataset, val_dataset = train_test_split(
        train_dataset, test_size=0.2, random_state=42
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Val size: {len(val_dataset)}")
    logger.info(f"Test size: {len(test_dataset)}")

    # Check data format
    sample = train_dataset[0]
    logger.info(f"Input features dim: {sample.x.shape[1]}")
    logger.info(f"Position dim: {sample.pos.shape[1]}")
    logger.info(f"FASER features dim: {sample.x_faser.shape[0]}")
    logger.info(f"Number of nodes: {sample.num_nodes}")
    logger.info(f"Graph label: {sample.y_graph}")

    # Get number of classes
    num_graph_classes = len(
        torch.unique(torch.tensor([d.y_graph for d in train_dataset]))
    )
    logger.info(f"Number of graph classes: {num_graph_classes}")

    # Compute class weights
    graph_class_weights = compute_class_weights(train_dataset, num_graph_classes)
    graph_class_weights = graph_class_weights.to(device)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Create model with FASER features
    model = NeutrinoGravNetFASER(
        input_dim=sample.x.shape[1],
        num_graph_classes=num_graph_classes,
        faser_dim=sample.x_faser.shape[0],
    ).to(device)

    logger.info(f"Model: {model}")
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0

    # Initialize metrics storage
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "train_weighted_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_weighted_acc": [],
    }

    for epoch in range(args.num_epochs):
        # Train
        train_loss, train_acc, train_weighted_acc = train_epoch(
            model, train_loader, optimizer, device, graph_class_weights
        )

        # Validate
        val_loss, val_acc, val_weighted_acc = validate_epoch(
            model, val_loader, device, graph_class_weights
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["train_weighted_acc"].append(train_weighted_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_weighted_acc"].append(val_weighted_acc)

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, WAcc: {train_weighted_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, WAcc: {val_weighted_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                weights_path / "best_model.pt",
            )
            logger.info(f"Saved best model at epoch {epoch + 1}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                weights_path / f"checkpoint_epoch_{epoch + 1}.pt",
            )

    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")

    # Save training metrics
    logger.info(f"Saving training metrics to {weights_path / 'training_metrics.npz'}")
    np.savez(
        weights_path / "training_metrics.npz",
        train_loss=np.array(metrics["train_loss"]),
        train_acc=np.array(metrics["train_acc"]),
        train_weighted_acc=np.array(metrics["train_weighted_acc"]),
        val_loss=np.array(metrics["val_loss"]),
        val_acc=np.array(metrics["val_acc"]),
        val_weighted_acc=np.array(metrics["val_weighted_acc"]),
    )

    # Test with best model
    logger.info("Testing best model...")
    checkpoint = torch.load(weights_path / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_weighted_acc = validate_epoch(
        model, test_loader, device, graph_class_weights
    )
    logger.info(
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test WAcc: {test_weighted_acc:.4f}"
    )

    # Compute per-class accuracy on test set
    logger.info("Computing per-class test accuracy...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            try:
                out = model(data.x, data.pos, data.batch, data.x_faser)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(data.y_graph.cpu().numpy())
            except RuntimeError:
                continue

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    class_names = ["NC", "CC νe", "CC νμ"]
    logger.info("Per-class test accuracy:")
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        mask = all_targets == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == i).mean()
            class_accuracies.append(class_acc)
            logger.info(f"  {class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
        else:
            class_accuracies.append(0.0)
            logger.info(f"  {class_name}: No samples")

    # Compute weighted average
    class_counts = np.bincount(all_targets, minlength=num_graph_classes)
    weighted_acc = np.sum(np.array(class_accuracies) * class_counts) / len(all_targets)
    logger.info(
        f"Weighted test accuracy: {weighted_acc:.4f} ({weighted_acc * 100:.2f}%)"
    )

    # Save test metrics
    np.savez(
        weights_path / "test_metrics.npz",
        test_loss=test_loss,
        test_acc=test_acc,
        class_accuracies=np.array(class_accuracies),
        weighted_acc=weighted_acc,
        predictions=all_preds,
        targets=all_targets,
    )
    logger.info(f"Test metrics saved to {weights_path / 'test_metrics.npz'}")


if __name__ == "__main__":
    main()
