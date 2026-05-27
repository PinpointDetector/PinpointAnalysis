#!/usr/bin/env python3
"""
Training script for NeutrinoGravNetNodesFaser model on calorimeter data with FASER spectrometer information.
Performs node-level hit classification only.

Usage:
    python -m analysis.gravnet.train_gravnet_nodes_faser --batch-size 8 --num-epochs 100
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

from analysis.gravnet.model import NeutrinoGravNetNodesFaser
from analysis.utils.utils import get_torch_path, get_weights_path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_class_weights(dataset, num_classes, is_node_level=False):
    """Compute class weights for imbalanced dataset."""
    counts = torch.zeros(num_classes)

    if is_node_level:
        # Count node labels across all graphs efficiently using bincount
        for data in dataset:
            if hasattr(data, "y") and data.y is not None:
                # Use bincount for efficient counting
                node_counts = torch.bincount(data.y, minlength=num_classes)
                counts += node_counts[:num_classes]
    else:
        # Count graph labels
        for data in dataset:
            counts[data.y_graph] += 1

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    label_type = "Node" if is_node_level else "Graph"
    logger.info(f"{label_type} class distribution: {counts.numpy()}")
    logger.info(f"{label_type} class weights: {weights.numpy()}")
    return weights


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    node_class_weights,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    # Node metrics
    node_correct = 0
    node_total = 0
    # node_weighted_correct = 0
    # node_weighted_total = 0

    train_bar = tqdm(loader, desc="Training")
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            # Forward pass with FASER data
            node_out = model(data.x, data.pos, data.batch, data.x_faser)

            # Node loss (if node labels are available)
            if hasattr(data, "y") and data.y is not None:
                node_loss = F.cross_entropy(node_out, data.y, weight=node_class_weights)

                # Node accuracy
                node_pred = node_out.argmax(dim=1)
                node_correct += (node_pred == data.y).sum().item()
                node_total += data.y.size(0)

                # Node weighted accuracy
                # node_weights = node_class_weights[data.y]
                # node_weighted_correct += (
                #     ((node_pred == data.y).float() * node_weights).sum().item()
                # )
                # node_weighted_total += node_weights.sum().item()
            else:
                logger.warning("No node labels found in batch")
                continue

            loss = node_loss

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if grad_norm > 50.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")

        optimizer.step()

        total_loss += loss.item() * data.num_nodes

        # Update progress bar
        if node_total > 0:
            current_loss = total_loss / node_total
            current_node_acc = 100 * node_correct / node_total
            # current_node_wacc = 100 * node_weighted_correct / node_weighted_total

            postfix = {
                "Loss": f"{current_loss:.4f}",
                "NAcc": f"{current_node_acc:.2f}%",
                # "NWAcc": f"{current_node_wacc:.2f}%",
            }

            train_bar.set_postfix(postfix)

    # Compute averages
    avg_loss = total_loss / node_total if node_total > 0 else 0.0
    node_acc = node_correct / node_total if node_total > 0 else 0.0
    # node_weighted_acc = (
    #     node_weighted_correct / node_weighted_total if node_weighted_total > 0 else 0.0
    # )

    return (
        avg_loss,
        node_acc,
        # node_weighted_acc,
        0.0,  # placeholder for weighted acc
    )


def validate_epoch(
    model,
    loader,
    device,
    node_class_weights,
    num_classes=None,
):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0

    # Node metrics
    node_correct = 0
    node_total = 0
    node_weighted_correct = 0
    node_weighted_total = 0

    # Per-class metrics
    all_node_preds = []
    all_node_targets = []

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation")
        for data in val_bar:
            data = data.to(device)

            try:
                # Forward pass with FASER data
                node_out = model(data.x, data.pos, data.batch, data.x_faser)

                # Node loss (if node labels are available)
                if hasattr(data, "y") and data.y is not None:
                    node_loss = F.cross_entropy(
                        node_out, data.y, weight=node_class_weights
                    )

                    # Node accuracy
                    node_pred = node_out.argmax(dim=1)
                    node_correct += (node_pred == data.y).sum().item()
                    node_total += data.y.size(0)

                    # Node weighted accuracy
                    node_weights = node_class_weights[data.y]
                    node_weighted_correct += (
                        ((node_pred == data.y).float() * node_weights).sum().item()
                    )
                    node_weighted_total += node_weights.sum().item()

                    # Store for per-class accuracy
                    all_node_preds.extend(node_pred.cpu().numpy())
                    all_node_targets.extend(data.y.cpu().numpy())
                else:
                    continue

                loss = node_loss

                total_loss += loss.item() * data.num_nodes

                # Update progress bar
                if node_total > 0:
                    current_loss = total_loss / node_total
                    current_node_acc = 100 * node_correct / node_total
                    current_node_wacc = (
                        100 * node_weighted_correct / node_weighted_total
                    )

                    postfix = {
                        "Loss": f"{current_loss:.4f}",
                        "NAcc": f"{current_node_acc:.2f}%",
                        "NWAcc": f"{current_node_wacc:.2f}%",
                    }

                    val_bar.set_postfix(postfix)

            except RuntimeError as e:
                continue

    # Compute averages
    avg_loss = total_loss / node_total if node_total > 0 else 0.0
    node_acc = node_correct / node_total if node_total > 0 else 0.0
    node_weighted_acc = (
        node_weighted_correct / node_weighted_total if node_weighted_total > 0 else 0.0
    )

    # Compute per-class accuracies
    per_class_acc = []
    if num_classes is not None and len(all_node_preds) > 0:
        all_node_preds = np.array(all_node_preds)
        all_node_targets = np.array(all_node_targets)
        for i in range(num_classes):
            mask = all_node_targets == i
            if mask.sum() > 0:
                class_acc = (all_node_preds[mask] == i).mean()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)

    return (
        avg_loss,
        node_acc,
        node_weighted_acc,
        per_class_acc,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train NeutrinoGravNetNodesFaser model"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-n", "--num-epochs", type=int, default=50)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-d",
        "--data-type",
        type=str,
        default="all",
        choices=["all", "good"],
        help="Use all events or only good events",
    )
    args = parser.parse_args()

    # Setup paths
    torch_path = get_torch_path()
    output_path = get_torch_path() / f"gravnet_nodes_faser_{args.data_type}_events"
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = get_weights_path() / f"gravnet_nodes_faser_{args.data_type}_events"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data (from pointnetpp_faser directory with FASER features)
    logger.info("Loading datasets with FASER data...")
    nue_dataset = torch.load(
        torch_path / f"10000/pointnetpp_faser_{args.data_type}_events/nue.pt",
        weights_only=False,
    )
    num_dataset = torch.load(
        torch_path / f"10001/pointnetpp_faser_{args.data_type}_events/num.pt",
        weights_only=False,
    )
    nun_dataset = torch.load(
        torch_path / f"10003/pointnetpp_faser_{args.data_type}_events/nun.pt",
        weights_only=False,
    )

    logger.info(
        f"Loaded {len(nue_dataset)} nue, {len(num_dataset)} num, {len(nun_dataset)} nun events"
    )
    print(len(nue_dataset), len(num_dataset), len(nun_dataset))
    num_nue_events = 400
    num_num_events = len(num_dataset)
    num_nun_events = 1000
    print(num_nue_events, num_num_events, num_nun_events)
    dataset = (
        nue_dataset[:num_nue_events]
        + num_dataset[:num_num_events]
        + nun_dataset[:num_nun_events]
    )

    # num_events = 2000
    # dataset = (
    #     nue_dataset[:num_events] + num_dataset[:num_events] + nun_dataset[:num_events]
    # )

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
    if hasattr(sample, "y") and sample.y is not None:
        logger.info(f"Node labels available: {sample.y.shape}")
    else:
        logger.info("Node labels not available")

    # Check if node labels are available
    has_node_labels = hasattr(train_dataset[0], "y") and train_dataset[0].y is not None
    if has_node_labels:
        # Get unique node labels across all training data
        all_node_labels = []
        for data in train_dataset:
            if hasattr(data, "y") and data.y is not None:
                all_node_labels.extend(data.y.tolist())
        num_node_classes = len(torch.unique(torch.tensor(all_node_labels)))
        logger.info(f"Number of node classes: {num_node_classes}")
    else:
        logger.error("Node labels not found in dataset")
        return

    # Compute class weights
    node_class_weights = compute_class_weights(
        train_dataset, num_node_classes, is_node_level=True
    )
    node_class_weights = node_class_weights.to(device)

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

    # Create model with FASER features and node classification only
    model = NeutrinoGravNetNodesFaser(
        input_dim=sample.x.shape[1],
        num_node_classes=num_node_classes,
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
        "train_node_acc": [],
        "train_node_weighted_acc": [],
        "val_loss": [],
        "val_node_acc": [],
        "val_node_weighted_acc": [],
    }

    for epoch in range(args.num_epochs):
        # Train
        (
            train_loss,
            train_node_acc,
            train_node_wacc,
        ) = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            node_class_weights,
        )

        # Validate
        (
            val_loss,
            val_node_acc,
            val_node_wacc,
            val_per_class_acc,
        ) = validate_epoch(
            model,
            val_loader,
            device,
            node_class_weights,
            num_classes=num_node_classes,
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_node_acc"].append(train_node_acc)
        metrics["train_node_weighted_acc"].append(train_node_wacc)
        metrics["val_loss"].append(val_loss)
        metrics["val_node_acc"].append(val_node_acc)
        metrics["val_node_weighted_acc"].append(val_node_wacc)

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Node Acc: {train_node_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Node Acc: {val_node_acc:.4f}/{val_node_wacc:.4f}"
        )

        # Log per-class validation accuracy
        if val_per_class_acc:
            node_class_names = ["other", "Electron", "Muon"]
            logger.info("Per-class validation accuracy:")
            for i, (class_name, class_acc) in enumerate(
                zip(node_class_names[:num_node_classes], val_per_class_acc)
            ):
                logger.info(f"  {class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%)")

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
                    "val_node_acc": val_node_acc,
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
                    "val_node_acc": val_node_acc,
                },
                weights_path / f"checkpoint_epoch_{epoch + 1}.pt",
            )

    logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")

    # Save training metrics
    logger.info(f"Saving training metrics to {weights_path / 'training_metrics.npz'}")
    np.savez(
        weights_path / "training_metrics.npz",
        **{key: np.array(value) for key, value in metrics.items()},
    )

    # Test with best model
    logger.info("Testing best model...")
    checkpoint = torch.load(weights_path / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        test_loss,
        test_node_acc,
        test_node_wacc,
        test_per_class_acc,
    ) = validate_epoch(
        model,
        test_loader,
        device,
        node_class_weights,
        num_classes=num_node_classes,
    )
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Node Acc: {test_node_acc:.4f}, WAcc: {test_node_wacc:.4f}")

    # Compute per-class accuracy on test set
    logger.info("Computing per-class test accuracy...")
    model.eval()
    all_node_preds = []
    all_node_targets = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            try:
                node_out = model(data.x, data.pos, data.batch, data.x_faser)

                # Node predictions
                if hasattr(data, "y") and data.y is not None:
                    node_pred = node_out.argmax(dim=1)
                    all_node_preds.extend(node_pred.cpu().numpy())
                    all_node_targets.extend(data.y.cpu().numpy())
            except RuntimeError:
                continue

    # Node-level per-class accuracy
    if len(all_node_preds) > 0:
        all_node_preds = np.array(all_node_preds)
        all_node_targets = np.array(all_node_targets)

        node_class_names = ["Electron", "Muon", "Hadron"]  # Adjust as needed
        logger.info("Per-class node test accuracy:")
        node_class_accuracies = []
        for i, class_name in enumerate(node_class_names[:num_node_classes]):
            mask = all_node_targets == i
            if mask.sum() > 0:
                class_acc = (all_node_preds[mask] == i).mean()
                node_class_accuracies.append(class_acc)
                logger.info(f"  {class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
            else:
                node_class_accuracies.append(0.0)
                logger.info(f"  {class_name}: No samples")
    else:
        node_class_accuracies = []
        logger.warning("No node predictions made")

    # Save test metrics
    np.savez(
        weights_path / "test_metrics.npz",
        test_loss=test_loss,
        test_node_acc=test_node_acc,
        test_node_weighted_acc=test_node_wacc,
    )
    logger.info(f"Test metrics saved to {weights_path / 'test_metrics.npz'}")


if __name__ == "__main__":
    main()
