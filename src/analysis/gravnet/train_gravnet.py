#!/usr/bin/env python3
"""
Training script for NeutrinoGravNet model on calorimeter data.

Usage:
    python -m analysis.gravnet.train_gravnet --data-dir data/processed --batch-size 8 --epochs 100
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

from analysis.gravnet.model import NeutrinoGravNet
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

    train_bar = tqdm(loader, desc="Training")
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            # Forward pass
            out = model(data.x, data.pos, data.batch)

            # Check for batch size mismatch
            num_graphs = data.y_graph.size(0)
            if out.size(0) != num_graphs:
                print(
                    f"\nSkipping batch {batch_idx}: output size {out.size(0)} != target size {num_graphs}"
                )
                continue

            # Graph loss
            graph_loss = F.cross_entropy(
                out, data.y_graph, weight=graph_class_weights.to(device)
            )

            loss = graph_loss
        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        if grad_norm > 20.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")

        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        # Compute accuracy
        pred = out.argmax(dim=1)
        correct += (pred == data.y_graph).sum().item()
        total += data.num_graphs

        # Update progress bar
        current_loss = total_loss / total
        current_acc = 100 * correct / total
        train_bar.set_postfix(
            {
                "Loss": f"{current_loss:.4f}",
                "Acc": f"{current_acc:.2f}%",
            }
        )

    return total_loss / total, correct / total


def validate_epoch(model, loader, device, graph_class_weights):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation")
        for data in val_bar:
            data = data.to(device)

            try:
                # Forward pass
                out = model(data.x, data.pos, data.batch)

                # Check for batch size mismatch
                num_graphs = data.y_graph.size(0)
                if out.size(0) != num_graphs:
                    continue

                # Graph loss
                graph_loss = F.cross_entropy(
                    out, data.y_graph, weight=graph_class_weights.to(device)
                )

                loss = graph_loss
                total_loss += loss.item() * data.num_graphs

                # Compute accuracy
                pred = out.argmax(dim=1)
                correct += (pred == data.y_graph).sum().item()
                total += data.num_graphs

                # Update progress bar
                current_loss = total_loss / total
                current_acc = 100 * correct / total
                val_bar.set_postfix(
                    {
                        "Loss": f"{current_loss:.4f}",
                        "Acc": f"{current_acc:.2f}%",
                    }
                )
            except RuntimeError as e:
                continue

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train NeutrinoGravNet model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument(
        "-g", "--gpu", type=str, default="cuda:0", help="GPU device to use"
    )
    args = parser.parse_args()

    # Setup paths
    torch_path = get_torch_path()
    output_path = get_torch_path() / "gravnet"
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = get_weights_path() / "gravnet"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    nue_dataset = torch.load(torch_path / "10000/pointnetpp/nue.pt", weights_only=False)
    num_dataset = torch.load(torch_path / "10001/pointnetpp/num.pt", weights_only=False)
    nun_dataset = torch.load(torch_path / "10003/pointnetpp/nun.pt", weights_only=False)

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
    logger.info(f"Number of nodes: {sample.num_nodes}")
    logger.info(f"Graph label: {sample.y_graph}")

    # Get number of classes
    num_graph_classes = len(
        torch.unique(torch.tensor([d.y_graph for d in train_dataset]))
    )
    logger.info(f"Number of graph classes: {num_graph_classes}")

    # Compute class weights
    graph_class_weights = compute_class_weights(train_dataset, num_graph_classes)

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

    # Create model
    model = NeutrinoGravNet(
        input_dim=sample.x.shape[1],
        num_graph_classes=num_graph_classes,
        dropout=args.dropout,
    ).to(device)

    logger.info(f"Model: {model}")
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, graph_class_weights
        )

        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, device, graph_class_weights
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
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

    # Test with best model
    logger.info("Testing best model...")
    checkpoint = torch.load(weights_path / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate_epoch(
        model, test_loader, device, graph_class_weights
    )
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
