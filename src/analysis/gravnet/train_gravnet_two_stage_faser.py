#!/usr/bin/env python3
"""
Two-stage training pipeline for neutrino event classification with FASER spectrometer information.

Stage 1: Train node-level hit classification
Stage 2: Use node predictions to create augmented dataset with particle probabilities
Stage 3: Train graph-level event classification using augmented features

Usage:
    python -m analysis.gravnet.train_gravnet_two_stage_faser --batch-size 8 --stage1-epochs 50 --stage2-epochs 50
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetFASER, NeutrinoGravNetNodesFaser
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


# ============================================================================
# STAGE 1: Node Classification Training
# ============================================================================


def train_node_epoch(model, loader, optimizer, device, node_class_weights):
    """Train node classification for one epoch."""
    model.train()
    total_loss = 0
    node_correct = 0
    node_total = 0

    train_bar = tqdm(loader, desc="Stage 1 - Training Nodes")
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            node_out = model(data.x, data.pos, data.batch, data.x_faser)

            if hasattr(data, "y") and data.y is not None:
                node_loss = F.cross_entropy(node_out, data.y, weight=node_class_weights)
                node_pred = node_out.argmax(dim=1)
                node_correct += (node_pred == data.y).sum().item()
                node_total += data.y.size(0)
            else:
                logger.warning("No node labels found in batch")
                continue

            loss = node_loss

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if grad_norm > 50.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")
        optimizer.step()

        total_loss += loss.item() * data.num_nodes

        if node_total > 0:
            current_loss = total_loss / node_total
            current_node_acc = 100 * node_correct / node_total
            train_bar.set_postfix(
                {"Loss": f"{current_loss:.4f}", "NAcc": f"{current_node_acc:.2f}%"}
            )

    avg_loss = total_loss / node_total if node_total > 0 else 0.0
    node_acc = node_correct / node_total if node_total > 0 else 0.0
    return avg_loss, node_acc


def validate_node_epoch(model, loader, device, node_class_weights, num_classes=None):
    """Validate node classification for one epoch."""
    model.eval()
    total_loss = 0
    node_correct = 0
    node_total = 0
    node_weighted_correct = 0
    node_weighted_total = 0

    all_node_preds = []
    all_node_targets = []

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Stage 1 - Validating Nodes")
        for data in val_bar:
            data = data.to(device)

            try:
                node_out = model(data.x, data.pos, data.batch, data.x_faser)

                if hasattr(data, "y") and data.y is not None:
                    node_loss = F.cross_entropy(
                        node_out, data.y, weight=node_class_weights
                    )
                    node_pred = node_out.argmax(dim=1)
                    node_correct += (node_pred == data.y).sum().item()
                    node_total += data.y.size(0)

                    node_weights = node_class_weights[data.y]
                    node_weighted_correct += (
                        ((node_pred == data.y).float() * node_weights).sum().item()
                    )
                    node_weighted_total += node_weights.sum().item()

                    all_node_preds.extend(node_pred.cpu().numpy())
                    all_node_targets.extend(data.y.cpu().numpy())
                else:
                    continue

                loss = node_loss
                total_loss += loss.item() * data.num_nodes

                if node_total > 0:
                    current_loss = total_loss / node_total
                    current_node_acc = 100 * node_correct / node_total
                    current_node_wacc = (
                        100 * node_weighted_correct / node_weighted_total
                    )
                    val_bar.set_postfix(
                        {
                            "Loss": f"{current_loss:.4f}",
                            "NAcc": f"{current_node_acc:.2f}%",
                            "NWAcc": f"{current_node_wacc:.2f}%",
                        }
                    )

            except RuntimeError:
                continue

    avg_loss = total_loss / node_total if node_total > 0 else 0.0
    node_acc = node_correct / node_total if node_total > 0 else 0.0
    node_weighted_acc = (
        node_weighted_correct / node_weighted_total if node_weighted_total > 0 else 0.0
    )

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

    return avg_loss, node_acc, node_weighted_acc, per_class_acc


# ============================================================================
# STAGE 2: Create Dataset with Particle Probabilities
# ============================================================================


def create_dataset_with_particle_probs(model, dataset, device, dataset_name="dataset"):
    """Create augmented dataset with particle probability predictions."""
    model.eval()
    new_dataset = []

    logger.info(f"Creating {dataset_name} with particle probabilities...")
    with torch.no_grad():
        for data in tqdm(dataset, desc=f"Stage 2 - Processing {dataset_name}"):
            data = data.to(device)

            if data.x.size(0) == 0:
                continue

            try:
                node_out = model(data.x, data.pos, data.batch, data.x_faser)
                node_prob = torch.softmax(node_out, dim=1)

                # Augment features with particle probabilities
                new_x = torch.cat([data.x, node_prob], dim=1)

                new_data = Data(
                    x=new_x,
                    pos=data.pos,
                    y_graph=data.y_graph,
                    x_faser=data.x_faser,
                )
                new_dataset.append(new_data)
            except RuntimeError:
                continue

    logger.info(
        f"Created {len(new_dataset)} graphs with particle probabilities for {dataset_name}"
    )
    return new_dataset


# ============================================================================
# STAGE 3: Graph Classification Training
# ============================================================================


def train_graph_epoch(model, loader, optimizer, device, graph_class_weights):
    """Train graph classification for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    weighted_correct = 0
    weighted_total = 0

    train_bar = tqdm(loader, desc="Stage 3 - Training Graphs")
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            out = model(data.x, data.pos, data.batch, data.x_faser)

            num_graphs = data.y_graph.size(0)
            if out.size(0) != num_graphs:
                print(
                    f"\nSkipping batch {batch_idx}: output size {out.size(0)} != target size {num_graphs}"
                )
                continue

            graph_loss = F.cross_entropy(out, data.y_graph, weight=graph_class_weights)
            loss = graph_loss

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if grad_norm > 5.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        pred = out.argmax(dim=1)
        correct += (pred == data.y_graph).sum().item()
        total += data.num_graphs

        class_weights = graph_class_weights[data.y_graph]
        weighted_correct += (
            ((pred == data.y_graph).float() * class_weights).sum().item()
        )
        weighted_total += class_weights.sum().item()

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


def validate_graph_epoch(model, loader, device, graph_class_weights):
    """Validate graph classification for one epoch."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    weighted_correct = 0
    weighted_total = 0

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Stage 3 - Validating Graphs")
        for data in val_bar:
            data = data.to(device)

            try:
                out = model(data.x, data.pos, data.batch, data.x_faser)

                num_graphs = data.y_graph.size(0)
                if out.size(0) != num_graphs:
                    continue

                graph_loss = F.cross_entropy(
                    out, data.y_graph, weight=graph_class_weights
                )

                loss = graph_loss
                total_loss += loss.item() * data.num_graphs

                pred = out.argmax(dim=1)
                correct += (pred == data.y_graph).sum().item()
                total += data.num_graphs

                class_weights = graph_class_weights[data.y_graph]
                weighted_correct += (
                    ((pred == data.y_graph).float() * class_weights).sum().item()
                )
                weighted_total += class_weights.sum().item()

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
            except RuntimeError:
                continue

    return total_loss / total, correct / total, weighted_correct / weighted_total


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage training pipeline for neutrino event classification"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=50)
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
    weights_path = get_weights_path() / f"gravnet_two_stage_faser_{args.data_type}"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Original Data
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Loading original datasets with FASER data...")
    logger.info("=" * 80)

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

    num_nue_events = 400
    num_num_events = len(num_dataset)
    num_nun_events = 1000
    dataset = (
        nue_dataset[:num_nue_events]
        + num_dataset[:num_num_events]
        + nun_dataset[:num_nun_events]
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

    # ========================================================================
    # STAGE 1: Train Node Classification Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STAGE 1: Training Node Classification Model")
    logger.info("=" * 80)

    # Get node labels
    all_node_labels = []
    for data in train_dataset:
        if hasattr(data, "y") and data.y is not None:
            all_node_labels.extend(data.y.tolist())
    num_node_classes = len(torch.unique(torch.tensor(all_node_labels)))
    logger.info(f"Number of node classes: {num_node_classes}")

    # Compute node class weights
    node_class_weights = compute_class_weights(
        train_dataset, num_node_classes, is_node_level=True
    )
    node_class_weights = node_class_weights.to(device)

    # Create data loaders for stage 1
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create node classification model
    node_model = NeutrinoGravNetNodesFaser(
        input_dim=sample.x.shape[1],
        num_node_classes=num_node_classes,
        faser_dim=sample.x_faser.shape[0],
    ).to(device)

    logger.info(
        f"Node model parameters: {sum(p.numel() for p in node_model.parameters() if p.requires_grad)}"
    )

    # Optimizer and scheduler for stage 1
    node_optimizer = Adam(node_model.parameters(), lr=1e-4, weight_decay=1e-5)
    node_scheduler = ReduceLROnPlateau(
        node_optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training loop for stage 1
    best_node_val_loss = float("inf")
    stage1_metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_weighted_acc": [],
    }

    for epoch in range(args.stage1_epochs):
        train_loss, train_acc = train_node_epoch(
            node_model, train_loader, node_optimizer, device, node_class_weights
        )

        val_loss, val_acc, val_weighted_acc, val_per_class_acc = validate_node_epoch(
            node_model,
            val_loader,
            device,
            node_class_weights,
            num_classes=num_node_classes,
        )

        node_scheduler.step(val_loss)

        stage1_metrics["train_loss"].append(train_loss)
        stage1_metrics["train_acc"].append(train_acc)
        stage1_metrics["val_loss"].append(val_loss)
        stage1_metrics["val_acc"].append(val_acc)
        stage1_metrics["val_weighted_acc"].append(val_weighted_acc)

        logger.info(
            f"Stage 1 - Epoch {epoch + 1}/{args.stage1_epochs} - "
            f"Train Loss: {train_loss:.4f}, Node Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Node Acc: {val_acc:.4f}/{val_weighted_acc:.4f}"
        )

        if val_per_class_acc:
            node_class_names = ["other", "Electron", "Muon"]
            logger.info("Per-class validation accuracy:")
            for i, (class_name, class_acc) in enumerate(
                zip(node_class_names[:num_node_classes], val_per_class_acc)
            ):
                logger.info(f"  {class_name}: {class_acc:.4f} ({class_acc * 100:.2f}%)")

        if val_loss < best_node_val_loss:
            best_node_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": node_model.state_dict(),
                    "optimizer_state_dict": node_optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                weights_path / "stage1_best_model.pt",
            )
            logger.info(f"Saved best node model at epoch {epoch + 1}")

    # Save stage 1 metrics
    np.savez(
        weights_path / "stage1_metrics.npz",
        **{key: np.array(value) for key, value in stage1_metrics.items()},
    )

    # ========================================================================
    # STAGE 2: Create Augmented Datasets with Particle Probabilities
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STAGE 2: Creating Augmented Datasets with Particle Probabilities")
    logger.info("=" * 80)

    # Load best node model
    checkpoint = torch.load(weights_path / "stage1_best_model.pt")
    node_model.load_state_dict(checkpoint["model_state_dict"])

    # Create augmented datasets
    train_dataset_aug = create_dataset_with_particle_probs(
        node_model, train_dataset, device, "train"
    )
    val_dataset_aug = create_dataset_with_particle_probs(
        node_model, val_dataset, device, "val"
    )
    test_dataset_aug = create_dataset_with_particle_probs(
        node_model, test_dataset, device, "test"
    )

    # Save augmented datasets
    output_path = weights_path / "augmented_datasets"
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving augmented datasets to {output_path}")
    torch.save(train_dataset_aug, output_path / "train_particle_prob.pt")
    torch.save(val_dataset_aug, output_path / "val_particle_prob.pt")
    torch.save(test_dataset_aug, output_path / "test_particle_prob.pt")
    logger.info("Augmented datasets saved successfully")

    # ========================================================================
    # STAGE 3: Train Graph Classification Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STAGE 3: Training Graph Classification Model")
    logger.info("=" * 80)

    # Get graph labels
    num_graph_classes = len(
        torch.unique(torch.tensor([d.y_graph for d in train_dataset_aug]))
    )
    logger.info(f"Number of graph classes: {num_graph_classes}")

    # Compute graph class weights
    graph_class_weights = compute_class_weights(
        train_dataset_aug, num_graph_classes, is_node_level=False
    )
    graph_class_weights = graph_class_weights.to(device)

    # Create data loaders for stage 3
    train_loader_aug = DataLoader(
        train_dataset_aug, batch_size=args.batch_size, shuffle=True
    )
    val_loader_aug = DataLoader(
        val_dataset_aug, batch_size=args.batch_size, shuffle=False
    )
    test_loader_aug = DataLoader(
        test_dataset_aug, batch_size=args.batch_size, shuffle=False
    )

    # Create graph classification model
    sample_aug = train_dataset_aug[0]
    graph_model = NeutrinoGravNetFASER(
        input_dim=sample_aug.x.shape[1],
        num_graph_classes=num_graph_classes,
        faser_dim=sample_aug.x_faser.shape[0],
    ).to(device)

    logger.info(
        f"Graph model parameters: {sum(p.numel() for p in graph_model.parameters() if p.requires_grad)}"
    )

    # Optimizer and scheduler for stage 3
    graph_optimizer = Adam(graph_model.parameters(), lr=1e-4, weight_decay=1e-5)
    graph_scheduler = ReduceLROnPlateau(
        graph_optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Training loop for stage 3
    best_graph_val_loss = float("inf")
    stage3_metrics = {
        "train_loss": [],
        "train_acc": [],
        "train_weighted_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_weighted_acc": [],
    }

    for epoch in range(args.stage2_epochs):
        train_loss, train_acc, train_weighted_acc = train_graph_epoch(
            graph_model, train_loader_aug, graph_optimizer, device, graph_class_weights
        )

        val_loss, val_acc, val_weighted_acc = validate_graph_epoch(
            graph_model, val_loader_aug, device, graph_class_weights
        )

        graph_scheduler.step(val_loss)

        stage3_metrics["train_loss"].append(train_loss)
        stage3_metrics["train_acc"].append(train_acc)
        stage3_metrics["train_weighted_acc"].append(train_weighted_acc)
        stage3_metrics["val_loss"].append(val_loss)
        stage3_metrics["val_acc"].append(val_acc)
        stage3_metrics["val_weighted_acc"].append(val_weighted_acc)

        logger.info(
            f"Stage 3 - Epoch {epoch + 1}/{args.stage2_epochs} - "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, WAcc: {train_weighted_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, WAcc: {val_weighted_acc:.4f}"
        )

        if val_loss < best_graph_val_loss:
            best_graph_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": graph_model.state_dict(),
                    "optimizer_state_dict": graph_optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                weights_path / "stage3_best_model.pt",
            )
            logger.info(f"Saved best graph model at epoch {epoch + 1}")

    # Save stage 3 metrics
    np.savez(
        weights_path / "stage3_metrics.npz",
        **{key: np.array(value) for key, value in stage3_metrics.items()},
    )

    # ========================================================================
    # Final Testing
    # ========================================================================
    logger.info("=" * 80)
    logger.info("FINAL TESTING")
    logger.info("=" * 80)

    # Test stage 3 graph classification model
    checkpoint = torch.load(weights_path / "stage3_best_model.pt")
    graph_model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_weighted_acc = validate_graph_epoch(
        graph_model, test_loader_aug, device, graph_class_weights
    )
    logger.info(
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test WAcc: {test_weighted_acc:.4f}"
    )

    # Compute per-class accuracy
    graph_model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data in test_loader_aug:
            data = data.to(device)
            try:
                out = graph_model(data.x, data.pos, data.batch, data.x_faser)
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

    # Save test metrics
    np.savez(
        weights_path / "test_metrics.npz",
        test_loss=test_loss,
        test_acc=test_acc,
        test_weighted_acc=test_weighted_acc,
        class_accuracies=np.array(class_accuracies),
        predictions=all_preds,
        targets=all_targets,
    )

    logger.info("=" * 80)
    logger.info("TWO-STAGE TRAINING PIPELINE COMPLETED")
    logger.info("=" * 80)
    logger.info(f"All weights saved to: {weights_path}")


if __name__ == "__main__":
    main()
