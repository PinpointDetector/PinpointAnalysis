#!/usr/bin/env python3
"""
Training script for NeutrinoGravNetFASERWithNodeClassification model on calorimeter data with FASER spectrometer information.
Performs both node-level hit classification and graph-level event classification.

Usage:
    python -m analysis.gravnet.train_gravnet_faser_with_nodes --batch-size 8 --epochs 100
"""

import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import uproot
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetNodesFaser
from analysis.utils.utils import get_root_path, get_torch_path, get_weights_path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_class_weights(
    dataset, num_classes, is_node_level=False, label_attr="y_graph"
):
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
        # Count graph labels or other attributes
        for data in dataset:
            label = getattr(data, label_attr)
            counts[label] += 1

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
    """Train for one epoch on node-level labels."""
    model.train()
    total_loss = 0

    # Node metrics
    node_correct = 0
    node_total = 0
    node_weighted_correct = 0
    node_weighted_total = 0

    train_bar = tqdm(loader, desc="Training", disable=not sys.stdout.isatty())
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            # Forward pass with FASER data
            node_out = model(data.x, data.pos, data.batch, data.x_faser)

            # Node loss (requires node labels)
            if not (hasattr(data, "y") and data.y is not None):
                print(f"\nSkipping batch {batch_idx}: no node labels available")
                continue

            node_loss = F.cross_entropy(node_out, data.y, weight=node_class_weights)

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
        current_loss = total_loss / node_total if node_total > 0 else 0
        current_node_acc = 100 * node_correct / node_total if node_total > 0 else 0
        current_node_wacc = (
            100 * node_weighted_correct / node_weighted_total
            if node_weighted_total > 0
            else 0
        )

        postfix = {
            "Loss": f"{current_loss:.4f}",
            "NAcc": f"{current_node_acc:.2f}%",
            "NWAcc": f"{current_node_wacc:.2f}%",
        }

        train_bar.set_postfix(postfix)

        # Print log output every 100 batches if progress bar is disabled
        if not sys.stdout.isatty() and (batch_idx + 1) % 100 == 0:
            log_msg = f"Batch {batch_idx + 1}: Loss={current_loss:.4f}, NAcc={current_node_acc:.2f}%, NWAcc={current_node_wacc:.2f}%"
            logger.info(log_msg)

    # Compute averages
    avg_loss = total_loss / node_total if node_total > 0 else 0.0
    node_acc = node_correct / node_total if node_total > 0 else 0.0
    node_weighted_acc = (
        node_weighted_correct / node_weighted_total if node_weighted_total > 0 else 0.0
    )

    return avg_loss, node_acc, node_weighted_acc


def validate_epoch(
    model,
    loader,
    device,
    node_class_weights,
):
    """Validate for one epoch on node-level labels."""
    model.eval()
    total_loss = 0

    # Node metrics
    node_correct = 0
    node_total = 0
    node_weighted_correct = 0
    node_weighted_total = 0

    # Per-class node predictions
    all_node_preds = []
    all_node_targets = []

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation", disable=not sys.stdout.isatty())
        for data in val_bar:
            data = data.to(device)

            try:
                # Forward pass with FASER data
                node_out = model(data.x, data.pos, data.batch, data.x_faser)

                # Node loss (requires node labels)
                if not (hasattr(data, "y") and data.y is not None):
                    continue

                node_loss = F.cross_entropy(node_out, data.y, weight=node_class_weights)

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

                total_loss += node_loss.item() * data.num_nodes

                # Store predictions for per-class accuracy
                all_node_preds.extend(node_pred.cpu().numpy())
                all_node_targets.extend(data.y.cpu().numpy())

                # Update progress bar
                current_loss = total_loss / node_total if node_total > 0 else 0
                current_node_acc = (
                    100 * node_correct / node_total if node_total > 0 else 0
                )
                current_node_wacc = (
                    100 * node_weighted_correct / node_weighted_total
                    if node_weighted_total > 0
                    else 0
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

    return avg_loss, node_acc, node_weighted_acc, all_node_preds, all_node_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--node-loss-weight", type=float, default=1.0)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-d", "--data-type", type=str, default="all", choices=["all", "good"]
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        default=[10000, 10001, 10003],
        help="Run numbers to load (default: 10000 10001 10003)",
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
        "--num-events-per-run",
        type=int,
        default=None,
        help="Total number of events to use per run (default: all)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Additional suffix to append to output directory name",
    )
    args = parser.parse_args()

    # Print all arguments
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info("=" * 80)
    for arg, value in sorted(vars(args).items()):
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 80)

    # Build output directory suffix
    suffix = f"{args.data_type}_events"

    # Add "_det" if any run >= 10024
    if any(run >= 10024 for run in args.runs):
        suffix += "_det"

    # Add "_nut" if any run converts to "nut"
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

    if any(get_str_from_run(run) == "nut" for run in args.runs):
        suffix += "_nut"

    # Add "_5mm" or "_8mm" based on (run//4) % 2
    scint_size = None
    for run in args.runs:
        run_scint_size = "_5mm" if ((run - 10000) // 4) % 2 == 0 else "_8mm"
        if scint_size is None:
            scint_size = run_scint_size
        elif scint_size != run_scint_size:
            raise ValueError(
                f"Conflicting scintillator sizes: {scint_size} vs {run_scint_size} for runs {args.runs}"
            )
    suffix += scint_size

    # Add "{num_scint}_scint_layer" where num_scint = (run//8) % 3
    num_scint = None
    for run in args.runs:
        run_num_scint = ((run - 10000) // 8) % 3
        if num_scint is None:
            num_scint = run_num_scint
        elif num_scint != run_num_scint:
            raise ValueError(
                f"Conflicting number of scintillator layers: {num_scint} vs {run_num_scint} for runs {args.runs}"
            )
    suffix += f"_{num_scint}_scint_layer"

    # Add custom suffix if provided
    if args.suffix:
        suffix += f"_{args.suffix}"

    # Setup paths
    torch_path = get_torch_path()
    # output_path = get_torch_path() / f"gravnet_nodes_graph_faser_{suffix}"
    # output_path.mkdir(parents=True, exist_ok=True)
    weights_path = get_weights_path() / f"gravnet_muon_hits_{suffix}"
    weights_path.mkdir(parents=True, exist_ok=True)

    # Add file handler to logger
    log_file = weights_path / "training.log"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")

    # Device
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data (from pointnetpp_faser directory with FASER features)
    logger.info("Loading datasets with FASER data...")

    dataset = []
    for run in args.runs:
        run_str = get_str_from_run(run)
        run_path = torch_path / f"{run}/pointnetpp_faser_{args.data_type}_events"

        # Get chunks to load
        if args.chunks is None:
            # Load all available chunks
            chunk_files = sorted(run_path.glob(f"{run_str}_*.pt"))
            chunks_to_load = [int(f.stem.split("_")[-1]) for f in chunk_files]
        else:
            chunks_to_load = args.chunks

        logger.info(f"Loading {len(chunks_to_load)} chunks for run {run} ({run_str})")

        # Load and concatenate chunks
        run_loaded = 0
        for chunk in chunks_to_load:
            chunk_file = run_path / f"{run_str}_{chunk:03d}.pt"
            if chunk_file.exists():
                chunk_data = torch.load(chunk_file, weights_only=False)
                if args.num_events is not None:
                    chunk_data = chunk_data[: args.num_events]

                if args.num_events_per_run is not None:
                    remaining = args.num_events_per_run - run_loaded
                    if remaining <= 0:
                        logger.info(
                            f"Reached num-events-per-run={args.num_events_per_run} for run {run}"
                        )
                        break
                    chunk_data = chunk_data[:remaining]

                if run_str == "nut":
                    # select leptonic tau decays for nut runs
                    logging.info("Select leptonic tau decays")
                    root_path = (
                        get_root_path() / f"{run:05d}/{run:05d}_{chunk:03d}.root"
                    )
                    truth_file = run_path / f"{run_str}_{chunk:03d}_truth.parq"
                    truth_df = pd.read_parquet(truth_file)
                    traj_df = ak.to_dataframe(
                        uproot.open(root_path)["trajectories"].arrays(library="ak"),
                        how="outer",
                    )

                    all_event_ids = truth_df["event_id"].unique()
                    traj_df.query("evtID in @all_event_ids", inplace=True)
                    lepton_event_ids = traj_df.query(
                        "(abs(trackPDG) == 13) & (trackPID == 1)"
                    )["evtID"].unique()
                    truth_df["idx"] = range(len(truth_df))
                    idx = truth_df.query("event_id in @lepton_event_ids")["idx"].values
                    chunk_data = [chunk_data[i] for i in idx]
                    logging.info(
                        f"Selected {len(lepton_event_ids)} leptonic tau events from {len(all_event_ids)} total events in chunk {chunk}"
                    )

                dataset.extend(chunk_data)
                run_loaded += len(chunk_data)
                logger.info(f"  Loaded chunk {chunk}: {len(chunk_data)} events")
            else:
                logger.warning(f"  Chunk file not found: {chunk_file}")

    logger.info(f"Total loaded events: {len(dataset)}")

    # Remap node labels: 0->0 (other), 1->0 (electrons to other), 2->1 (muons)
    logger.info("Remapping node labels: combining electrons with other particles")
    lookup = torch.tensor([0, 0, 1])
    for data in dataset:
        if hasattr(data, "y") and data.y is not None:
            data.y = lookup[data.y]

    # Old method (commented out)
    # nue_dataset = torch.load(
    #     torch_path / f"10000/pointnetpp_faser_{args.data_type}_events/nue.pt",
    #     weights_only=False,
    # )
    # num_dataset = torch.load(
    #     torch_path / f"10001/pointnetpp_faser_{args.data_type}_events/num.pt",
    #     weights_only=False,
    # )
    # nun_dataset = torch.load(
    #     torch_path / f"10003/pointnetpp_faser_{args.data_type}_events/nun.pt",
    #     weights_only=False,
    # )
    #
    # logger.info(
    #     f"Loaded {len(nue_dataset)} nue, {len(num_dataset)} num, {len(nun_dataset)} nun events"
    # )
    # num_events = 2000
    # dataset = (
    #     nue_dataset[:num_events] + num_dataset[:num_events] + nun_dataset[:num_events]
    # )

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
    logger.info(f"Graph label: {sample.y_graph}")
    if hasattr(sample, "y") and sample.y is not None:
        logger.info(f"Node labels available: {sample.y.shape}")
    else:
        logger.info("Node labels not available")

    # Get number of classes
    num_graph_classes = len(
        torch.unique(torch.tensor([d.y_graph for d in train_dataset]))
    )
    logger.info(f"Number of graph classes: {num_graph_classes}")

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
        logger.info(
            "Node class labels: 0=Other particles (includes electrons), 1=Muons"
        )
    else:
        num_node_classes = 2  # Default: other particles + muons
        logger.warning(
            f"Node labels not found, assuming {num_node_classes} node classes"
        )

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

    # Create model with FASER features and node classification only
    model = NeutrinoGravNetNodesFaser(
        input_dim=sample.x.shape[1],
        num_node_classes=num_node_classes,
        faser_dim=sample.x_faser.shape[0],
        dropout=0.2,
        k=12,
    ).to(device)

    logger.info(f"Model: {model}")
    logger.info(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
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
        train_loss, train_node_acc, train_node_wacc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            node_class_weights,
        )

        # Validate
        val_loss, val_node_acc, val_node_wacc, _, _ = validate_epoch(
            model,
            val_loader,
            device,
            node_class_weights,
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
        log_msg = (
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Node Acc: {train_node_acc:.4f}/{train_node_wacc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Node Acc: {val_node_acc:.4f}/{val_node_wacc:.4f}"
        )
        logger.info(log_msg)

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

    # Evaluate on validation set with best model
    logger.info("Evaluating best model on validation set...")
    checkpoint = torch.load(weights_path / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    (
        val_loss_final,
        val_node_acc_final,
        val_node_wacc_final,
        all_node_preds,
        all_node_targets,
    ) = validate_epoch(
        model,
        val_loader,
        device,
        node_class_weights,
    )
    logger.info(f"Final Val Loss: {val_loss_final:.4f}")
    logger.info(
        f"Final Val Node Acc: {val_node_acc_final:.4f}, WAcc: {val_node_wacc_final:.4f}"
    )

    # Compute per-class accuracy on validation set
    logger.info("Computing per-class validation accuracy...")

    # Node-level per-class accuracy
    node_class_accuracies = []
    if len(all_node_preds) > 0:
        all_node_preds_arr = np.array(all_node_preds)
        all_node_targets_arr = np.array(all_node_targets)

        logger.info("Per-class node validation accuracy:")
        num_node_classes_final = int(all_node_targets_arr.max()) + 1
        for i in range(num_node_classes_final):
            mask = all_node_targets_arr == i
            if mask.sum() > 0:
                class_acc = (all_node_preds_arr[mask] == i).mean()
                node_class_accuracies.append(class_acc)
                logger.info(f"  Class {i}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
            else:
                node_class_accuracies.append(0.0)
                logger.info(f"  Class {i}: No samples")

    # Save validation metrics
    save_dict = {
        "val_loss": val_loss_final,
        "val_node_acc": val_node_acc_final,
        "val_node_weighted_acc": val_node_wacc_final,
        "node_class_accuracies": np.array(node_class_accuracies)
        if node_class_accuracies
        else np.array([]),
        "node_predictions": np.array(all_node_preds)
        if all_node_preds
        else np.array([]),
        "node_targets": np.array(all_node_targets)
        if all_node_targets
        else np.array([]),
    }

    np.savez(
        weights_path / "final_val_metrics.npz",
        **save_dict,
    )
    logger.info(
        f"Final validation metrics saved to {weights_path / 'final_val_metrics.npz'}"
    )


if __name__ == "__main__":
    main()
