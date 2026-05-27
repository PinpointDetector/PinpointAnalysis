#!/usr/bin/env python3
"""
Training script for NeutrinoGravNetFASER model on calorimeter data with FASER spectrometer information.
Performs graph-level event classification with optional charm prediction.

Usage:
    python -m analysis.gravnet.train_gravnet_graph_faser_prob --batch-size 8 --num-epochs 100
"""

import argparse
import logging
import sys

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

from analysis.gravnet.model import NeutrinoGravNetFASER
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
    graph_class_weights,
    charm_class_weights,
    graph_loss_weight=1.0,
    charm_loss_weight=1.0,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_graph_loss = 0

    # Graph metrics
    graph_correct = 0
    graph_total = 0
    graph_weighted_correct = 0
    graph_weighted_total = 0

    # Charm metrics
    charm_correct = 0
    charm_total = 0
    charm_weighted_correct = 0
    charm_weighted_total = 0

    train_bar = tqdm(loader, desc="Training", disable=not sys.stdout.isatty())
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            # Forward pass with FASER data
            output = model(data.x, data.pos, data.batch, data.x_faser)

            # Handle charm prediction output
            if model.predict_charm:
                graph_out, charm_out = output
            else:
                graph_out = output
                charm_out = None

            # Check for batch size mismatch
            num_graphs = data.y_graph.size(0)
            if graph_out.size(0) != num_graphs:
                print(
                    f"\nSkipping batch {batch_idx}: output size {graph_out.size(0)} != target size {num_graphs}"
                )
                continue

            # Graph loss
            graph_loss = F.cross_entropy(
                graph_out, data.y_graph, weight=graph_class_weights
            )

            # Charm loss (if charm prediction is enabled)
            charm_loss = torch.tensor(0.0, device=device)
            if model.predict_charm and hasattr(data, "charm"):
                charm_loss = F.cross_entropy(
                    charm_out, data.charm, weight=charm_class_weights
                )

                # Charm accuracy
                charm_pred = charm_out.argmax(dim=1)
                charm_correct += (charm_pred == data.charm).sum().item()
                charm_total += data.charm.size(0)

                # Charm weighted accuracy
                charm_weights = charm_class_weights[data.charm]
                charm_weighted_correct += (
                    ((charm_pred == data.charm).float() * charm_weights).sum().item()
                )
                charm_weighted_total += charm_weights.sum().item()

            # Combined loss
            loss = graph_loss_weight * graph_loss + charm_loss_weight * charm_loss

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if grad_norm > 50.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")

        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_graph_loss += graph_loss.item() * data.num_graphs

        # Graph accuracy
        graph_pred = graph_out.argmax(dim=1)
        graph_correct += (graph_pred == data.y_graph).sum().item()
        graph_total += data.num_graphs

        # Graph weighted accuracy
        g_class_weights = graph_class_weights[data.y_graph]
        graph_weighted_correct += (
            ((graph_pred == data.y_graph).float() * g_class_weights).sum().item()
        )
        graph_weighted_total += g_class_weights.sum().item()

        # Update progress bar
        current_loss = total_loss / graph_total
        current_graph_acc = 100 * graph_correct / graph_total
        current_graph_wacc = 100 * graph_weighted_correct / graph_weighted_total

        postfix = {
            "Loss": f"{current_loss:.4f}",
            "GAcc": f"{current_graph_acc:.2f}%",
            "GWAcc": f"{current_graph_wacc:.2f}%",
        }

        if charm_total > 0:
            current_charm_acc = 100 * charm_correct / charm_total
            current_charm_wacc = 100 * charm_weighted_correct / charm_weighted_total
            postfix["CAcc"] = f"{current_charm_acc:.2f}%"
            postfix["CWAcc"] = f"{current_charm_wacc:.2f}%"

        train_bar.set_postfix(postfix)

        # Print log output every 100 batches if progress bar is disabled
        if not sys.stdout.isatty() and (batch_idx + 1) % 100 == 0:
            log_msg = f"Batch {batch_idx + 1}: Loss={current_loss:.4f}, GAcc={current_graph_acc:.2f}%, GWAcc={current_graph_wacc:.2f}%"
            if charm_total > 0:
                log_msg += (
                    f", CAcc={current_charm_acc:.2f}%, CWAcc={current_charm_wacc:.2f}%"
                )
            logger.info(log_msg)

    # Compute averages
    avg_loss = total_loss / graph_total
    avg_graph_loss = total_graph_loss / graph_total
    graph_acc = graph_correct / graph_total
    graph_weighted_acc = graph_weighted_correct / graph_weighted_total
    charm_acc = charm_correct / charm_total if charm_total > 0 else 0.0
    charm_weighted_acc = (
        charm_weighted_correct / charm_weighted_total
        if charm_weighted_total > 0
        else 0.0
    )

    return (
        avg_loss,
        avg_graph_loss,
        graph_acc,
        graph_weighted_acc,
        charm_acc,
        charm_weighted_acc,
    )


def validate_epoch(
    model,
    loader,
    device,
    graph_class_weights,
    charm_class_weights,
    graph_loss_weight=1.0,
    charm_loss_weight=1.0,
):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_graph_loss = 0
    total_charm_loss = 0

    # Graph metrics
    graph_correct = 0
    graph_total = 0
    graph_weighted_correct = 0
    graph_weighted_total = 0

    # Charm metrics
    charm_correct = 0
    charm_total = 0
    charm_weighted_correct = 0
    charm_weighted_total = 0

    # Per-class graph predictions
    all_graph_preds = []
    all_graph_targets = []

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation", disable=not sys.stdout.isatty())
        for data in val_bar:
            data = data.to(device)

            try:
                # Forward pass with FASER data
                output = model(data.x, data.pos, data.batch, data.x_faser)

                # Handle charm prediction output
                if model.predict_charm:
                    graph_out, charm_out = output
                else:
                    graph_out = output
                    charm_out = None

                # Check for batch size mismatch
                num_graphs = data.y_graph.size(0)
                if graph_out.size(0) != num_graphs:
                    continue

                # Graph loss
                graph_loss = F.cross_entropy(
                    graph_out, data.y_graph, weight=graph_class_weights
                )

                # Charm loss (if charm prediction is enabled)
                charm_loss = torch.tensor(0.0, device=device)
                if model.predict_charm and hasattr(data, "charm"):
                    charm_loss = F.cross_entropy(
                        charm_out, data.charm, weight=charm_class_weights
                    )

                    # Charm accuracy
                    charm_pred = charm_out.argmax(dim=1)
                    charm_correct += (charm_pred == data.charm).sum().item()
                    charm_total += data.charm.size(0)

                    # Charm weighted accuracy
                    charm_weights = charm_class_weights[data.charm]
                    charm_weighted_correct += (
                        ((charm_pred == data.charm).float() * charm_weights)
                        .sum()
                        .item()
                    )
                    charm_weighted_total += charm_weights.sum().item()

                # Combined loss
                loss = graph_loss_weight * graph_loss + charm_loss_weight * charm_loss

                total_loss += loss.item() * data.num_graphs
                total_graph_loss += graph_loss.item() * data.num_graphs
                total_charm_loss += charm_loss.item() * data.num_graphs

                # Graph accuracy
                graph_pred = graph_out.argmax(dim=1)
                graph_correct += (graph_pred == data.y_graph).sum().item()
                graph_total += data.num_graphs

                # Store predictions for per-class accuracy
                all_graph_preds.extend(graph_pred.cpu().numpy())
                all_graph_targets.extend(data.y_graph.cpu().numpy())

                # Graph weighted accuracy
                g_class_weights = graph_class_weights[data.y_graph]
                graph_weighted_correct += (
                    ((graph_pred == data.y_graph).float() * g_class_weights)
                    .sum()
                    .item()
                )
                graph_weighted_total += g_class_weights.sum().item()

                # Update progress bar
                current_loss = total_loss / graph_total
                current_graph_acc = 100 * graph_correct / graph_total
                current_graph_wacc = 100 * graph_weighted_correct / graph_weighted_total

                postfix = {
                    "Loss": f"{current_loss:.4f}",
                    "GAcc": f"{current_graph_acc:.2f}%",
                    "GWAcc": f"{current_graph_wacc:.2f}%",
                }

                if charm_total > 0:
                    current_charm_acc = 100 * charm_correct / charm_total
                    current_charm_wacc = (
                        100 * charm_weighted_correct / charm_weighted_total
                    )
                    postfix["CAcc"] = f"{current_charm_acc:.2f}%"
                    postfix["CWAcc"] = f"{current_charm_wacc:.2f}%"

                val_bar.set_postfix(postfix)

            except RuntimeError:
                continue

    # Compute averages
    avg_loss = total_loss / graph_total
    avg_graph_loss = total_graph_loss / graph_total
    avg_charm_loss = total_charm_loss / graph_total
    graph_acc = graph_correct / graph_total
    graph_weighted_acc = graph_weighted_correct / graph_weighted_total
    charm_acc = charm_correct / charm_total if charm_total > 0 else 0.0
    charm_weighted_acc = (
        charm_weighted_correct / charm_weighted_total
        if charm_weighted_total > 0
        else 0.0
    )

    # Compute and print per-class graph accuracies
    if len(all_graph_preds) > 0:
        all_graph_preds_arr = np.array(all_graph_preds)
        all_graph_targets_arr = np.array(all_graph_targets)

        num_classes = int(all_graph_targets_arr.max()) + 1
        print("\nPer-class graph accuracies:")
        for i in range(num_classes):
            mask = all_graph_targets_arr == i
            if mask.sum() > 0:
                class_acc = (all_graph_preds_arr[mask] == i).mean()
                print(f"  Class {i}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
            else:
                print(f"  Class {i}: No samples")

    return (
        avg_loss,
        avg_graph_loss,
        avg_charm_loss,
        graph_acc,
        graph_weighted_acc,
        charm_acc,
        charm_weighted_acc,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Training script for graph-level classification with optional charm prediction"
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--graph-loss-weight", type=float, default=1.0)
    parser.add_argument("--charm-loss-weight", type=float, default=1.0)
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-d", "--data-type", type=str, default="all", choices=["all", "good"]
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        default=[10028, 10029, 10030, 10031],
        help="Run numbers to load (default: 10028, 10029, 10030, 10031)",
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
        "--suffix",
        type=str,
        default="",
        help="Additional suffix to append to output directory name",
    )
    parser.add_argument(
        "--use-particle-prob",
        action="store_true",
        default=False,
        help="Use particle probabilities predicted from previous network.",
    )
    parser.add_argument(
        "--predict-charm",
        action="store_true",
        default=False,
        help="Predict charm interaction flag in addition to graph classification",
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
        if run > 10000:
            run = run - 10000
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
    tungsten_size = None
    for run in args.runs:
        run_tungsten_size = "_5mm" if ((run - 10000) // 4) % 2 == 0 else "_8mm"
        if tungsten_size is None:
            tungsten_size = run_tungsten_size
        elif tungsten_size != run_tungsten_size:
            raise ValueError(
                f"Conflicting scintillator sizes: {tungsten_size} vs {run_tungsten_size} for runs {args.runs}"
            )
    suffix += tungsten_size

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

    # Add particle probability suffix if enabled
    if args.use_particle_prob:
        suffix += "_particle_prob"

    # Add custom suffix if provided
    if args.suffix:
        suffix += f"_{args.suffix}"

    # Setup paths
    torch_path = get_torch_path()
    output_path = get_torch_path() / f"gravnet_graph_faser_{suffix}"
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = get_weights_path() / f"gravnet_graph_faser_{suffix}"
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
    particle_suffix = "_particle_prob" if args.use_particle_prob else ""
    logger.info(
        "Loading datasets with FASER data"
        + (" and particle probabilities" if args.use_particle_prob else "")
        + "..."
    )

    dataset = []
    for run in args.runs:
        run_str = get_str_from_run(run - 10000)
        run_path = torch_path / f"{run}/pointnetpp_faser_{args.data_type}_events"

        # Get chunks to load
        if args.chunks is None:
            # Load all available chunks
            chunk_files = sorted(run_path.glob(f"{run_str}_*{particle_suffix}.pt"))
            chunks_to_load = [
                int(f.stem.replace(particle_suffix, "").split("_")[-1])
                for f in chunk_files
            ]
        else:
            chunks_to_load = args.chunks

        logger.info(f"Loading {len(chunks_to_load)} chunks for run {run} ({run_str})")

        # Load and concatenate chunks
        for chunk in chunks_to_load:
            chunk_file = run_path / f"{run_str}_{chunk:03d}{particle_suffix}.pt"
            if chunk_file.exists():
                chunk_data = torch.load(chunk_file, weights_only=False)
                if args.num_events is not None:
                    chunk_data = chunk_data[: args.num_events]

                if run_str == "nut":
                    # select only hadronic tau interactions for nut runs
                    logging.info("Select only hadronic tau interactions")
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
                    leptonic_event_ids = traj_df.query(
                        "(abs(trackPDG) in [11, 13]) & (trackPID == 1)"
                    )["evtID"].unique()
                    hadronic_event_ids = np.setdiff1d(all_event_ids, leptonic_event_ids)
                    truth_df.reset_index(inplace=True)
                    truth_df["my_index"] = range(len(truth_df))
                    hadronic_events_idx = truth_df.query(
                        "event_id in @hadronic_event_ids"
                    )["my_index"].values
                    if args.num_events is not None:
                        hadronic_events_idx = [
                            i for i in hadronic_events_idx if i < args.num_events
                        ]
                    chunk_data = [chunk_data[i] for i in hadronic_events_idx]
                    logging.info(
                        f"Selected {len(hadronic_events_idx)} hadronic tau events from {len(all_event_ids)} total events in chunk {chunk}"
                    )
                dataset.extend(chunk_data)
                logger.info(f"  Loaded chunk {chunk}: {len(chunk_data)} events")
            else:
                logger.warning(f"  Chunk file not found: {chunk_file}")

    logger.info(f"Total loaded events: {len(dataset)}")

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

    # Get number of classes
    logging.info(
        f"Graph classes: {torch.unique(torch.tensor([d.y_graph for d in train_dataset]))}"
    )
    num_graph_classes = len(
        torch.unique(torch.tensor([d.y_graph for d in train_dataset]))
    )
    logger.info(f"Number of graph classes: {num_graph_classes}")

    # Compute class weights
    graph_class_weights = compute_class_weights(
        train_dataset, num_graph_classes, is_node_level=False
    )
    graph_class_weights = graph_class_weights.to(device)

    # Compute charm class weights (binary classification)
    charm_class_weights = None
    if args.predict_charm:
        if not hasattr(train_dataset[0], "charm"):
            logger.warning(
                "WARNING: --predict-charm is enabled but the dataset does not contain 'charm' labels!"
            )
            logger.warning(
                "Charm prediction will be disabled. Please regenerate the dataset with charm labels."
            )
            args.predict_charm = False
            charm_class_weights = torch.ones(2, device=device)
        else:
            logger.info("Computing charm class weights...")
            charm_class_weights = compute_class_weights(
                train_dataset, num_classes=2, is_node_level=False, label_attr="charm"
            )
            charm_class_weights = charm_class_weights.to(device)
    else:
        charm_class_weights = torch.ones(2, device=device)

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

    # Create model with FASER features (graph-level classification only)
    model = NeutrinoGravNetFASER(
        input_dim=sample.x.shape[1],
        num_graph_classes=num_graph_classes,
        faser_dim=sample.x_faser.shape[0],
        predict_charm=args.predict_charm,
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
        "train_graph_loss": [],
        "train_graph_acc": [],
        "train_graph_weighted_acc": [],
        "val_loss": [],
        "val_graph_loss": [],
        "val_graph_acc": [],
        "val_graph_weighted_acc": [],
    }

    for epoch in range(args.num_epochs):
        # Train
        (
            train_loss,
            train_graph_loss,
            train_graph_acc,
            train_graph_wacc,
            train_charm_acc,
            train_charm_wacc,
        ) = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            graph_class_weights,
            charm_class_weights,
            args.graph_loss_weight,
            args.charm_loss_weight,
        )

        # Validate
        (
            val_loss,
            val_graph_loss,
            val_charm_loss,
            val_graph_acc,
            val_graph_wacc,
            val_charm_acc,
            val_charm_wacc,
        ) = validate_epoch(
            model,
            val_loader,
            device,
            graph_class_weights,
            charm_class_weights,
            args.graph_loss_weight,
            args.charm_loss_weight,
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Store metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_graph_loss"].append(train_graph_loss)
        metrics["train_graph_acc"].append(train_graph_acc)
        metrics["train_graph_weighted_acc"].append(train_graph_wacc)
        metrics["val_loss"].append(val_loss)
        metrics["val_graph_loss"].append(val_graph_loss)
        metrics["val_graph_acc"].append(val_graph_acc)
        metrics["val_graph_weighted_acc"].append(val_graph_wacc)

        # Log
        log_msg = (
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_loss:.4f} (Graph: {train_graph_loss:.4f}), "
            f"Train Graph: {train_graph_acc:.4f}, {train_graph_wacc:.4f} - "
            f"Val Loss: {val_loss:.4f} (Graph: {val_graph_loss:.4f}), "
            f"Val Graph: {val_graph_acc:.4f}, {val_graph_wacc:.4f}"
        )
        if args.predict_charm:
            log_msg += f", Train Charm: {train_charm_wacc:.4f}, Val Charm: {val_charm_wacc:.4f}"
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
                    "val_graph_acc": val_graph_acc,
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
                    "val_graph_acc": val_graph_acc,
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
        val_graph_loss_final,
        val_charm_loss_final,
        val_graph_acc_final,
        val_graph_wacc_final,
        val_charm_acc_final,
        val_charm_wacc_final,
    ) = validate_epoch(
        model,
        val_loader,
        device,
        graph_class_weights,
        charm_class_weights,
        args.graph_loss_weight,
        args.charm_loss_weight,
    )
    logger.info(
        f"Final Val Loss: {val_loss_final:.4f} (Graph: {val_graph_loss_final:.4f}, Charm: {val_charm_loss_final:.4f})"
    )
    logger.info(
        f"Final Val Graph Acc: {val_graph_acc_final:.4f}, WAcc: {val_graph_wacc_final:.4f}"
    )
    if args.predict_charm:
        logger.info(f"Final Val Charm WAcc: {val_charm_wacc_final:.4f}")

    # Compute per-class accuracy on validation set
    logger.info("Computing per-class validation accuracy...")
    model.eval()
    all_graph_preds = []
    all_graph_targets = []
    all_charm_preds = []
    all_charm_targets = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            try:
                output = model(data.x, data.pos, data.batch, data.x_faser)

                # Handle charm prediction output
                if model.predict_charm:
                    graph_out, charm_out = output
                else:
                    graph_out = output
                    charm_out = None

                # Graph predictions
                graph_pred = graph_out.argmax(dim=1)
                all_graph_preds.extend(graph_pred.cpu().numpy())
                all_graph_targets.extend(data.y_graph.cpu().numpy())

                # Charm predictions
                if model.predict_charm and hasattr(data, "charm"):
                    charm_pred = charm_out.argmax(dim=1)
                    all_charm_preds.extend(charm_pred.cpu().numpy())
                    all_charm_targets.extend(data.charm.cpu().numpy())
            except RuntimeError:
                continue

    all_graph_preds = np.array(all_graph_preds)
    all_graph_targets = np.array(all_graph_targets)

    # Graph-level per-class accuracy
    logger.info("Per-class graph validation accuracy:")
    graph_class_accuracies = []
    num_graph_classes_final = int(all_graph_targets.max()) + 1
    for i in range(num_graph_classes_final):
        mask = all_graph_targets == i
        if mask.sum() > 0:
            class_acc = (all_graph_preds[mask] == i).mean()
            graph_class_accuracies.append(class_acc)
            logger.info(f"  Class {i}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
        else:
            graph_class_accuracies.append(0.0)
            logger.info(f"  Class {i}: No samples")

    # Charm-level per-class accuracy (if available)
    charm_class_accuracies = []
    if len(all_charm_preds) > 0:
        all_charm_preds = np.array(all_charm_preds)
        all_charm_targets = np.array(all_charm_targets)

        logger.info("Per-class charm validation accuracy:")
        for i in range(2):
            mask = all_charm_targets == i
            if mask.sum() > 0:
                class_acc = (all_charm_preds[mask] == i).mean()
                charm_class_accuracies.append(class_acc)
                logger.info(f"  Class {i}: {class_acc:.4f} ({class_acc * 100:.2f}%)")
            else:
                charm_class_accuracies.append(0.0)
                logger.info(f"  Class {i}: No samples")

    # Save validation metrics
    save_dict = {
        "val_loss": val_loss_final,
        "val_graph_loss": val_graph_loss_final,
        "val_charm_loss": val_charm_loss_final,
        "val_graph_acc": val_graph_acc_final,
        "val_graph_weighted_acc": val_graph_wacc_final,
        "val_charm_acc": val_charm_acc_final,
        "graph_class_accuracies": np.array(graph_class_accuracies),
        "graph_predictions": all_graph_preds,
        "graph_targets": all_graph_targets,
    }

    if len(all_charm_preds) > 0:
        save_dict["charm_class_accuracies"] = np.array(charm_class_accuracies)
        save_dict["charm_predictions"] = np.array(all_charm_preds)
        save_dict["charm_targets"] = np.array(all_charm_targets)

    np.savez(
        weights_path / "final_val_metrics.npz",
        **save_dict,
    )
    logger.info(
        f"Final validation metrics saved to {weights_path / 'final_val_metrics.npz'}"
    )


if __name__ == "__main__":
    main()
