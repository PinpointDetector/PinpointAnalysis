#!/usr/bin/env python3
"""
Training script for NeutrinoGravNetRegressionFASER model.

Predicts log10(E_nu) and logit(y) where y = E_lepton / E_nu (Bjorken inelasticity).
E_lepton and E_roe are derived at inference as y·E_nu and (1-y)·E_nu, so energy
conservation E_lepton + E_roe = E_nu holds exactly by construction.

Usage:
    python -m analysis.gravnet.train_gravnet_regression_faser -r 10000 -b 8 --num-epochs 50 --pooling sum
"""

import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetRegressionFASER
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


def compute_targets(data):
    """
    Compute reparametrised training targets from a batch.

    Returns targets [batch_size, 2]:
        col 0: log10(E_nu)
        col 1: logit(y) = log(E_lepton / E_roe)  where y = E_lepton / E_nu
    """
    E_nu = data.E_nu.clamp(min=1e-6)
    E_lepton = data.E_lepton.clamp(min=1e-6)
    E_roe = data.E_roe.clamp(min=1e-6)
    log_E_nu = torch.log10(E_nu)
    logit_y = torch.log(E_lepton / E_roe)
    return torch.stack([log_E_nu, logit_y], dim=1)  # [batch_size, 2]


def preds_to_physical(preds, targets):
    """
    Convert model outputs and targets to physical energies for metric reporting.

    Args:
        preds:   [N, 2] — (log10_E_nu_pred, logit_y_pred)
        targets: [N, 2] — (log10_E_nu_true, logit_y_true)

    Returns:
        preds_linear, targets_linear: both [N, 3] — (E_nu, E_lepton, E_roe) in TeV
    """
    E_nu_pred = 10 ** preds[:, 0]
    y_pred = torch.sigmoid(preds[:, 1])
    E_lepton_pred = y_pred * E_nu_pred
    E_roe_pred = (1 - y_pred) * E_nu_pred

    E_nu_true = 10 ** targets[:, 0]
    y_true = torch.sigmoid(targets[:, 1])
    E_lepton_true = y_true * E_nu_true
    E_roe_true = (1 - y_true) * E_nu_true

    preds_linear = torch.stack([E_nu_pred, E_lepton_pred, E_roe_pred], dim=1)
    targets_linear = torch.stack([E_nu_true, E_lepton_true, E_roe_true], dim=1)
    return preds_linear, targets_linear


def train_epoch(model, loader, optimizer, device, loss_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_events = 0

    all_preds = []
    all_targets = []

    w_enu, w_logit = loss_weights

    train_bar = tqdm(loader, desc="Training", disable=not sys.stdout.isatty())
    for batch_idx, data in enumerate(train_bar):
        data = data.to(device)
        optimizer.zero_grad()

        try:
            predictions = model(data.x, data.pos, data.batch, data.x_faser)
            targets = compute_targets(data)

            loss = (w_enu * F.mse_loss(predictions[:, 0], targets[:, 0]) +
                    w_logit * F.mse_loss(predictions[:, 1], targets[:, 1]))

        except RuntimeError as e:
            print(f"\nSkipping batch {batch_idx} due to error: {e}")
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        if grad_norm > 50.0:
            logger.warning(f"Large gradient norm: {grad_norm:.2f}")

        optimizer.step()

        batch_size = predictions.size(0)
        total_loss += loss.item() * batch_size
        total_events += batch_size

        all_preds.append(predictions.detach())
        all_targets.append(targets.detach())

        current_loss = total_loss / total_events
        train_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

        if not sys.stdout.isatty() and (batch_idx + 1) % 100 == 0:
            logger.info(f"Batch {batch_idx + 1}: Loss={current_loss:.4f}")

    avg_loss = total_loss / total_events
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    rmse = torch.sqrt(torch.tensor(avg_loss))

    preds_linear, targets_linear = preds_to_physical(all_preds, all_targets)
    rel_err = torch.abs(preds_linear - targets_linear) / targets_linear.clamp(min=1e-6)
    mean_rel_err = rel_err.mean(dim=0)  # [3]
    std_rel_err = rel_err.std(dim=0)    # [3]

    return avg_loss, rmse.item(), mean_rel_err.cpu(), std_rel_err.cpu()


def validate_epoch(model, loader, device, loss_weights):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    total_events = 0

    all_preds = []
    all_targets = []

    w_enu, w_logit = loss_weights

    with torch.no_grad():
        val_bar = tqdm(loader, desc="Validation", disable=not sys.stdout.isatty())
        for data in val_bar:
            data = data.to(device)

            try:
                predictions = model(data.x, data.pos, data.batch, data.x_faser)
                targets = compute_targets(data)

                loss = (w_enu * F.mse_loss(predictions[:, 0], targets[:, 0]) +
                        w_logit * F.mse_loss(predictions[:, 1], targets[:, 1]))

                batch_size = predictions.size(0)
                total_loss += loss.item() * batch_size
                total_events += batch_size

                all_preds.append(predictions)
                all_targets.append(targets)

                current_loss = total_loss / total_events
                val_bar.set_postfix({"Loss": f"{current_loss:.4f}"})

            except RuntimeError:
                continue

    avg_loss = total_loss / total_events
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    rmse = torch.sqrt(torch.tensor(avg_loss))

    preds_linear, targets_linear = preds_to_physical(all_preds, all_targets)
    rel_err = torch.abs(preds_linear - targets_linear) / targets_linear.clamp(min=1e-6)
    mean_rel_err = rel_err.mean(dim=0)
    std_rel_err = rel_err.std(dim=0)

    return avg_loss, rmse.item(), mean_rel_err.cpu(), std_rel_err.cpu()


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
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint .pt file to resume training from",
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
    suffix = f"{args.data_type}_events_{args.pooling}_reparam"
    if args.suffix:
        suffix += f"_{args.suffix}"

    # Setup paths
    torch_path = get_torch_path()
    weights_path = get_weights_path() / f"gravnet_regression_faser_{suffix}"
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
    for run in args.runs:
        run_str = get_str_from_run(run)
        run_path = torch_path / f"{run}/pointnetpp_faser_{args.data_type}_events"

        if args.chunks is None:
            chunk_files = sorted(run_path.glob(f"{run_str}_*.pt"))
            chunks_to_load = [int(f.stem.split("_")[-1]) for f in chunk_files]
        else:
            chunks_to_load = args.chunks

        logger.info(f"Loading {len(chunks_to_load)} chunks for run {run} ({run_str})")

        for chunk in chunks_to_load:
            chunk_file = run_path / f"{run_str}_{chunk:03d}.pt"
            if chunk_file.exists():
                chunk_data = torch.load(chunk_file, weights_only=False)
                if args.num_events is not None:
                    chunk_data = chunk_data[: args.num_events]
                dataset.extend(chunk_data)
                logger.info(f"  Loaded chunk {chunk}: {len(chunk_data)} events")
            else:
                logger.warning(f"  Chunk file not found: {chunk_file}")

    logger.info(f"Total loaded events: {len(dataset)}")

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

    # Compute per-target loss weights from training set variance.
    # logit(y) has ~30x larger variance than log10(E_nu), so without weighting
    # the logit MSE dominates the gradient and E_nu gets undertrained.
    # Weights are inversely proportional to each target's variance, normalised to sum to 1.
    log_enu_vals = torch.stack(
        [torch.log10(d.E_nu.clamp(1e-6)) for d in train_dataset]
    )
    logit_y_vals = torch.stack(
        [torch.log(d.E_lepton.clamp(1e-6) / d.E_roe.clamp(1e-6))
         for d in train_dataset]
    )
    w_enu = 1.0 / log_enu_vals.var().clamp(min=1e-6)
    w_logit = 1.0 / logit_y_vals.var().clamp(min=1e-6)
    total_w = w_enu + w_logit
    w_enu = (w_enu / total_w).item()
    w_logit = (w_logit / total_w).item()
    loss_weights = (w_enu, w_logit)
    logger.info(f"Loss weights: w_enu={w_enu:.4f}, w_logit={w_logit:.4f}")

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

    # Create model (2 outputs: log10 E_nu, logit y)
    model = NeutrinoGravNetRegressionFASER(
        input_dim=sample.x.shape[1],
        num_targets=2,
        faser_dim=sample.x_faser.shape[0],
        pooling=args.pooling,
        dropout=0.2,
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
        "train_rmse": [],
        "val_loss": [],
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
        train_loss, train_rmse, train_rel_err, train_resolution = train_epoch(
            model, train_loader, optimizer, device, loss_weights
        )

        val_loss, val_rmse, val_rel_err, val_resolution = validate_epoch(
            model, val_loader, device, loss_weights
        )

        scheduler.step(val_loss)

        metrics["train_loss"].append(train_loss)
        metrics["train_rmse"].append(train_rmse)
        metrics["val_loss"].append(val_loss)
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
            f"Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f} - "
            f"Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f} - "
            f"Val RelErr: {rel_err_str}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_rmse": val_rmse,
            "val_rel_err": val_rel_err.numpy(),
            "val_resolution": val_resolution.numpy(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "loss_weights": loss_weights,
            "parametrisation": "reparam",
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
