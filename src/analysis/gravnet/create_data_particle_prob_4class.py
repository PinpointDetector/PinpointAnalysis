"""
Augment base graph data with 4-class node softmax probabilities from
train_gravnet_nodes_faser_4class.py checkpoint.

Each graph's data.x gains 4 extra columns: [P(other), P(secondary_e), P(primary_EM_e), P(mu)].
Output files: {run_str}_{chunk:03d}_particle_prob.pt in the same directory as input.

Usage:
    python -m analysis.gravnet.create_data_particle_prob_4class \
        -w /path/to/gravnet_nodes_faser_all_events_4class/best_model.pt \
        -r 10000 -g cuda:0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet.model import NeutrinoGravNetNodesFaser
from analysis.utils.torch_utils import get_device, load_weights
from analysis.utils.utils import get_torch_path
from analysis.utils.validation_utils import get_accuracy_dict

HIT_LABELS = ["other", "secondary_e", "primary_EM_e", "mu"]
NUM_NODE_CLASSES = 4


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


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_faser: bool = False,
) -> tuple[list, list, list]:
    """Run inference; returns (targets, predictions, probabilities) per graph."""
    model.eval()
    all_targets, all_predictions, all_probabilities = [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            if data.x.size(0) == 0:
                continue
            if use_faser:
                out = model(data.x, data.pos, data.batch, data.x_faser)
            else:
                out = model(data.x, data.pos, data.batch)
            prob = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            all_targets.append(data.pdg_label.cpu().numpy())
            all_predictions.append(pred.cpu().numpy())
            all_probabilities.append(prob.cpu().numpy())
    return all_targets, all_predictions, all_probabilities


def create_graph_data_with_particle_prob(
    weights_path: str | Path,
    run: int,
    chunk: int,
    use_truth_selection: bool = False,
    num_events: int | None = None,
    device: str | torch.device = "cuda:0",
    recreate: bool = False,
) -> None:
    """
    Create particle-probability-augmented graph data for a single chunk.

    Loads the 4-class node classifier, runs inference on base .pt chunk,
    appends softmax probabilities to data.x, saves *_particle_prob.pt.
    """
    device = get_device(device)
    model = NeutrinoGravNetNodesFaser(input_dim=1, num_node_classes=NUM_NODE_CLASSES, faser_dim=5)
    model = model.to(device)
    load_weights(model, device=device, weights_path=weights_path)

    run_str  = get_str_from_run(run)
    logging.info(f"Processing run {run} chunk {chunk} ({run_str}).")

    data_path = get_torch_path() / str(run)
    if use_truth_selection:
        data_path = data_path / "pointnetpp_faser_good_events"
    else:
        data_path = data_path / "pointnetpp_faser_all_events"

    input_path  = data_path / f"{run_str}_{chunk:03d}.pt"
    output_path = data_path / f"{run_str}_{chunk:03d}_particle_prob.pt"

    if not input_path.exists():
        logging.warning(f"Input file {input_path} does not exist. Skipping.")
        return

    if output_path.exists() and not recreate:
        logging.info(f"Output {output_path} already exists. Skipping (use --recreate to overwrite).")
        return

    logging.info("Loading dataset and computing node probabilities.")
    dataset = torch.load(input_path, weights_only=False)
    if num_events is not None:
        dataset = dataset[:num_events]

    data_loader = DataLoader(dataset)
    y_true, y_pred, y_prob = evaluate(
        model=model, test_loader=data_loader, device=device, use_faser=True
    )

    # Log per-class accuracy for this chunk
    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    acc_dict = get_accuracy_dict(y_true=y_true_flat, y_pred=y_pred_flat, class_names=HIT_LABELS)
    logging.info(f"Node classification accuracy for chunk {chunk}:")
    for label, acc in acc_dict.items():
        logging.info(f"  {label}: {acc * 100:.1f}%")

    # Build augmented dataset: data.x gains 4 probability columns
    logging.info("Creating augmented dataset (data.x shape: [N, 5]).")
    new_dataset = []
    for data, node_prob in zip(dataset, y_prob):
        new_x = torch.cat([data.x, torch.tensor(node_prob, dtype=torch.float)], dim=1)
        new_data = Data(
            x=new_x,
            y=data.y_graph,
            pos=data.pos if hasattr(data, "pos") else None,
            y_graph=data.y_graph,
            x_faser=data.x_faser if hasattr(data, "x_faser") else None,
        )
        for attr in ["E_nu", "E_roe", "E_lepton", "charm", "true_pos", "true_pos_centered"]:
            if hasattr(data, attr):
                setattr(new_data, attr, getattr(data, attr))
        new_dataset.append(new_data)

    logging.info(f"Saving {len(new_dataset)} augmented graphs to {output_path}.")
    torch.save(new_dataset, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment graph data with 4-class node softmax probabilities"
    )
    parser.add_argument(
        "-w", "--weights-path", type=str, required=True,
        help="Path to 4-class node classifier best_model.pt",
    )
    parser.add_argument(
        "-r", "--runs", nargs="+", type=int, required=True,
        help="Run numbers to process",
    )
    parser.add_argument(
        "-c", "--chunks", nargs="*", type=int, default=None,
        help="Chunk numbers to process (default: all)",
    )
    parser.add_argument(
        "-n", "--num-events", type=int, default=None,
        help="Cap events per chunk (default: all)",
    )
    parser.add_argument(
        "--recreate", action="store_true", default=False,
        help="Overwrite existing _particle_prob.pt files",
    )
    parser.add_argument(
        "--use-truth-selection", action="store_true", default=False,
        help="Use good_events directory instead of all_events",
    )
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-l", "--log-level", type=str, default="INFO",
        choices=["INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(f"Processing runs {args.runs} with 4-class classifier.")

    for run in args.runs:
        run_str  = get_str_from_run(run)
        data_path = get_torch_path() / str(run)
        data_path = data_path / ("pointnetpp_faser_good_events" if args.use_truth_selection
                                  else "pointnetpp_faser_all_events")

        if args.chunks is None:
            files  = list(data_path.glob(f"{run_str}_*.pt"))
            files  = [f for f in files if "_particle_prob" not in f.stem]
            chunks = sorted([int(f.stem.split("_")[-1]) for f in files])
            logging.info(f"Run {run}: found {len(chunks)} base chunks.")
        else:
            chunks = args.chunks

        for chunk in chunks:
            create_graph_data_with_particle_prob(
                weights_path=args.weights_path,
                run=run,
                chunk=chunk,
                use_truth_selection=args.use_truth_selection,
                num_events=args.num_events,
                device=args.gpu,
                recreate=args.recreate,
            )


if __name__ == "__main__":
    main()
