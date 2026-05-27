import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analysis.gravnet import model
from analysis.gravnet.model import (
    NeutrinoGravNetNodesFaser,
    NeutrinoGravNetNodesGraphFaser,
)
from analysis.utils.torch_geometric_utils import evaluate_pointnetpp_model
from analysis.utils.torch_utils import get_device, load_weights
from analysis.utils.utils import get_torch_path
from analysis.utils.validation_utils import get_accuracy_dict


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_faser: bool = False,
) -> tuple[list, list, list]:
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

            all_targets.append(data.y.cpu().numpy())
            all_predictions.append(pred.cpu().numpy())
            all_probabilities.append(prob.cpu().numpy())

    return all_targets, all_predictions, all_probabilities


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
    Create graph data with particle probabilities for a single chunk.

    Args:
        weights_path: Path to model weights
        run: Run number to process
        chunk: Chunk number to process
        use_truth_selection: If True, use good events selection
        num_events: Number of events to process (None = all)
        device: Device to use for inference
        recreate: If True, overwrite existing files
    """
    hit_labels = ["other", "e", "mu"]

    device = get_device(device)
    # model = NeutrinoGravNetNodesFaser(input_dim=1, num_node_classes=3, faser_dim=5)
    model = NeutrinoGravNetNodesGraphFaser(
        input_dim=1,
        num_graph_classes=3,
        num_node_classes=3,
        faser_dim=5,
        predict_charm=False,
        dropout=0.2,
    )
    model = model.to(device)
    load_weights(model, device=device, weights_path=weights_path)

    run_str = get_str_from_run(run)
    logging.info(f"Processing run {run} chunk {chunk} ({run_str}).")

    data_path = get_torch_path() / str(run)
    if use_truth_selection:
        data_path = data_path / "pointnetpp_faser_good_events"
    else:
        data_path = data_path / "pointnetpp_faser_all_events"

    input_path = data_path / f"{run_str}_{chunk:03d}.pt"
    output_path = data_path / f"{run_str}_{chunk:03d}_particle_prob.pt"

    if not input_path.exists():
        logging.warning(f"Input file {input_path} does not exist. Skipping...")
        return

    if output_path.exists() and not recreate:
        logging.info(f"File {output_path} already exists. Skipping...")
        return

    logging.info("Loading dataset and computing node probabilities.")
    dataset = torch.load(input_path, weights_only=False)
    if num_events is not None:
        dataset = dataset[:num_events]

    data_loader = DataLoader(dataset)
    # y_true, y_pred, y_prob = evaluate(
    #     model=model, test_loader=data_loader, device=device, use_faser=True
    # )

    y_true, y_pred, y_prob, _, _, _ = evaluate_pointnetpp_model(
        model=model,
        test_loader=data_loader,
        device=device,
        use_faser=True,
    )

    y_true_flat = np.concatenate(y_true)
    y_pred_flat = np.concatenate(y_pred)
    acc_dict = get_accuracy_dict(
        y_true=y_true_flat, y_pred=y_pred_flat, class_names=hit_labels
    )
    logging.info(f"Node classification accuracy for chunk {chunk}:")
    for label, acc in acc_dict.items():
        logging.info(f"  {label}: {acc * 100:.1f}%")

    logging.info("Creating new dataset with node probabilities.")
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

        # Copy optional attributes if present
        optional_attrs = [
            "E_nu",
            "E_roe",
            "charm",
            "true_pos",
            "E_lepton",
            "true_pos_centered",
        ]
        for attr in optional_attrs:
            if hasattr(data, attr):
                setattr(new_data, attr, getattr(data, attr))

        new_dataset.append(new_data)

    logging.info(f"Saving new dataset to {output_path}.")
    torch.save(new_dataset, output_path)
    logging.info(f"Saved {len(new_dataset)} graphs for run {run} chunk {chunk}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add particle probabilities to graph data for neutrino events"
    )
    parser.add_argument(
        "-w", "--weights-path", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument(
        "-r",
        "--runs",
        nargs="+",
        type=int,
        required=True,
        help="Run numbers to process",
    )
    parser.add_argument(
        "-c",
        "--chunks",
        nargs="*",
        type=int,
        required=False,
        help="Chunk numbers to process (if not provided, all chunks will be processed)",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        default=None,
        required=False,
        help="Number of events to process per chunk",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--use-truth-selection",
        action="store_true",
        default=False,
        help="Apply good event selection based on truth information (default: False).",
    )
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0", help="GPU device")
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(
        f"Creating graph data with particle probabilities for runs {args.runs}."
    )

    # Process each run and chunk
    for run in args.runs:
        logging.info(f"Processing run {run}.")

        if args.chunks is None:
            # Find all available chunks for this run
            data_path = get_torch_path() / str(run)
            if args.use_truth_selection:
                data_path = data_path / "pointnetpp_faser_good_events"
            else:
                data_path = data_path / "pointnetpp_faser_all_events"

            run_str = get_str_from_run(run)
            files = list(data_path.glob(f"{run_str}_*.pt"))
            # Exclude files that already have _particle_prob suffix
            files = [f for f in files if "_particle_prob" not in f.stem]
            chunks = sorted([int(f.stem.split("_")[-1]) for f in files])
            logging.info(f"Found {len(chunks)} chunks for run {run}.")
        else:
            chunks = args.chunks

        for chunk in chunks:
            logging.info(f"Processing chunk {chunk}.")
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
