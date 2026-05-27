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
    run_label_dict = {
        10000: "nue",
        10001: "num",
        10002: "nut",
        10003: "nun",
    }
    if run not in run_label_dict.keys():
        raise ValueError(f"Run {run} not found in run label dictionary.")
    return run_label_dict[run]


def create_graph_data_with_particle_prob(
    weights_path: str | Path,
    runs: list[int] = [10000, 10001, 10003],
    use_truth_selection: bool = False,
    num_events: int = -1,
    device: str | torch.device = "cuda:0",
    recreate: bool = False,
) -> None:
    hit_labels = ["other", "e", "mu"]

    device = get_device(device)
    model = NeutrinoGravNetNodesFaser(input_dim=1, num_node_classes=3, faser_dim=5)
    model = model.to(device)
    load_weights(model, device=device, weights_path=weights_path)

    for run in runs:
        run_str = get_str_from_run(run)
        logging.info(f"Processing run {run} ({run_str}).")
        data_path = get_torch_path() / str(run)
        if use_truth_selection:
            data_path = data_path / "pointnetpp_faser_good_events"
        else:
            data_path = data_path / "pointnetpp_faser_all_events"
        input_path = data_path / f"{run_str}.pt"
        output_path = data_path / f"{run_str}_particle_prob.pt"

        if output_path.exists() and not recreate:
            logging.info(f"File {output_path} already exists. Skipping...")
            continue

        logging.info("Get node probabilities.")
        dataset = torch.load(input_path, weights_only=False)[:num_events]
        data_loader = DataLoader(dataset)
        y_true, y_pred, y_prob = evaluate(
            model=model, test_loader=data_loader, device=device, use_faser=True
        )

        y_true_flat = np.concatenate(y_true)
        y_pred_flat = np.concatenate(y_pred)
        acc_dict = get_accuracy_dict(
            y_true=y_true_flat, y_pred=y_pred_flat, class_names=hit_labels
        )
        for label, acc in acc_dict.items():
            logging.info(f"{label}: {acc * 100:.1f}%")

        logging.info("Create new dataset with node probabilities.")
        new_dataset = []
        for data, node_prob in zip(dataset, y_prob):
            new_x = torch.cat(
                [data.x, torch.tensor(node_prob, dtype=torch.float)], dim=1
            )
            new_data = Data(
                x=new_x,
                y=data.y_graph,
                pos=data.pos if hasattr(data, "pos") else None,
                y_graph=data.y_graph,
                x_faser=data.x_faser if hasattr(data, "x_faser") else None,
            )
            new_dataset.append(new_data)

        logging.info(f"Saving new dataset to {output_path}.")
        torch.save(new_dataset, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights-path", type=str, required=True)
    parser.add_argument(
        "-r", "--runs", nargs="+", type=int, default=[10000, 10001, 10003]
    )
    parser.add_argument("-n", "--num-events", type=int, default=-1, required=False)
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
    parser.add_argument("-g", "--gpu", type=str, default="cuda:0")
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    logging.info(
        f"Creating graph data with particle probabilities for runs {args.runs}."
    )

    create_graph_data_with_particle_prob(
        weights_path=args.weights_path,
        runs=args.runs,
        use_truth_selection=args.use_truth_selection,
        num_events=args.num_events,
        device=args.gpu,
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
