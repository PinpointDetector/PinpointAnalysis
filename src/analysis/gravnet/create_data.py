import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from analysis.utils.preprocessing_utils import get_good_event_ids
from analysis.utils.utils import get_parquet_path, get_torch_path

# Suppress SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)


def create_event_graph(
    hits_df: pd.DataFrame,
    faser_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    label: int,
    use_faser: bool = True,
) -> tuple[Data, np.ndarray]:
    """
    Create PointNet++ compatible graph data for a single event.

    Args:
        hits_df: DataFrame with pixel hit information
        faser_df: DataFrame with FASER spectrometer information
        truth_df: DataFrame with truth information
        label: Graph-level label (event classification)
        use_faser: Whether to include FASER spectrometer features

    Returns:
        data: PyG Data object with pos (positions) and x (features) separated
        mean_features: Mean values used for normalization
    """
    # Extract positions (x, y, z) and center at event centroid
    # This makes learning translation-invariant (relative geometry matters, not absolute position)
    pos = hits_df[["x", "y", "z"]].values
    pos_mean = pos.mean(axis=0)
    pos_centered = pos - pos_mean

    # Optionally normalize by event scale (makes it scale-invariant too)
    # pos_std = pos_centered.std()
    # pos_normalized = pos_centered / (pos_std + 1e-5)
    pos_normalized = pos_centered / 100.0  # fixed scale normalization

    pos_tensor = torch.tensor(pos_normalized, dtype=torch.float)

    # Extract features (energy/nhits)
    # Use log-normalized hit count - simple and effective
    n_hits_log = np.log10(hits_df["n_hits"].values)
    x_tensor = torch.tensor(n_hits_log.reshape(-1, 1), dtype=torch.float)

    # Store normalization parameters for reference
    mean_features = np.array([pos_mean[0], pos_mean[1], pos_mean[2]])
    # std_features = np.array([n_hits_log.mean(), n_hits_log.std()])

    # Node-level labels: 0: other, 1: electron, 2: muon
    # Merge primary and non-primary electrons
    hits_df["label"] = hits_df["pdg_label"].replace({2: 1, 3: 2})
    y_node = torch.tensor(hits_df["label"].values.astype(np.int64), dtype=torch.long)
    pdg_label_node = torch.tensor(hits_df["pdg_label"].values.astype(np.int64), dtype=torch.long)

    # Graph-level label (event classification)
    y_graph = torch.tensor([label], dtype=torch.long)

    # FASER spectrometer features (event-level/graph-level)
    if use_faser:
        # FASER spectrometer hits features: nhits_0, nhits_1, nhits_2, x, y
        x_faser = faser_df[
            ["nhits_0", "nhits_1", "nhits_2", "faser_x", "faser_y"]
        ].values[0]
        # Normalize x and y positions using the same normalization as pixel hits
        x_faser[3] = (x_faser[3] - pos_mean[0]) / 100.0
        x_faser[4] = (x_faser[4] - pos_mean[1]) / 100.0
        x_faser_tensor = torch.tensor(x_faser, dtype=torch.float)
    else:
        x_faser_tensor = None

    # Extract truth information from the first row (single event)
    truth_info = truth_df.iloc[0]
    E_nu = torch.tensor(truth_info["E_nu"] / 1e6, dtype=torch.float)
    E_lepton = torch.tensor(truth_info["E_lepton"] / 1e6, dtype=torch.float)
    E_roe = torch.tensor(
        (truth_info["E_nu"] - truth_info["E_lepton"]) / 1e6, dtype=torch.float
    )
    charm = torch.tensor(int(truth_info["charm"]), dtype=torch.long)

    # True neutrino interaction position minus pos_mean (centered and normalized)
    true_pos = np.array([truth_info["vx"], truth_info["vy"], truth_info["vz"]])
    true_pos_normalized = torch.tensor(true_pos / 100.0, dtype=torch.float)
    true_pos_centered = torch.tensor((true_pos - pos_mean) / 100.0, dtype=torch.float)

    # Create graph data (no edge_index needed for PointNet++)
    data = Data(
        x=x_tensor,  # [N, 1] node features (nhits)
        pos=pos_tensor,  # [N, 3] positions (x, y, z)
        y=y_node,  # [N] node labels
        pdg_label=pdg_label_node,  # [N] 4-class labels (0=other, 1=secondary_e, 2=primary_EM_e, 3=muon)
        y_graph=y_graph,  # [1] graph label
        x_faser=x_faser_tensor,  # [5] FASER spectrometer features (event-level)
        E_nu=E_nu,  # neutrino energy (TeV)
        E_lepton=E_lepton,  # lepton energy (TeV)
        E_roe=E_roe,  # rest-of-event energy (TeV)
        charm=charm,  # charm flag (boolean as long)
        true_pos=true_pos_normalized,  # [3] true neutrino interaction position normalized by 100
        true_pos_centered=true_pos_centered,  # [3] true neutrino interaction position centered at pos_mean
    )

    return data, mean_features


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
    # run_label_dict = {
    #     10000: "nue",
    #     10001: "num",
    #     10002: "nut",
    #     10003: "nun",
    # }
    # if run not in run_label_dict.keys():
    #     raise ValueError(f"Run {run} not found in run label dictionary.")
    # return run_label_dict[run]


def get_label_from_run(run: int) -> int:
    """Convert run number to classification label."""
    # run_label_dict = {
    #     10003: 0,  # NC
    #     10000: 1,  # CC nue
    #     10001: 2,  # CC num
    #     # 10002: 2,  # nut (not used)
    # }
    # if run not in run_label_dict.keys():
    #     raise ValueError(f"Run {run} not found in run label dictionary.")
    # return run_label_dict[run]
    if run % 4 == 3:
        return 0  # NC
    elif run % 4 == 0:
        return 1  # CC nue
    elif run % 4 == 1:
        return 2  # CC num
    elif run % 4 == 2:
        return 3  # CC nut


def create_graph_data(
    run: int,
    chunk: int,
    recreate: bool = False,
    num_events: int | None = None,
    use_faser: bool = True,
    apply_good_event_selection: bool = False,
    bin_size: str = "200um",
) -> None:
    """
    Create PointNet++ compatible graph data for a single chunk.

    Args:
        run: Run number to process
        chunk: Chunk number to process
        recreate: If True, recreate existing data files
        num_events: Number of events to process per chunk
        use_faser: If True, include FASER spectrometer features
        apply_good_event_selection: If True, apply good event selection based on truth information
    """
    run_str = get_str_from_run(run)
    label = get_label_from_run(run)
    run_path = get_parquet_path() / f"{run}/{bin_size}_bins"
    output_path = get_torch_path() / f"{run}"
    if use_faser:
        output_path = output_path / "pointnetpp_faser"
    else:
        output_path = output_path / "pointnetpp"
    if apply_good_event_selection:
        output_path = output_path.parent / f"{output_path.name}_good_events"
    else:
        output_path = output_path.parent / f"{output_path.name}_all_events"
    if bin_size != "200um":
        output_path = output_path.parent / f"{output_path.name}_{bin_size}"
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = output_path / f"{run_str}_{chunk:03d}.pt"
    truth_path = output_path / f"{run_str}_{chunk:03d}_truth.parq"
    means_path = output_path / f"{run_str}_{chunk:03d}_means.npy"

    if data_path.exists() and not recreate:
        logging.info(
            f"Graph data for run {run} chunk {chunk} already exists at {data_path}."
        )
        return

    logging.info(f"Creating PointNet++ data for run {run} chunk {chunk} ({run_str})...")

    # Read single chunk files
    hits_file = run_path / f"{run:05d}_{chunk:03d}_hits.parq"
    truth_file = run_path / f"{run:05d}_{chunk:03d}_truth.parq"
    faser_file = run_path / f"{run:05d}_{chunk:03d}_faser.parq"

    if not hits_file.exists() or not truth_file.exists() or not faser_file.exists():
        logging.warning(f"Missing files for run {run} chunk {chunk}, skipping...")
        return

    hits_df = pd.read_parquet(hits_file)
    truth_df = pd.read_parquet(truth_file)
    faser_df = pd.read_parquet(faser_file)

    # Filter good events
    good_event_ids = get_good_event_ids(
        truth_df, apply_truth_cuts=apply_good_event_selection
    )
    if num_events is not None:
        good_event_ids = good_event_ids[:num_events]
    hits_df = hits_df[hits_df["event_id"].isin(good_event_ids)]
    truth_df = truth_df[truth_df["event_id"].isin(good_event_ids)]
    faser_df = faser_df[faser_df["event_id"].isin(good_event_ids)]

    logging.info(f"Processing {len(good_event_ids)} events...")

    # Create graph data for each event
    graphs, means = [], []
    valid_event_ids = []
    for event_id in tqdm(good_event_ids):
        event_hits_df = hits_df.query(f"event_id == {event_id}")

        # Skip events with too few nodes (< 100)
        if len(event_hits_df) < 100:
            continue

        faser_event_df = faser_df.query(f"event_id == {event_id}")
        event_truth_df = truth_df.query(f"event_id == {event_id}")

        event_graph_data, event_means = create_event_graph(
            hits_df=event_hits_df,
            faser_df=faser_event_df,
            truth_df=event_truth_df,
            label=label,
            use_faser=use_faser,
        )

        graphs.append(event_graph_data)
        means.append(event_means)
        valid_event_ids.append(event_id)

    # Filter truth and faser dataframes to only include valid events
    truth_df = truth_df[truth_df["event_id"].isin(valid_event_ids)]
    faser_df = faser_df[faser_df["event_id"].isin(valid_event_ids)]

    logging.info(
        f"Kept {len(graphs)}/{len(good_event_ids)} events (skipped {len(good_event_ids) - len(graphs)} events with < 100 nodes)."
    )

    # Save data
    logging.info(f"Saving dataset to {data_path}.")
    torch.save(graphs, data_path)
    truth_df.to_parquet(truth_path)
    np.save(means_path, np.array(means))

    logging.info(f"Saved {len(graphs)} graphs for run {run} chunk {chunk}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create PointNet++ compatible graph data for neutrino events"
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
        help="Recreate existing data files",
    )
    parser.add_argument(
        "--bin-size",
        type=str,
        default="200um",
        help="Bin size subdirectory to read from, e.g. '200um' or '100um' (default: 200um)",
    )
    parser.add_argument(
        "--use-faser",
        action="store_true",
        default=True,
        help="Include FASER spectrometer data (nhits_0, nhits_1, nhits_2, x, y) as separate graph-level features.",
    )
    parser.add_argument(
        "--good-event-selection",
        action="store_true",
        default=False,
        help="Apply good event selection based on truth information (default: False).",
    )
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
        f"FASER spectrometer data: {'enabled' if args.use_faser else 'disabled'}"
    )

    # Process each run and chunk
    for run in args.runs:
        logging.info(f"Processing run {run}.")
        if args.chunks is None:
            run_path = get_parquet_path() / f"{run}/{args.bin_size}_bins"
            files = list(run_path.glob(f"{run:05d}_*_hits.parq"))
            chunks = sorted([int(f.stem.split("_")[1]) for f in files])
        else:
            chunks = args.chunks

        for chunk in chunks:
            logging.info(f"Processing chunk {chunk}.")
            create_graph_data(
                run=run,
                chunk=chunk,
                recreate=args.recreate,
                num_events=args.num_events,
                use_faser=args.use_faser,
                apply_good_event_selection=args.good_event_selection,
                bin_size=args.bin_size,
            )


if __name__ == "__main__":
    main()
