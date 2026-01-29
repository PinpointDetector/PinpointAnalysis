import argparse
import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn_graph
from tqdm import tqdm

from analysis.utils.geometry import Geometry, get_rebinned_geometry
from analysis.utils.units import um
from analysis.utils.utils import get_parquet_path, get_root_path, get_torch_path


def get_rebinned_df(
    hits_df: pd.DataFrame,
    old_geometry: Geometry,
    new_geometry: Geometry,
    layer_var: str = "hit_layerID",
    pixel_x_var: str = "hit_colID",
    pixel_y_var: str = "hit_rowID",
    energy_var: str = "hit_edep",
) -> pd.DataFrame:
    """
    Get hits with different bin size.
    This is useful to cluster pixels and reduce total number of pixels.
    """
    x_bin_factor = new_geometry.pixel_x_size / old_geometry.pixel_x_size
    y_bin_factor = new_geometry.pixel_y_size / old_geometry.pixel_y_size
    hits_resampled = hits_df.copy()
    hits_resampled.loc[:, "new_pixel_x"] = (
        hits_df[pixel_x_var].values // x_bin_factor
    ).astype(int)
    hits_resampled.loc[:, "new_pixel_y"] = (
        hits_df[pixel_y_var].values // y_bin_factor
    ).astype(int)

    aggregator_dict = {
        "energy": (energy_var, "sum"),
        "n_hits": (energy_var, "count"),
    }

    resampled = (
        hits_resampled.groupby(["event_id", layer_var, "new_pixel_x", "new_pixel_y"])
        .agg(**aggregator_dict)
        .reset_index()
    )

    resampled.rename(
        columns={"new_pixel_x": "pixel_x", "new_pixel_y": "pixel_y"}, inplace=True
    )
    return resampled


def create_parquet_from_root(run: int, chunk: int, recreate: bool = False) -> None:
    hits_file = get_parquet_path() / f"{run}/{run}_{chunk}_hits.parq"
    truth_file = get_parquet_path() / f"{run}/{run}_{chunk}_truth.parq"
    if hits_file.exists() and truth_file.exists() and not recreate:
        logging.info(f"Hits and truth files already exist in {hits_file.parent}.")
        return

    input_path = get_root_path() / f"{run}/{run:05d}_{chunk:03d}.root"
    if not input_path.exists():
        raise FileNotFoundError(f"Input ROOT file {input_path} does not exist.")
    root_file = uproot.open(input_path)

    # truth neutrino
    truth_columns = ["evtID", "initPDG", "initX", "initY", "initZ", "initE"]
    truth_df = root_file["event"].arrays(truth_columns, library="pd")
    truth_df.rename(
        columns={
            "evtID": "event_id",
            "initPDG": "pdg_nu",
            "initE": "E_nu",
            "initX": "vx",
            "initY": "vy",
            "initZ": "vz",
        },
        inplace=True,
    )

    # truth lepton
    primaries_columns = ["evtID", "trackID", "PDG", "E", "Px", "Py", "Pz"]
    primaries_df = root_file["primaries"].arrays(primaries_columns, library="pd")
    primaries_df.query("trackID == 1", inplace=True)
    primaries_df = primaries_df[[i for i in primaries_df.columns if i != "trackID"]]
    primaries_df = primaries_df.rename(
        columns={
            "evtID": "event_id",
            "E": "E_lepton",
            "PDG": "pdg_lepton",
            "Px": "px_lepton",
            "Py": "py_lepton",
            "Pz": "pz_lepton",
        }
    )

    if len(truth_df) != len(primaries_df):
        raise ValueError("Dataframes should have same length!")

    truth_df = pd.merge(truth_df, primaries_df, left_on="event_id", right_on="event_id")

    # geometry
    geom_columns = ["pixel_Xpos", "pixel_Ypos", "pixel_Zpos"]
    geom = root_file["geometry"].arrays(geom_columns, library="np")

    # pixel hits
    columns = [
        "event_id",
        "hit_rowID",
        "hit_colID",
        "hit_layerID",
        "hit_pdgc",
        "hit_edep",
    ]
    df: pd.DataFrame = ak.to_dataframe(
        root_file["Hits/pixelHits"].arrays(columns, library="ak"),
        how="outer",
    )

    # remove hits that are outside of the detector range
    max_row_index = 8596
    max_col_index = 12788
    df.query(
        f"hit_rowID < {max_row_index} and hit_colID < {max_col_index}", inplace=True
    )

    # map pixel indices to positions
    df.loc[:, "x"] = geom["pixel_Xpos"][0][df["hit_colID"].astype(int).values]
    df.loc[:, "y"] = geom["pixel_Ypos"][0][df["hit_rowID"].astype(int).values]
    df.loc[:, "z"] = geom["pixel_Zpos"][0][df["hit_layerID"].astype(int).values]
    # write to parquet
    (get_parquet_path() / f"{run}").mkdir(parents=True, exist_ok=True)
    df.to_parquet(hits_file)
    truth_df.to_parquet(truth_file)
    logging.info(f"Created hits file {hits_file} and truth file {truth_file}.")


def create_rebinned_parquet(
    run: int, chunk: int, pixel_size: float, recreate: bool = False
) -> None:
    output_path = get_parquet_path() / f"{run}/{pixel_size:03.0f}um_bins"
    output_hits_file = output_path / f"{run}_{chunk}_hits.parq"
    output_truth_file = output_path / f"{run}_{chunk}_truth.parq"

    if output_hits_file.exists() and not recreate:
        logging.info(f"Output path {output_path} already exists.")
        return

    input_hits_file = get_parquet_path() / f"{run}/{run}_{chunk}_hits.parq"
    input_truth_file = get_parquet_path() / f"{run}/{run}_{chunk}_truth.parq"

    if not input_hits_file.exists():
        raise FileNotFoundError(f"Input parquet file {input_hits_file} does not exist.")
    hits_df = pd.read_parquet(input_hits_file)

    geo = Geometry()
    new_geo = get_rebinned_geometry(pixel_size=pixel_size * um, old_geometry=geo)
    new_df = get_rebinned_df(hits_df=hits_df, old_geometry=geo, new_geometry=new_geo)
    output_path.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(output_hits_file)
    output_truth_file.symlink_to(input_truth_file)
    logging.info(f"Created rebinned hits file {output_hits_file}.")


def get_good_event_ids(df: pd.DataFrame):
    # Require distance of 10 mm from detector edge, so that shower is (fully) contained
    geo = Geometry()
    df = df.query(f"(vx.abs() < {geo.xmax} - 10) & (vy.abs() < {geo.ymax} - 10)")
    return df["event_id"].values


def create_pixel_graph(
    hits_df: pd.DataFrame,
    k_neighbors: int = 128,
    layer_scale: float = 1.0,
    layer_var: str = "hit_layerID",
) -> Data:
    # Use pre-computed normalized features

    start_df = hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")
    mean_x, mean_y = start_df[["pixel_x", "pixel_y"]].mean().astype(int).values

    pixel_x = hits_df["pixel_x"].values - mean_x
    pixel_y = hits_df["pixel_y"].values - mean_y
    layer_id = hits_df[layer_var].values
    nhits = hits_df["n_hits"].values

    x = torch.tensor(
        np.stack([pixel_x, pixel_y, layer_id, nhits], axis=1), dtype=torch.float
    )
    y = torch.tensor(hits_df["label"].values.astype(np.int64)[0], dtype=torch.long)

    pos = np.stack([pixel_x, pixel_y, layer_id * layer_scale], axis=1)
    pos_tensor = torch.tensor(pos, dtype=torch.float)
    edge_index = knn_graph(pos_tensor, k=k_neighbors, loop=False)
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos_tensor)

    return data, mean_x, mean_y


def create_pixel_graph_with_energy(
    hits_df: pd.DataFrame,
    energy: float,
    k_neighbors: int = 128,
    layer_scale: float = 1.0,
    layer_var: str = "hit_layerID",
) -> Data:
    """
    Create graph data with both label and energy.
    Energy is stored as a scalar attribute in the Data object.
    """
    start_df = hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")
    mean_x, mean_y = start_df[["pixel_x", "pixel_y"]].mean().astype(int).values

    pixel_x = hits_df["pixel_x"].values - mean_x
    pixel_y = hits_df["pixel_y"].values - mean_y
    layer_id = hits_df[layer_var].values
    nhits = hits_df["n_hits"].values

    x = torch.tensor(
        np.stack([pixel_x, pixel_y, layer_id, nhits], axis=1), dtype=torch.float
    )
    y = torch.tensor(hits_df["label"].values.astype(np.int64)[0], dtype=torch.long)

    # Store energy as a scalar tensor
    energy_tensor = torch.tensor([energy], dtype=torch.float)

    pos = np.stack([pixel_x, pixel_y, layer_id * layer_scale], axis=1)
    pos_tensor = torch.tensor(pos, dtype=torch.float)
    edge_index = knn_graph(pos_tensor, k=k_neighbors, loop=False)

    # Data class accepts arbitrary attributes
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        energy=energy_tensor,  # Add energy attribute
        pos=pos_tensor,
    )

    return data, mean_x, mean_y


def create_graph_data(
    pixel_size: float,
    num_events: int | None = None,
    recreate: bool = False,
    min_hits: int = 128,
    k_neighbors: int = 128,
) -> None:
    logging.info("Creating graph data torch data.")

    torch_path = get_torch_path() / f"gcn_{pixel_size:03.0f}um_bins_10000_10003"
    torch_path.mkdir(parents=True, exist_ok=True)
    output_torch_path = torch_path / "gcn.pt"
    output_truth_path = torch_path / "graph_truth.parq"

    if output_torch_path.exists() and not recreate:
        logging.info(f"Output path {output_torch_path} already exists.")
        return

    # FIXME: make this an indenpendent function, used by cnn + graph data
    nue_run = 10000
    nue_path = get_parquet_path() / f"{nue_run}/{pixel_size:03.0f}um_bins"
    nue_hits_files = sorted(nue_path.glob(f"{nue_run}_*_hits.parq"))
    nue_truth_files = sorted(nue_path.glob(f"{nue_run}_*_truth.parq"))
    nue_hits_df = pd.concat([pd.read_parquet(f) for f in nue_hits_files])
    nue_truth_df = pd.concat([pd.read_parquet(f) for f in nue_truth_files])

    num_run = 10001
    num_path = get_parquet_path() / f"{num_run}/{pixel_size:03.0f}um_bins"
    num_hits_files = sorted(num_path.glob(f"{num_run}_*_hits.parq"))
    num_truth_files = sorted(num_path.glob(f"{num_run}_*_truth.parq"))
    num_hits_df = pd.concat([pd.read_parquet(f) for f in num_hits_files])
    num_truth_df = pd.concat([pd.read_parquet(f) for f in num_truth_files])

    nun_run = 10003
    nun_path = get_parquet_path() / f"{nun_run}/{pixel_size:03.0f}um_bins"
    nun_hits_files = sorted(nun_path.glob(f"{nun_run}_*_hits.parq"))
    nun_truth_files = sorted(nun_path.glob(f"{nun_run}_*_truth.parq"))
    nun_hits_df = pd.concat([pd.read_parquet(f) for f in nun_hits_files])
    nun_truth_df = pd.concat([pd.read_parquet(f) for f in nun_truth_files])

    # event selection
    good_nue_event_ids = get_good_event_ids(nue_truth_df)
    good_num_event_ids = get_good_event_ids(num_truth_df)
    good_nun_event_ids = get_good_event_ids(nun_truth_df)
    print(f"Selected {len(good_nue_event_ids)} good nue events.")
    print(f"Selected {len(good_num_event_ids)} good num events.")
    print(f"Selected {len(good_nun_event_ids)} good nun events.")

    nue_hits_df = nue_hits_df.loc[nue_hits_df["event_id"].isin(good_nue_event_ids)]
    nue_truth_df = nue_truth_df.loc[nue_truth_df["event_id"].isin(good_nue_event_ids)]
    num_hits_df = num_hits_df.loc[num_hits_df["event_id"].isin(good_num_event_ids)]
    num_truth_df = num_truth_df.loc[num_truth_df["event_id"].isin(good_num_event_ids)]
    nun_hits_df = nun_hits_df.loc[nun_hits_df["event_id"].isin(good_nun_event_ids)]
    nun_truth_df = nun_truth_df.loc[nun_truth_df["event_id"].isin(good_nun_event_ids)]

    if num_events is not None:
        nue_event_ids = nue_hits_df["event_id"].unique()[:num_events]
        nue_hits_df = nue_hits_df.loc[nue_hits_df["event_id"].isin(nue_event_ids)]
        nue_truth_df = nue_truth_df.loc[nue_truth_df["event_id"].isin(nue_event_ids)]
        # nue_hits_df = nue_hits_df.query(f"event_id < {num_events}")
        # nue_truth_df = nue_truth_df.query(f"event_id < {num_events}")
        num_event_ids = num_hits_df["event_id"].unique()[:num_events]
        num_hits_df = num_hits_df.loc[num_hits_df["event_id"].isin(num_event_ids)]
        num_truth_df = num_truth_df.loc[num_truth_df["event_id"].isin(num_event_ids)]
        nun_event_ids = nun_hits_df["event_id"].unique()[:num_events]
        nun_hits_df = nun_hits_df.loc[nun_hits_df["event_id"].isin(nun_event_ids)]
        nun_truth_df = nun_truth_df.loc[nun_truth_df["event_id"].isin(nun_event_ids)]

    nue_hits_df.loc[:, "label"] = 0
    num_hits_df.loc[:, "label"] = 1
    nun_hits_df.loc[:, "label"] = 2

    # create unique event ids
    num_truth_df["event_id"] += 10_000
    num_hits_df["event_id"] += 10_000
    nun_hits_df["event_id"] += 20_000
    nun_truth_df["event_id"] += 20_000

    hits_df = pd.concat([nue_hits_df, num_hits_df, nun_hits_df])
    truth_df = pd.concat([nue_truth_df, num_truth_df, nun_truth_df])
    event_ids = hits_df["event_id"].unique()

    print(f"Creating graphs from {len(event_ids)} events...")

    min_hits = max(min_hits, k_neighbors + 1)

    dataset = []
    x_centers = []
    y_centers = []
    good_event_ids = []
    for event_id in tqdm(event_ids):
        event_hits = hits_df[hits_df["event_id"] == event_id].copy()

        # Skip events with too few hits
        if len(event_hits) < min_hits:
            continue

        graph_data, x_center, y_center = create_pixel_graph(
            event_hits, k_neighbors=k_neighbors
        )
        dataset.append(graph_data)
        x_centers.append(x_center)
        y_centers.append(y_center)
        good_event_ids.append(event_id)

    print(f"Created dataset with {len(dataset)} events")

    logging.info(f"Saving dataset to {output_torch_path}")
    torch.save(dataset, output_torch_path)
    truth_df = truth_df[truth_df["event_id"].isin(good_event_ids)]
    truth_df.to_parquet(output_truth_path)
    np.save(torch_path / "graph_x_centers.npy", x_centers)
    np.save(torch_path / "graph_y_centers.npy", y_centers)

    return dataset, truth_df, np.array(x_centers), np.array(y_centers)


def create_graph_data_with_energy(
    pixel_size: float,
    num_events: int | None = None,
    recreate: bool = False,
    min_hits: int = 128,
    k_neighbors: int = 128,
) -> None:
    """
    Create graph dataset with both event labels and energy regression targets.
    Each graph Data object contains:
    - x: node features
    - edge_index: kNN edges
    - y: event class label (0, 1, 2)
    - energy: neutrino energy for regression
    - pos: node positions
    """
    logging.info("Creating graph data with energy regression torch data.")

    torch_path = get_torch_path() / f"gcn_{pixel_size:03.0f}um_bins_10000_10003"
    torch_path.mkdir(parents=True, exist_ok=True)
    output_torch_path = torch_path / "gcn_with_energy.pt"
    output_truth_path = torch_path / "graph_truth_with_energy.parq"

    if output_torch_path.exists() and not recreate:
        logging.info(f"Output path {output_torch_path} already exists.")
        return

    # Load data (same as create_graph_data)
    nue_run = 10000
    nue_path = get_parquet_path() / f"{nue_run}/{pixel_size:03.0f}um_bins"
    nue_hits_files = sorted(nue_path.glob(f"{nue_run}_*_hits.parq"))
    nue_truth_files = sorted(nue_path.glob(f"{nue_run}_*_truth.parq"))
    nue_hits_df = pd.concat([pd.read_parquet(f) for f in nue_hits_files])
    nue_truth_df = pd.concat([pd.read_parquet(f) for f in nue_truth_files])

    num_run = 10001
    num_path = get_parquet_path() / f"{num_run}/{pixel_size:03.0f}um_bins"
    num_hits_files = sorted(num_path.glob(f"{num_run}_*_hits.parq"))
    num_truth_files = sorted(num_path.glob(f"{num_run}_*_truth.parq"))
    num_hits_df = pd.concat([pd.read_parquet(f) for f in num_hits_files])
    num_truth_df = pd.concat([pd.read_parquet(f) for f in num_truth_files])

    nun_run = 10003
    nun_path = get_parquet_path() / f"{nun_run}/{pixel_size:03.0f}um_bins"
    nun_hits_files = sorted(nun_path.glob(f"{nun_run}_*_hits.parq"))
    nun_truth_files = sorted(nun_path.glob(f"{nun_run}_*_truth.parq"))
    nun_hits_df = pd.concat([pd.read_parquet(f) for f in nun_hits_files])
    nun_truth_df = pd.concat([pd.read_parquet(f) for f in nun_truth_files])

    # Event selection
    good_nue_event_ids = get_good_event_ids(nue_truth_df)
    good_num_event_ids = get_good_event_ids(num_truth_df)
    good_nun_event_ids = get_good_event_ids(nun_truth_df)
    print(f"Selected {len(good_nue_event_ids)} good nue events.")
    print(f"Selected {len(good_num_event_ids)} good num events.")
    print(f"Selected {len(good_nun_event_ids)} good nun events.")

    nue_hits_df = nue_hits_df.loc[nue_hits_df["event_id"].isin(good_nue_event_ids)]
    nue_truth_df = nue_truth_df.loc[nue_truth_df["event_id"].isin(good_nue_event_ids)]
    num_hits_df = num_hits_df.loc[num_hits_df["event_id"].isin(good_num_event_ids)]
    num_truth_df = num_truth_df.loc[num_truth_df["event_id"].isin(good_num_event_ids)]
    nun_hits_df = nun_hits_df.loc[nun_hits_df["event_id"].isin(good_nun_event_ids)]
    nun_truth_df = nun_truth_df.loc[nun_truth_df["event_id"].isin(good_nun_event_ids)]

    if num_events is not None:
        nue_event_ids = nue_hits_df["event_id"].unique()[:num_events]
        nue_hits_df = nue_hits_df.loc[nue_hits_df["event_id"].isin(nue_event_ids)]
        nue_truth_df = nue_truth_df.loc[nue_truth_df["event_id"].isin(nue_event_ids)]
        num_event_ids = num_hits_df["event_id"].unique()[:num_events]
        num_hits_df = num_hits_df.loc[num_hits_df["event_id"].isin(num_event_ids)]
        num_truth_df = num_truth_df.loc[num_truth_df["event_id"].isin(num_event_ids)]
        nun_event_ids = nun_hits_df["event_id"].unique()[:num_events]
        nun_hits_df = nun_hits_df.loc[nun_hits_df["event_id"].isin(nun_event_ids)]
        nun_truth_df = nun_truth_df.loc[nun_truth_df["event_id"].isin(nun_event_ids)]

    nue_hits_df.loc[:, "label"] = 0
    num_hits_df.loc[:, "label"] = 1
    nun_hits_df.loc[:, "label"] = 2

    # Create unique event ids
    num_truth_df["event_id"] += 10_000
    num_hits_df["event_id"] += 10_000
    nun_hits_df["event_id"] += 20_000
    nun_truth_df["event_id"] += 20_000

    hits_df = pd.concat([nue_hits_df, num_hits_df, nun_hits_df])
    truth_df = pd.concat([nue_truth_df, num_truth_df, nun_truth_df])

    # Create event_id to energy mapping
    energy_map = dict(zip(truth_df["event_id"], truth_df["E_nu"]))
    event_ids = hits_df["event_id"].unique()

    print(f"Creating graphs with energy from {len(event_ids)} events...")

    min_hits = max(min_hits, k_neighbors + 1)

    dataset = []
    x_centers = []
    y_centers = []
    good_event_ids = []

    for event_id in tqdm(event_ids):
        event_hits = hits_df[hits_df["event_id"] == event_id].copy()

        # Skip events with too few hits
        if len(event_hits) < min_hits:
            continue

        # Get energy for this event
        energy = energy_map[event_id]

        graph_data, x_center, y_center = create_pixel_graph_with_energy(
            event_hits, energy=energy, k_neighbors=k_neighbors
        )
        dataset.append(graph_data)
        x_centers.append(x_center)
        y_centers.append(y_center)
        good_event_ids.append(event_id)

    print(f"Created dataset with {len(dataset)} events")

    logging.info(f"Saving dataset to {output_torch_path}")
    torch.save(dataset, output_torch_path)
    truth_df = truth_df[truth_df["event_id"].isin(good_event_ids)]
    truth_df.to_parquet(output_truth_path)
    np.save(torch_path / "graph_with_energy_x_centers.npy", x_centers)
    np.save(torch_path / "graph_with_energy_y_centers.npy", y_centers)

    return dataset, truth_df, np.array(x_centers), np.array(y_centers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-r",
    #     "--run",
    #     type=int,
    #     required=True,
    # )
    # parser.add_argument(
    #     "-c",
    #     "--chunk",
    #     type=int,
    #     required=True,
    # )
    parser.add_argument(
        "-p", "--pixel_size", type=float, default=200.0, help="New pixel size in um."
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate output files."
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="INFO",
        choices=["info", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "-n",
        "--num-events",
        type=int,
        default=None,
        help="Maximum number of events to process.",
    )
    parser.add_argument(
        "-k",
        "--k-neighbors",
        type=int,
        default=128,
        help="Number of k-nearest neighbors for graph construction.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    # create_parquet_from_root(run=args.run, chunk=args.chunk, recreate=args.recreate)
    # create_rebinned_parquet(
    #     run=args.run,
    #     chunk=args.chunk,
    #     pixel_size=args.pixel_size,
    #     recreate=args.recreate,
    # )
    create_graph_data(
        pixel_size=args.pixel_size,
        num_events=args.num_events,
        recreate=args.recreate,
        k_neighbors=args.k_neighbors,
    )
