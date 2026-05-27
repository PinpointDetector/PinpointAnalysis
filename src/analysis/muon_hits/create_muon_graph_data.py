import argparse
import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot
from sklearn.preprocessing import StandardScaler
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
    hits_resampled["fromMuon"] = 0
    hits_resampled.loc[hits_resampled["hit_pdgc"].abs() == 13, "fromMuon"] = 1

    aggregator_dict = {
        "energy": (energy_var, "sum"),
        "n_hits": (energy_var, "count"),
        "fromMuon": ("fromMuon", "any"),
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
    geo_file = get_parquet_path() / f"{run}/geometry.parq"
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
    output_path = get_parquet_path() / f"{run}/muon_{pixel_size:03.0f}um_bins"
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
    # geo = Geoetry()
    # df = df.query(f"(vx.abs() < {geo.xmax} - 10) & (vy.abs() < {geo.ymax} - 10)")

    # Take only events in the fiducial volume
    df = df.query("vx**2 + vy**2 < 100**2")
    return df["event_id"].values


def create_pixel_graph(
    hits_df: pd.DataFrame,
    k_neighbors: int = 16,
    max_layer_distance: int = 2,
    layer_scale: float = 2.5,
) -> Data:
    # Use pre-computed normalized features

    # start_df = hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")
    # mean_x, mean_y = start_df[["pixel_x", "pixel_y"]].mean().astype(int).values

    # pixel_x = hits_df["pixel_x"].values - mean_x
    # pixel_y = hits_df["pixel_y"].values - mean_y
    # layer_id = hits_df[layer_var].values
    # nhits = hits_df["n_hits"].values

    # x = torch.tensor(
    #     np.stack([pixel_x, pixel_y, layer_id, nhits], axis=1), dtype=torch.float
    # )
    # y = torch.tensor(hits_df["label"].values.astype(np.int64)[0], dtype=torch.long)

    # pos = np.stack([pixel_x, pixel_y, layer_id * layer_scale], axis=1)
    # pos_tensor = torch.tensor(pos, dtype=torch.float)
    # edge_index = knn_graph(pos_tensor, k=k_neighbors, loop=False)
    # data = Data(x=x, edge_index=edge_index, y=y, pos=pos_tensor)

    # return data, mean_x, mean_y

    # TODO normalize
    node_features_norm = np.stack(
        [
            hits_df["pixel_x_norm"].values,
            hits_df["pixel_y_norm"].values,
            hits_df["layer_id_norm"].values,
            # hits_df["n_hits_norm"].values,
        ],
        axis=1,
    )

    x = torch.tensor(node_features_norm, dtype=torch.float)
    y = torch.tensor(hits_df["fromMuon"].values.astype(np.int64), dtype=torch.long)

    # Use original (unnormalized) positions for graph construction
    pixel_x = hits_df["pixel_x"].values.astype(np.float32)
    pixel_y = hits_df["pixel_y"].values.astype(np.float32)
    layer_id = hits_df["hit_layerID"].values.astype(np.float32)

    pos = np.stack([pixel_x, pixel_y, layer_id * layer_scale], axis=1)
    pos_tensor = torch.tensor(pos, dtype=torch.float)
    edge_index = knn_graph(pos_tensor, k=k_neighbors, loop=False)

    # Filter edges that span too many layers (optional but helps with locality)
    src_layers = layer_id[edge_index[0].numpy()]
    dst_layers = layer_id[edge_index[1].numpy()]
    layer_diff = np.abs(src_layers - dst_layers)
    valid_edges = layer_diff <= max_layer_distance
    edge_index = edge_index[:, valid_edges]

    # Create graph data
    data = Data(x=x, edge_index=edge_index, y=y, pos=pos_tensor)

    return data


def create_graph_data(
    pixel_size: float,
    num_events: int | None = None,
    recreate: bool = False,
    min_hits: int = 50,
    k_neighbors: int = 16,
) -> None:
    logging.info("Creating graph data torch data.")

    torch_path = get_torch_path() / f"muon_gcn_{pixel_size:03.0f}um_bins_10000_10003"
    torch_path.mkdir(parents=True, exist_ok=True)
    output_torch_path = torch_path / "gcn.pt"
    output_truth_path = torch_path / "graph_truth.parq"
    output_scaler_path = torch_path / "scaler_params_muon_gcn.parq"

    if output_torch_path.exists() and not recreate:
        logging.info(f"Output path {output_torch_path} already exists.")
        return

    # FIXME: make this an indenpendent function, used by cnn + graph data
    nue_run = 10000
    nue_path = get_parquet_path() / f"{nue_run}/muon_{pixel_size:03.0f}um_bins"
    nue_hits_files = sorted(nue_path.glob(f"{nue_run}_*_hits.parq"))
    nue_truth_files = sorted(nue_path.glob(f"{nue_run}_*_truth.parq"))
    nue_hits_df = pd.concat([pd.read_parquet(f) for f in nue_hits_files])
    nue_truth_df = pd.concat([pd.read_parquet(f) for f in nue_truth_files])

    num_run = 10001
    num_path = get_parquet_path() / f"{num_run}/muon_{pixel_size:03.0f}um_bins"
    num_hits_files = sorted(num_path.glob(f"{num_run}_*_hits.parq"))
    num_truth_files = sorted(num_path.glob(f"{num_run}_*_truth.parq"))
    num_hits_df = pd.concat([pd.read_parquet(f) for f in num_hits_files])
    num_truth_df = pd.concat([pd.read_parquet(f) for f in num_truth_files])

    nun_run = 10003
    nun_path = get_parquet_path() / f"{nun_run}/muon_{pixel_size:03.0f}um_bins"
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

    nue_hits_df["fromMuon"] = 0
    nun_hits_df["fromMuon"] = 0

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

    print("Computing global normalization parameters...")
    pixel_x = hits_df["pixel_x"].values.astype(np.float32)
    pixel_y = hits_df["pixel_y"].values.astype(np.float32)
    layer_id = hits_df["hit_layerID"].values.astype(np.float32)

    # Stack features for normalization
    node_features = np.stack([pixel_x, pixel_y, layer_id], axis=1)

    # Fit scaler on all data
    scaler = StandardScaler()
    node_features_norm = scaler.fit_transform(node_features)

    # Add normalized columns to DataFrame
    hits_df = hits_df.copy()
    hits_df["pixel_x_norm"] = node_features_norm[:, 0]
    hits_df["pixel_y_norm"] = node_features_norm[:, 1]
    hits_df["layer_id_norm"] = node_features_norm[:, 2]

    if output_scaler_path is not None:
        scaler_df = pd.DataFrame(
            {"mean": scaler.mean_, "scale": scaler.scale_},
            index=["pixel_x", "pixel_y", "layer_id"],
        )
        scaler_df.to_parquet(output_scaler_path)
        print(f"Saved normalization parameters to {output_scaler_path}")
        print("Normalization statistics:")
        print(scaler_df)

    print(f"Creating graphs from {len(event_ids)} events...")

    min_hits = max(min_hits, k_neighbors + 1)

    dataset = []
    good_event_ids = []
    for event_id in tqdm(event_ids):
        event_hits = hits_df[hits_df["event_id"] == event_id].copy()

        if len(event_hits) < min_hits:
            continue

        graph_data = create_pixel_graph(event_hits, k_neighbors=k_neighbors)
        dataset.append(graph_data)
        good_event_ids.append(event_id)

    print(len(dataset))
    print(dataset[0].x)
    print(dataset[0].y)

    # Calculate class balance
    total_nodes = sum([data.y.size(0) for data in dataset])
    muon_nodes = sum([data.y.sum().item() for data in dataset])
    print(f"Total nodes: {total_nodes:,}")
    print(f"Muon nodes: {muon_nodes:,} ({100 * muon_nodes / total_nodes:.1f}%)")
    print(
        f"Non-muon nodes: {total_nodes - muon_nodes:,} ({100 * (1 - muon_nodes / total_nodes):.1f}%)"
    )

    print(f"Created dataset with {len(dataset)} events")

    logging.info(f"Saving dataset to {output_torch_path}")
    torch.save(dataset, output_torch_path)
    truth_df = truth_df[truth_df["event_id"].isin(good_event_ids)]
    truth_df.to_parquet(output_truth_path)


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
        default=16,
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
