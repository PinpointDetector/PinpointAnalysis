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


def loop_events(
    hits_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    faser_df: pd.DataFrame,
    good_event_ids: np.ndarray,
    label: int,
    data_path: Path,
    truth_path: Path,
    use_faser: bool = True,
    use_truth: bool = True,
) -> None:
    graphs = []
    valid_event_ids = []
    for event_id in tqdm(good_event_ids):
        event_hits_df = hits_df.query(f"event_id == {event_id}")

        # Skip events with too few nodes (< 100)
        if len(event_hits_df) < 100:
            continue

        faser_event_df = faser_df.query(f"event_id == {event_id}")
        event_truth_df = truth_df.query(f"event_id == {event_id}")

        event_graph_data = create_true_event_graph(
            hits_df=event_hits_df,
            faser_df=faser_event_df,
            truth_df=event_truth_df,
            label=label,
            use_faser=use_faser,
            use_truth_features=use_truth,
        )

        graphs.append(event_graph_data)
        valid_event_ids.append(event_id)

    # Filter truth and faser dataframes to only include valid events
    truth_df = truth_df[truth_df["event_id"].isin(valid_event_ids)]
    # faser_df = faser_df[faser_df["event_id"].isin(valid_event_ids)]
    # tau_df = tau_df[tau_df["evtID"].isin(valid_event_ids)]

    logging.info(f"Saved {len(graphs)}/{len(good_event_ids)} events.")

    # Save data
    logging.info(f"Saving dataset to {data_path}.")
    torch.save(graphs, data_path)
    truth_df.to_parquet(truth_path)

    logging.info(f"Saved {len(graphs)} graphs.")


def get_data(run: int, chunk=int) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    path = get_parquet_path() / f"{run:05d}"
    hits_path = path / f"{run:05d}_{chunk:03d}_hits.parq"
    truth_path = path / f"{run:05d}_{chunk:03d}_truth.parq"
    faser_path = path / f"{run:05d}_{chunk:03d}_faser.parq"
    tau_path = path / f"{run:05d}_{chunk:03d}_tau.parq"

    hits_df = pd.read_parquet(hits_path)
    truth_df = pd.read_parquet(truth_path)
    faser_df = pd.read_parquet(faser_path)
    if tau_path.exists():
        tau_df = pd.read_parquet(tau_path)
    else:
        tau_df = None

    return hits_df, truth_df, faser_df, tau_df


def get_str_from_run(run: int) -> str:
    """Convert run number to string label."""
    if run >= 10000:
        run -= 10000
    if run % 4 == 0:
        return "nue"
    elif run % 4 == 1:
        return "num"
    elif run % 4 == 2:
        return "nut"
    elif run % 4 == 3:
        return "nun"


def get_label_from_run(run: int) -> int:
    """Convert run number to classification label."""
    if run >= 10000:
        run -= 10000
    if run % 4 == 3:
        return 0  # NC
    elif run % 4 == 0:
        return 1  # CC nue
    elif run % 4 == 1:
        return 2  # CC num
    elif run % 4 == 2:
        return 3  # CC nut


def create_true_event_graph(
    hits_df: pd.DataFrame,
    faser_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    label: int,
    use_faser: bool = True,
    use_truth_features: bool = True,
) -> tuple[Data]:
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
    # x_tensor = torch.tensor(n_hits_log.reshape(-1, 1), dtype=torch.float)
    # x_tensor = pos_tensor
    features = hits_df[["x", "y", "z", "pdg_label"]].values
    x_tensor = torch.tensor(features, dtype=torch.float)

    # Store normalization parameters for reference
    mean_features = np.array([pos_mean[0], pos_mean[1], pos_mean[2]])
    # std_features = np.array([n_hits_log.mean(), n_hits_log.std()])

    # Node-level labels: 0: other, 1: electron, 2: muon
    # Merge primary and non-primary electrons
    # hits_df["label"] = hits_df["pdg_label"].replace({2: 1, 3: 2})
    hits_df["label"] = hits_df["pdg_label"]
    y_node = torch.tensor(hits_df["label"].values.astype(np.int64), dtype=torch.long)

    # Graph-level label (event classification)
    y_graph = torch.tensor([label], dtype=torch.long)

    # FASER spectrometer features (event-level/graph-level)
    x_event = np.array([])
    if use_faser:
        # FASER spectrometer hits features: nhits_0, nhits_1, nhits_2, x, y
        x_faser = faser_df[
            [
                "nhits_0",
                "nhits_1",
                "nhits_2",
                "x_faser",
                "y_faser",
                "dx_dz_faser",
                "dy_dz_faser",
            ]
        ].values[0]
        # Normalize x and y positions using the same normalization as pixel hits
        x_faser[3] = (x_faser[3] - pos_mean[0]) / 100.0
        x_faser[4] = (x_faser[4] - pos_mean[1]) / 100.0
        x_event = np.append(x_event, x_faser)
    if use_truth_features:
        x_truth = truth_df[
            ["E_vis", "pT_miss", "p_jet", "pT_jet", "p_l", "pT_l"]
        ].values[0]
        x_truth[0] *= 1e-3
        x_truth[2] *= 1e-3
        x_truth[4] *= 1e-3
        x_event = np.append(x_event, x_truth)
    if len(x_event) > 0:
        x_event_tensor = torch.tensor(x_event, dtype=torch.float)
    else:
        x_event_tensor = None

    # Extract truth information from the first row (single event)
    truth_info = truth_df.iloc[0]
    E_nu = torch.tensor(truth_info["E_nu"] * 1e-3, dtype=torch.float)
    E_lepton = torch.tensor(truth_info["E_lepton"] * 1e-3, dtype=torch.float)
    E_vis = torch.tensor(truth_info["E_vis"] * 1e-3, dtype=torch.float)
    pT_miss = torch.tensor(truth_info["pT_miss"], dtype=torch.float)
    p_jet = torch.tensor(truth_info["p_jet"] * 1e-3, dtype=torch.float)
    pT_jet = torch.tensor(truth_info["pT_jet"], dtype=torch.float)
    p_l = torch.tensor(truth_info["p_l"] * 1e-3, dtype=torch.float)
    pT_l = torch.tensor(truth_info["pT_l"], dtype=torch.float)

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
        y_graph=y_graph,  # [1] graph label
        x_event=x_event_tensor,  # [7 + 6 = 13] FASER spectrometer features (event-level)
        E_nu=E_nu,  # neutrino energy (TeV)
        E_lepton=E_lepton,  # lepton energy (TeV)
        E_vis=E_vis,
        pT_miss=pT_miss,
        p_jet=p_jet,
        pT_jet=pT_jet,
        p_l=p_l,
        pT_l=pT_l,
        charm=charm,  # charm flag (boolean as long)
        true_pos=true_pos_normalized,  # [3] true neutrino interaction position normalized by 100
        true_pos_centered=true_pos_centered,  # [3] true neutrino interaction position centered at pos_mean
    )

    return data


def create_graph_data(
    run: int,
    chunk: int,
    recreate: bool = False,
    num_events: int | None = None,
    use_faser: bool = True,
    use_truth: bool = True,
    pixel_size: float | None = None,
    apply_good_event_selection: bool = False,
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

    # Create output directories
    output_path = get_torch_path() / f"{run}"
    if pixel_size is not None:
        output_path = output_path / f"{pixel_size}um_bins"
    model_dir = "gravnet"
    if use_faser:
        model_dir += "_faser"
    if use_truth:
        model_dir += "_truth"
    if apply_good_event_selection:
        model_dir += "_good_events"
    output_path = output_path / model_dir
    output_path.mkdir(parents=True, exist_ok=True)

    data_path = output_path / f"{run_str}_{chunk:03d}.pt"
    truth_path = output_path / f"{run_str}_{chunk:03d}_truth.parq"

    if data_path.exists() and not recreate:
        logging.info(
            f"Graph data for run {run} chunk {chunk} already exists at {data_path}."
        )
        return
    logging.info(f"Creating GravNet data for run {run} chunk {chunk} ({run_str})")

    # Read input files
    hits_df, truth_df, faser_df, tau_df = get_data(run=run, chunk=chunk)
    # if tau_df is not None:
    #     # Select hadronic tau decays
    #     nut_had_event_ids = tau_df.query("is_had == 1")["evtID"].unique()
    #     print(
    #         f"tau -> had: {len(nut_had_event_ids) / tau_df['evtID'].nunique() * 100:.1f} %"
    #     )
    #     hits_df = hits_df.loc[hits_df["event_id"].isin(nut_had_event_ids)]
    #     truth_df = truth_df.loc[truth_df["event_id"].isin(nut_had_event_ids)]
    #     faser_df = faser_df.loc[faser_df["event_id"].isin(nut_had_event_ids)]

    # Event selection
    good_event_ids = get_good_event_ids(
        truth_df, apply_truth_cuts=apply_good_event_selection
    )
    if num_events is not None:
        good_event_ids = good_event_ids[:num_events]
    hits_df = hits_df[hits_df["event_id"].isin(good_event_ids)]
    truth_df = truth_df[truth_df["event_id"].isin(good_event_ids)]
    faser_df = faser_df[faser_df["event_id"].isin(good_event_ids)]
    if tau_df is not None:
        tau_df = tau_df[tau_df["evtID"].isin(good_event_ids)]
    logging.info(f"Selected {len(good_event_ids)} good events events")

    # Calcualte truth variables
    if run_str == "num" or run_str == "nue":
        truth_df["E_vis"] = truth_df["E_nu"]
        truth_df["pT_miss"] = 0
        truth_df["p_jet"] = truth_df["E_nu"] - truth_df["E_lepton"]
        truth_df["pT_jet"] = np.sqrt(
            (truth_df["px_nu"] - truth_df["px_lepton"]) ** 2
            + (truth_df["py_nu"] - truth_df["py_lepton"]) ** 2
        )
        truth_df["p_l"] = truth_df["E_lepton"]
        truth_df["pT_l"] = np.sqrt(
            truth_df["px_lepton"] ** 2 + truth_df["py_lepton"] ** 2
        )
    elif run_str == "nut":
        # tau_df = tau_df.query("abs(PDG) == 16")[["evtID", "Px", "Py", "Pz"]].rename(
        #     columns={
        #         "evtID": "event_id",
        #         "Px": "px_miss",
        #         "Py": "py_miss",
        #         "Pz": "pz_miss",
        #     }
        # )
        # truth_df = pd.merge(truth_df, tau_df, on="event_id", how="left")

        # truth_df["E_miss"] = np.sqrt(
        #     truth_df["px_miss"] ** 2
        #     + truth_df["py_miss"] ** 2
        #     + truth_df["pz_miss"] ** 2
        # )

        # truth_df["E_vis"] = truth_df["E_nu"] - truth_df["E_miss"]
        # truth_df["pT_miss"] = np.sqrt(
        #     truth_df["px_miss"] ** 2 + truth_df["py_miss"] ** 2
        # )
        # truth_df["p_jet"] = truth_df["E_nu"] - truth_df["E_miss"]
        # truth_df["pT_jet"] = np.sqrt(
        #     (truth_df["px_nu"] - truth_df["px_miss"]) ** 2
        #     + (truth_df["py_nu"] - truth_df["py_miss"]) ** 2
        # )
        # truth_df["p_l"] = -1
        # truth_df["pT_l"] = -1
        # tau_df = tau_df.loc[~((tau_df["PDG"].abs() == 11) & (tau_df["Pz"] < 0.2))]
        n_tau_events = tau_df["evtID"].nunique()

        # tau -> had. decay
        tau_had_event_ids = tau_df.query("is_had == True")["evtID"].unique()

        print(f"tau -> had: {len(tau_had_event_ids) / n_tau_events * 100:.1f} %")
        truth_had_df = truth_df.loc[truth_df["event_id"].isin(tau_had_event_ids)]
        tau_had_df = tau_df.loc[tau_df["evtID"].isin(tau_had_event_ids)]
        tau_had_df = tau_had_df.query("abs(PDG) == 16")[
            ["evtID", "Px", "Py", "Pz"]
        ].rename(
            columns={
                "evtID": "event_id",
                "Px": "px_miss",
                "Py": "py_miss",
                "Pz": "pz_miss",
            }
        )
        truth_had_df = pd.merge(truth_had_df, tau_had_df, on="event_id", how="left")

        truth_had_df["p_miss"] = np.sqrt(
            truth_had_df["px_miss"] ** 2
            + truth_had_df["py_miss"] ** 2
            + truth_had_df["pz_miss"] ** 2
        )
        truth_had_df["E_vis"] = truth_had_df["E_nu"] - truth_had_df["p_miss"]
        truth_had_df["pT_miss"] = np.sqrt(
            truth_had_df["px_miss"] ** 2 + truth_had_df["py_miss"] ** 2
        )
        truth_had_df["p_jet"] = truth_had_df["E_nu"] - truth_had_df["p_miss"]
        truth_had_df["pT_jet"] = np.sqrt(
            (truth_had_df["px_nu"] - truth_had_df["px_miss"]) ** 2
            + (truth_had_df["py_nu"] - truth_had_df["py_miss"]) ** 2
        )
        truth_had_df["p_l"] = -1
        truth_had_df["pT_l"] = -1

        # tau -> nu_tau + mu + nu_mu decay
        tau_mu_event_ids = tau_df.query("is_mu == True")["evtID"].unique()
        tau_mu_df = tau_df.loc[tau_df["evtID"].isin(tau_mu_event_ids)]
        truth_mu_df = truth_df.loc[truth_df["event_id"].isin(tau_mu_event_ids)]
        print(f"tau -> mu: {len(truth_mu_df) / n_tau_events * 100:.1f} %")
        # neutrinos from tau -> nu_tau + mu + nu_mu decay
        nu_tau_mu_df = (
            tau_mu_df.loc[tau_mu_df["PDG"].abs().isin([14, 16])].groupby("evtID").sum()
        )
        nu_tau_mu_df = nu_tau_mu_df.reset_index()
        nu_tau_mu_df = nu_tau_mu_df[["evtID", "Px", "Py", "Pz"]].rename(
            columns={
                "evtID": "event_id",
                "Px": "px_miss",
                "Py": "py_miss",
                "Pz": "pz_miss",
            }
        )
        # muon from tau -> nu_tau + mu + nu_mu decay
        mu_tau_mu_df = (
            tau_mu_df.loc[tau_mu_df["PDG"].abs() == 13].groupby("evtID").sum()
        )
        mu_tau_mu_df = mu_tau_mu_df.reset_index()
        mu_tau_mu_df = mu_tau_mu_df[["evtID", "Px", "Py", "Pz"]].rename(
            columns={"evtID": "event_id", "Px": "px_mu", "Py": "py_mu", "Pz": "pz_mu"}
        )
        # merge
        tau_mu_df = pd.merge(nu_tau_mu_df, mu_tau_mu_df, how="inner", on="event_id")
        truth_mu_df = pd.merge(truth_df, tau_mu_df, how="inner", on="event_id")
        # Calculate kinematic variables
        truth_mu_df["p_miss"] = np.sqrt(
            truth_mu_df["px_miss"] ** 2
            + truth_mu_df["py_miss"] ** 2
            + truth_mu_df["pz_miss"] ** 2
        )
        truth_mu_df["E_vis"] = truth_mu_df["E_nu"] - truth_mu_df["p_miss"]
        truth_mu_df["pT_miss"] = np.sqrt(
            truth_mu_df["px_miss"] ** 2 + truth_mu_df["py_miss"] ** 2
        )
        truth_mu_df["p_jet"] = truth_mu_df["E_nu"] - truth_mu_df["E_lepton"]
        truth_mu_df["pT_jet"] = np.sqrt(
            (truth_mu_df["px_nu"] - truth_mu_df["px_lepton"]) ** 2
            + (truth_mu_df["py_nu"] - truth_mu_df["py_lepton"]) ** 2
        )
        truth_mu_df["p_l"] = np.sqrt(
            truth_mu_df["px_mu"] ** 2
            + truth_mu_df["py_mu"] ** 2
            + truth_mu_df["pz_mu"] ** 2
        )
        truth_mu_df["pT_l"] = np.sqrt(
            truth_mu_df["px_mu"] ** 2 + truth_mu_df["py_mu"] ** 2
        )

        # tau -> nu_tau + e + nu_e decay
        tau_e_event_ids = tau_df.query("is_e == True")["evtID"].unique()
        tau_e_df = tau_df.loc[tau_df["evtID"].isin(tau_e_event_ids)]
        truth_e_df = truth_df.loc[truth_df["event_id"].isin(tau_e_event_ids)]
        # neutrinos from tau -> nu_tau + e + nu_e decay
        nu_tau_e_df = (
            tau_e_df.loc[tau_e_df["PDG"].abs().isin([14, 16])].groupby("evtID").sum()
        )
        nu_tau_e_df = nu_tau_e_df.reset_index()
        nu_tau_e_df = nu_tau_e_df[["evtID", "Px", "Py", "Pz"]].rename(
            columns={
                "evtID": "event_id",
                "Px": "px_miss",
                "Py": "py_miss",
                "Pz": "pz_miss",
            }
        )
        # e from tau -> nu_tau + e + nu_e decay
        e_tau_e_df = tau_e_df.loc[tau_e_df["PDG"].abs() == 11].groupby("evtID").sum()
        e_tau_e_df = e_tau_e_df.reset_index()
        e_tau_e_df = e_tau_e_df[["evtID", "Px", "Py", "Pz"]].rename(
            columns={"evtID": "event_id", "Px": "px_e", "Py": "py_e", "Pz": "pz_e"}
        )
        tau_e_df = pd.merge(nu_tau_e_df, e_tau_e_df, how="inner", on="event_id")
        truth_e_df = pd.merge(truth_df, tau_e_df, how="inner", on="event_id")
        print(f"tau -> e: {len(truth_e_df) / n_tau_events * 100:.1f} %")
        # Calculate kinematic variables
        truth_e_df["p_miss"] = np.sqrt(
            truth_e_df["px_miss"] ** 2
            + truth_e_df["py_miss"] ** 2
            + truth_e_df["pz_miss"] ** 2
        )
        truth_e_df["E_vis"] = truth_e_df["E_nu"] - truth_e_df["p_miss"]
        truth_e_df["pT_miss"] = np.sqrt(
            truth_e_df["px_miss"] ** 2 + truth_e_df["py_miss"] ** 2
        )
        truth_e_df["p_jet"] = truth_e_df["E_nu"] - truth_e_df["E_lepton"]
        truth_e_df["pT_jet"] = np.sqrt(
            (truth_e_df["px_nu"] - truth_e_df["px_lepton"]) ** 2
            + (truth_e_df["py_nu"] - truth_e_df["py_lepton"]) ** 2
        )
        truth_e_df["p_l"] = np.sqrt(
            truth_e_df["px_e"] ** 2 + truth_e_df["py_e"] ** 2 + truth_e_df["pz_e"] ** 2
        )
        truth_e_df["pT_l"] = np.sqrt(truth_e_df["px_e"] ** 2 + truth_e_df["py_e"] ** 2)

    elif run_str == "nun":
        truth_df["E_vis"] = truth_df["E_nu"] - truth_df["E_lepton"]
        truth_df["pT_miss"] = np.sqrt(
            truth_df["px_lepton"] ** 2 + truth_df["py_lepton"] ** 2
        )
        truth_df["p_jet"] = truth_df["E_nu"] - truth_df["E_lepton"]
        truth_df["pT_jet"] = np.sqrt(
            (truth_df["px_nu"] - truth_df["px_lepton"]) ** 2
            + (truth_df["py_nu"] - truth_df["py_lepton"]) ** 2
        )
        truth_df["p_l"] = -1
        truth_df["pT_l"] = -1
    else:
        raise ValueError(f"Unknow run {run_str}")

    # Create graph data for each event
    if run_str in ["num", "nue", "nun"]:
        loop_events(
            hits_df=hits_df,
            truth_df=truth_df,
            faser_df=faser_df,
            good_event_ids=good_event_ids,
            label=label,
            data_path=data_path,
            truth_path=truth_path,
            use_faser=use_faser,
            use_truth=use_truth,
        )
    elif run_str == "nut":
        # tau -> e nu_tau nu_e
        nut_e_data_path = output_path / f"{run_str}_{chunk:03d}_nut_e.pt"
        nut_e_truth_path = output_path / f"{run_str}_{chunk:03d}_nut_e_truth.parq"
        loop_events(
            hits_df=hits_df,
            truth_df=truth_e_df,
            faser_df=faser_df,
            good_event_ids=tau_e_event_ids,
            label=3,
            data_path=nut_e_data_path,
            truth_path=nut_e_truth_path,
            use_faser=use_faser,
            use_truth=use_truth,
        )
        # tau -> mu nu_tau nu_mu
        nut_mu_data_path = output_path / f"{run_str}_{chunk:03d}_nut_mu.pt"
        nut_mu_truth_path = output_path / f"{run_str}_{chunk:03d}_nut_mu_truth.parq"
        loop_events(
            hits_df=hits_df,
            truth_df=truth_mu_df,
            faser_df=faser_df,
            good_event_ids=tau_mu_event_ids,
            label=4,
            data_path=nut_mu_data_path,
            truth_path=nut_mu_truth_path,
            use_faser=use_faser,
            use_truth=use_truth,
        )
        # tau -> had
        nut_had_data_path = output_path / f"{run_str}_{chunk:03d}_nut_had.pt"
        nut_had_truth_path = output_path / f"{run_str}_{chunk:03d}_nut_had_truth.parq"
        loop_events(
            hits_df=hits_df,
            truth_df=truth_had_df,
            faser_df=faser_df,
            good_event_ids=tau_had_event_ids,
            label=5,
            data_path=nut_had_data_path,
            truth_path=nut_had_truth_path,
            use_faser=use_faser,
            use_truth=use_truth,
        )


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
        "--use-faser",
        action="store_true",
        default=False,
        help="Include FASER spectrometer data (nhits_0, nhits_1, nhits_2, x, y) as separate graph-level features.",
    )
    parser.add_argument(
        "--use-truth",
        action="store_true",
        default=False,
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
    parser.add_argument(
        "-p",
        "--pixel-size",
        type=float,
        required=False,
        help="Rebinned pixel size in um.",
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
            run_path = get_parquet_path() / f"{run}/200um_bins"
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
                pixel_size=args.pixel_size,
                apply_good_event_selection=args.good_event_selection,
                use_truth=args.use_truth,
            )


if __name__ == "__main__":
    main()
