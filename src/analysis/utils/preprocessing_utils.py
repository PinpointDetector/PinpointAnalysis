import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from analysis.utils.geometry import Geometry, get_geometry, get_rebinned_geometry
from analysis.utils.units import um
from analysis.utils.utils import get_parquet_path, get_root_path


def extrapolate_hits(df: pd.DataFrame, z_target: float) -> pd.DataFrame:
    df = df.copy()

    z_cols = ["z_0", "z_1", "z_2"]
    x_cols = ["x_0", "x_1", "x_2"]
    y_cols = ["y_0", "y_1", "y_2"]

    x_targets = []
    y_targets = []
    for idx, row in df.iterrows():
        if (
            (row["hit_counts_0"] > 0)
            & (row["hit_counts_1"] > 0)
            & (row["hit_counts_2"] > 0)
        ):
            z_pos = row[z_cols].values
            x_pos = row[x_cols].values
            y_pos = row[y_cols].values

            x_fit = np.polyfit(z_pos, x_pos, deg=2)  # Quadratic fit for x
            y_fit = np.polyfit(z_pos, y_pos, deg=1)  # Linear fit for y

            x_targets.append(np.polyval(x_fit, z_target))
            y_targets.append(np.polyval(y_fit, z_target))
        else:
            x_targets.append(0)
            y_targets.append(0)

    df["xp"] = x_targets
    df["yp"] = y_targets

    return df


def create_parquet_from_root(
    run: int, chunk: int, input_path: Path | None = None, recreate: bool = False
) -> None:
    run_path = get_parquet_path() / f"{run:05d}"
    hits_file = run_path / f"{run:05d}_{chunk:03d}_hits.parq"
    truth_file = run_path / f"{run:05d}_{chunk:03d}_truth.parq"
    faser_file = run_path / f"{run:05d}_{chunk:03d}_faser.parq"
    primaries_file = run_path / f"{run:05d}_{chunk:03d}_primaries.parq"
    geo_path = run_path / "geometry.pkl"
    if (
        hits_file.exists()
        and truth_file.exists()
        and faser_file.exists()
        and primaries_file.exists()
        and geo_path.exists()
        and not recreate
    ):
        logging.info(f"Hits, truth, and geometry files already exist in {run_path}.")
        return

    if input_path is None:
        input_path = get_root_path() / f"{run}/{run:05d}_{chunk:03d}.root"
    if not input_path.exists():
        raise FileNotFoundError(f"Input ROOT file {input_path} does not exist.")

    _ = get_geometry(geo_path=geo_path, root_path=input_path)

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

    # Get events which contain a charmed hadron (nu -> X + c)
    charm_pdg_ids = [421, 421, 431, 4122, 4112, 4212, 4222, 4132, 4232]
    charm_event_ids = primaries_df.loc[primaries_df["PDG"].abs().isin(charm_pdg_ids)][
        "evtID"
    ].unique()
    truth_df["charm"] = 0
    truth_df.loc[truth_df["event_id"].isin(charm_event_ids), "charm"] = 1

    primary_lepton_df = primaries_df.query("trackID == 1")
    # remove trackID column
    primary_lepton_df = primary_lepton_df[
        [i for i in primary_lepton_df.columns if i != "trackID"]
    ]
    primary_lepton_df = primary_lepton_df.rename(
        columns={
            "evtID": "event_id",
            "E": "E_lepton",
            "PDG": "pdg_lepton",
            "Px": "px_lepton",
            "Py": "py_lepton",
            "Pz": "pz_lepton",
        }
    )

    if len(truth_df) != len(primary_lepton_df):
        raise ValueError("Dataframes should have same length!")

    truth_df = pd.merge(
        truth_df, primary_lepton_df, left_on="event_id", right_on="event_id"
    )

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
        "hit_fromPrimaryLepton",
        "hit_fromPrimaryEMShower",
        "hit_fromCharmedHadron",
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

    # FASER positions in each spectrometer layer
    faser_pos_columns = ["event_id", "trackerID", "energy", "x", "y", "z", "pz"]
    faser_pos_df = ak.to_dataframe(
        root_file["Hits/faserHits"].arrays(faser_pos_columns, library="ak"),
        how="outer",
    )
    faser_pos_df.sort_values(
        ["event_id", "trackerID", "energy"], ascending=[True, True, False], inplace=True
    )

    # Number of hits per layer
    faser_nhits_df = (
        faser_pos_df.query("energy > 10e3").groupby(["event_id", "trackerID"]).size()
    )
    faser_nhits_df = pd.DataFrame(faser_nhits_df, columns=["hit_counts"]).reset_index()
    faser_nhits_event_df = faser_nhits_df.pivot(
        index="event_id", columns="trackerID", values=["hit_counts"]
    )
    faser_nhits_event_df.columns = [
        f"{col}_{int(tracker)}" for col, tracker in faser_nhits_event_df.columns
    ]

    # Take hit with highest energy in each layer per event
    faser_pos_df = faser_pos_df[
        ~faser_pos_df.duplicated(subset=["event_id", "trackerID"], keep="first")
    ]
    faser_pos_df = faser_pos_df.fillna({"trackerID": 0})
    # Create dataframe with one row per event and columns for each tracker layer's position
    faser_pos_event_df = (
        faser_pos_df.query("~trackerID.isna()")
        .reset_index()
        .pivot(
            index="event_id",
            columns="trackerID",
            values=["x", "y", "z", "pz"],
        )
    )
    faser_pos_event_df.columns = [
        f"{col}_{int(tracker)}" for col, tracker in faser_pos_event_df.columns
    ]
    faser_pos_event_df = faser_pos_event_df.reset_index()

    faser_df = pd.merge(
        faser_pos_event_df,
        faser_nhits_event_df.reset_index(),
        on="event_id",
        how="left",
    )
    faser_df.fillna(0, inplace=True)
    faser_df = extrapolate_hits(faser_df, z_target=750)
    faser_df.rename(
        {
            "hit_counts_0": "nhits_0",
            "hit_counts_1": "nhits_1",
            "hit_counts_2": "nhits_2",
            "xp": "faser_x",
            "yp": "faser_y",
        },
        axis=1,
        inplace=True,
    )

    # primaries (for easier access later)
    # primaries_columns = ["evtID", "trackID", "PDG", "E", "Px", "Py", "Pz"]
    # primaries_df = root_file["primaries"].arrays(primaries_columns, library="pd")
    primaries_df = primaries_df.rename(
        columns={
            "evtID": "event_id",
            "E": "E",
            "PDG": "pdg",
            "Px": "px",
            "Py": "py",
            "Pz": "pz",
        }
    )

    # write to parquet
    (get_parquet_path() / f"{run}").mkdir(parents=True, exist_ok=True)
    df.to_parquet(hits_file)
    truth_df.to_parquet(truth_file)
    faser_df.to_parquet(faser_file)
    primaries_df.to_parquet(primaries_file)
    logging.info(f"Created hits file {hits_file} and truth file {truth_file}.")


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

    queries = [
        "(abs(hit_pdgc) != 11) & (abs(hit_pdgc) != 13)",
        "(abs(hit_pdgc) == 11) & (hit_fromPrimaryEMShower == 0)",
        "(abs(hit_pdgc) == 11) & (hit_fromPrimaryEMShower == 1)",
        "(abs(hit_pdgc) == 13)",
    ]
    for label, query in enumerate(queries):
        hits_resampled.loc[hits_resampled.eval(query), "pdg_label"] = label

    aggregator_dict = {
        "energy": (energy_var, "sum"),
        "n_hits": (energy_var, "count"),
        "pdg_label": ("pdg_label", "max"),
    }

    resampled = (
        hits_resampled.groupby(["event_id", layer_var, "new_pixel_x", "new_pixel_y"])
        .agg(**aggregator_dict)
        .reset_index()
    )

    # resampled["hit_label"] = 0
    # resampled.loc[resampled["from_muon"] == 1, "hit_label"] = 1
    # resampled.loc[resampled["from_electron"] == 1, "hit_label"] = 2

    resampled.rename(
        columns={"new_pixel_x": "pixel_x", "new_pixel_y": "pixel_y"}, inplace=True
    )

    # If there are n pixels, sometimes we get index n, which is out of bounds
    resampled = resampled[
        (resampled["pixel_x"] < new_geometry.num_x_pixels)
        & (resampled["pixel_y"] < new_geometry.num_y_pixels)
    ]
    resampled.loc[:, "x"] = new_geometry.pixel_xpos[
        resampled["pixel_x"].astype(int).values
    ]
    resampled.loc[:, "y"] = new_geometry.pixel_ypos[
        resampled["pixel_y"].astype(int).values
    ]
    resampled.loc[:, "z"] = new_geometry.pixel_zpos[
        resampled["hit_layerID"].astype(int).values
    ]
    return resampled


def create_rebinned_parquet_file(
    run: int, chunk: int, pixel_size: float, recreate: bool = False
) -> None:
    """
    pixel_size: in um
    """
    output_path = get_parquet_path() / f"{run}/{pixel_size:03.0f}um_bins"
    output_hits_file = output_path / f"{run}_{chunk:03d}_hits.parq"
    output_truth_file = output_path / f"{run}_{chunk:03d}_truth.parq"
    output_faser_file = output_path / f"{run}_{chunk:03d}_faser.parq"
    output_geometry_file = output_path / "geometry.pkl"

    if output_hits_file.exists() and not recreate:
        logging.info(f"Output path {output_path} already exists.")
        return

    input_geo_file = get_parquet_path() / f"{run}/geometry.pkl"
    input_hits_file = get_parquet_path() / f"{run}/{run}_{chunk:03d}_hits.parq"
    input_truth_file = get_parquet_path() / f"{run}/{run}_{chunk:03d}_truth.parq"
    input_faser_file = get_parquet_path() / f"{run}/{run}_{chunk:03d}_faser.parq"

    if not input_hits_file.exists():
        raise FileNotFoundError(f"Input parquet file {input_hits_file} does not exist.")
    hits_df = pd.read_parquet(input_hits_file)

    geo = get_geometry(geo_path=input_geo_file)
    new_geo = get_rebinned_geometry(
        pixel_size=pixel_size * um, old_geometry=geo, geometry_path=output_geometry_file
    )
    new_df = get_rebinned_df(hits_df=hits_df, old_geometry=geo, new_geometry=new_geo)
    output_path.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(output_hits_file)
    output_truth_file.symlink_to(input_truth_file)
    output_faser_file.symlink_to(input_faser_file)
    logging.info(f"Created rebinned hits file {output_hits_file}.")


def get_good_event_ids(
    df: pd.DataFrame, geo: Geometry | None = None, apply_truth_cuts: bool = False
) -> np.ndarray:
    # Require distance of 10 mm from detector edge, so that shower is (fully) contained
    # if geo is None:
    #     geo = Geometry()
    # df = df.query(f"(vx.abs() < {geo.xmax} - 20) & (vy.abs() < {geo.ymax} - 20)")

    # Take only events in the fiducial volume
    # df = df.query("vx**2 + vy**2 < 100**2")

    if apply_truth_cuts:
        df.loc[:, "E_ratio"] = df["E_lepton"] / df["E_nu"]
        df.loc[:, "tan_theta_lepton"] = (
            np.sqrt(df["px_lepton"].values ** 2 + df["py_lepton"].values ** 2)
            / df["pz_lepton"].values
        )
        df = df.query(
            "(E_ratio > 0.2) & (tan_theta_lepton < 0.025) & (vx**2 + vy**2 < 100**2)"
        )
    else:
        if geo is None:
            geo = Geometry()
        df = df.query(f"(vx.abs() < {geo.xmax} - 20) & (vy.abs() < {geo.ymax} - 20)")

    return df["event_id"].values
