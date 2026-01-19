import argparse
from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from analysis.utils.geometry import Geometry, get_rebinned_geometry
from analysis.utils.utils import get_parquet_path, get_root_path


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
        print(f"Hits and truth files already exist in {hits_file.parent}.")
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
    primaries_df.rename(
        columns={
            "evtID": "event_id",
            "E": "E_lepton",
            "PDG": "pdg_lepton",
            "Px": "px_lepton",
            "Py": "py_lepton",
            "Pz": "pz_lepton",
        },
        inplace=True,
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
    df["x"] = df.apply(lambda row: geom["pixel_Xpos"][0][int(row["hit_colID"])], axis=1)
    df["y"] = df.apply(lambda row: geom["pixel_Ypos"][0][int(row["hit_rowID"])], axis=1)
    df["z"] = df.apply(
        lambda row: geom["pixel_Zpos"][0][int(row["hit_layerID"])], axis=1
    )

    # write to parquet
    df.to_parquet(hits_file)
    truth_df.to_parquet(truth_file)


def create_rebinned_parquet(
    run: int, chunk: int, pixel_size: float, recreate: bool = False
) -> None:
    output_path = (
        get_parquet_path() / f"{run}/{pixel_size:03d}um_bins/{run}_{chunk}_hits.parq"
    )
    if output_path.exists() and not recreate:
        print(f"Output path {output_path} already exists.")
        return

    input_path = get_parquet_path() / f"{run}/{run}_{chunk}_hits.parq"
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file {input_path} does not exist.")

    hits_df = pd.read_parquet(input_path)

    geo = Geometry()
    new_geo = get_rebinned_geometry(pixel_size=pixel_size, old_geometry=geo)
    new_df = get_rebinned_df(hits_df=hits_df, old_geometry=geo, new_geometry=new_geo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--run",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--chunk",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-p", "--pixel_size", type=float, default=500.0, help="New pixel size."
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Recreate output files."
    )
    args = parser.parse_args()

    create_parquet_from_root(run=args.run, chunk=args.chunk, recreate=args.recreate)
    create_rebinned_parquet(
        run=args.run,
        chunk=args.chunk,
        pixel_size=args.pixel_size,
        recreate=args.recreate,
    )
