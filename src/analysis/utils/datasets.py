import logging

import hist
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from analysis.utils.geometry import Geometry


def create_voxels(
    hits_df: pd.DataFrame,
    bins: tuple[int, int, int] = (128, 128, 200),
    layer_var: str = "hit_layerID",
) -> tuple[int, int, np.ndarray]:
    if len(hits_df) == 0:
        logging.warning("Empty hits dataframe received in create_voxels.")
        return 0, 0, np.zeros(bins)

    x_half_size = bins[0] // 2
    y_half_size = bins[1] // 2

    mean_x, mean_y = (
        hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")[["pixel_x", "pixel_y"]]
        .mean()
        .astype(int)
        .values
    )

    x_min, x_max = mean_x - x_half_size, mean_x + x_half_size
    y_min, y_max = mean_y - y_half_size, mean_y + y_half_size

    h = hist.Hist(
        hist.axis.Variable(np.arange(x_min - 0.5, x_max + 0.5, 1)),  # x pixel axis
        hist.axis.Variable(np.arange(y_min - 0.5, y_max + 0.5, 1)),  # y pixel axis
        hist.axis.Variable(np.arange(-0.5, bins[2] + 0.5, 1)),  # layer axis
        storage=hist.storage.Weight(),
    )
    h.fill(
        hits_df["pixel_x"],
        hits_df["pixel_y"],
        hits_df[layer_var],
        weight=hits_df["n_hits"],
    )
    voxels = h.values()
    if voxels.shape != bins:
        raise ValueError(f"Voxel shape {voxels.shape} does not match expected {bins}")
    return mean_x, mean_y, voxels


def create_projections(
    hits_df: pd.DataFrame,
    bins: tuple[int, int, int] = (128, 128, 150),
    layer_var: str = "hit_layerID",
) -> tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]:
    if len(hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")) == 0:
        return None, None, None, None

    x_half_size = bins[0] // 2
    y_half_size = bins[1] // 2

    mean_x, mean_y = (
        hits_df.query(f"{layer_var} >= 4 & {layer_var} <= 14")[["pixel_x", "pixel_y"]]
        .mean()
        .astype(int)
        .values
    )

    x_min, x_max = mean_x - x_half_size, mean_x + x_half_size
    y_min, y_max = mean_y - y_half_size, mean_y + y_half_size

    h_zx = hist.Hist(
        hist.axis.Variable(np.arange(-0.5, bins[2] + 0.5, 1)),  # layer axis
        hist.axis.Variable(np.arange(x_min - 0.5, x_max + 0.5, 1)),  # x pixel axis
        storage=hist.storage.Weight(),
    )

    h_zy = hist.Hist(
        hist.axis.Variable(np.arange(-0.5, bins[2] + 0.5, 1)),  # layer axis
        hist.axis.Variable(np.arange(y_min - 0.5, y_max + 0.5, 1)),  # y pixel axis
        storage=hist.storage.Weight(),
    )

    h_zx.fill(hits_df[layer_var], hits_df["pixel_x"], weight=hits_df["n_hits"])
    h_zy.fill(hits_df[layer_var], hits_df["pixel_y"], weight=hits_df["n_hits"])

    zx_projection = h_zx.values()
    zy_projection = h_zy.values()

    return mean_x, mean_y, zx_projection, zy_projection


class CNN3DDataset(Dataset):
    """Dataset that stores binned pixel counts per event"""

    def __init__(
        self,
        data_list: list[tuple[pd.DataFrame, pd.DataFrame, int]],
        bins: tuple[int, int, int],
        geo: Geometry,
    ):
        self.voxels = []
        self.labels = []
        self.e_nu = []
        self.e_lepton = []
        self.vx = []
        self.vy = []
        self.delta_vx = []
        self.delta_vy = []
        self.mean_x = []
        self.mean_y = []

        for hits_df, truth_df, label in data_list:
            event_ids = hits_df["event_id"].unique()
            logging.info(f"Processing {len(event_ids)} events")

            for event_id in event_ids:
                event_hits = hits_df.loc[hits_df["event_id"] == event_id]
                event_truth = truth_df[truth_df["event_id"] == event_id].iloc[0]
                # if len(event_truth) != 1:
                #     logging.warning(
                #         f"Warning: Expected exactly one row, "
                #         f"but got {len(event_truth)}, skipping event {event_id}."
                #     )
                #     continue

                mean_x, mean_y, voxel_data = create_voxels(event_hits, bins)

                voxel_tensor = torch.FloatTensor(voxel_data).unsqueeze(0)
                voxel_tensor = torch.log(voxel_tensor + 1)
                self.voxels.append(voxel_tensor)

                # truth variables
                self.labels.append(label)
                self.e_nu.append(np.float32(event_truth["E_nu"]))
                self.e_lepton.append(np.float32(event_truth["E_lepton"]))

                vx = np.float32(event_truth["vx"])
                vy = np.float32(event_truth["vy"])
                self.vx.append(vx)
                self.vy.append(vy)
                mean_x_pos = geo.get_x_pos(mean_x)
                mean_y_pos = geo.get_y_pos(mean_y)
                self.mean_x.append(mean_x_pos)
                self.mean_y.append(mean_y_pos)
                self.delta_vx.append(vx - mean_x_pos)
                self.delta_vy.append(vy - mean_y_pos)

        logging.info(f"Created dataset with {len(self.voxels)} samples")

    def __len__(self):
        return len(self.voxels)

    def __getitem__(self, idx):
        voxel_tensor = self.voxels[idx]
        targets = {
            "label": self.labels[idx],
            "E_nu": np.float32(self.e_nu[idx]),
            "E_lepton": np.float32(self.e_lepton[idx]),
            "delta_vx": np.float32(self.delta_vx[idx]),
            "delta_vy": np.float32(self.delta_vy[idx]),
            "vx": np.float32(self.vx[idx]),
            "vy": np.float32(self.vy[idx]),
        }

        return voxel_tensor, targets


class CNNProjectionDataset(Dataset):
    """Dataset that stores binned pixel counts per event"""

    def __init__(
        self,
        data_list: list[tuple[pd.DataFrame, pd.DataFrame, int]],
        bins: tuple[int, int, int],
        geo: Geometry,
    ):
        self.projections = []
        self.labels = []
        self.e_nu = []
        self.e_lepton = []
        self.vx = []
        self.vy = []
        self.delta_vx = []
        self.delta_vy = []

        for hits_df, truth_df, label in data_list:
            event_ids = hits_df["event_id"].unique()
            logging.info(f"Processing {len(event_ids)} events")

            for event_id in event_ids:
                event_hits = hits_df.loc[hits_df["event_id"] == event_id]
                event_truth = truth_df[truth_df["event_id"] == event_id].iloc[0]

                mean_x, mean_y, zx_proj, zy_proj = create_projections(event_hits, bins)
                if (
                    mean_x is None
                    or mean_y is None
                    or zx_proj is None
                    or zy_proj is None
                ):
                    # FIXME: Because of this the truth dataframe has more entries
                    continue

                zx_tensor = torch.FloatTensor(zx_proj).unsqueeze(0)
                zx_tensor = torch.log(zx_tensor + 1)

                zy_tensor = torch.FloatTensor(zy_proj).unsqueeze(0)
                zy_tensor = torch.log(zy_tensor + 1)

                self.projections.append((zx_tensor, zy_tensor))

                # truth variables
                self.labels.append(label)
                self.e_nu.append(np.float32(event_truth["E_nu"]))
                self.e_lepton.append(np.float32(event_truth["E_lepton"]))

                vx = np.float32(event_truth["vx"])
                vy = np.float32(event_truth["vy"])
                mean_x_pos = geo.get_x_pos(mean_x)
                mean_y_pos = geo.get_y_pos(mean_y)
                self.vx.append(vx)
                self.vy.append(vy)
                self.delta_vx.append(mean_x_pos - vx)
                self.delta_vy.append(mean_y_pos - vy)

        logging.info(f"Created dataset with {len(self.projections)} samples")

    def __len__(self):
        return len(self.projections)

    def __getitem__(self, idx):
        zx_tensor, zy_tensor = self.projections[idx]
        targets = {
            "label": self.labels[idx],
            "E_nu": np.float32(self.e_nu[idx]),
            "E_lepton": np.float32(self.e_lepton[idx]),
            "delta_vx": np.float32(self.delta_vx[idx]),
            "delta_vy": np.float32(self.delta_vy[idx]),
            "vx": np.float32(self.vx[idx]),
            "vy": np.float32(self.vy[idx]),
        }

        return (zx_tensor, zy_tensor), targets
