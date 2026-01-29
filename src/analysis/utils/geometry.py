from copyreg import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot

from analysis.utils.units import mm, um


@dataclass
class Geometry:
    pixel_x_size: float = 20.8 * um
    """Pixel size in mm"""

    pixel_y_size: float = 22.8 * um
    """Pixel size in mm"""

    num_x_pixels: int = 12788
    """Number of pixels in one layer"""

    num_y_pixels: int = 8596
    """Number of pixels in one layer"""

    num_layers: int = 150
    """Number of layers in the detector"""

    tungsten_thickness: float = 5 * mm

    silicon_thickness: float = 10 * um

    box_thickness: float = 5 * mm

    scintillator_thickness: float = 5 * mm

    num_scintillators: int = 0

    _layer_thickness: float | None = None

    z_offset: float = 0  # in mm
    """Z offset of the detector in mm"""

    _pixel_xpos: np.ndarray | None = None
    _pixel_ypos: np.ndarray | None = None
    _pixel_zpos: np.ndarray | None = None

    @property
    def pixel_xpos(self) -> np.ndarray:
        """X positions of the pixel centers in mm"""
        if self._pixel_xpos is None:
            self._pixel_xpos = (
                np.arange(0, self.num_x_pixels * self.pixel_x_size, self.pixel_x_size)
                - self.detector_x_size / 2
                + self.pixel_x_size / 2
            )
        return self._pixel_xpos

    @property
    def pixel_ypos(self) -> np.ndarray:
        """X positions of the pixel centers in mm"""
        if self._pixel_ypos is None:
            self._pixel_ypos = (
                np.arange(0, self.num_y_pixels * self.pixel_y_size, self.pixel_y_size)
                - self.detector_y_size / 2
                + self.pixel_y_size / 2
            )
        return self._pixel_ypos

    @property
    def pixel_zpos(self) -> np.ndarray:
        """X positions of the pixel centers in mm"""
        if self._pixel_zpos is None:
            self._pixel_zpos = (
                np.arange(
                    0, self.num_layers * self.layer_thickness, self.layer_thickness
                )
                + self.tungsten_thickness
                + self.box_thickness
                + self.silicon_thickness / 2
            )
        return self._pixel_zpos

    @property
    def layer_thickness(self) -> float:
        """Layer thickness in mm"""
        if self._layer_thickness is None:
            self._layer_thickness = (
                self.tungsten_thickness + self.silicon_thickness + self.box_thickness
            )
            if self.num_scintillators > 0:
                self._layer_thickness += (
                    self.num_scintillators * self.scintillator_thickness
                    + self.tungsten_thickness
                )
        return self._layer_thickness

    @property
    def detector_size(self) -> float:
        if self.detector_x_size != self.detector_y_size:
            raise ValueError("Detector is not symmetric, please specify x or y size.")
        return self.detector_x_size

    @property
    def detector_x_size(self) -> float:
        """Total size of the detector in mm"""
        return self.pixel_x_size * self.num_x_pixels

    @property
    def detector_y_size(self) -> float:
        """Total size of the detector in mm"""
        return self.pixel_y_size * self.num_y_pixels

    @property
    def detector_thickness(self) -> float:
        """Total depth of the detector in mm"""
        return self.num_layers * self.layer_thickness

    @property
    def xmin(self) -> float:
        """Minimum x position in mm"""
        return -self.detector_x_size / 2

    @property
    def xmax(self) -> float:
        """Maximum x position in mm"""
        return self.detector_x_size / 2

    @property
    def ymin(self) -> float:
        """Minimum y position in mm"""
        return -self.detector_y_size / 2

    @property
    def ymax(self) -> float:
        """Maximum y position in mm"""
        return self.detector_y_size / 2

    @property
    def zmin(self) -> float:
        """Minimum z position in mm"""
        return self.z_offset

    @property
    def zmax(self) -> float:
        """Maximum z position in mm"""
        return self.z_offset + self.detector_thickness

    def get_z_pos(self, layer: int) -> float:
        return layer * self.layer_thickness + self.z_offset

    def get_layer(self, pos: float) -> float:
        return (pos - self.z_offset) / self.layer_thickness

    def get_pos(self, pixel: int) -> float:
        if (self.num_x_pixels != self.num_y_pixels) or (
            self.pixel_x_size != self.pixel_y_size
        ):
            raise ValueError(
                "Detector is not symmetric, please specify x or y position."
            )
        return (pixel - self.num_x_pixels // 2) * self.pixel_x_size

    def get_x_pos(self, pixel: int) -> float:
        return (pixel - self.num_x_pixels // 2) * self.pixel_x_size

    def get_y_pos(self, pixel: int) -> float:
        return (pixel - self.num_y_pixels // 2) * self.pixel_y_size

    def get_pixel(self, pos: float) -> int:
        if (self.num_x_pixels != self.num_y_pixels) or (
            self.pixel_x_size != self.pixel_y_size
        ):
            raise ValueError("Detector is not symmetric, please specify x or y pixel.")
        return int(pos / self.pixel_x_size + self.num_x_pixels // 2)

    def get_x_pixel(self, pos: float) -> int:
        return int(pos / self.pixel_x_size + self.num_x_pixels // 2)

    def get_y_pixel(self, pos: float) -> int:
        return int(pos / self.pixel_y_size + self.num_y_pixels // 2)


def get_rebinned_geometry(pixel_size: float, old_geometry: Geometry) -> Geometry:
    return Geometry(
        pixel_x_size=pixel_size,
        pixel_y_size=pixel_size,
        num_x_pixels=int(
            old_geometry.num_x_pixels * old_geometry.pixel_x_size / pixel_size
        ),
        num_y_pixels=int(
            old_geometry.num_y_pixels * old_geometry.pixel_y_size / pixel_size
        ),
        num_layers=old_geometry.num_layers,
        tungsten_thickness=old_geometry.tungsten_thickness,
        silicon_thickness=old_geometry.silicon_thickness,
        box_thickness=old_geometry.box_thickness,
        scintillator_thickness=old_geometry.scintillator_thickness,
        num_scintillators=old_geometry.num_scintillators,
        _layer_thickness=old_geometry.layer_thickness,
        z_offset=old_geometry.z_offset,
        _pixel_zpos=old_geometry._pixel_zpos,
    )


def get_geometry(geo_path: str | Path, root_path: str | Path | None = None) -> Geometry:
    if geo_path.exists():
        with open(geo_path, "rb") as f:
            geo = pickle.load(f)
    else:
        if root_path is None or not Path(root_path).is_file():
            raise ValueError(
                f"Root path {root_path} was not provided or does not exist."
            )
        geo_df = uproot.open(root_path)["geometry"].arrays(library="np")

        xpos = geo_df["pixel_Xpos"][0]
        num_x_pixels = len(xpos)
        xsize = (xpos[-1] - xpos[0]) / (num_x_pixels - 1)

        ypos = geo_df["pixel_Ypos"][0]
        num_y_pixels = len(ypos)
        ysize = (ypos[-1] - ypos[0]) / (num_y_pixels - 1)

        zpos = geo_df["pixel_Zpos"][0]
        num_layers = geo_df["nLayers"][0]
        zsize = (zpos[-1] - zpos[0]) / (num_layers - 1)

        num_scintillators = geo_df["sim_flag"][0] + 1

        geo = Geometry(
            pixel_x_size=xsize * mm,
            pixel_y_size=ysize * mm,
            num_x_pixels=num_x_pixels,
            num_y_pixels=num_y_pixels,
            num_layers=num_layers,
            num_scintillators=num_scintillators,
            tungsten_thickness=geo_df["tungsten_thickness"][0] * mm,
            silicon_thickness=geo_df["silicon_thickness"][0] * um,
            _layer_thickness=zsize * mm,
            _pixel_xpos=xpos,
            _pixel_ypos=ypos,
            _pixel_zpos=zpos,
        )
        with open(geo_path, "wb") as f:
            pickle.dump(geo, f)
    return geo
