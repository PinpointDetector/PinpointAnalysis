from dataclasses import dataclass

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

    num_scintillator_layers: int = 0

    z_offset: float = 0  # in mm
    """Z offset of the detector in mm"""

    @property
    def layer_thickness(self) -> float:
        """Layer thickness in mm"""
        layer_thickness = (
            self.tungsten_thickness + self.silicon_thickness + self.box_thickness
        )
        if self.num_scintillator_layers > 0:
            layer_thickness += (
                self.tungsten_thickness
                + self.num_scintillator_layers * self.scintillator_thickness
            )
        return layer_thickness

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
        tungsten_thickness=old_geometry.tungsten_thickness,
        silicon_thickness=old_geometry.silicon_thickness,
        num_layers=old_geometry.num_layers,
    )


# TODO
# def get_geometry_from_root(geom_data: dict[str, list[float]]) -> Geometry:
#     pass
