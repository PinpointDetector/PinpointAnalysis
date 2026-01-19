from typing import Final

# Length units (base: mm)
mm: Final[float] = 1.0
cm: Final[float] = 1e1
m: Final[float] = 1e3
um: Final[float] = 1e-3

# Derived units
mm2: Final[float] = mm**2
cm2: Final[float] = cm**2
cm3: Final[float] = cm**3

# Energy units (base: GeV)
GeV: Final[float] = 1.0
MeV: Final[float] = 1e-3
keV: Final[float] = 1e-6
eV: Final[float] = 1e-9

# Mass units (base: kg)
kg: Final[float] = 1.0
g: Final[float] = 1e-3

# Time units (base: s)
s: Final[float] = 1.0
ns: Final[float] = 1e-9

# Charge (base: fC)
fC: Final[float] = 1.0
C: Final[float] = 1e15

# Angle units (base: rad)
rad: Final[float] = 1.0
mrad: Final[float] = 1e-3
urad: Final[float] = 1e-6

# Cross  units (base: fb)
fb: Final[float] = 1.0
pb: Final[float] = 1e-3


# Conversion functions
def to_cm(value_mm: float) -> float:
    """Convert mm to cm"""
    return value_mm / cm


def to_m(value_mm: float) -> float:
    """Convert mm to m"""
    return value_mm / m


def to_MeV(value_GeV: float) -> float:
    """Convert GeV to MeV"""
    return value_GeV / MeV
