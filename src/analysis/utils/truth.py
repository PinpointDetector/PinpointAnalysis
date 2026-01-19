from dataclasses import dataclass

import pandas as pd

from analysis.utils.units import MeV, mm
from analysis.utils.utils import get_label_from_pdg


@dataclass
class TruthParticle:
    nu_pdg: int
    nu_e: float
    x: float
    y: float
    z: float
    l_pdg: int
    l_px: float
    l_py: float
    l_pz: float

    @property
    def is_cc(self):
        if abs(self.l_pdg) in [11, 13, 15]:
            return True
        elif abs(self.l_pdg) in [12, 14, 16]:
            return False
        else:
            raise ValueError(f"Invalid lepton pdg: {self.l_pdg}")

    def __str__(self):
        return (
            f"{self.nu_pdg} -> {self.l_pdg},"
            f" E_nu={self.nu_e:.1f} GeV,"
            f" E_lep={self.l_pz:.1f} GeV, "
            f" pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
        )

    def get_latex_title(self):
        return (
            rf"${get_label_from_pdg(self.nu_pdg)} \rightarrow {get_label_from_pdg(self.l_pdg)},"
            rf"E_{{\nu}} = {self.nu_e:.1f} \, \mathrm{{GeV}}, "
            rf"E_{{l}} = {self.l_pz:.1f} \, \mathrm{{GeV}}$, "
            rf"$({self.x:.1f}, {self.y:.1f}, {self.z:.1f}) \mathrm{{mm}}$"
        )


def get_truth_particle(truth_df: pd.Series) -> TruthParticle:
    if isinstance(truth_df, pd.DataFrame):
        if len(truth_df) == 1:
            truth_df = truth_df.iloc[0]
        else:
            raise ValueError("Expected a single row DataFrame or Series.")
    elif isinstance(truth_df, pd.Series):
        pass
    else:
        raise TypeError("Expected a pandas Series or single row DataFrame.")
    return TruthParticle(
        nu_pdg=truth_df["pdg_nu"],
        nu_e=truth_df["E_nu"] * MeV,
        l_pdg=truth_df["pdg_lepton"],
        l_px=truth_df["px_lepton"] * MeV,
        l_py=truth_df["py_lepton"] * MeV,
        l_pz=truth_df["pz_lepton"] * MeV,
        x=truth_df["vx"] * mm,
        y=truth_df["vy"] * mm,
        z=truth_df["vz"] * mm,
    )
