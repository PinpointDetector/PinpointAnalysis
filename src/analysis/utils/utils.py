from analysis.utils.colors import TangoColors


def get_label_from_pdg(pdg: int) -> str:
    pdg_latex_dict = {
        11: r"e^{-}",
        12: r"\nu_{e}",
        13: r"\mu^{-}",
        14: r"\nu_{\mu}",
        15: r"\tau^{-}",
        16: r"\nu_{\tau}",
        -16: r"\overline{\nu}_{\tau}",
        -15: r"\tau^{+}",
        -14: r"\overline{\nu}_{\mu}",
        -13: r"\mu^{+}",
        -12: r"\overline{\nu}_{e}",
        -11: r"e^{+}",
        211: r"\pi^{+}",
        321: r"K^{+}",
        411: r"D^{+}",
        431: r"D_{s}^{+}",
        2212: "p",
        22: r"\gamma",
        111: r"\pi^{0}",
        130: r"K_{L}^{0}",
        310: r"K_{S}^{0}",
        311: r"K^{0}",
        421: r"D^{0}",
        221: r"\eta",
        223: r"\omega(782)",
        2112: "n",
        3212: r"\Sigma^{0}",
        3122: r"\Lambda",
    }
    return pdg_latex_dict.get(pdg, str(pdg))


def get_color_from_pdg(pdg: int) -> str:
    pdg_color_dict = {
        11: TangoColors.scarlet_red,
        13: TangoColors.orange,
        15: TangoColors.butter,
        211: TangoColors.chameleon,
        321: TangoColors.sky_blue,
        2212: TangoColors.plum,
        411: TangoColors.chocolate,
        431: TangoColors.slate_dark,
        12: "black",
        14: "black",
        16: "black",
    }
    return pdg_color_dict.get(abs(pdg), "white")
