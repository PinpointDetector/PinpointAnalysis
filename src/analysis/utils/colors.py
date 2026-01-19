from cycler import cycler


class TangoColors:
    """Tango colors"""

    sky_blue_light = "#729fcf"
    sky_blue = "#3465a4"
    sky_blue_dark = "#204a87"

    scarlet_red_light = "#ef2929"
    scarlet_red = "#cc0000"
    scarlet_red_dark = "#a40000"

    orange_light = "#fcaf3e"
    orange = "#f57900"
    orange_dark = "#ce5c00"

    aluminium_light = "#eeeeec"
    aluminium = "#d3d7cf"
    aluminium_dark = "#babdb6"

    butter_light = "#fce94f"
    butter = "#edd400"
    butter_dark = "#c4a000"

    chameleon_light = "#8ae234"
    chameleon = "#73d216"
    chameleon_dark = "#4e9a06"

    chocolate_light = "#e9b96e"
    chocolate = "#c17d11"
    chocolate_dark = "#8f5902"

    plum_light = "#ad7fa8"
    plum = "#75507b"
    plum_dark = "#5c3566"

    slate_light = "#888a85"
    slate = "#555753"
    slate_dark = "#2e3436"

    heather_gray = "#9aa297"

    white = "#FFFFFF"

    default_colors = [
        sky_blue,
        scarlet_red,
        orange,
        chameleon,
        plum,
        chocolate,
        butter,
        slate,
        aluminium,
    ]

    cycler = cycler(color=default_colors)
