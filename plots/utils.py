import seaborn as sb
import matplotlib.pyplot as plt

from tueplots import figsizes, fonts, fontsizes


def setup_tueplots(
    nrows: int,
    ncols: int,
    rel_width: float = 1.0,
    hw_ratio: float | None = None,
    default_smaller: int = -2,
    use_tex: bool = True,
    tight_layout=False,
    constrained_layout=False,
    **kwargs
):
    if use_tex:
        font_config = fonts.iclr2024_tex(family="serif")
    else:
        font_config = fonts.iclr2024(family="serif")
    if hw_ratio is not None:
        kwargs["height_to_width_ratio"] = hw_ratio
    size = figsizes.iclr2024(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        tight_layout=tight_layout,
        constrained_layout=constrained_layout,
        **kwargs
    )
    fontsize_config = fontsizes.iclr2024(default_smaller=default_smaller)
    rc_params = {
        **font_config,
        **size,
        **fontsize_config,
    }
    rc_params.update({"text.latex.preamble": r"\usepackage{amsfonts}"})
    plt.rcParams.update(rc_params)
    plt.rcParams.update({
       "axes.prop_cycle": plt.cycler(
           color=["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                  "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]
       ),
       "patch.facecolor": "#0173B2"
    })
    sb.color_palette("colorblind")
