from __future__ import annotations

import os

import pandas as pd

from cairosvg import svg2png

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import colorcet as cc

import bokeh
from bokeh.io import export_svg
from bokeh.models import NumeralTickFormatter
from bokeh.themes.theme import Theme

import holoviews as hv

from SSMuLA.landscape_global import LIB_NAMES
from SSMuLA.util import checkNgen_folder

bokeh.io.output_notebook()
hv.extension("bokeh", "matplotlib")

# Select a colorcet colormap, for example, 'fire' or 'CET_CBL1'
colormap = cc.cm["glasbey_category10"]

# Extract a list of hex codes from the colormap
glasbey_category10 = [mcolors.to_hex(colormap(i)) for i in range(colormap.N)]


JSON_THEME = Theme(
    json={
        "attrs": {
            "Title": {
                "align": "center",
                "text_font_size": "12px",
                "text_color": "black",
                "text_font": "arial",
            },  # title centered and bigger
            "Axis": {
                "axis_label_text_font_style": "normal",
                "axis_label_text_color": "black",
                "major_label_text_color": "black",
                "axis_label_text_font": "arial",
                "major_label_text_font": "arial",
            },  # no italic labels
            "Legend": {
                "title_text_font_style": "normal",
                "title_text_color": "black",
                "label_text_color": "black",
                "label_text_font": "arial",
            },
            "ColorBar": {
                "title_text_font_style": "normal",
                "major_label_text_color": "black",
                "major_label_text_font": "arial",
                "title_text_color": "black",
                "title_text_font": "arial",
            },
        }
    }
)

hv.renderer("bokeh").theme = JSON_THEME

# Grey for heatmap
HM_GREY = "#76777B"

# blue, orange, green, yellow, purple, gray
FZL_PALETTE = {
    "blue": "#4bacc6",
    "orange": "#f79646ff",
    "light_orange": "#ffbb78",
    "red": "#ff8888",
    "maroon": "#7A303F",
    "green": "#9bbb59",
    "yellow": "#f9be00",
    "purple": "#8064a2",
    "brown": "#ae682f",
    "dark_brown": "#6e4a2eff",
    "gray": "#666666",
    "light_gray": "#D3D3D3",
    "light_blue": "#849895",
    "light_green": "#9DAE88",
    "light_yellow": "#F1D384",
    "light_brown": "#C7B784",
    "black": "#000000",
}

GRAY_COLORS = {
    "gray-blue": "#749aa3",
    "gray-orange": "#c58a6c",
    "gray-green": "#8b946e",
    "gray-yellow": "#d6b969",
    "gray-purple": "#897a8f",
    "gray-brown": "#8b6e57",
}

LIGHT_COLORS = {"yellow": "#F1D384"}

PLOTEXTENTIONS = [".svg", ".png"]
PLOTTYPES = [t[1:] for t in PLOTEXTENTIONS]

LIB_COLORS = {
    n: c
    for (n, c) in zip(
        LIB_NAMES,
        [
            FZL_PALETTE["orange"],
            FZL_PALETTE["light_orange"],
            FZL_PALETTE["brown"],
            FZL_PALETTE["yellow"],
            FZL_PALETTE["maroon"],
            FZL_PALETTE["purple"],
        ]
        + sns.color_palette("crest", 9).as_hex()
        + [FZL_PALETTE["gray"]],
    )
}

LIB_COLORS_GLASBEY = {
    n: c
    for (n, c) in zip(
        LIB_NAMES,
        glasbey_category10[:6]
        + glasbey_category10[12:15]
        + glasbey_category10[6:10]
        + [glasbey_category10[15]]
        + glasbey_category10[10:12],
    )
}

LIB_COLORS_CODON = {"DHFR": "#ffbb78"}  # light orange

MLDE_COLORS = (
    [
        FZL_PALETTE["orange"],
        FZL_PALETTE["yellow"],
    ]
    + sns.color_palette("crest", 9).as_hex()
    + [FZL_PALETTE["gray"]]
)

SIMPLE_ZS_COLOR_MAP = {
    "none": FZL_PALETTE["gray"],
    "ed_score": FZL_PALETTE["blue"],
    "ev_score": FZL_PALETTE["green"],
    "esm_score": FZL_PALETTE["purple"],
    "esmif_score": FZL_PALETTE["yellow"],
    "coves_score": FZL_PALETTE["brown"],
    "Triad_score": FZL_PALETTE["orange"],
}

ZS_COLOR_MAP = {
    "none": FZL_PALETTE["gray"],
    "ev_score": FZL_PALETTE["green"],
    "esm_score": FZL_PALETTE["purple"],
    "esmif_score": FZL_PALETTE["yellow"],
    "coves_score": FZL_PALETTE["brown"],
    "Triad_score": FZL_PALETTE["orange"],
    "Triad-esmif_score": FZL_PALETTE["light_blue"],
    "ev-esm_score": FZL_PALETTE["light_green"],
    "ev-esm-esmif_score": FZL_PALETTE["light_yellow"],
    "Triad-ev-esm-esmif_score": FZL_PALETTE["light_brown"],
    "two-best-comb_score": FZL_PALETTE["light_gray"],
}


# define plot hooks
def one_decimal_x(plot, element):
    plot.handles["plot"].xaxis[0].formatter = NumeralTickFormatter(format="0.0")


def one_decimal_y(plot, element):
    plot.handles["plot"].yaxis[0].formatter = NumeralTickFormatter(format="0.0")


def fixmargins(plot, element):
    plot.handles["plot"].min_border_right = 30
    plot.handles["plot"].min_border_left = 65
    plot.handles["plot"].min_border_top = 20
    plot.handles["plot"].min_border_bottom = 65
    plot.handles["plot"].outline_line_color = "black"
    plot.handles["plot"].outline_line_alpha = 1
    plot.handles["plot"].outline_line_width = 1
    plot.handles["plot"].toolbar.autohide = True


def render_hv(hv_plot) -> bokeh.plotting.Figure:
    """Render a holoviews plot as a bokeh plot"""
    return hv.render(hv_plot)


def save_bokeh_hv(
    plot_obj,
    plot_name: str,
    plot_path: str,
    bokehorhv: str = "hv",
    dpi: int = 300,
    scale: int = 2,
    skippng: bool = False,
):

    """
    A function for exporting bokeh plots as svg

    Args:
    - plot_obj: hv or bokeh plot object
    - plot_name: str: name of the plot
    - plot_path: str: path to save the plot without the plot_name
    - bokehorhv: str: 'hv' or 'bokeh'
    - dpi: int: dpi
    - scale: int: scale
    - skippng: bool: skip png
    """

    plot_name = plot_name.replace(" ", "_")
    plot_path = checkNgen_folder(plot_path.replace(" ", "_"))
    plot_noext = os.path.join(plot_path, plot_name)

    if bokehorhv == "hv":

        # save as html legend
        hv.save(plot_obj, plot_noext + ".html")

        # hv.save(plot_obj, plot_noext, mode="auto", fmt='svg', dpi=300, toolbar='disable')

        plot_obj = hv.render(plot_obj, backend="bokeh")

    plot_obj.toolbar_location = None
    plot_obj.toolbar.logo = None

    plot_obj.output_backend = "svg"
    export_svg(plot_obj, filename=plot_noext + ".svg", timeout=1200)

    if not skippng:
        svg2png(
            write_to=plot_noext + ".png",
            dpi=dpi,
            scale=scale,
            bytestring=open(plot_noext + ".svg").read().encode("utf-8"),
        )
    else:
        print("Skipping png export")


def save_svg(fig, plot_title: str, path2folder: str, ifshow: bool = True):
    """
    A function for saving svg plots
    """

    plot_title_no_space = plot_title.replace(" ", "_")
    plt.savefig(
        os.path.join(checkNgen_folder(path2folder), f"{plot_title_no_space}.svg"),
        bbox_inches="tight",
        dpi=300,
        format="svg",
    )

    if ifshow:
        plt.show()


def save_plt(fig, plot_title: str, path2folder: str):

    """
    A helper function for saving plt plots
    Args:
    - fig: plt.figure: the figure to save
    - plot_title: str: the title of the plot
    - path2folder: str: the path to the folder to save the plot
    """

    for ext in PLOTEXTENTIONS:
        plot_title_no_space = plot_title.replace(" ", "_")
        plt.savefig(
            os.path.join(checkNgen_folder(path2folder), f"{plot_title_no_space}{ext}"),
            bbox_inches="tight",
            dpi=300,
        )

    plt.close()


def plot_fit_dist(
    fitness: pd.Series,
    label: str,
    color: str = "",
    spike_length: float | None = None,
    ignore_line_label: bool = False,
) -> hv.Distribution:
    """
    Plot the fitness distribution

    Args:
    - fitness: pd.Series: fitness values
    - label: str: label
    - color: str: color
    - ignore_line_label: bool: ignore line label

    Returns:
    - hv.Distribution: plot of the fitness distribution
    """

    if label == "codon":
        cap_label = f"{label.capitalize()}-level"
    elif label == "AA":
        cap_label = f"{label.upper()}-level"
    else:
        cap_label = label

    if color == "":
        color = FZL_PALETTE["blue"]

    if ignore_line_label:
        mean_label = {}
        median_label = {}
    else:
        mean_label = {"label": f"Mean {label}"}
        median_label = {"label": f"Median {label}"}

    hv_dist = hv.Distribution(fitness, label=cap_label).opts(
        width=400,
        height=400,
        color=color,
        line_color=None,
    )

    # get y_range for spike height
    y_range = (
        hv.renderer("bokeh").instance(mode="server").get_plot(hv_dist).state.y_range
    )

    # set spike length to be 5% of the y_range
    if spike_length is None:
        spike_length = (y_range.end - y_range.start) * 0.05

    return (
        hv_dist
        * hv.Spikes([fitness.mean()], **mean_label).opts(
            line_dash="dotted",
            line_color=color,
            line_width=1.6,
            spike_length=spike_length,
        )
        * hv.Spikes([fitness.median()], **median_label).opts(
            line_color=color, line_width=1.6, spike_length=spike_length
        )
    )


def plot_zs_violin(
    all_df: pd.DataFrame,
    zs: str,
    encoding_list: list[str],
    model: str,
    n_sample: int,
    n_top: int,
    metric: str,
    plot_name: str,
) -> hv.Violin:

    return hv.Violin(
        all_df[
            (all_df["zs"] == zs)
            & (all_df["encoding"].isin(encoding_list))
            & (all_df["model"] == model)
            & (all_df["n_sample"] == n_sample)
            & (all_df["n_top"] == n_top)
        ]
        .sort_values(["lib", "n_mut_cutoff"], ascending=[True, False])
        .copy(),
        kdims=["lib", "n_mut_cutoff"],
        vdims=[metric],
    ).opts(
        width=1200,
        height=400,
        violin_color="n_mut_cutoff",
        show_legend=True,
        legend_position="top",
        legend_offset=(0, 5),
        title=plot_name,
        ylim=(0, 1),
        hooks=[one_decimal_x, one_decimal_y, fixmargins, lib_ncut_hook],
    )


def lib_ncut_hook(plot, element):

    plot.handles["plot"].x_range.factors = [
        (lib, n_mut) for lib in LIB_NAMES for n_mut in ["single", "double", "all"]
    ]
    plot.handles["xaxis"].major_label_text_font_size = "0pt"
    # plot.handles['xaxis'].group_text_font_size = '0pt'
    # plot.handles['yaxis'].axis_label_text_font_size = '10pt'
    # plot.handles['yaxis'].axis_label_text_font_style = 'normal'
    # plot.handles['xaxis'].axis_label_text_font_style = 'normal'


def generate_related_color(reference_idx, base_idx, target_idx, palette="colorblind"):
    """
    Generate a color that has the same relationship to palette[target_idx]
    as palette[base_idx] has to palette[reference_idx].
    
    Parameters:
    - reference_idx: Index of the reference color in the palette.
    - base_idx: Index of the color that is related to reference_idx.
    - target_idx: Index of the color to which the transformation is applied.
    - palette: Name of the Seaborn palette (default: "colorblind").
    
    Returns:
    - A tuple representing the new RGB color.
    """
    import seaborn as sns
    import numpy as np

    # Get the palette
    colors = sns.color_palette(palette)

    # Compute transformation
    color_shift = np.array(colors[base_idx]) - np.array(colors[reference_idx])

    # Apply the transformation
    new_color = np.array(colors[target_idx]) + color_shift

    # Clip to valid RGB range [0,1]
    new_color = np.clip(new_color, 0, 1)

    return tuple(new_color)