from __future__ import annotations

import os

import pandas as pd

from cairosvg import svg2png

import seaborn as sns

import matplotlib.pyplot as plt

import bokeh
from bokeh.io import export_svg
from bokeh.models import NumeralTickFormatter
from bokeh.themes.theme import Theme

bokeh.io.output_notebook()

import holoviews as hv

hv.extension("bokeh", "matplotlib")

from SSMuLA.landscape_global import LIB_NAMES
from SSMuLA.util import checkNgen_folder

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
PRESENTATION_PALETTE_SATURATE = {
    "blue": "#4bacc6",
    "orange": "#f79646ff",
    "light_orange": "#00faf4",
    "red": "#ff8888",
    "green": "#9bbb59",
    "yellow": "#f9be00",
    "purple": "#8064a2",
    "brown": "#6e4a2eff",
    "gray": "#666666",
}

LIGHT_COLORS = {"yellow": "#F1D384"}

PLOTEXTENTIONS = [".svg", ".png"]
PLOTTYPES = [t[1:] for t in PLOTEXTENTIONS]

LIB_COLORS = {
    n: c
    for (n, c) in zip(
        LIB_NAMES,
        [
            PRESENTATION_PALETTE_SATURATE["orange"],
            PRESENTATION_PALETTE_SATURATE["light_orange"],
            PRESENTATION_PALETTE_SATURATE["brown"],
            PRESENTATION_PALETTE_SATURATE["yellow"],
        ]
        + sns.color_palette("crest", 9).as_hex()
        + [PRESENTATION_PALETTE_SATURATE["gray"]],
    )
}

LIB_COLORS_CODON = {"DHFR": "#00faf4"} # light orange

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
    ignore_line_label: bool = False
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
        color = PRESENTATION_PALETTE_SATURATE["blue"]

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