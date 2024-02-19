from __future__ import annotations

import os

import pandas as pd

from cairosvg import svg2png

import bokeh
from bokeh.io import show, export_svg, export_png
from bokeh.plotting import show
from bokeh.themes.theme import Theme

bokeh.io.output_notebook()

import holoviews as hv
from holoviews import opts

hv.extension("bokeh", "matplotlib")

from SSMuLA.util import checkNgen_folder

theme = Theme(
    json={
        "attrs": {
            "Title": {
                "align": "center",
                "text_font_size": "15px",
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

hv.renderer("bokeh").theme = theme

# Grey for heatmap
HM_GREY = "#76777B"

# blue, orange, green, yellow, purple, gray
PRESENTATION_PALETTE_SATURATE = {
    "blue": "#4bacc6",
    "orange": "#f79646ff",
    "green": "#9bbb59",
    "yellow": "#f9be00",
    "purple": "#8064a2",
    "gray": "#666666",
}

PLOTEXTENTIONS = [".svg", ".png"]
PLOTTYPES = [t[1:] for t in PLOTEXTENTIONS]

CODON_AA_COLOER_DICT = {"codon": PRESENTATION_PALETTE_SATURATE["blue"], 
                        "AA": PRESENTATION_PALETTE_SATURATE["orange"]}


def render_hv(hv_plot) -> bokeh.plotting.Figure:
    """Render a holoviews plot as a bokeh plot"""
    return hv.render(hv_plot)


def save_bokeh_hv(
    plot_obj,
    plot_name: str,
    plot_path: str,
    bokehorhv: str = "hv",
    height: int = 400,
    width: int = 400,
    dpi: int = 300,
    scale: int = 2,
):

    """A function for exporting bokeh plots as svg"""

    plot_name = plot_name.replace(" ", "_")
    plot_path = checkNgen_folder(plot_path.replace(" ", "_"))
    plot_noext = os.path.join(plot_path, plot_name)

    if bokehorhv == "hv":

        # save as html legend
        hv.save(plot_obj, plot_noext + ".html")

        plot_obj = hv.renderer("bokeh").instance(dpi=300).get_plot(plot_obj).state
    
    plot_obj.toolbar_location = None
    plot_obj.toolbar.logo = None

    plot_obj.output_backend = "svg"
    export_svg(plot_obj, filename=plot_noext + ".svg", timeout=1200)

    svg2png(
            # url=plotpath,
            write_to=plot_noext + ".png",
            # output_height=height*2,
            # output_width=width*2,
            dpi=dpi,
            scale=scale,
            bytestring=open(plot_noext + ".svg").read().encode("utf-8"),
            
        )
    
def plot_fit_dist(fitness: pd.Series, codon_aa: str) -> hv.Distribution:
        """
        Plot the fitness distribution

        Args:
        - fitness: pd.Series: fitness values
        - codon_aa: str: codon or amino acid
        
        Returns:
        - hv.Distribution: plot of the fitness distribution
        """

        if codon_aa == "codon":
            cap_codon_aa = codon_aa.capitalize()
        else:
            cap_codon_aa = codon_aa.upper()    
        
        return (
            hv.Distribution(fitness, label=f"{cap_codon_aa}-level").opts(
                width=400,
                height=400,
                color=CODON_AA_COLOER_DICT[codon_aa],
                line_color=None,
            )
            * hv.Spikes([fitness.mean()], label=f"Mean {codon_aa} fitness").opts(
                line_dash="dotted", line_color=CODON_AA_COLOER_DICT[codon_aa], line_width=1.6
            )
            * hv.Spikes([fitness.median()], label=f"Median {codon_aa} fitness").opts(
                line_color=CODON_AA_COLOER_DICT[codon_aa], line_width=1.6
            )
        )