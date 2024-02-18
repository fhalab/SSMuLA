from __future__ import annotations

import os

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
PRESENTATION_PALETTE_SATURATE6 = [
    "#4bacc6",
    "#f79646ff",
    "#9bbb59",
    "#f9be00",
    "#8064a2",
    "#666666",
]

# blue, orange, green, yellow, gray
PRESENTATION_PALETTE_SATURATE5 = [
    "#4bacc6",
    "#f79646ff",
    "#9bbb59",
    "#f9be00",
    "#666666",
]

PLOTEXTENTIONS = [".svg", ".png"]
PLOTTYPES = [t[1:] for t in PLOTEXTENTIONS]


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
):

    """A function for exporting bokeh plots as svg"""

    plot_name = plot_name.replace(" ", "_")
    plot_path = checkNgen_folder(plot_path.replace(" ", "_"))
    plot_noext = os.path.join(plot_path, plot_name)

    if bokehorhv == "hv":

        # save as html legend
        hv.save(plot_obj, plot_noext + ".html")

        # hv.save(plot_obj, plot_noext + ".png", fmt="png", dpi=300)
        hv.save(plot_obj, plot_noext + ".png", backend='bokeh', fmt="png", dpi=300)

        # hv.renderer('matplotlib').instance(dpi=300).save(plot_obj, plot_noext)

        plot_obj = hv.renderer("bokeh").instance(dpi=300).get_plot(plot_obj).state
    
    plot_obj.toolbar_location = None
    plot_obj.toolbar.logo = None

   # export_png(plot_obj, filename=plot_noext + ".png", timeout=1200, height=height, width=width)
    
    plot_obj.output_backend = "svg"
    export_svg(plot_obj, filename=plot_noext + ".svg", timeout=1200)



    
