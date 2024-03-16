"""
A script to visualize the results cross landscapes.
"""

import os
import numpy as np
import pandas as pd

import holoviews as hv
from holoviews import dim


hv.extension("bokeh")

from SSMuLA.landscape_global import LIB_NAMES, TrpB_names
from SSMuLA.vis import (
    save_bokeh_hv,
    JSON_THEME,
    LIB_COLORS,
    one_decimal_x,
    one_decimal_y,
    fixmargins,
)
from SSMuLA.util import get_file_name, checkNgen_folder

hv.renderer("bokeh").theme = JSON_THEME


de_metric_map = {
    "mean_all": "all simulations fitness mean",
    "median_all": "all simulations fitness median",
    "mean_top96": "top 96 simulations fitness mean",
    "median_top96":	"top 96 simulations fitness median",
    "mean_top384": "top 384 simulations fitness mean",
    "median_top384": "top 384 simulations fitness median",
    "fraction_max": "fraction reached max fitness",
}


def de_sum_hook(plot,element):
    plot.handles['plot'].x_range.factors = [(lib, de) for lib in LIB_NAMES for de in ["single_step_DE", "recomb_SSM", "top96_SSM"]]

class DESumVis:
    """A class to visualize the summary of DE simulations."""
    def __init__(self, 
                 input_csv = "results/simulations/DE-active/scale2max/all_landscape_de_summary.csv",
                 ) -> None:
        
        self._input_csv = input_csv

        for metric, metric_dets in de_metric_map.items():

            title = f"DE from active variant {metric_dets}"

            self._plot_de_sum(metric, metric_dets, title)


    def _plot_de_sum(self, metric: str, metric_dets: str, title: str):
            
            """A method to plot the summary of DE simulations."""

            save_bokeh_hv(
                hv.Bars(self.df, kdims=["lib", "de_type"], vdims=metric).opts(
                width=1200,
                height=400,
                show_legend=True,
                legend_position="top",
                legend_offset=(0, 5),
                ylabel=metric_dets.capitalize(),
                multi_level=False,
                title=title,
                xlabel="Library",
                hooks=[fixmargins, one_decimal_y, de_sum_hook],
            ),
                plot_name=title,
                plot_path=self.output_folder,
            )

    @property
    def df(self):

        df = pd.read_csv(self._input_csv)
        # clean up the nan if not full N
        # Fill NaNs in 'mean_top96' and 'median_top96' from 'mean_all' and 'median_all'
        df['mean_top96'] = df['mean_top96'].fillna(df['mean_all'])
        df['median_top96'] = df['median_top96'].fillna(df['median_all'])

        # Fill NaNs in 'mean_top384' and 'median_top384' from 'mean_top96' and 'median_top96'
        df['mean_top384'] = df['mean_top384'].fillna(df['mean_top96'])
        df['median_top384'] = df['median_top384'].fillna(df['median_top96'])

        return self.df
    
    @property
    def output_folder(self):
        return checkNgen_folder(os.path.join(os.path.dirname(self._input_csv), "summary"))
