"""A script for plotting de simluation results"""

# General imports
from glob import glob

import os
import re
import pickle
import datetime

# Data manipulation
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import bokeh
from bokeh.io import export_svg
from bokeh.models import NumeralTickFormatter
from bokeh.themes.theme import Theme


from bokeh.io import output_notebook

output_notebook()

import holoviews as hv
from holoviews import opts, dim

import panel as pn

pn.config.comms = "vscode"

# Large data plotting
import datashader as ds
from holoviews.operation.datashader import datashade, rasterize

# Making graphs
import networkx as nx
import itertools
from tqdm.auto import tqdm
from multiprocessing import Pool

hv.extension("bokeh")

from SSMuLA import growth_analysis as ga
from SSMuLA.landscape_global import LIB_NAMES, TrpB_names
from SSMuLA.vis import save_bokeh_hv, JSON_THEME, LIB_COLORS
from SSMuLA.util import get_file_name, checkNgen_folder

hv.renderer("bokeh").theme = JSON_THEME

# order of de simluation from simple to complex
SIM_ORDER = ["SSM_recomb", "single_step_DE", "SSM_top96"]

sim_line_styles = {
    "SSM_recomb": "solid",
    "single_step_DE": "dashed",
    "SSM_top96": "dotted",
}

# Plot Hooks
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


def de_violin(
    slice_df: pd.DataFrame,
    lib_name: str,
    plot_name: str,
    plot_folder: str,
    v_width: int = 400,
):

    """
    A function to plot a violin plot of the DE simulation results

    Args:
    - slice_df (pd.DataFrame): A dataframe containing the DE simulation results
    - lib_name (str): The name of the library to plot
    - plot_name (str): The name of the plot
    - plot_folder (str): The folder to save the plot
    - v_width (int): Width of the violin plot
    """

    print(f"Plotting DE max fitness achieved violin...")

    violin = hv.Violin(
        slice_df,
        kdims=["simulation", "lib"],
        vdims=["final_fitness"],
    ).opts(
        violin_color=dim("lib").str(),
        cmap=[LIB_COLORS[lib_name]],
        width=v_width,
        height=300,
        inner=None,
        violin_width=0.8,
        title=plot_name,
        hooks=[fixmargins, one_decimal_y],
        ylabel="Max fitness achieved",
    )

    save_bokeh_hv(
        violin,
        plot_name=plot_name,
        plot_path=checkNgen_folder(os.path.join(plot_folder, "violin")),
        bokehorhv="hv",
        dpi=300,
        scale=2,
    )

    return violin


def de_ecdf(slice_df: pd.DataFrame, lib_name: str, plot_name: str, plot_folder: str):

    """ 
    A function to plot an ECDF of the DE simulation results
    
    Args:
    - slice_df (pd.DataFrame): A dataframe containing the DE simulation results
    - lib_name (str): The name of the library to plot
    - plot_name (str): The name of the plot
    - plot_folder (str): The folder to save the plot
    """

    print(f"Plotting DE max fitness achieved ECDF...")


    # Initialize an empty HoloViews Overlay container
    overlay = hv.Overlay()

    # Initialize a dictionary to store legend labels
    legend_labels = {}

    # Iterate over simulation types and libraries, create individual traces, and overlay them
    for sim, style in sim_line_styles.items():
        # for lib_name, lib_color in LIB_COLORS.items():
        selection = slice_df[
            (slice_df["simulation"] == sim) & (slice_df["lib"] == lib_name)
        ]
        if not selection.empty:
            curve = hv.Curve(
                selection.sort_values(["simulation", "lib", "final_fitness"]),
                kdims="final_fitness",
                vdims=["final_fitness ECDF", "lib", "simulation"],
            )
            overlay *= curve.opts(
                line_dash=style,
                color=LIB_COLORS[lib_name],
                width=500,
                height=300,
                title=plot_name,
                hooks=[fixmargins, one_decimal_x, one_decimal_y],
                xlabel="Max fitness achieved",
                ylabel="ECDF",
            )
            legend_labels[sim] = hv.Curve([0], label=sim).opts(
                line_dash=style, color=LIB_COLORS[lib_name]
            )

    layout = overlay.opts(
        title=f"{lib_name} max fitness achieved no imputed",
        hooks=[fixmargins, one_decimal_x, one_decimal_y],
        xlabel="Max fitness achieved",
        ylabel="ECDF",
    )

    # Add the legend to the layout
    save_bokeh_hv(
        layout
        * hv.NdOverlay(legend_labels).opts(
            legend_position="right", legend_offset=(10, 100), legend_limit=50
        ),
        plot_name=plot_name,
        plot_path=checkNgen_folder(os.path.join(plot_folder, "ecdf")),
        bokehorhv="hv",
        dpi=300,
        scale=2,
    )


class VisDESims:

    """
    Class for visualizing DE simulation results
    """

    def __init__(
        self,
        lib_name: str,
        append_title: str = "max fitness achieved",
        v_width: int = 400,
        sim_folder: str = "results/simulations",
        de_sub_folder: str = "DE-no_stop_codon",
        fit_scale_sub_folder: str = "scale2max",
        vis_folder: str = "results/simulations_vis",
    ) -> None:

        """ """

        self._lib_name = lib_name
        self._append_title = append_title
        self._v_width = v_width
        self._sim_folder = os.path.normpath(sim_folder)
        self._de_sub_folder = de_sub_folder
        self._fit_scale_sub_folder = fit_scale_sub_folder
        self._vis_folder = checkNgen_folder(os.path.normpath(vis_folder))

        print("Visualizing DE simulation results for {} {} {}...".format(lib_name, de_sub_folder, vis_folder))
        self._plot_violin_ecdf()

    def _plot_violin_ecdf(self):

        """
        Plot a violin plot and ecdf of the DE simulation results

        Args:
        - lib_name (str): The name of the library to plot
        - append_title (str): Additional title to append to the plot title
        - v_width (int): Width of the violin plot
        """

        slice_df = (
            self.all_df[self.all_df["lib"] == self._lib_name]
            .sort_values(["simulation", "lib", "final_fitness"])
            .copy()
        )

        plot_name = f"{self._lib_name} {self._append_title}"
        plot_folder = checkNgen_folder(
            os.path.join(
                self._vis_folder, self._de_sub_folder, self._fit_scale_sub_folder
            )
        )

        self._violin = de_violin(
            slice_df=slice_df,
            lib_name=self._lib_name,
            plot_name=plot_name,
            plot_folder=plot_folder,
            v_width=self._v_width,
        )

        self._ecdf = de_ecdf(
            slice_df=slice_df,
            lib_name=self._lib_name,
            plot_name=plot_name,
            plot_folder=plot_folder,
        )

    @property
    def all_df(self) -> pd.DataFrame:

        """
        Combine all DE simulation results into a single dataframe

        Args:
        - de_folder_full_path (str): Path to the DE simulation results

        Returns:
        - all_df (pd.DataFrame): A dataframe containing all DE simulation results
        """

        all_de_sim_files = glob(f"{os.path.normpath(self.de_folder_full_path)}/*.csv")

        dfs = []

        for res in all_de_sim_files:
            lib_name, sim_name = get_file_name(res).split("-")

            df = pd.read_csv(res)
            df["simulation"] = sim_name
            df["lib"] = lib_name
            dfs.append(df)

        all_df = pd.concat(dfs).reset_index(drop=True)
        all_df["final_fitness"] = all_df["final_fitness"].astype(float)

        # sort sim to be in custom order from easy to hard
        all_df["simulation"] = pd.Categorical(
            all_df["simulation"], categories=SIM_ORDER, ordered=True
        )

        return all_df.sort_values("simulation")

    @property
    def de_folder_full_path(self) -> str:
        """Return the path to the DE simulation results"""
        return os.path.join(
            self._sim_folder, self._de_sub_folder, self._fit_scale_sub_folder
        )

    @property
    def de_violin(self) -> hv.Violin:
        """Return the violin plot"""
        return self._violin

    @property
    def de_ecdf(self) -> hv.Curve:
        """Return the ecdf plot"""
        return self._ecdf


def run_plot_de(
    de_opts: list = ["DE-active", "DE-no_stop_codons", "DE-all"],
    sim_folder: str = "results/simulations", 
    vis_folder: str = "results/simulations_vis",
    v_width: int = 400,
):

    """Run the DE simulation plotting"""

    for de_sub_folder in de_opts:
        for fit_scale_sub_folder in ["scale2parent", "scale2max"]:
            for lib in LIB_NAMES:
                # if "TrpB" in lib:
                #     v_with = 1280
                # else:
                #     v_with = 400

                vis = VisDESims(
                    lib_name=lib,
                    append_title="max fitness achieved",
                    v_width=v_width,
                    sim_folder=sim_folder,
                    de_sub_folder=de_sub_folder,
                    fit_scale_sub_folder=fit_scale_sub_folder,
                    vis_folder=vis_folder,
                )