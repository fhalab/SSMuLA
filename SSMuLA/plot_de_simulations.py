"""A script for plotting de simluation results"""

# General imports
from glob import glob

import os
from copy import deepcopy

import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


# Data manipulation
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

# order of de simluation from simple to complex
SIM_ORDER = deepcopy(["recomb_SSM", "single_step_DE", "top96_SSM"])

sim_line_styles = deepcopy(
    {
        "recomb_SSM": "solid",
        "single_step_DE": "dashed",
        "top96_SSM": "dotted",
    }
)


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

    if lib_name == "TrpB":
        v_width = 1280
        cmap = [LIB_COLORS[lib_name] for lib_name in TrpB_names]
        xrotation = 45
    else:
        v_width = 400
        cmap = [LIB_COLORS[lib_name]]
        xrotation = 0

    violin = hv.Violin(
        slice_df,
        kdims=["simulation", "lib"],
        vdims=["final_fitness"],
    ).opts(
        violin_color=dim("lib").str(),
        cmap=cmap,
        width=v_width,
        height=300,
        violin_width=0.8,
        title=plot_name,
        hooks=[fixmargins, one_decimal_y],
        ylabel="Max fitness achieved",
        xrotation=xrotation,
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

    if lib_name == "TrpB":
        # Initialize a dictionary to store legend labels
        sim_legend_labels = {}
        lib_legend_labels = {}

        # Iterate over simulation types and libraries, create individual traces, and overlay them
        for sim, style in sim_line_styles.items():
            for lib_name in TrpB_names:
                selection = slice_df[
                    (slice_df["simulation"] == sim) & (slice_df["lib"] == lib_name)
                ]
                if not selection.empty:
                    curve = hv.Curve(
                        selection.sort_values(["simulation", "lib", "final_fitness", "final_fitness ECDF"]),
                        kdims="final_fitness",
                        vdims=["final_fitness ECDF", "lib", "simulation"],
                    )
                    overlay *= curve.opts(
                        line_dash=style,
                        color=LIB_COLORS[lib_name],
                        width=1200,
                        height=800,
                        title=plot_name,
                        hooks=[fixmargins, one_decimal_x, one_decimal_y],
                        xlabel="Max fitness achieved",
                        ylabel="ECDF",
                    )

                sim_legend_labels[sim] = hv.Curve([0], label=sim).opts(
                    line_dash=style, color="gray"
                )
                lib_legend_labels[lib_name] = hv.Curve([0], label=lib_name).opts(
                    line_dash="solid", color=LIB_COLORS[lib_name]
                )

        legend_labels = {**deepcopy(sim_legend_labels), **deepcopy(lib_legend_labels)}
        print(legend_labels)

        layout = overlay.opts(
            title=plot_name,
            hooks=[fixmargins, one_decimal_x, one_decimal_y],
            xlabel="Max fitness achieved",
            ylabel="ECDF",
        )
    else:
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
                    selection.sort_values(["simulation", "lib", "final_fitness", "final_fitness ECDF"]),
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
        skippng=(lib_name == "GB1" or lib_name == "TrpB"),
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

        """
        Args:
        - lib_name (str): The name of the library to plot
        - append_title (str): Additional title to append to the plot title
        - v_width (int): Width of the violin plot
        - sim_folder (str): Path to the DE simulation results
        - de_sub_folder (str): Subfolder of the DE simulation results
        - fit_scale_sub_folder (str): Subfolder of the DE simulation results
        - vis_folder (str): Path to save the DE simulation plots
        """

        self._lib_name = lib_name
        self._append_title = append_title
        self._v_width = v_width
        self._sim_folder = os.path.normpath(sim_folder)
        self._de_sub_folder = de_sub_folder
        self._fit_scale_sub_folder = fit_scale_sub_folder
        self._vis_folder = checkNgen_folder(os.path.normpath(vis_folder))

        print(
            "Visualizing DE simulation results for {} {} {}...".format(
                lib_name, de_sub_folder, vis_folder
            )
        )
        self._plot_violin_ecdf()

    def _plot_violin_ecdf(self):

        """
        Plot a violin plot and ecdf of the DE simulation results

        Args:
        - lib_name (str): The name of the library to plot
        - append_title (str): Additional title to append to the plot title
        - v_width (int): Width of the violin plot
        """

        if self._lib_name == "TrpB":
            # all trpb
            slice_df = (
                self.all_df[self.all_df["lib"].isin(TrpB_names)]
                .sort_values(["simulation", "lib", "final_fitness"])
                .copy()
            )
        else:
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

        all_de_sim_files = sorted(glob(f"{os.path.normpath(self.de_folder_full_path)}/*.csv"))

        all_df = pd.DataFrame()

        for res in all_de_sim_files:
            # do not include the summary file
            if "all_landscape" not in res:
                lib_name, sim_name = get_file_name(res).split("-")

                df = pd.read_csv(res)
                df["simulation"] = sim_name
                df["lib"] = lib_name
                all_df = all_df._append(df)

        all_df = all_df.reset_index(drop=True)
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
    scale_types: list = ["scale2max", "scale2parent"],
    de_opts: list = ["DE-active", "DE-all"],
    sim_folder: str = "results/simulations",
    vis_folder: str = "results/simulations_vis",
    v_width: int = 400,
):

    """
    Run the DE simulation plotting

    Args:
    - scale_types (list): The scale types of fitness
    - de_opts (list): The DE options to plot
    - sim_folder (str): Path to the DE simulation results
    - vis_folder (str): Path to save the DE simulation plots
    - v_width (int): Width of the violin plot
    """

    for de_sub_folder in de_opts:
        for fit_scale_sub_folder in scale_types:
            for lib in LIB_NAMES + ["TrpB"]:
                if "TrpB" in lib:
                    v_width = 1280
                else:
                    v_width = 400

                vis = VisDESims(
                    lib_name=lib,
                    append_title="max fitness achieved",
                    v_width=v_width,
                    sim_folder=sim_folder,
                    de_sub_folder=de_sub_folder,
                    fit_scale_sub_folder=fit_scale_sub_folder,
                    vis_folder=vis_folder,
                )