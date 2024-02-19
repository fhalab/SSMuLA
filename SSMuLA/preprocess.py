"""
Contains classes for dataprocessing in this work
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from Bio.Seq import Seq

import holoviews as hv

from SSMuLA.landscape_global import ACTIVE_THRESH_DICT, LIB_INFO_DICT, calc_active_cutoff
from SSMuLA.vis import save_bokeh_hv, plot_fit_dist
from SSMuLA.util import checkNgen_folder

# Import custom modules
from SSMuLA.util import checkNgen_folder

hv.extension('bokeh')
# hv.renderer('bokeh').theme = JSON_THEME


class ProcessDHFR:
    """
    Class to clean up the DHFR data
    """

    def __init__(self, input_csv: str = "data/DHFR/fitness_landscape/DHFR.csv") -> None:

        """
        Args:
        - input_csv, str: path to the input csv file
        """

        self._input_csv = input_csv

        # append the active cutoffs
        self._df_avg_aa_append, self._avg_aa_active_cutoff = calc_active_cutoff(
            self.df_avg_aa, ["fitness"]
        )

        # save the appended dataframe
        self._df_avg_aa_append.to_csv(self.output_csv, index=False)

        self._overlay_fit_dist()

    def _overlay_fit_dist(
        self,
        title: str = "DHFR fitness distribution",
    ) -> hv.Distribution:
        
        """
        Plot the fitness distribution

        Args:
        - dist1, hv.Distribution: fitness distribution 1
        - dist2, hv.Distribution: fitness distribution 2
        - active_cut_calc, float: calculated active cutoff
        - active_cut_def, float: defined active cutoff

        Returns:
        - hv.Distribution: plot of the fitness distribution
        """

        # Overlay the two plots
        overlay_dist = (
            self.codon_fit_dist
            * self.avg_aa_fit_dist
            * hv.Spikes([self.avg_aa_active_cutoff], label = "Calculated active cutoff").opts(
                color="gray", line_width=1.6
            )
            * hv.Spikes([ACTIVE_THRESH_DICT["DHFR"]], label = "Defined active cutoff").opts(
                color="gray", line_dash="dashed", line_width=1.6
            )
        )

        # Customize the plot options
        overlay_dist.opts(
            legend_position="top_right",
            title=title,
            xlabel="Fitness",
        )

        # Display the plot with the legend
        save_bokeh_hv(
            overlay_dist,
            plot_name=title,
            plot_path="results/fitness_distribution",
            bokehorhv="hv",
        )
        return overlay_dist

    @property
    def lib_info(self) -> dict:
        """Return the library information"""
        return LIB_INFO_DICT["DHFR"]

    @property
    def split_AA_cols(self) -> list:
        """Return the columns for the split amino acids"""
        return [f"AA{str(i)}" for i in self.lib_info["positions"].keys()]

    @property
    def input_df(self) -> pd.DataFrame:
        """Return the input dataframe"""
        return pd.read_csv(self._input_csv)

    @property
    def df_aa(self) -> pd.DataFrame:

        """Return the input dataframe with amino acid translations"""

        df = self.input_df.copy()

        # Translate the sequence to amino acids
        df["AAs"] = df["seq"].apply(lambda x: "".join(Seq(x).translate()))

        return df[["AAs", "seq", "fitness"]].copy()

    @property
    def df_split_aa(self) -> pd.DataFrame:

        """Return the input dataframe with amino acid translations
        and split into individual amino acids"""

        df = self.df_aa.copy()

        # Split combo into individual amino acids
        df[self.split_AA_cols] = df["AAs"].apply(lambda x: pd.Series(list(x)))

        return df[["AAs", *self.split_AA_cols, "seq", "fitness"]].copy()

    @property
    def df_avg_aa(self) -> pd.DataFrame:

        """Return the average fitness of each amino acid"""

        df = self.df_aa.copy()[["AAs", "fitness"]]
        # Group by amino acid and take the average fitness
        df = df.groupby("AAs")["fitness"].mean().reset_index()
        # Split combo into individual amino acids
        df[self.split_AA_cols] = df["AAs"].apply(lambda x: pd.Series(list(x)))
        return df[["AAs", *self.split_AA_cols, "fitness"]].copy()

    @property
    def df_avg_aa_append(self) -> pd.DataFrame:
        """Return the average fitness of each amino acid with the active cutoff appended"""
        return self._df_avg_aa_append

    @property
    def avg_aa_active_cutoff(self) -> float:
        """Return the active cutoff for the average amino acid"""
        return self._avg_aa_active_cutoff

    @property
    def codon_fit(self) -> pd.Series:
        """Return the fitness of based on codon as a series"""
        return self.input_df["fitness"]

    @property
    def avg_aa_fit(self) -> pd.Series:
        """Return the fitness of based on codon as a series"""
        return self.df_avg_aa["fitness"]

    @property
    def codon_fit_dist(self) -> hv.Distribution:
        """Return the fitness distribution based on codon"""
        return plot_fit_dist(self.codon_fit, "codon")

    @property
    def avg_aa_fit_dist(self) -> hv.Distribution:
        """Return the fitness distribution based on average amino acid"""
        return plot_fit_dist(self.avg_aa_fit, "AA")
    
    @property
    def output_csv(self) -> str:
        """Return the path to the output csv"""
        output_csv = self._input_csv.replace("fitness_landscape", "processed")
        # check if the folder exists
        checkNgen_folder(output_csv)
        return output_csv