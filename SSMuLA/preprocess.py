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
hv.extension('bokeh')

from SSMuLA.landscape_global import ACTIVE_THRESH_DICT, LIB_INFO_DICT, append_active_cutoff
from SSMuLA.vis import save_bokeh_hv, plot_fit_dist
from SSMuLA.util import checkNgen_folder, get_file_name


# hv.renderer('bokeh').theme = JSON_THEME

class ProcessData:
    """
    A parent class to process the data
    """

    def __init__(self, input_csv: str) -> None:
        
        """
        Args:
        - input_csv, str: path to the input csv file
        """

        self._input_csv = input_csv
    
    @property
    def input_df(self) -> pd.DataFrame:
        """Return the input dataframe"""
        return pd.read_csv(self._input_csv)
    
    @property
    def output_csv(self) -> str:
        """Return the path to the output csv"""
        output_csv = self._input_csv.replace("fitness_landscape", "processed")
        # check if the folder exists
        checkNgen_folder(output_csv)
        return output_csv
    
    @property
    def lib_name(self) -> dict:
        """Return the library name"""
        return get_file_name(self._input_csv)

    @property
    def lib_info(self) -> dict:
        """Return the library information"""
        return LIB_INFO_DICT[self.lib_name]
    
    @property
    def parent_aa(self) -> str:
        """Return the parent amino acid"""
        return "".join(list(self.lib_info["AAs"].values()))
    
    @property
    def parent_codon(self) -> float:
        """Return the parent codon"""
        return "".join(list(self.lib_info["codons"].values()))


class ProcessDHFR(ProcessData):
    """
    Class to clean up the DHFR data
    """

    def __init__(self, input_csv: str = "data/DHFR/fitness_landscape/DHFR.csv") -> None:

        """
        Args:
        - input_csv, str: path to the input csv file
        """

        super().__init__(input_csv)

        self._input_csv = input_csv

        # append the active cutoffs
        self._df_avg_aa_append, self._avg_aa_active_cutoff = append_active_cutoff(
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
            * hv.Spikes([self.parent_codon_fitness], label = "Parent codon").opts(
                color="black", line_dash="dotted", line_width=1.6
            )
            * hv.Spikes([self.parent_aa_fitness], label = "Averaged parent AA").opts(
                color="black", line_dash="dashed", line_width=1.6
            )
            * hv.Spikes([self.avg_aa_active_cutoff], label = "Calculated active").opts(
                color="gray", line_width=1.6
            )
            * hv.Spikes([ACTIVE_THRESH_DICT[self.lib_name]], label = "Defined active").opts(
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
    def split_AA_cols(self) -> list:
        """Return the columns for the split amino acids"""
        return [f"AA{str(i)}" for i in self.lib_info["positions"].keys()]

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
    def parent_codon_fitness(self) -> float:
        """Return the parent codon fitness"""
        return self.input_df[self.input_df["seq"] == self.parent_codon]["fitness"].values[0]

    @property
    def parent_aa_fitness(self) -> float:
        """Return the parent aa fitness"""
        return self.df_avg_aa[self.df_avg_aa["AAs"] == self.parent_aa]["fitness"].values[0]


class ProcessGB1(ProcessData):
    """
    Class to clean up the GB1 data
    """

    def __init__(self, input_csv: str = "data/GB1/fitness_landscape/GB1.csv") -> None:
        """
        Args:
        - input_csv, str: path to the input csv file
        """

        super().__init__(input_csv)

        # append the active cutoffs
        self._df_active_append, _ = append_active_cutoff(
            self.df_aa, ["fitness"], ACTIVE_THRESH_DICT[self.lib_name]
        )

        # save the appended dataframe
        self._df_active_append.to_csv(self.output_csv, index=False)

        self._fit_dist = plot_fit_dist(self._df_active_append["fitness"], label="GB1") * hv.Spikes(
            [ACTIVE_THRESH_DICT[self.lib_name]], label="Active"
        ).opts(color="gray", line_width=1.6) * hv.Spikes(
            [self.parent_aa_fitness], label="Parent AA"
        ).opts(color="black", line_width=1.6)


        # Display the plot with the legend
        save_bokeh_hv(
            plot_obj=self._fit_dist,
            plot_name="GB1 fitness distribution",
            plot_path="results/fitness_distribution",
            bokehorhv="hv",
        )

    @property
    def df_aa(self) -> pd.DataFrame:

        """Return the input dataframe with amino acid translations
        and split into individual amino acids"""

        return self.input_df.copy().rename(columns={'Variants': 'AAs', "Fitness": "fitness"})[["AAs", "fitness"]].copy()
    
    @property
    def df_split_aa(self) -> pd.DataFrame:

        """Return the input dataframe with amino acid translations
        and split into individual amino acids"""

        df = self.df_aa.copy()
        
        # Split combo into individual amino acids
        df[self.split_AA_cols] = df["AAs"].apply(lambda x: pd.Series(list(x)))

        return df[["AAs", *self.split_AA_cols, "fitness"]].copy()
    
    @property
    def parent_aa_fitness(self) -> float:
        """Return the parent aa fitness"""
        return self.df_aa[self.df_aa["AAs"] == self.parent_aa]["fitness"].values[0]
    
    @property
    def df_active_append(self) -> pd.DataFrame:
        """Return the active appended dataframe"""
        return self._df_active_append



class ProcessTrpB(ProcessData):

    """
    Class to clean up the TrpB data. Similiar to what Kadina did but normalize to WT fitness to be 0
    """

    def __init__(self, input_csv: str = "data/TrpB/fitness_landscape/TrpB.csv") -> None:

        """
        Args:
        - input_csv, str: path to the input csv file
        """

        super().__init__(input_csv)

