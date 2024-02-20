"""
Contains classes for dataprocessing in this work
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

from functools import reduce

import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from Bio.Seq import Seq

import seaborn as sns
import holoviews as hv

hv.extension("bokeh")

from SSMuLA.landscape_global import (
    ACTIVE_THRESH_DICT,
    LIB_INFO_DICT,
    append_active_cutoff,
)
from SSMuLA.vis import save_bokeh_hv, plot_fit_dist, PRESENTATION_PALETTE_SATURATE
from SSMuLA.util import checkNgen_folder, get_file_name


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

        self._ks, self._ks_p = ks_2samp(self.codon_fit, self.avg_aa_fit)
        print(f"Kolmogorov-Smirnov Statistic: {self._ks}")
        print(f"P-value: {self._ks_p}")

        self._overlay_fit_dist()

    def _overlay_fit_dist(self) -> hv.Distribution:

        """
        Plot the fitness distribution

        Returns:
        - hv.Distribution: plot of the fitness distribution
        """

        # Overlay the two plots
        overlay_dist = (
            self.codon_fit_dist
            * self.avg_aa_fit_dist
            * hv.Spikes([self.parent_codon_fitness], label="Parent codon").opts(
                color="black", line_dash="dotted", line_width=1.6
            )
            * hv.Spikes([self.parent_aa_fitness], label="Averaged parent AA").opts(
                color="black", line_dash="dashed", line_width=1.6
            )
            * hv.Spikes([self.avg_aa_active_cutoff], label="Calculated active").opts(
                color="gray", line_width=1.6
            )
            * hv.Spikes(
                [ACTIVE_THRESH_DICT[self.lib_name]], label="Defined active"
            ).opts(color="gray", line_dash="dashed", line_width=1.6)
        )

        # Customize the plot options
        overlay_dist.opts(
            legend_position="top_right",
            title=f"{self.lib_name} fitness distribution",
            xlabel="Fitness",
        )

        # Display the plot with the legend
        save_bokeh_hv(
            overlay_dist,
            plot_name=f"{self.lib_name} fitness distribution",
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
        return self.input_df[self.input_df["seq"] == self.parent_codon][
            "fitness"
        ].values[0]

    @property
    def parent_aa_fitness(self) -> float:
        """Return the parent aa fitness"""
        return self.df_avg_aa[self.df_avg_aa["AAs"] == self.parent_aa][
            "fitness"
        ].values[0]


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

        self._fit_dist = (
            plot_fit_dist(self._df_active_append["fitness"], label="GB1")
            * hv.Spikes([ACTIVE_THRESH_DICT[self.lib_name]], label="Active").opts(
                color="gray", line_width=1.6
            )
            * hv.Spikes([self.parent_aa_fitness], label="Parent AA").opts(
                color="black", line_width=1.6
            )
        ).opts(
            legend_position="top_right",
            title=f"{self.lib_name} fitness distribution",
            xlabel="Fitness"
        )

        # Save the plot with the legend
        save_bokeh_hv(
            plot_obj=self._fit_dist,
            plot_name=f"{self.lib_name} fitness distribution",
            plot_path="results/fitness_distribution",
            bokehorhv="hv",
        )

    @property
    def df_aa(self) -> pd.DataFrame:

        """Return the input dataframe with amino acid translations
        and split into individual amino acids"""

        return (
            self.input_df.copy()
            .rename(columns={"Variants": "AAs", "Fitness": "fitness"})[
                ["AAs", "fitness"]
            ]
            .copy()
        )

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


class PlotTrpB:

    """
    Plot TrpB fitness landscape
    """

    def __init__(
        self, folder: str = "data/TrpB/fitness_landscape", codon_aa: str = "AA"
    ) -> None:

        """
        Args:
        - folder, str: path to the folder containing the fitness landscapes

        """

        self._folder = os.path.normpath(folder)
        self._codon_aa = codon_aa

        (
            self._overlay_dist,
            self._dist_list,
            self._ks_list,
            self._ks_p_list,
        ) = self._overlay_fit_dist()

    def _overlay_fit_dist(self) -> hv.Distribution:

        """
        Plot the fitness distribution
        """

        colors = sns.color_palette("crest", self.trpb3_lib_numb).as_hex() + [
            PRESENTATION_PALETTE_SATURATE["orange"]
        ]

        dist_list = [None] * len(self.lib_aa_list)
        ks_list = [0] * self.trpb3_lib_numb
        ks_p_list = [0] * self.trpb3_lib_numb

        for i, lib in enumerate(self.lib_aa_list):

            print(f"Processing {lib} ...")

            lib_df = pd.read_csv(lib)
            lib_fit = lib_df["fitness"]

            wt_fit = lib_df[
                lib_df["AAs"]
                == "".join(list(LIB_INFO_DICT[get_file_name(lib)]["AAs"].values()))
            ]["fitness"].values[0]

            act_fit = lib_df[lib_df["active"] == True]["fitness"].min()

            dist_list[i] = plot_fit_dist(
                lib_fit, get_file_name(lib), colors[i], ignore_line_label=True
            ) * hv.Spikes([wt_fit]).opts(
                color="black", line_width=1.6
            ) * hv.Spikes([act_fit]).opts(
                color="gray", line_width=1.6
            )

            if i < self.trpb3_lib_numb:

                ks_list[i], ks_p_list[i] = ks_2samp(lib_fit, self.trpb4_fitness)

                print(f"Kolmogorov-Smirnov Statistic: {ks_list[i]}")
                print(f"P-value: {ks_p_list[i]}")

        # Overlay all plots
        overlay_dist = reduce(lambda x, y: x * y, dist_list)
        overlay_dist.opts(
            legend_limit=50,
            legend_position="top_right",
            title="TrpB fitness distribution",
            xlabel="Fitness",
            width=600,
        )

        save_bokeh_hv(
            plot_obj=overlay_dist,
            plot_name="TrpB fitness distribution",
            plot_path="results/fitness_distribution",
            bokehorhv="hv",
        )

        return overlay_dist, dist_list, ks_list, ks_p_list

    @property
    def lib_aa_list(self) -> list:
        """Return the list of libraries"""
        return sorted(
            list(
                set(glob(self._folder + "/*.csv"))
                - set(glob(self._folder + "/*codon*.csv"))
            )
        )

    @property
    def lib_condon_list(self) -> list:
        """Return the list of libraries"""
        return sorted(list(set(glob(self._folder + "/*codon*.csv"))))

    @property
    def trpb3_aa_csv(self) -> str:
        """Return the TrpB3 csv path"""
        return sorted(
            list(
                set(glob(self._folder + "/TrpB3*.csv"))
                - set(glob(self._folder + "/*codon*.csv"))
            )
        )

    @property
    def trpb3_lib_numb(self) -> int:
        """Return the number of three-site libraries for TrpB"""
        return len(self.trpb3_aa_csv)

    @property
    def trpb4_aa_csv(self) -> str:
        """Return the TrpB4 csv path"""
        return self._folder + "/TrpB4.csv"

    @property
    def trpb4_aa_df(self) -> pd.DataFrame:
        """Return the TrpB4 dataframe"""
        return pd.read_csv(self.trpb4_aa_csv)

    @property
    def trpb4_fitness(self) -> pd.Series:
        """Return the TrpB4 fitness"""
        return self.trpb4_aa_df["fitness"]

    @property
    def overlay_dist(self) -> hv.Distribution:
        """Return the overlay distribution"""
        return self._overlay_dist

    @property
    def dist_list(self) -> list:
        """Return the list of distributions"""
        return self._dist_list

    @property
    def ks_list(self) -> list:
        """Return the list of KS statistics"""
        return self._ks_list

    @property
    def ks_p_list(self) -> list:
        """Return the list of KS p-values"""
        return self._ks_p_list