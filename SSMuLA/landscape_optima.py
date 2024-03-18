"""A scrpit to calculate local optima"""

from __future__ import annotations

import os
from glob import glob

import itertools
from tqdm import tqdm
from multiprocessing import Pool

import numpy as np
import pandas as pd

import holoviews as hv

hv.extension("bokeh")

from SSMuLA.landscape_global import LIB_INFO_DICT, make_new_sequence, hamming
from SSMuLA.vis import save_bokeh_hv, JSON_THEME
from SSMuLA.util import get_file_name, checkNgen_folder

hv.renderer("bokeh").theme = JSON_THEME


def determine_optima(active_variant: str, df: pd.DataFrame, fit_col: str) -> int:
    """
    Determines how many single substitution variants are above a given variant in fitness.

    Args:
    - active_variant : str
        The sequence of the active/starting variant
    - df : pd.DataFrame
        The dataframe containing the fitness data
    - fit_col : str
        The column in the dataframe containing the fitness data

    Returns:
    - variant_rank : int
        The rank of the active variant in the dataframe
    """

    # slice out the variant and all the variants at hamming distance 1.
    # Then sort the dataframe by descending fitness
    temp = (
        df[
            (df["AAs"].apply(lambda x: hamming(x, active_variant) == 1))
            | (df["AAs"] == active_variant)
        ]
        .sort_values(fit_col, ascending=False)
        .reset_index(drop=True)
        .copy()
    )

    # determine the rank of the active variant or
    # how many variants are more active than it (0 is the best)
    variant_rank = temp[temp["AAs"] == active_variant].index[0]

    return variant_rank


# TODO comb with libdata for libstat
class LocOpt:
    """A class for calcaulting local optima"""

    def __init__(
        self,
        input_csv: str,
        output_folder: str = "results/local_optima",
        n_jobs: int = 256,
    ) -> None:
        """
        Args:
        - input_csv, str: pass in the input csv"""
        self._input_csv = input_csv

        output_folder = os.path.normpath(output_folder)
        self._output_folder = checkNgen_folder(output_folder)
        self._vis_folder = checkNgen_folder(
            output_folder.replace("local_optima", "local_optima_vis")
        )

        self._n_jobs = n_jobs

        self._loc_opt_df = self._find_loc_opt()

        self._hd2_escape_df, self._loc_opt_escape_df = self._append_escape()

        self._merged_escape_df = pd.merge(
            self.hd2_escape_df, self.loc_opt_escape_df, on="AAs"
        )

        # do the plottings
        self._loc_opt_scatter = self._plot_loc_opt()
        self._escape_hist = self._plot_escape()

        print("Calculating and analyzing local optima for {}...".format(self.lib_name))
        print(
            "Saving results and visualizations in {} {}".format(
                self._output_folder, self._vis_folder
            )
        )

        print(f"{self.frac_measured} of the library are measured.")
        print(
            "Number of active {} out of total {}, fraction {}".format(
                self.numb_active, self.numb_no_stop, self.frac_active
            )
        )
        print(
            "Local optima number: {}, fraction of active: {}, fraction of total: {}".format(
                self.numb_loc_opt, self.frac_loc_opt_active, self.frac_loc_opt_total
            )
        )

        # now test the escape
        print("Testing escape with double-site saturation mutagenesis...")
        print(
            "fraction of local optima that can be escaped: {:.2f}% (n={}))".format(
                self.frac_loc_opt_hd2_escape_numb, self.hd2_can_escape_numb
            )
        )

        print(
            "fraction of local optima still cannot be escaped: {:.2f}% (n={})".format(
                self.frac_loc_opt_hd2_cannot_escape_numb, self.hd2_cannot_escape_numb
            )
        )

        # save the merged loc opt with escape data
        self.merged_escape_df.to_csv(
            os.path.join(self._output_folder, f"{self.lib_name}_loc_opt_escape.csv"),
            index=False,
        )
        print("Saving local optima data to {}...".format(self._output_folder))

    def _find_loc_opt(self) -> pd.DataFrame:
        """
        Find local optima among the active variants by
        finding the rank of the variant among all its single substitutions and
        just taking those which are the maximum fitness
        """
        # Get the active variants and the args for multiprocessing
        active_variants = self.df[self.df["active"]]["AAs"].values
        pool_args = [(x, self.df, "fitness") for x in active_variants]

        # For every active variant determine its rank among its single mutants
        with Pool(self._n_jobs) as pool:
            results = pool.starmap(determine_optima, tqdm(pool_args))

        find_optima_dict = {
            active_variants[i]: results[i] for i in range(len(active_variants))
        }

        # Convert this data to a dataframe and
        # merge it with the original data to get the fitness information
        temp = pd.merge(
            self.df.copy(),
            pd.DataFrame(find_optima_dict, index=["n_greater"])
            .T.sort_values("n_greater", ascending=False)
            .reset_index()
            .rename(columns={"index": "AAs"}),
        )

        # Get the local optima by finding the variants where no single mutant is more fit
        loc_opt_df = temp[temp["n_greater"] == 0].reset_index(drop=True).copy()
        return loc_opt_df.sort_values("fitness", ascending=False).reset_index(drop=True)

    def _append_escape(self) -> pd.DataFrame:
        """
        Append escape variants to the dataframe
        """
        find_hd2_escape = {}
        double_site_escape = {}

        # Loop through the optima
        for opt_variant in tqdm(self.loc_opt_df["AAs"].values):

            # slice out the variant and all the variants at hamming distance 2.
            # Then sort the dataframe by descending fitness
            temp = (
                self.df[
                    (self.df["AAs"].apply(lambda x: hamming(x, opt_variant) == 2))
                    | (self.df["AAs"] == opt_variant)
                ]
                .sort_values("fitness", ascending=False)
                .reset_index(drop=True)
                .copy()
            )

            # determine the rank of the active variant/how many variants are more active than it.
            # If the variant is not 0 it can escape
            find_hd2_escape[opt_variant] = temp[temp["AAs"] == opt_variant].index[0]

            # This time save the entire DataFrame for some downstream analyses
            double_site_escape[opt_variant] = temp

        # merge the data with the TrpB_data to get the fitness information
        hd2_escape_df = pd.merge(
            self.df.copy(),
            pd.DataFrame(find_hd2_escape, index=["n_escape"])
            .T.sort_values("n_escape", ascending=False)
            .reset_index()
            .rename(columns={"index": "AAs"}),
        )

        # Get all possible pairs of positions that could be included in the escape double
        position_sets = list(itertools.combinations(range(self.numb_sites), 2))

        result_dict = {}

        # for every local optima
        for var_of_interest in tqdm(self.loc_opt_df["AAs"].values):

            # Get the dataframe for that variant
            result_dict[var_of_interest] = {}
            var_fit = self.df_seq_fit_dict[var_of_interest]
            temp = double_site_escape[var_of_interest].copy()

            # For every pair of positions
            for position1, position2 in position_sets:

                # find the mutants that escape the double-site mutant
                _temp = temp[temp["fitness"] > var_fit].reset_index(drop=True)
                _temp = _temp[
                    (_temp[f"AA{position1+1}"] == var_of_interest[position1])
                    & (_temp[f"AA{position2+1}"] == var_of_interest[position2])
                ]

                # save the number of mutants for that pair that escape
                result_dict[var_of_interest][(position1, position2)] = len(_temp)

        # Convert these results to a DataFrame
        loc_opt_escape = pd.DataFrame(result_dict).T
        loc_opt_escape.columns = [
            "".join(str(col)).strip() for col in loc_opt_escape.columns.values
        ]

        # Add some columns to the dataframe for the fraction that do and do not escape
        loc_opt_escape["frac pairs no escape"] = loc_opt_escape.apply(
            lambda x: list(x).count(0) / len(position_sets), axis=1
        )
        loc_opt_escape["frac pairs that escape"] = loc_opt_escape.apply(
            lambda x: 1 - list(x).count(0) / len(position_sets), axis=1
        )

        # Add the fitness of the local optima to the dataframe
        loc_opt_escape = loc_opt_escape.reset_index().rename(columns={"index": "AAs"})

        return hd2_escape_df, loc_opt_escape

    def _plot_loc_opt(self) -> None:
        """
        Plot the local optima
        """

        title = f"{self.lib_name} local optima"

        # add a column for rank
        temp = self.loc_opt_df.copy()
        temp["rank"] = temp["fitness"].rank(ascending=False)

        fig = (
            hv.Scatter(temp, kdims=["rank"], vdims=["fitness"]).opts(
                xlabel="Rank",
                ylabel="Fitness",
                title=title,
            )
            * hv.HLine(self.parent_aa_fitness).opts(color="grey")
        )

        save_bokeh_hv(
            fig,
            plot_name=title,
            plot_path=checkNgen_folder(os.path.join(self._vis_folder, "fitness")),
        )

        return fig

    def _plot_escape(self) -> None:
        """
        Plot the escape
        """

        title = f"{self.lib_name} local optima double-site escape"

        hist = hv.Histogram(
            np.histogram(
                self.loc_opt_escape_df["frac pairs that escape"].values, bins=6
            )
        ).opts(
            ylabel="Number of variants",
            xlabel="Fraction of pairs that escape",
            title=title,
        )

        save_bokeh_hv(
            hist,
            plot_name=title,
            plot_path=checkNgen_folder(os.path.join(self._vis_folder, "escape")),
        )

        return hist

    @property
    def lib_name(self) -> dict:
        """Return the library name"""
        return get_file_name(self._input_csv)

    @property
    def lib_info(self) -> dict:
        """Return the library information"""
        return LIB_INFO_DICT[self.lib_name]

    @property
    def numb_sites(self) -> int:
        """Return the number of sites"""
        return len(self.lib_info["AAs"])

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe without stop codons"""
        df = pd.read_csv(self._input_csv)
        return df[~df["AAs"].str.contains("\*")].copy()

    @property
    def numb_no_stop(self) -> int:
        """Return the length of the dataframe"""
        return len(self.df)

    @property
    def frac_measured(self) -> float:
        """Return the fraction of complete sequences"""
        return self.numb_no_stop / 20 ** self.numb_sites

    @property
    def numb_active(self) -> int:
        """Return the number of active variants"""
        return len(self.df[self.df["active"]])

    @property
    def frac_active(self) -> float:
        """Return the percent active"""
        return self.numb_active / self.numb_no_stop

    @property
    def parent_aa(self) -> str:
        """Return the parent amino acid"""
        return "".join(list(self.lib_info["AAs"].values()))

    @property
    def parent_aa_fitness(self) -> float:
        """Return the parent aa fitness"""
        return self.df[self.df["AAs"] == self.parent_aa]["fitness"].values[0]

    @property
    def df_seq_fit_dict(self) -> dict:
        """Return the dataframe as a dictionary for faster processing"""

        seqs = self.df["AAs"].values
        fit = self.df["fitness"].values

        return {seq: fit for seq, fit in zip(seqs, fit)}

    @property
    def loc_opt_df(self) -> pd.DataFrame:
        """Return the dataframe of local optima"""
        return self._loc_opt_df

    @property
    def numb_loc_opt(self) -> int:
        """Return the number of local optima"""
        return len(self._loc_opt_df)

    @property
    def frac_loc_opt_active(self) -> float:
        """Return the fraction of local optima of active variants"""
        return self.numb_loc_opt / self.numb_active

    @property
    def frac_loc_opt_total(self) -> float:
        """Return the fraction of local optima of total variants"""
        return self.numb_loc_opt / self.numb_no_stop

    @property
    def hd2_escape_df(self) -> pd.DataFrame:
        """Return the dataframe of escape variants"""
        return self._hd2_escape_df

    @property
    def loc_opt_escape_df(self) -> pd.DataFrame:
        """Return the dataframe of local optima escape variants"""
        return self._loc_opt_escape_df

    @property
    def hd2_can_escape_numb(self) -> pd.DataFrame:
        """Return the dataframe of local optima escape variants"""
        return len(self.hd2_escape_df[self.hd2_escape_df["n_escape"] != 0])

    @property
    def hd2_cannot_escape_numb(self) -> pd.DataFrame:
        """Return the dataframe of local optima escape variants"""
        return len(self.hd2_escape_df[self.hd2_escape_df["n_escape"] == 0])

    @property
    def frac_loc_opt_hd2_escape_numb(self) -> float:
        """Return the fraction of local optima that can escape"""
        return self.hd2_can_escape_numb / self.numb_loc_opt

    @property
    def frac_loc_opt_hd2_cannot_escape_numb(self) -> float:
        """Return the fraction of local optima that cannot escape"""
        return self.hd2_cannot_escape_numb / self.numb_loc_opt

    @property
    def merged_escape_df(self) -> pd.DataFrame:
        """Return the merged double site escape dataframe"""
        return self._merged_escape_df

    @property
    def loc_opt_scatter(self) -> hv.Scatter:
        """Return the loc opt scatter"""
        return self._loc_opt_scatter

    @property
    def escape_hist(self) -> hv.Histogram:
        """Return the escape hist"""
        return self._escape_hist


def run_loc_opt(
    input_folder: str = "data",
    fitness_process_type: str = "scale2max",
    output_folder: str = "results/local_optima",
    n_jobs: int = 256,
) -> None:
    """
    Run the local optima
    """

    summary_df = pd.DataFrame()

    for lib in sorted(
        glob(os.path.normpath(input_folder) + "/*/" + fitness_process_type + "/*.csv")
    ):

        print(f"Processing {lib}...")

        opt_class = LocOpt(
            lib,
            checkNgen_folder(os.path.join(output_folder, fitness_process_type)),
            n_jobs,
        )

        summary_df = summary_df._append(
            {
                "lib": opt_class.lib_name,
                "summary_type": fitness_process_type,
                "frac_measured": opt_class.frac_measured,
                "numb_active": opt_class.numb_active,
                "frac_active": opt_class.frac_active,
                "numb_loc_opt": opt_class.numb_loc_opt,
                "frac_loc_opt_active": opt_class.frac_loc_opt_active,
                "frac_loc_opt_total": opt_class.frac_loc_opt_total,
                "frac_loc_opt_hd2_escape_numb": opt_class.frac_loc_opt_hd2_escape_numb,
                "frac_loc_opt_hd2_cannot_escape_numb": opt_class.frac_loc_opt_hd2_cannot_escape_numb,
            },
            ignore_index=True,
        )

    summary_df_path = os.path.join(output_folder, f"{fitness_process_type}.csv")

    checkNgen_folder(summary_df_path)

    print("Saving summary_df_melt at {}...".format(summary_df_path))
    summary_df.to_csv(summary_df_path, index=False)