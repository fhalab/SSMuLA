"""A script for calculating and plotting pairwise epistasis"""

from __future__ import annotations

# General imports
from glob import glob
import os

# Data manipulation
import pandas as pd
import numpy as np

import itertools
from tqdm.auto import tqdm
from multiprocessing import Pool
from operator import itemgetter



import holoviews as hv
from holoviews import dim

hv.extension("bokeh")

from SSMuLA.aa_global import ALL_AAS
from SSMuLA.landscape_global import LIB_NAMES, LIB_POS_MAP, make_new_sequence, hamming
from SSMuLA.util import checkNgen_folder, get_file_name, get_dir_name
from SSMuLA.vis import (
    JSON_THEME,
    save_bokeh_hv,
    one_decimal_y,
    fixmargins,
    LIB_COLORS,
)

from bokeh.themes.theme import Theme

hv.renderer("bokeh").theme = JSON_THEME


EPISTASIS_TYPE = ["magnitude", "sign", "reciprocal sign"]


assign_epistasis_dict = {
    ### starting variant #1 ###
    ("00", "01", "10", "11"): "magnitude",  # upwards
    ("00", "10", "01", "11"): "magnitude",  # upwards
    ("00", "01", "11", "10"): "sign",  # upwards
    ("00", "10", "11", "01"): "sign",  # upwards
    ("00", "11", "01", "10"): "reciprocal sign",  # upwards
    ("00", "11", "10", "01"): "reciprocal sign",  # upwards
    ### starting variant #2 ###
    ("01", "00", "11", "10"): "magnitude",  # upwards
    ("10", "00", "11", "01"): "magnitude",  # upwards
    ("01", "00", "10", "11"): "sign",  # upwards
    ("10", "00", "01", "11"): "sign",  # upwards
    ("11", "00", "01", "10"): "reciprocal sign",  # downwards
    ("11", "00", "10", "01"): "reciprocal sign",  # downwards
    ### starting variant #3 ###
    ("01", "11", "00", "10"): "magnitude",  # downwards
    ("10", "11", "00", "01"): "magnitude",  # downwards
    ("11", "01", "00", "10"): "sign",  # downwards
    ("11", "10", "00", "01"): "sign",  # downwards
    ("01", "10", "00", "11"): "reciprocal sign",  # upwards
    ("10", "01", "00", "11"): "reciprocal sign",  # upwards
    ### starting variant #4 ###
    ("11", "01", "10", "00"): "magnitude",  # downwards
    ("11", "10", "01", "00"): "magnitude",  # downwards
    ("01", "11", "10", "00"): "sign",  # downwards
    ("10", "11", "01", "00"): "sign",  # downwards
    ("01", "10", "11", "00"): "reciprocal sign",  # downwards
    ("10", "01", "11", "00"): "reciprocal sign",  # downwards
}


class PairwiseEpistasis:

    """
    A class for calculating pairwise epistasis.
    """

    def __init__(
        self,
        input_csv: str,
        output_folder: str = "results/pairwise_epistasis",
        filter_min_by: str = "none",
        n_jobs: int = 256,
    ) -> None:

        """
        Args:
        - input_csv, str: The input CSV file, ie. data/TrpB/scale2max/TrpB4.csv
        - output_folder, str: The output folder.
        - filter_min_by, str: methods to filter the DataFrame by, 0 or active_min.
        - n_jobs, int: The number of jobs to run in parallel.
        """

        self._input_csv = input_csv
        self._output_folder = checkNgen_folder(output_folder)

        if filter_min_by in ["", "none", "None", None]:
            self._filter_min_by = "none"
        else:
            self._filter_min_by = filter_min_by
        self._n_jobs = n_jobs

        # generate and save the pairwise epistasis DataFrame
        self.filtered_epistasis_df.to_csv(self.output_csv, index=True)
        print(f"Saving pairwise epistasis in {self.output_csv}...")

    @property
    def lib_name(self) -> str:
        """The name of the library."""
        return get_file_name(self._input_csv)

    @property
    def fitness_process_type(self) -> str:
        """The fitness process type."""
        return get_dir_name(self._input_csv)

    @property
    def output_csv(self) -> str:

        """
        Get the output path of the dataframe.

        Returns:
        - str: The output folder.
        """

        output_csv = os.path.join(
            self._output_folder,
            self._filter_min_by,
            self.fitness_process_type,
            f"{self.lib_name}.csv",
        )

        # make sure the subfolder exists and create them if not
        checkNgen_folder(output_csv)

        return output_csv

    @property
    def df(self) -> pd.DataFrame:

        """
        Remove stop codons from the input DataFrame.

        Returns:
        - pd.DataFrame: The DataFrame with stop codons removed.
        """

        df = pd.read_csv(self._input_csv)

        return df[~df["AAs"].str.contains("\*")].copy()

    @property
    def active_fit_min(self) -> float:

        """
        Calculate the cutoff for active mutants based on
        1.96 standard deviations above the mean fitness of all stop-codon-containing sequences.

        Returns:
        - float: The cutoff value for active mutants.
        """

        return self.df[self.df["active"]]["fitness"].min()

    @property
    def epistasis_df(self) -> pd.DataFrame:

        """
        Calculate pairwise epistasis.

        Returns:
        - pd.DataFrame: The DataFrame with pairwise epistasis.
        """

        epistasis_df = get_epistasis_data(
            self.df, "AAs", "fitness", n_jobs=self._n_jobs
        )

        # add epsilon
        epistasis_df["epsilon"] = epistasis_df.apply(calc_epsilon, axis=1)

        return epistasis_df

    @property
    def filtered_epistasis_df(self) -> pd.DataFrame:

        """
        Filter the DataFrame by the minimum fitness.

        Returns:
        - pd.DataFrame: The filtered DataFrame.
        """

        if self._filter_min_by == "active_min":
            return filter_epistasis_results(
                self.epistasis_df, self.active_fit_min
            ).copy()
        elif self._filter_min_by == "none":
            return self.epistasis_df.copy()
        else:
            return filter_epistasis_results(
                self.epistasis_df, float(self._filter_min_by)
            ).copy()


def pairwise_epistasis(seq_ab: str, data_dict: dict) -> pd.DataFrame:
    """
    Classifies the epistasis type between the given sequence and all other double mutants.

    Args:
    - seq_ab, str: the starting sequence to compare to all other double mutants.
    - data_dict, dict: a dictionary of all sequences and their fitness values.

    Returns:
    - epistasis_results : pd.DataFrame
        A dataframe with the following columns:
            - start_seq: the starting sequence
            - positions: the positions that were mutated
            - res1_AA: the amino acid that was mutated at position 1
            - res2_AA: the amino acid that was mutated at position 2
            - fit_ab: the fitness of the starting sequence
            - fit_Ab: the fitness of the sequence with a mutation at position 1
            - fit_aB: the fitness of the sequence with a mutation at position 2
            - fit_AB: the fitness of the sequence with mutations at both positions
            - epistasis_type: the type of epistasis between the starting sequence and the double mutant
    """

    position_orders = list(itertools.combinations(range(len(seq_ab)), 2))
    epsilon_dict = {}

    fit_ab = data_dict[seq_ab]

    for temp_order in position_orders:

        for AA1 in ALL_AAS:

            if AA1 == seq_ab[temp_order[0]]:
                pass

            else:
                seq_Ab = make_new_sequence(seq_ab, AA1, temp_order[0])

                if seq_Ab in data_dict:
                    fit_Ab = data_dict[seq_Ab]
                else:
                    fit_Ab = np.nan

                for AA2 in ALL_AAS:
                    if AA2 == seq_ab[temp_order[1]]:
                        pass

                    else:
                        seq_aB = make_new_sequence(seq_ab, AA2, temp_order[1])
                        seq_AB = make_new_sequence(seq_Ab, AA2, temp_order[1])

                        # get fitness scores
                        if seq_aB in data_dict:
                            fit_aB = data_dict[seq_aB]
                        else:
                            fit_aB = np.nan

                        if seq_AB in data_dict:
                            fit_AB = data_dict[seq_AB]
                        else:
                            fit_AB = np.nan

                        # sort the entries by fitness
                        sorted_fitness_list = sorted(
                            [
                                ("00", fit_ab),
                                ("10", fit_Ab),
                                ("01", fit_aB),
                                ("11", fit_AB),
                            ],
                            key=itemgetter(1),
                        )

                        # abstract the sequences only from the list
                        sorted_entries = tuple([x[0] for x in sorted_fitness_list])

                        epistasis_type = assign_epistasis_dict[sorted_entries]

                        epsilon_dict[(f"{temp_order[0]}{temp_order[1]}", AA1, AA2)] = [
                            fit_ab,
                            fit_Ab,
                            fit_aB,
                            fit_AB,
                            epistasis_type,
                        ]

    epistasis_results = pd.DataFrame(epsilon_dict).T.reset_index()
    epistasis_results.insert(0, "start_seq", seq_ab)

    return epistasis_results


def get_epistasis_data(
    full_df: pd.DataFrame, seq_col: str, fitness_col: str, n_jobs: int
) -> pd.DataFrame:

    # Get all nonzero sequences
    active_variants = full_df[full_df["active"]][seq_col]

    # Conver the data to a dictionary for faster lookup
    data_dict = dict(zip(full_df[seq_col].values, full_df[fitness_col].values))
    pool_args = [(start_seq, data_dict) for start_seq in active_variants]

    with Pool(n_jobs) as pool:
        result = pool.starmap(pairwise_epistasis, tqdm(pool_args))

    all_epistasis_results = (
        pd.concat(result)
        .rename(
            columns={
                0: "fit_ab",
                1: "fit_Ab",
                2: "fit_aB",
                3: "fit_AB",
                4: "epistasis_type",
                "level_0": "positions",
                "level_1": "res1_AA",
                "level_2": "res2_AA",
            }
        )
        .set_index(["start_seq", "positions", "res1_AA", "res2_AA"])
    )

    # Set column types
    all_epistasis_results = all_epistasis_results.astype(
        {
            "fit_ab": "float64",
            "fit_Ab": "float64",
            "fit_aB": "float64",
            "fit_AB": "float64",
        }
    )

    return all_epistasis_results


def filter_epistasis_results(df, fit_min=0):

    df = df[
        (df["fit_ab"] >= fit_min)
        & (df["fit_Ab"] >= fit_min)
        & (df["fit_aB"] >= fit_min)
        & (df["fit_AB"] >= fit_min)
    ]

    return df


def calc_epsilon(row):
    return (
        np.log(row["fit_AB"] / row["fit_ab"])
        - np.log(row["fit_Ab"] / row["fit_ab"])
        - np.log(row["fit_aB"] / row["fit_ab"])
    )


class VisPairwiseEpistasis:

    """A class to visualize pairwise epistasis data."""

    def __init__(
        self,
        pairwise_csv: str,
        dets_folder: str = "results/pairwise_epistasis_dets",
        vis_folder: str = "results/pairwise_epistasis_vis",
        filter_min_by: str = "none",
    ):

        """
        Args:
        - pairwise_csv: The path to the pairwise epistasis data.
            ie. results/pairwise_epistasis/DHFR/scale2max/DHFR.csv
        - dets_folder: The folder to save the processed dataframe with more details.
        - vis_folder: The folder to save the visualizations.
        - filter_min_by: The minimum fitness to filter by.
        """

        self._pairwise_csv = pairwise_csv
        self._vis_folder = checkNgen_folder(vis_folder)
        self._dets_folder = checkNgen_folder(dets_folder)
        self._filter_min_by = filter_min_by

        print(
            "Visulizing pairwise epistasis for {} saved to {}".format(
                self.lib_name, self.vis_subfolder
            )
        )

        if self.df_len > 0:
            self._epsilon_pos = self._plot_epsilon_pos()
            self._epsilon_pos_type = self._plot_epsilon_pos_type()
            self._type_frac = self._plot_type_frac()
            self._quartile_type_frac = self._plot_quartile_type_frac()
        else:
            print(f"No data for {self.lib_name}. Skip plotting...")

        # save the df_quartile_grouped
        print(
            "Visulizing pairwise epistasis with details for {} saved to {}".format(
                self.lib_name, self.dets_subfolder
            )
        )     

        self.df_quartiles.to_csv(    
            os.path.join(self.dets_subfolder, f"{self.lib_name}.csv")
        )
        self.df_quartile_grouped.to_csv(    
            os.path.join(self.dets_subfolder, f"{self.lib_name}_grouped.csv")
        )

    def _plot_epsilon_pos(self) -> hv.Violin:

        """
        A method to plot the position vs. epsilon violin plot.
        """

        title = f"{self.lib_name} pairwise epsilon vs positions"

        vis = hv.Violin(self.df, kdims=["positions"], vdims=["epsilon"],).opts(
            violin_color=dim("positions").str(),
            cmap=[LIB_COLORS[self.lib_name]],
            height=300,
            width=400,
            show_legend=False,
            violin_width=0.8,
            hooks=[fixmargins],
            title=title,
        )

        save_bokeh_hv(vis, plot_name=title, plot_path=self.vis_subfolder)

        return vis

    def _pos_hook(self, plot, elements):
        """
        Define plot hook for position
        """
        plot.handles["plot"].x_range.factors = [
            (position, epistasis)
            for position in sorted(list(self.df["positions"].unique()))
            for epistasis in EPISTASIS_TYPE
        ]

    def _quart_hook(self, plot, element):
        """
        Define plot hook for quartile
        """
        plot.handles["plot"].x_range.factors = [
            (epistasis, quartile)
            for epistasis in EPISTASIS_TYPE
            for quartile in ["Q" + str(n) for n in range(1, 5)]
        ]

    def _plot_epsilon_pos_type(self) -> hv.Violin:

        """
        A method to plot the position vs. epsilon violin plot colored by type
        """

        title = f"{self.lib_name} pairwise epsilon vs positions by type"

        def hook(plot, elements):
            plot.handles["plot"].x_range.factors = [
                (position, epistasis)
                for position in sorted(list(self.df["positions"].unique()))
                for epistasis in EPISTASIS_TYPE
            ]

        vis = hv.Violin(
            self.df,
            kdims=["positions", "epistasis_type"],
            vdims=["epsilon"],
        ).opts(
            height=400,
            width=600,
            xrotation=90,
            violin_width=0.8,
            violin_color="epistasis_type",
            show_legend=True,
            legend_position="top",
            legend_offset=(0, 5),
            hooks=[fixmargins, one_decimal_y, self._pos_hook],
            title=title,
        )

        save_bokeh_hv(vis, plot_name=title, plot_path=self.vis_subfolder)

        return vis

    def _plot_type_frac(self) -> hv.Violin:

        """
        A method to plot the position vs. epistasis type fraction
        """

        title = f"{self.lib_name} pairwise epistasis type fraction"

        df = self.df_grouped[self.df_grouped["total"] > 0].copy()

        vis = hv.Violin(
            df, kdims=["positions", "epistasis_type"], vdims=["frac epistasis type"]
        ).opts(
            height=400,
            width=600,
            xrotation=90,
            violin_width=0.8,
            violin_color="epistasis_type",
            hooks=[fixmargins, one_decimal_y, self._pos_hook],
            title=title,
            show_legend=True,
            legend_position="top",
            legend_offset=(0, 5),
        )

        save_bokeh_hv(vis, plot_name=title, plot_path=self.vis_subfolder)

        return vis

    def _plot_quartile_type_frac(self) -> hv.Violin:

        """
        A method to plot the position vs. epistasis type fraction
        """

        title = f"{self.lib_name} pairwise epistasis type fraction by quartile"

        df = self.df_quartile_grouped.copy()

        vis = hv.Violin(
            self.df_quartile_grouped.sort_values(["epistasis type", "quartile"]),
            kdims=["epistasis type", "quartile"],
            vdims=["frac epistasis type"],
        ).opts(
            violin_color=dim("quartile").str(),
            height=400,
            width=600,
            violin_width=0.8,
            hooks=[fixmargins, one_decimal_y, self._quart_hook],
            show_legend=True,
            legend_position="top",
            legend_offset=(0, 5),
            ylabel="fraction of epistasis type",
            title=title,
        )

        save_bokeh_hv(vis, plot_name=title, plot_path=self.vis_subfolder)

        return vis

    @property
    def lib_name(self) -> str:
        """The name of the library."""
        return get_file_name(self._pairwise_csv)

    @property
    def fitness_process_type(self) -> str:
        """The fitness process type."""
        return get_dir_name(self._pairwise_csv)

    @property
    def dets_subfolder(self) -> str:
        """The subfolder to save more detailed dataframe."""
        return checkNgen_folder(
            os.path.join(
                self._dets_folder,
                self._filter_min_by,
                self.fitness_process_type
            )
        )
    
    @property
    def vis_subfolder(self) -> str:
        """The subfolder to save visualizations."""
        return checkNgen_folder(
            os.path.join(
                self._vis_folder,
                self._filter_min_by,
                self.fitness_process_type,
                self.lib_name,
            )
        )

    @property
    def df(self) -> pd.DataFrame:
        """The dataframe of the pairwise epistasis data."""

        # map int pos to real
        df = pd.read_csv(self._pairwise_csv).dropna()
        df["positions"] = df["positions"].map(LIB_POS_MAP[self.lib_name])

        return df.copy()

    @property
    def df_len(self) -> int:
        """The length of the dataframe."""
        return len(self.df)

    @property
    def epistasis_type_counts(self) -> dict:
        """The counts of the epistasis types."""
        if self.df_len > 0:
            return self.df["epistasis_type"].value_counts().to_dict()
        else:
            return {epistasis_type: 0 for epistasis_type in EPISTASIS_TYPE}

    @property
    def epistasis_type_fraction(self) -> dict:
        """The fraction of the epistasis types."""
        if self.df_len > 0:
            return (self.df["epistasis_type"].value_counts() / len(self.df)).to_dict()
        else:
            return {epistasis_type: 0 for epistasis_type in EPISTASIS_TYPE}

    @property
    def df_grouped(self) -> pd.DataFrame:

        """The dataframe of the pairwise epistasis data."""
        self.df["epistasis_type"] = self.df["epistasis_type"].astype("category")

        grouped_epistasis_df = pd.DataFrame(
            self.df.groupby(["positions", "start_seq", "epistasis_type"]).size()
        ).rename(columns={0: "count"})

        grouped_epistasis_df["total"] = grouped_epistasis_df.groupby(
            ["positions", "start_seq"]
        )["count"].transform("sum")
        grouped_epistasis_df["frac epistasis type"] = (
            grouped_epistasis_df["count"] / grouped_epistasis_df["total"]
        )

        return grouped_epistasis_df.copy()

    @property
    def df_quartiles(self) -> pd.DataFrame:

        """The dataframe of the pairwise epistasis data with quartiles."""

        df = self.df.copy()

        # Add quartiles
        temp = df.reset_index()[["start_seq", "fit_ab"]].drop_duplicates().copy()

        cutoffs = temp["fit_ab"].quantile([0.25, 0.5, 0.75, 1]).to_dict()
        quart_dict = {cutoff: f"Q{i+1}" for i, cutoff in enumerate(cutoffs.values())}

        def assign_quartile(x):
            for cutoff in quart_dict:
                if x <= cutoff:
                    return quart_dict[cutoff]

        df["quartile"] = df["fit_ab"].apply(assign_quartile)

        return df

    @property
    def df_quartile_grouped(self) -> pd.DataFrame:

        """
        The dataframe of the pairwise epistasis data
        with quartiles and grouped by quartiles.
        """

        temp = self.df_quartiles.copy()

        temp["epistasis_type"] = temp["epistasis_type"].astype("category")

        df = pd.DataFrame(
            temp.groupby(["start_seq", "quartile", "epistasis_type"]).size()
        ).rename(columns={0: "count"})

        df["total"] = df.groupby(["start_seq", "quartile"])["count"].transform("sum")
        df["frac epistasis type"] = df["count"] / df["total"]

        df = df[df["total"] > 0].copy()

        df.index.names = ["start_seq", "quartile", "epistasis type"]

        return df

    @property
    def epsilon_pos(self) -> hv.Violin:
        """The position vs. epsilon violin plot."""
        return self._epsilon_pos

    @property
    def epsilon_pos_type(self) -> hv.Violin:
        """The position vs. epsilon violin plot colored by type."""
        return self._epsilon_pos_type

    @property
    def type_frac(self) -> hv.Violin:
        """The position vs. epistasis type fraction."""
        return self._type_frac

    @property
    def quartile_type_frac(self) -> hv.Violin:
        """The position vs. epistasis type fraction by quartile."""
        return self._quartile_type_frac


def run_pairwise_epistasis(
    input_folder: str = "data",
    fitness_process_type: str = "scale2max",
    filter_min_by: str = "none",
    output_folder: str = "results/pairwise_epistasis",
    n_jobs: int = 256,
):

    """
    Run pairwise epistasis on all CSV files in a folder.

    Args:
    - input_folder, str: The input folder.
    - fitness_process_type, str: The fitness process type.
    - filter_min_by, str: The minimum fitness to filter by.
    - output_folder, str: The output folder.
    - n_jobs, int: The number of jobs to run in parallel.
    """

    for lib in glob(
        os.path.normpath(input_folder) + "/*/" + fitness_process_type + "/*.csv"
    ):
        print(f"Processing {lib}...")
        PairwiseEpistasis(
            lib, filter_min_by=filter_min_by, output_folder=output_folder, n_jobs=n_jobs
        )


def plot_pairwise_epistasis(
    fitness_process_type: str = "scale2max",
    filter_min_by: str = "none",
    input_folder: str = "results/pairwise_epistasis",
    output_folder: str = "results/pairwise_epistasis_vis",
    dets_folder: str = "results/pairwise_epistasis_dets",
):

    """
    Plot pairwise epistasis on all CSV files in a folder.

    Args:
    - fitness_process_type, str: The fitness process type.
    - input_folder, str: The input folder.
    - output_folder, str: The output folder.
    - dets_folder, str: The folder to save the processed dataframe with more details.
    """

    summary_df = pd.DataFrame(columns=["lib", "summary_type", *EPISTASIS_TYPE])

    for lib in sorted(
        glob(
            os.path.join(
                os.path.normpath(input_folder), filter_min_by, fitness_process_type
            )
            + "/*.csv"
        )
    ):

        print(f"Plotting {lib}...")

        vis_class = VisPairwiseEpistasis(
            lib,
            filter_min_by=filter_min_by,
            dets_folder=dets_folder,
            vis_folder=output_folder
            )

        count_dict = vis_class.epistasis_type_counts
        fract_dict = vis_class.epistasis_type_fraction

        for et, ed in zip(["count", "fraction"], [count_dict, fract_dict]):  

            summary_df = summary_df.append(
                {
                    "lib": vis_class.lib_name,
                    "summary_type": et,
                    "magnitude": ed.get("magnitude", 0),
                    "sign": ed.get("sign", 0),
                    "reciprocal sign": ed.get("reciprocal sign", 0),
                },
                ignore_index=True,
            )

    summary_df_melt = pd.melt(
        summary_df,
        id_vars=["lib", "summary_type"],
        value_vars=["magnitude", "sign", "reciprocal sign"],
        var_name="epistasis_type",
        value_name="value",
    ).sort_values(["lib", "summary_type"])

    summary_df_path = os.path.join(
        output_folder, filter_min_by, f"{fitness_process_type}.csv"
    )

    checkNgen_folder(summary_df_path)

    print("Saving summary_df_melt at {}...".format(summary_df_path))

    summary_df_melt.to_csv(summary_df_path, index=False)

    # make bar plots base on that and save to the same directory
    def hook(plot, element):
        plot.handles["plot"].x_range.factors = [
            (lib, epistasis) for lib in LIB_NAMES for epistasis in EPISTASIS_TYPE
        ]

    # Create the Holoviews Bars element
    save_bokeh_hv(
        hv.Bars(
            summary_df_melt[summary_df_melt["summary_type"] == "fraction"],
            kdims=["lib", "epistasis_type"],
            vdims="value",
        ).opts(
            width=1200,
            height=400,
            show_legend=True,
            legend_position="top",
            legend_offset=(0, 5),
            ylabel="Fraction",
            multi_level=False,
            title="Fraction of pairwise epistasis types",
            xlabel="Library",
            hooks=[fixmargins, one_decimal_y, hook],
        ),
        plot_name=fitness_process_type,
        plot_path=os.path.join(output_folder, filter_min_by),
    )
    