"""A script for calculating pairwise epistasis"""

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


from SSMuLA.aa_global import ALL_AAS
from SSMuLA.util import checkNgen_folder


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
        filter_min_by: str = "active_min",
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
        self._filter_min_by = filter_min_by
        self._n_jobs = n_jobs

        # generate and save the pairwise epistasis DataFrame
        self.filtered_epistasis_df.to_csv(self.output_csv, index=True)
        print(f"Saving pairwise epistasis in {self.output_csv}...")

    @property
    def output_csv(self) -> str:

        """
        Get the output path of the dataframe.

        Returns:
        - str: The output folder.
        """

        output_csv = self._input_csv.replace("data", self._output_folder)

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

        return df[~df["AAs"].str.contains("\*")]

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
        elif self._filter_min_by == "" or None:
            return self.epistasis_df.copy()
        else:
            return filter_epistasis_results(
                self.epistasis_df, float(self._filter_min_by)
            ).copy()


def make_new_sequence(input_seq: str, new_AA: str, position: int) -> str:
    """
    Make a new sequence by replacing the amino acid at a specific position.

    Args:
        - input_seq (str): The input sequence.
        - new_AA (str): The new amino acid to replace.
        - position (int): The position in the sequence to replace.

    Returns:
        - str: The new sequence with the replaced amino acid.
    """
    seq_list = list(input_seq)
    seq_list[position] = new_AA
    return "".join(seq_list)


def hamming(str1: str, str2: str) -> int:
    """
    Calculate the Hamming distance between two strings.

    Args:
        - str1 (str): The first string.
        - str2 (str): The second string.

    Returns:
        - int: The Hamming distance between the two strings.
    """
    assert len(str1) == len(str2)

    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance


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


def run_pairwise_epistasis(
    input_folder: str = "data",
    fitness_process_type: str = "scale2max",
    filter_min_by: str = "active_min",
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

    for lib in glob(os.path.normpath(input_folder) + "/*/" + fitness_process_type + "/*.csv"):
        print(f"Processing {lib}...")
        PairwiseEpistasis(
            lib, filter_min_by=filter_min_by, output_folder=output_folder, n_jobs=n_jobs
        )


######## FOR PLOTTING #####
def hook(plot, element):
    plot.handles["plot"].x_range.factors = [
        (position, epistasis)
        for position in ["01", "02", "03", "12", "13", "23"]
        for epistasis in ["magnitude", "sign", "reciprocal sign"]
    ]