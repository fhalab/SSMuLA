"""A script for generating mut file needed for triad"""

import re
import os

from copy import deepcopy
from glob import glob

import pandas as pd
import numpy as np
from itertools import product

from SSMuLA.aa_global import ALL_AAS
from SSMuLA.landscape_global import LIB_INFO_DICT, LIB_NAMES, TrpB_names
from SSMuLA.util import checkNgen_folder, get_file_name


# TrpB_TRIAD_FOLDER = "/home/shared_data/triad_structures"
TrpB_LIB_FOLDER = "data/TrpB/scaled2max"
# TrpB3_TRIAD_TXT = deepcopy(sorted(list(glob(f"{TrpB_TRIAD_FOLDER}/*3*/*/*.txt"))))
TrpB4_TRIAD_TXT = deepcopy(sorted(list(glob("triad/TrpB4/*.txt"))))
# /disk2/fli/SSMuLA/triad/TrpB4

lib_triad_pair = {}

# append the other two lib
for lib in LIB_NAMES:
    if lib != "TrpB4":
        lib_triad_pair[f"data/{lib}/scaled2max/{lib}.csv"] = f"triad/{lib}/{lib}_fixed.txt"

sorted_lib_triad_pair = deepcopy(dict(sorted(lib_triad_pair.items(), key=lambda x: x[0])))

TrpB4_lib_triad_pair = {
    os.path.join(TrpB_LIB_FOLDER, "TrpB4.csv"): triad for triad in TrpB4_TRIAD_TXT
}


class TriadLib:
    """
    A class for common traid things for a given lib
    """

    def __init__(self, input_csv: str, triad_folder: str = "triad") -> None:

        """
        Args:
        - input_csv: str, the path to the input csv
        - output_folder: str, the path to the output folder
        """

        self._input_csv = input_csv
        self._triad_folder = os.path.normpath(triad_folder)

    @property
    def lib_name(self) -> str:

        """
        A property for the library name
        """
        return get_file_name(self._input_csv)

    @property
    def site_num(self) -> int:

        """
        A property for the site number
        """
        return len(LIB_INFO_DICT[self.lib_name]["positions"])

    @property
    def wt_aas(self) -> list:
        """
        A property for the wildtype amino acids
        """
        return list(LIB_INFO_DICT[self.lib_name]["AAs"].values())

    @property
    def prefixes(self) -> list:
        """
        A property for the prefixes
        """
        return [f"A_{pos}" for pos in LIB_INFO_DICT[self.lib_name]["positions"].values()]

    @property
    def df(self) -> pd.DataFrame:
        """
        A property for the dataframe and drop stop codons
        """
        return pd.read_csv(self._input_csv)

    @property
    def df_no_stop(self) -> pd.DataFrame:
        """
        A property for the dataframe and drop stop codons
        """
        return self.df[~self.df["AAs"].str.contains("\*")]

    @property
    def variants(self) -> list:
        """
        A AA sequence for the variants
        """
        return self.df_no_stop["AAs"].values.tolist()

    @property
    def mut_numb(self) -> int:
        """
        A property for the number of mutations
        """
        return len(self.df_no_stop)


class TriadGenMutFile(TriadLib):
    """
    A class for generating a mut file for triad
    """

    def __init__(self, input_csv: str, triad_folder: str = "triad") -> None:

        """
        Args:
        - input_csv: str, the path to the input csv
        - output_folder: str, the path to the output folder
        """

        super().__init__(input_csv, triad_folder)

        print(f"Generating {self.mut_path} from {self._input_csv}...")
        self._mutation_encodings = self._generate_mut_file()

    def _generate_mut_file(self) -> None:
        """
        Generate the mut file
        """

        # Loop over variants
        mutation_encodings = []

        for variant in self.variants:

            # Loop over each character in the variant
            mut_encoding_list = []
            for j, (var_char, wt_char) in enumerate(zip(variant, self.wt_aas)):

                # If the var_char does not equal the wt_char, append
                if var_char != wt_char:
                    mut_encoding_list.append(self.prefixes[j] + var_char)

            # If the mut_encoding_list has no entries, continue (this is wild type)
            if len(mut_encoding_list) == 0:
                continue

            # Otherwise, append to mutation_encodings
            else:
                mutation_encodings.append("+".join(mut_encoding_list) + "\n")

        # check before saving
        assert len(mutation_encodings) == self.mut_numb - 1

        # Save the mutants
        with open(self.mut_path, "w") as f:
            f.writelines(mutation_encodings)

        return mutation_encodings

    @property
    def mut_path(self) -> str:
        """
        A property for the mut file path
        """
        sub_folder = checkNgen_folder(os.path.join(self._triad_folder, self.lib_name))
        return os.path.join(sub_folder, f"{self.lib_name}.mut")

    @property
    def mut_encoding(self) -> list:
        """
        A property for the mutation encodings
        """
        return self._mutation_encodings


class ParseTriadResults(TriadLib):
    """
    A class for parsing the triad results
    """

    def __init__(
        self,
        input_csv: str,
        triad_txt: str,
        triad_folder: str = "triad",
    ) -> None:

        """
        Args:
        - input_csv: str, the path to the input csv
        - triad_txt: str, the path to the triad txt file
        - triad_folder: str, the parent folder to all triad data
        """

        super().__init__(input_csv, triad_folder)

        self._triad_txt = triad_txt

        print(f"Parsing {self._triad_txt} and save to {self.triad_csv}...")

        # extract triad score into dataframe
        self._triad_df = self._get_triad_score()

        # save the triad dataframe
        self._triad_df.to_csv(self.triad_csv, index=False)

    def _get_triad_score(self) -> float:

        """
        A function to load the output of a triad analysis and get a score

        Args:
        - triad_output_file: str, the path to the triad output file
        - WT_combo: str, the wildtype combo
        - num_seqs: int, the number of sequences to load
        """

        # Load the output file
        with open(self._triad_txt) as f:

            # Set some flags for starting analysis
            solutions_started = False
            record_start = False

            # Begin looping over the file
            summary_lines = []
            for line in f:

                # Start looking at data once we hit "solution"
                # if "Solution" in line:
                if "All sequences:" in line:
                    solutions_started = True

                # Once we have "Index" we can start recording the rest
                if solutions_started and "Index" in line:
                    record_start = True

                # Record appropriate data
                if record_start:

                    # Strip the newline and split on whitespace
                    summary_line = line.strip().split()

                    if summary_line[0] == "Average":
                        break
                    else:
                        # Otherwise, append the line
                        summary_lines.append(summary_line)

        # Build the dataframe with col ['Index', 'Tags', 'Score', 'Seq', 'Muts']
        all_results = pd.DataFrame(summary_lines[1:], columns=summary_lines[0])
        all_results["Triad_score"] = all_results["Score"].astype(float)

        wt_chars = self.wt_aas
        reconstructed_combos = [
            "".join(
                [char if char != "-" else wt_chars[i] for i, char in enumerate(seq)]
            )
            for seq in all_results.Seq.values
        ]
        all_results["AAs"] = reconstructed_combos

        # Get the order
        all_results["Triad_rank"] = np.arange(1, len(all_results) + 1)

        return all_results[["AAs", "Triad_score", "Triad_rank"]]

    @property
    def triad_csv(self) -> str:
        """
        A property for the triad csv
        """
        return os.path.join(self._triad_folder, self.lib_name, f"{self.lib_name}.csv")

    @property
    def triad_df(self) -> pd.DataFrame:
        """
        A property for the triad dataframe
        """
        return self._triad_df


def run_traid_gen_mut_file():
    """Run the triad gen mut file function for all libraries"""
    for lib in glob("data/*/scale2max/*.csv"):
        TriadGenMutFile(input_csv=lib)


def run_parse_triad_results(triad_folder: str = "triad"):

    """
    Run the parse triad results function for all libraries
    Currently TrpB only
    """

    for lib, triad_txt in sorted_lib_triad_pair.items():
        ParseTriadResults(input_csv=lib, triad_txt=triad_txt, triad_folder=triad_folder)

    # need to merge
    trpb4_df = []
    for lib, triad_txt in TrpB4_lib_triad_pair.items():
        trpb4_df.append(
            ParseTriadResults(
                input_csv=lib, triad_txt=triad_txt, triad_folder=triad_folder
            ).triad_df
        )

    pd.concat(trpb4_df).to_csv(
        os.path.join(triad_folder, "TrpB4", "TrpB4.csv"), index=False
    )