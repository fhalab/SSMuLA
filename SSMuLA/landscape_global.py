"""
This file contains variables specifically to each dataset
"""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import pandas as pd

from SSMuLA.util import get_file_name

DEFALT_SPLITS = ["single", "double", "multi"]

ACTIVE_THRESH_DICT ={
    "DHFR": -0.5,
    "ParD": 0,
    "GB1" : 0.01,
    "TrpB": None
}

COMBO_COLS = ["AAs", "fitness", "active"]

ParD_names = ["ParD2", "ParD3"]

# the number for which the PDB file has as the first amino acid
# no M included while in seq M is added to be the first amino acid
TrpB_PDB_START_AA_IND = 2

TrpB_names = deepcopy([*["TrpB3" + string for string in [chr(ord("A") + i) for i in range(9)]], "TrpB4"])

LIB_NAMES = deepcopy(["DHFR", *ParD_names, "GB1", *TrpB_names])

# Dictionary with positions for mutation, parent codons, and parent amino acids
LIB_INFO_DICT = deepcopy({
    "DHFR": {
        "positions": {1: 26, 2: 27, 3: 28},
        "codons": {1: "GCC", 2: "GAT", 3: "CTC"}, 
        "AAs": {1: "A", 2: "D", 3: "L"}
        },
    "ParD2": {
        "positions": {1: 61, 2: 64, 3: 80},
        "codons": {1: "", 2: "", 3: ""}, 
        "AAs": {1: "I", 2: "L", 3: "K"}
    },
    "ParD3": {
        "positions": {1: 61, 2: 64, 3: 80},
        "codons": {1: "", 2: "", 3: ""}, 
        "AAs": {1: "D", 2: "K", 3: "E"}
    },
    "GB1": {
        "positions": {1: 39, 2: 40, 3: 41, 4: 54},
        "codons": {1: "", 2: "", 3: "", 4: ""}, 
        "AAs": {1: "V", 2: "D", 3: "G", 4: "V"}
        },    
    "TrpB3A": {
        "positions": {1: 104, 2: 105, 3: 106},
        "codons": {1: "GCT", 2: "GAA", 3: "ACG"}, 
        "AAs": {1: "A", 2: "E", 3: "T"}
        },
    "TrpB3B": {
        "positions": {1: 105, 2: 106, 3: 107},
        "codons": {1: "GAA", 2: "ACG", 3: "GGT"}, 
        "AAs": {1: "E", 2: "T", 3: "G"}
    },
    "TrpB3C": {
        "positions": {1: 106, 2: 107, 3: 108},
        "codons": {1: "ACG", 2: "GGT", 3: "GCT"}, 
        "AAs": {1: "T", 2: "G", 3: "A"}
    },
    "TrpB3D": {
        "positions": {1: 117, 2: 118, 3: 119},
        "codons": {1: "ACC", 2: "GCA", 3: "GCA"}, 
        "AAs": {1: "T", 2: "A", 3: "A"}
        },
    "TrpB3E": {
        "positions": {1: 184, 2: 185, 3: 186},
        "codons": {1: "TTC", 2: "GGC", 3: "TCT"}, 
        "AAs": {1: "F", 2: "G", 3: "S"}
    },
    "TrpB3F": {
        "positions": {1: 162, 2: 166, 3: 301},
        "codons": {1: "CTG", 2: "ATT", 3: "TAC"}, 
        "AAs": {1: "L", 2: "I", 3: "Y"}
    },
    "TrpB3G": {
        "positions": {1: 227, 2: 228, 3: 301},
        "codons": {1: "GTG", 2: "AGC", 3: "TAC"}, 
        "AAs": {1: "V", 2: "S", 3: "Y"}
    },
    "TrpB3H": {
        "positions": {1: 228, 2: 230, 3: 231},
        "codons": {1: "AGC", 2: "GGT", 3: "TCT"}, 
        "AAs": {1: "S", 2: "G", 3: "S"}
    },
    "TrpB3I": {
        "positions": {1: 182, 2: 183, 3: 184},
        "codons": {1: "TAC", 2: "GTG", 3: "TTC"}, 
        "AAs": {1: "Y", 2: "V", 3: "F"}
    },
    "TrpB4": {
        "positions": {1: 183, 2: 184, 3: 227, 4: 228},
        "codons": {1: "GTG", 2: "TTC", 3: "GTG", 4: "AGC"},
        "AAs": {1: "V", 2: "F", 3: "V", 4: "S"}
    }
})


LIB_POS_0_IDX = deepcopy({
    lib: {
        str(int(pos_number) - 1): pos_loc
        for pos_number, pos_loc in dets["positions"].items()
    }
    for lib, dets in LIB_INFO_DICT.items()
})


def map_lib_pos() -> dict:
    """
    Map the position of the mutation to the position of the mutation in the library

    Returns:
    - dict: dictionary with the library and the position of the mutation
        ie. 'DHFR': {1: '26/27', 2: '26/28', 12: '27/28'},
    """
    lib_pos = {}
    for lib, pos in LIB_POS_0_IDX.items():
        lib_pos[lib] = {}
        for k, v in zip(
            [int("".join(p)) for p in list(combinations(pos.keys(), 2))],
            ["/".join(p) for p in list(combinations(list(map(str, pos.values())), 2))],
        ):
            lib_pos[lib][k] = v

    return lib_pos


LIB_POS_MAP = deepcopy(map_lib_pos())


PARENT_COMBO_DICT = deepcopy({lib: "".join(list(dets["AAs"].values())) for lib, dets in LIB_INFO_DICT.items()})


# Dictionary encoding the ODs over time for libraries DEFGHI
OD_DICT = {
    'TrpB3A': {
        'rep_1': {
            0: 0.1,
            18: 0.72,
            20: 0.78,
            24: 0.94,
            44: 2.55
        },
        'rep_2': {
            0: 0.1,
            18: 0.75,
            20: 0.83,
            24: 1.01,
            44: 2.70
        },
    },
    'TrpB3B': {
        'rep_1': {
            0: 0.1,
            18: 0.75,
            20: 0.83,
            24: 1.09,
            44: 3.30
        },
        'rep_2': {
            0: 0.1,
            18: 0.84,
            20: 0.98,
            24: 1.50,
            44: 3.85
        },
    },
    'TrpB3C': {
        'rep_1': {
            0: 0.1,
            18: 0.74,
            20: 0.78,
            24: 0.86,
            44: 1.95
        },
        'rep_2': {
            0: 0.1,
            18: 0.76,
            20: 0.84,
            24: 0.92,
            44: 4.15
        },
    },
    'TrpB3D': {
        'rep_1': {
            0: 0.05,
            12: 0.19,
            16: 0.29,
            20: 0.51,
            24: 0.85,
            36: 1.42
        },
        'rep_2': {
            0: 0.05,
            12: 0.18,
            16: 0.28,
            20: 0.49,
            24: 0.97,
            36: 1.81
        }
    },
    'TrpB3E': {
        'rep_1': {
            0: 0.05,
            12: 0.2,
            16: 0.27,
            20: 0.47,
            24: 0.91,
            36: 1.41
        },
        'rep_2': {
            0: 0.05,
            12: 0.2,
            16: 0.26,
            20: 0.44,
            24: 0.94,
            36: 1.54
        }
    },
    'TrpB3F': {
        'rep_1': {
            0: 0.05,
            12: 0.17,
            16: 0.20,
            20: 0.23,
            24: 0.27,
            36: 0.79
        },
        'rep_2': {
            0: 0.05,
            12: 0.17,
            16: 0.20,
            20: 0.24,
            24: 0.27,
            36: 0.79
        }
    },
    'TrpB3G': {
        'rep_1': {
            0: 0.05,
            12: 0.14,
            16: 0.18,
            20: 0.23,
            24: 0.44,
            36: 1.95
        },
        'rep_2': {
            0: 0.05,
            12: 0.14,
            16: 0.18,
            20: 0.23,
            24: 0.44,
            36: 1.95
        }
    },
    'TrpB3H': {
        'rep_1': {
            0: 0.05,
            12: 0.15,
            16: 0.19,
            20: 0.26,
            24: 0.67,
            36: 2.90
        },
        'rep_2': {
            0: 0.05,
            12: 0.14,
            16: 0.18,
            20: 0.26,
            24: 0.58,
            36: 1.85
        }
    },
    'TrpB3I': {
        'rep_1': {
            0: 0.05,
            12: 0.36,
            16: 0.83,
            20: 1.24,
            24: 0.7,
            36: 1.95
        },
        'rep_2': {
            0: 0.05,
            12: 0.39,
            16: 0.87,
            20: 1.36,
            24: 2.1,
            36: 2.25
        }
    },
    'TrpB4': {
        'rep_1': {
            0: 0.025,
            12: 0.19,
            16: 0.51,
            20: 1.26,
            24: 1.50,
            28: 1.675,
            36: 1.75
        },
        'rep_2': {
            0: 0.025,
            12: 0.19,
            16: 0.52,
            20: 1.34,
            24: 1.625,
            28: 1.75,
            36: 1.875

        }
    },
}

TIMEPOINT_DICT = {
    'TrpB3A': {
        'T0': 0,
        'T1': 18,
        'T2': 20,
        'T3': 24,
        'T4': 44
    },
    'TrpB3B': {
        'T0': 0,
        'T1': 18,
        'T2': 20,
        'T3': 24,
        'T4': 44
    },
    'TrpB3C': {
        'T0': 0,
        'T1': 18,
        'T2': 20,
        'T3': 24,
        'T4': 44
    },
    'TrpB3D': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'TrpB3E': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'TrpB3F': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'TrpB3G': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'TrpB3H': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'TrpB3I': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 36
    },
    'TrpB4': {
        'T0': 0,
        'T1': 12,
        'T2': 16,
        'T3': 20,
        'T4': 24,
        'T5': 28,
        'T6': 36
    }
}

n_mut_cutoff_dict = {0: "all", 1: "single", 2: "double", 3: "triple", 4: "quadruple"}

def append_active_cutoff(
    df: pd.DataFrame, fitness_cols: list = ["fitness"], def_cutoff: float | None = None
) -> tuple[pd.DataFrame, list[float]]:

    """
    Calculate the cutoff for active mutants based on
    1.96 standard deviations above the mean fitness of all stop-codon-containing sequences

    Args:
    - df, pd.DataFrame: input dataframe
    - fitness_cols, list: fitness columns

    Returns:
    - pd.DataFrame: input dataframe with active column
    - list[float]: cutoff value for each fitness column
    """
    
    if def_cutoff is None:
        fit_cutoffs = [None] * len(fitness_cols)

        stop_df = df[df["AAs"].str.contains("\*")]

        for i, c in enumerate(fitness_cols):
            avg_stop = stop_df[c].mean()
            std_stop = stop_df[c].std()
            fit_cutoffs[i] = 1.96 * std_stop + avg_stop
    else:
        fit_cutoffs = [def_cutoff] * len(fitness_cols)

    # Generate a list of lambda functions
    lambda_funcs = [
        lambda row, c=c, fit_min=fit_min: row[c] > fit_min
        for c, fit_min in zip(fitness_cols, fit_cutoffs)
    ]

    # Apply the lambda functions to the DataFrame
    df["active"] = df.apply(lambda row: all(func(row) for func in lambda_funcs), axis=1)

    return df, fit_cutoffs


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


class LibData:
    """
    A parent class to get the library information
    """

    def __init__(self, input_csv: str, scale_fit: str) -> None:
        """
        Args:
        - input_csv, str: path to the input csv file,
            ie. data/DHFR/fitness_landscape/DHFR.csv for preprocessed
                data/DHFR/scale2max/DHFR.csv for scaled to max = 1
        """
        
        self._input_csv = input_csv
        self._scale_fit = scale_fit

    @property
    def lib_name(self) -> dict:
        """Return the library name"""
        return get_file_name(self._input_csv)

    @property
    def lib_info(self) -> dict:
        """Return the library information"""
        return LIB_INFO_DICT[self.lib_name]
    
    @property
    def protein_name(self) -> str:
        """
        Returns the protein name
        """
        if "TrpB" in self.lib_name:
            return "TrpB"
        elif "ParD" in self.lib_name:
            return "ParD"
        else:
            return self.lib_name

    @property
    def parent_aa(self) -> str:
        """Return the parent amino acid"""
        return "".join(list(self.lib_info["AAs"].values()))

    @property
    def parent_codon(self) -> float:
        """Return the parent codon"""
        return "".join(list(self.lib_info["codons"].values()))
    
    @property
    def n_site(self) -> int:
        """Return the number of sites"""
        return len(self.lib_info["positions"])

    @property
    def scale_type(self) -> str:
        """Return the scale type"""
        if self._scale_fit in ["max", "parent"]:
            return f"scale2{self._scale_fit}"
        else:
            return "processed"
    
    @property
    def input_df(self) -> pd.DataFrame:
        """Return the input dataframe"""
        return pd.read_csv(self._input_csv)
    
    @property
    def df_length(self):
        return len(self.input_df)

    @property
    def split_aa_cols(self) -> list:
        """Return the columns for the split amino acids"""
        return [f"AA{str(i)}" for i in self.lib_info["positions"].keys()]



def lib2prot(lib_name: str) -> str:
    """
    Return the protein name from the library name

    Args:
    - lib_name, str: the library name

    Returns:
    - str: the protein name
    """
    if "TrpB" in lib_name:
        return "TrpB"
    else:
        return lib_name