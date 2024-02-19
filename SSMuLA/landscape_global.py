"""
This file contains variables specifically to each dataset
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

DEFALT_SPLITS = ["single", "double", "multi"]

ACTIVE_THRESH_DICT ={
    "DHFR": -0.5,
    "GB1" : 0.01,
    "TrpB4": None
}

COMBO_COLS = ["AAs", "fitness", "active"]

# the number for which the PDB file has as the first amino acid
# no M included while in seq M is added to be the first amino acid
TrpB_PDB_START_AA_IND = 2

TrpB_names = deepcopy([*["TrpB3" + string for string in [chr(ord("A") + i) for i in range(9)]], "TrpB4"])

"""
TrpB3D_Johnston_2023_dict = {
    "mut_reg_seq": "TAA",  # seq
    0: "T117", # position of T
    1: "A118", # position of A
    2: "A119", # position of A
    # {117: "T", 118: "A", 119: "A"}
}

TrpB3G_Johnston_2023_dict = {
    "mut_reg_seq": "VSY",  # seq
    0: "V227", # position of V
    1: "S228", # position of S
    2: "Y301", # position of Y
    # {227: "V", 228: "S", 301: "Y"}
}

TrpB3H_Johnston_2023_dict = {
    "mut_reg_seq": "SGS",  # seq
    0: "S228", # position of S
    1: "G230", # position of G
    2: "S231", # position of S
    # {228: "S", 231: "S"}, # should be 228, 230, and 231
}

TrpB3I_Johnston_2023_dict = {
    "mut_reg_seq": "YVF",  # seq
    0: "Y182", # position of Y
    1: "V183", # position of V
    2: "F184", # position of F
    # {182: "Y", 183: "V", 184: "F"}
}

TrpB4_Johnston_2023_dict= {
    "mut_reg_seq": "VFVS",  # seq
    0: "V183", # position of V
    1: "F184", # position of F
    2: "V227", # position of V
    3: "S228", # position of S
    #(V183, F184, V227, S228)
}


GB1_dict = {
    "mut_reg_seq": "VDGV",  # seq
    0: "V39", # position of V
    1: "D40", # position of D
    2: "G41", # position of G
    3: "V54", # position of V
    # V39, D40, G41 and V54
}


"""
# Dictionary with positions for mutation, parent codons, and parent amino acids
LIB_INFO_DICT = deepcopy({
    "DHFR": {
        "positions": {1: 26, 2: 27, 3: 28},
        "codons": {1: "GCC", 2: "GAT", 3: "CTC"}, 
        "AAs": {1: "A", 2: "D", 3: "L"}
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

PARENT_COMBO_DICT = deepcopy({lib: "".join(list(dets["AAs"].values())) for lib, dets in LIB_INFO_DICT.items()})

def calc_active_cutoff(
    df: pd.DataFrame, fitness_cols: list = ["fitness"]
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

    fit_cutoffs = [None] * len(fitness_cols)

    stop_df = df[df["AAs"].str.contains("\*")]

    for i, c in enumerate(fitness_cols):
        avg_stop = stop_df[c].mean()
        std_stop = stop_df[c].std()
        fit_cutoffs[i] = 1.96 * std_stop + avg_stop

    # Generate a list of lambda functions
    lambda_funcs = [
        lambda row, c=c, fit_min=fit_min: row[c] > fit_min
        for c, fit_min in zip(fitness_cols, fit_cutoffs)
    ]

    # Apply the lambda functions to the DataFrame
    df["active"] = df.apply(lambda row: all(func(row) for func in lambda_funcs), axis=1)

    return df, fit_cutoffs