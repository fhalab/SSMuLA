"""Util functions"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score

from Bio import SeqIO


def checkNgen_folder(folder_path: str) -> str:

    """
    Check if the folder and its subfolder exists
    create a new directory if not
    Args:
    - folder_path: str, the folder path
    """

    # if input path is file
    if bool(os.path.splitext(folder_path)[1]):
        folder_path = os.path.dirname(folder_path)

    split_list = os.path.normpath(folder_path).split("/")
    for p, _ in enumerate(split_list):
        subfolder_path = "/".join(split_list[: p + 1])
        if not os.path.exists(subfolder_path):
            print(f"Making {subfolder_path} ...")
            os.mkdir(subfolder_path)
    return folder_path


def pickle_save(what2save, where2save: str) -> None:

    """
    Save variable to a pickle file
    Args:
    - what2save, the varible that needs to be saved
    - where2save: str, the .pkl path for saving
    """

    with open(where2save, "wb") as f:
        pickle.dump(what2save, f)


def pickle_load(path2load: str):

    """
    Load pickle file
    Args:
    - path2load: str, the .pkl path for loading
    """

    with open(path2load, "rb") as f:
        return pickle.load(f)


def get_file_name(file_path: str) -> str:

    """
    Extract file name without the extension
    Args:
    - file_path: str, ie. data/graph_nx/Tm9D8s/Tm9D8s_3siteA_fixed/WT.pdb
    Returns:
    - str, ie WT
    """

    return os.path.splitext(os.path.basename(file_path))[0]


def get_dir_name(file_path: str) -> str:

    """
    Extract dir name
    Args:
    - file_path: str, ie. data/graph_nx/Tm9D8s/Tm9D8s_3siteA_fixed/WT.pdb
    Returns:
    - str, ie Tm9D8s_3siteA_fixed
    """

    return os.path.basename(os.path.dirname(file_path))


def get_dirNfile_name(file_path: str) -> [str, str]:

    """
    Extract file name without the extension and direct dir name
    Args:
    - file_path: str, ie. data/graph_nx/Tm9D8s/Tm9D8s_3siteA_fixed/WT.pdb
    Returns:
    - str, ie ['Tm9D8s_3siteA_fixed', 'WT']
    """

    return (
        os.path.basename(os.path.dirname(file_path)),
        os.path.splitext(os.path.basename(file_path))[0],
    )


def get_fulldirNfile_name(file_path: str) -> [str, str]:

    """
    Extract file name without the extension and full dir name
    Args:
    - file_path: str, ie. data/graph_nx/Tm9D8s/Tm9D8s_3siteA_fixed/WT.pdb
    Returns:
    - str, ie ['data/graph_nx/Tm9D8s/Tm9D8s_3siteA_fixed', 'WT']
    """

    return os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0]


def ndcg_scale(true: np.ndarray, pred: np.ndarray):
    """Calculate the ndcg_score with neg correction"""

    if min(true) < 0:
        true = true - min(true)
    return ndcg_score(true[None, :], pred[None, :])


def ecdf_transform(data: pd.Series) -> pd.Series:

    """
    Transform a series of fitness values into an empirical cumulative distribution function

    Args:
    - data: pd.Series, the fitness values

    Returns:
    - pd.Series, the ECDF
    """

    return data.rank(method="first") / len(data)



def csv2fasta(csv: str) -> None:
    """
    A function for converting a csv file to a fasta file
    ie /disk2/fli/SSMuLA/ev_esm2/DHFR/DHFR.csv

    """
    df = pd.read_csv(csv)

    for col in ["muts", "seq"]:
        if col not in df.columns:
            raise ValueError(f"{col} column not found")

    fasta = csv.replace(".csv", ".fasta")
    with open(fasta, "w") as f:
        for mut, seq in zip(df["muts"].values, df["seq"].values):
            f.write(f">{mut}\n{seq}\n")


def read_fasta(fasta_filename):
    """
    Reads a FASTA file using Biopython and returns the sequence of the first record.

    Args:
    - fasta_filename (str): The path to the FASTA file.

    Returns:
    - str: The sequence from the first record in the FASTA file.
    """
    with open(fasta_filename, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            return str(record.seq)
