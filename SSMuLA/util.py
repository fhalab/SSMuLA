"""Util functions"""

from __future__ import annotations

import os
import sys
import pickle
import random

import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score


def checkNgen_folder(folder_path: str) -> str:

    """
    Check if the folder and its subfolder exists
    create a new directory if not
    Args:
    - folder_path: str, the folder path
    """

    # if input path is file, get the folder path
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