"""
A script for generating zs scores gratefully adapted from EmreGuersoy's work
"""

# Import packages
import os
from glob import glob

import numpy as np
import pandas as pd
import random
from collections import Counter


import argparse
from pathlib import Path

from tqdm import tqdm
from typing import List, Tuple, Optional

import warnings


from SSMuLA.zs_models import ZeroShotPrediction, ESM, EvMutation
from SSMuLA.zs_data import DataProcessor
from SSMuLA.landscape_global import lib2prot
from SSMuLA.util import get_file_name, checkNgen_folder


# TODO clean up evmutation model path


def calc_zs(
    fit_df_path: str,
    scalefit: str = "scale2max",
    output_folder: str = "results/zs",
    zs_model_names: str = "all",
    ev_model_folder: str = "data/evmodels",
    regen_esm: str = False,
    rerun_zs: str = False,
) -> pd.DataFrame:

    """
    A function for calculating zs scores and adding them to the fitness csv

    Args:
    - input_folder: str, input folder of landscape
        ie, 'data/DHFR,
        contains csv and fasta with wt seq
    - output_folder: str = "results/zs", with landscape subfolders
    - zs_model_names: str, name(s) of zero-shot models to use seperated by comma,
        available: 'esm', 'ev'
        developing: 'ddg', 'Bert'
        all: 'all' runs all currently available models
        ie, 'esm, ev'
    - ev_model_folder: str = "data/evmodels", folder for evmodels,
        with dataset name and dataset name.model
        ie. data/evmodels/AAV2_Bryant_2021/AAV2_Bryant_2021.model
    - regen_esm: str = False, if regenerate esm logits or load directly
    - rerun_zs: str = False, if append new zs to current csv or create new output
    """

    # deal with the /
    input_folder = os.path.normpath(fit_df_path.split(scalefit)[0])
    output_folder = os.path.normpath(output_folder)

    landscape_name = os.path.basename(input_folder)

    fasta_path = os.path.join(input_folder, landscape_name + ".fasta")

    ev_model_path = os.path.join(
        os.path.normpath(ev_model_folder), landscape_name + ".model"
    )

    # check if file exists
    assert os.path.exists(fit_df_path), f"{fit_df_path} does not exist"
    assert os.path.exists(fasta_path), f"{fasta_path} does not exist"

    output_folder = checkNgen_folder(output_folder)
    landscape_output_folder = checkNgen_folder(
        os.path.join(output_folder, landscape_name)
    )

    zs_df_path = os.path.join(landscape_output_folder, landscape_name + ".csv")

    # Create an instance of the DataProcessor class
    data_processor = DataProcessor()

    # Call the prepare_zero_shot method
    data = data_processor.prepare_zero_shot(
        fit_df_path, fasta_path, _combo=True, _pos=True
    )

    # init df
    existing_zs_df = data

    # check if exist
    if os.path.exists(zs_df_path):
        if rerun_zs:
            print(f"{zs_df_path} exists. Remove for rerun_zs = {rerun_zs}")
            os.remove(zs_df_path)
        else:
            print(
                f"{zs_df_path} exists. Append new zs {zs_model_names} for rerun_zs = {rerun_zs}"
            )
            existing_zs_df = pd.read_csv(zs_df_path)

    # Ref Sequence
    wt = data_processor.get_Seq(fasta_path)

    if zs_model_names == "all":
        zs_model_list = ["esm", "ev"]
    else:
        zs_model_list = zs_model_names.split(",")

    # init zs_df_list
    zs_df_list = []

    print(f"zs_model_list: {zs_model_list}")

    # TODO ESM load logits directly

    for zs_model_name in zs_model_list:

        # get max numb of muts
        max_numb_mut = max(data["combo"].str.len())

        # Access Model
        if "esm" in zs_model_name:
            esm = ESM(data, wt)
            logits_path = os.path.join(landscape_output_folder, landscape_name + "_logits.npy")

            if os.path.exists(logits_path) and not(regen_esm):
                print(f"{logits_path} exists and regen_esm = {regen_esm}. Loading...")
                log_reprs = np.load(logits_path)
            else:
                print(f"Generating {logits_path}...")
                log_reprs = esm._get_logits()
                np.save(logits_path, log_reprs)

            score_esm = esm._get_n_score(list(range(max_numb_mut+1))[1:])
            zs_df_list.append(score_esm)
            print(f"score_esm:\n{score_esm.head()}")

        elif "ev" in zs_model_name:
            if os.path.exists(ev_model_path):
                ev = EvMutation(data, wt, model_path=ev_model_path)
                score_ev = ev._get_n_score(list(range(max_numb_mut+1))[1:])
                zs_df_list.append(score_ev)
                print(f"score_ev:\n{score_ev.head()}")
            else:
                print(f"{ev_model_path} does not exist yet. Skipping...")

        # TODO add remaining zs_models

        elif zs_model_name == "ddg":
            pass
        else:
            print("Model currently not available")
            continue

    # Add muts from data to df
    print(f"zs_df_list:\n{zs_df_list}")

    for zs_df in zs_df_list:

        df = pd.merge(
            existing_zs_df,
            zs_df[list(zs_df.columns.difference(existing_zs_df.columns)) + ["muts"]],
            left_on="muts",
            right_on="muts",
            how="outer",
        )
        
        existing_zs_df = df

    return df.to_csv(zs_df_path, index=False)


# TODO FIX EV MODEL PATH
def calc_all_zs(
    landscape_folder: str = "data",
    scalefit = "scale2max",
    dataset_list: list[str] = [],
    output_folder: str = "results/zs",
    zs_model_names: str = "all",
    ev_model_folder: str = "data",
    regen_esm: str = False,
    rerun_zs: str = False,
):
    """
    A function for calc same list of zs scores for all landscape datasets

    Args:
    - landscape_folder: str = "data", folder path for all landscape data
    - dataset_list: list[str] = [], a list of encoders over write dataset_folder,
        ie. ['TrpB3I_Johnston_2023']
    - output_folder: str = "results/zs",
    - zs_model_names: str = "all",
    - ev_model_folder: str = "data/evmodels", folder for evmodels,
        with dataset name and dataset name.model
        ie. data/evmodels/AAV2_Bryant_2021/AAV2_Bryant_2021.model
    - regen_esm: str = False, if regenerate esm logits or load directly
    - rerun_zs: str = False, if append new zs to current csv or create new output
    """

    if len(dataset_list) == 0:
        landscape_paths = glob(os.path.normpath(landscape_folder) + "/*/" + scalefit + "/*.csv")
        print(f"Landscape paths: {landscape_paths}")
    else:
        landscape_paths = [
            os.path.join(landscape_folder, dataset) for dataset in dataset_list
        ]

    for landscape_path in landscape_paths:
        print(f"Calc zs {zs_model_names} for {landscape_path}...")
        _ = calc_zs(
            fit_df_path=landscape_path,
            scalefit=scalefit,
            output_folder=output_folder,
            zs_model_names=zs_model_names,
            ev_model_folder=ev_model_folder,
            regen_esm=regen_esm,
            rerun_zs=rerun_zs,
        )