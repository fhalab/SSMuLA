"""
A script for handling the calculation of the hamming distance cutoff fitness
"""

import os
from glob import glob

import pandas as pd
import numpy as np

from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from SSMuLA.vis import LIB_COLORS, save_plt
from SSMuLA.landscape_global import LIB_INFO_DICT, hamming
from SSMuLA.util import checkNgen_folder, get_file_name

# Define the function that will be executed in parallel
def process_aa(aa, all_aas, all_fitnesses):
    hm2_fits = []
    for aa2, fitness in zip(all_aas, all_fitnesses):
        if hamming(aa, aa2) > 2:
            continue
        hm2_fits.append(fitness)
    return aa, np.mean(hm2_fits), np.std(hm2_fits)


def get_hd_avg_fit(
    df_csv: str,
    hd_dir: str = "results/hd",
    num_processes: None | int = None,
):

    df = pd.read_csv(df_csv)

    # no stop codons
    df = df[~df["AAs"].str.contains("\*")].copy()

    # only active variants
    active_df = df[df["active"]].copy()

    all_aas = active_df["AAs"].tolist()
    all_fitnesses = active_df.loc[active_df["AAs"].isin(all_aas), "fitness"].tolist()

    hm2_dict = {}
    # Set number of processes; if None, use all available cores
    if num_processes is None:
        num_processes = int(np.round(os.cpu_count() * 0.8))

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_aa, aa, all_aas, all_fitnesses) for aa in all_aas
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            aa, mean, std = future.result()
            hm2_dict[aa] = {"mean": mean, "std": std}

    mean_df = pd.DataFrame.from_dict(hm2_dict, orient="index")

    # Set the index name to 'aa'
    mean_df.index.name = "AAs"

    checkNgen_folder(hd_dir)
    mean_df.to_csv(os.path.join(hd_dir, get_file_name(df_csv) + ".csv"))

    return hm2_dict


def run_hd_avg_fit(
    data_dir: str = "data", 
    scalefit: str = "max", 
    hd_dir: str = "results/hd",
    num_processes: None | int = None,
    all_lib: bool = True,
    lib_list: list[str] = [],
):

    """
    Run the calculation of the average fitness for all sequences within a Hamming distance of 2

    Args:
    - data_dir: str, the directory containing the data
    - scalefit: str, the scale of the fitness values
    - hd_dir: str, the directory to save the results
    - num_processes: None | int, the number of processes to use
    - all_lib: bool, whether to use all libraries
    - lib_list: list[str], the list of libraries to use
    """

    if all_lib or len(lib_list) == 0:
        df_csvs = sorted(
            glob(f"{os.path.normpath(data_dir)}/*/scale2{scalefit}/*.csv")
        )
    else:
        df_csvs = [f"{os.path.normpath(data_dir)}/{lib}/scale2{scalefit}/{lib}.csv" for lib in lib_list]
    
    for df_csv in df_csvs:
        print(f"Processing {df_csv} ...")
        df = get_hd_avg_fit(df_csv, hd_dir)

        del df

    # You can now specify the number of processes when calling the main function


# For example, to use 4 processes:
# result_dict = main(df, num_processes=4)


def plot_all_hd2(hd_dir: str = "results/hd"):

    all_dfs = []
    parent_mean = {}

    for lib, lib_dict in LIB_INFO_DICT.items():

        df = pd.read_csv(os.path.join(hd_dir, lib + ".csv"))
        df["lib"] = lib
        all_dfs.append(df)

        parent_mean[lib] = df[df["AAs"] == "".join(lib_dict["AAs"].values())][
            "mean"
        ].values[0]

    all_df = pd.concat(all_dfs)

    fig = plt.figure(figsize=(16, 8))  # Adjust the size as needed
    ax = sns.violinplot(x="lib", y="mean", data=all_df, hue="lib", palette=LIB_COLORS)

    for lib in LIB_INFO_DICT.keys():

        # Find the position of the violin to add the line to
        position = all_df["lib"].unique().tolist().index(lib)
        ax.axhline(
            parent_mean[lib],
            color=LIB_COLORS[lib],
            linestyle="--",
            linewidth=2,
            xmin=position / len(LIB_INFO_DICT),
            xmax=(position + 1) / len(LIB_INFO_DICT),
            label=f"{lib} parent",
        )

    ax.set_title("Average fitness of HD2 with different backgrounds")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    save_plt(fig, plot_title="average_fitness_hd2", path2folder=hd_dir)


def correlate_hd2fit(aa, all_aas, all_fitnesses, all_ifactive):

    """
    A function to correlate the Hamming distance of a sequence
    with all other sequences with the fitness values
    """

    hms = [-1*hamming(aa, aa2) for aa2 in all_aas]
    rho = spearmanr(all_fitnesses, hms)[0]

    fpr, tpr, _ = roc_curve(all_ifactive, hms, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return aa, rho, roc_auc


def get_hd_avg_metric(
    df_csv: str,
    hd_dir: str = "results/hd_corr",
    num_processes: None | int = None,
):

    df = pd.read_csv(df_csv)

    # no stop codons
    df = df[~df["AAs"].str.contains("\*")].copy()

    all_aas = df["AAs"].tolist()
    all_fitnesses = df["fitness"].values
    all_ifactive = df["active"].values

    # only active variants
    active_df = df[df["active"]].copy()

    all_active_aas = active_df["AAs"].tolist()

    hm_dict = {}
    # Set number of processes; if None, use all available cores
    if num_processes is None:
        num_processes = int(np.round(os.cpu_count() * 0.8))

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(correlate_hd2fit, aa, all_aas, all_fitnesses, all_ifactive) for aa in all_active_aas
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            aa, rho, roc_auc = future.result()
            hm_dict[aa] = {"rho": rho, "roc_auc": roc_auc}

    hm_df = pd.DataFrame.from_dict(hm_dict, orient="index")

    # Set the index name to 'aa'
    hm_df.index.name = "AAs"

    checkNgen_folder(hd_dir)
    hm_df.to_csv(os.path.join(hd_dir, get_file_name(df_csv) + ".csv"))

    return hm_df["rho"].mean(), hm_df["roc_auc"].mean()


def run_hd_avg_metric(
    data_dir: str = "data", 
    scalefit: str = "max", 
    hd_dir: str = "results/hd_corr",
    num_processes: None | int = None,
    all_lib: bool = True,
    lib_list: list[str] = [],
):

    """
    Run the calculation of the average fitness for all sequences within a Hamming distance of 2

    Args:
    - data_dir: str, the directory containing the data
    - scalefit: str, the scale of the fitness values
    - hd_dir: str, the directory to save the results
    - num_processes: None | int, the number of processes to use
    - all_lib: bool, whether to use all libraries
    - lib_list: list[str], the list of libraries to use
    """

    hd_avg_metric = pd.DataFrame(columns=["lib", "rho", "roc_auc"])

    if all_lib or len(lib_list) == 0:
        df_csvs = sorted(
            glob(f"{os.path.normpath(data_dir)}/*/scale2{scalefit}/*.csv")
        )
    else:
        df_csvs = [f"{os.path.normpath(data_dir)}/{lib}/scale2{scalefit}/{lib}.csv" for lib in lib_list]
    
    for df_csv in df_csvs:
        print(f"Processing {df_csv} ...")
        rho, roc_aud = get_hd_avg_metric(df_csv, hd_dir)

        hd_avg_metric = hd_avg_metric._append(
            {
                "lib": get_file_name(df_csv),
                "rho": rho,
                "roc_auc": roc_aud,
            },
            ignore_index=True,
        )
    
    checkNgen_folder(hd_dir)
    hd_avg_metric.to_csv(os.path.join(hd_dir, "hd_avg_metric.csv"))

    return hd_avg_metric