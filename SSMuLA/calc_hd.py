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
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from SSMuLA.landscape_global import LIB_INFO_DICT, hamming, lib2prot
from SSMuLA.zs_analysis import (
    ZS_METRIC_MAP_TITLE,
    ZS_METRIC_MAP_LABEL,
    ZS_METRIC_BASELINE,
)
from SSMuLA.vis import LIB_COLORS, FZL_PALETTE, LIB_COLORS_GLASBEY, save_plt, save_svg
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
    hd_dir: str = "results/hd_fit",
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
        df_csvs = sorted(glob(f"{os.path.normpath(data_dir)}/*/scale2{scalefit}/*.csv"))
    else:
        df_csvs = [
            f"{os.path.normpath(data_dir)}/{lib}/scale2{scalefit}/{lib}.csv"
            for lib in lib_list
        ]

    for df_csv in df_csvs:
        print(f"Processing {df_csv} ...")
        df = get_hd_avg_fit(df_csv, hd_dir)

        del df


def correlate_hd2fit(aa, all_aas, all_fitnesses, all_ifactive):

    """
    A function to correlate the Hamming distance of a sequence
    with all other sequences with the fitness values
    """

    hms = [-1 * hamming(aa, aa2) for aa2 in all_aas]
    rho = spearmanr(all_fitnesses, hms)[0]

    fpr, tpr, _ = roc_curve(all_ifactive, hms, pos_label=1)
    rocauc = auc(fpr, tpr)

    return aa, rho, rocauc


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
            executor.submit(correlate_hd2fit, aa, all_aas, all_fitnesses, all_ifactive)
            for aa in all_active_aas
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            aa, rho, rocauc = future.result()
            hm_dict[aa] = {"rho": rho, "rocauc": rocauc}

    hm_df = pd.DataFrame.from_dict(hm_dict, orient="index")

    # Set the index name to 'aa'
    hm_df.index.name = "AAs"

    checkNgen_folder(hd_dir)
    hm_df.to_csv(os.path.join(hd_dir, get_file_name(df_csv) + ".csv"))

    return hm_df["rho"].mean(), hm_df["rocauc"].mean()


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

    hd_avg_metric = pd.DataFrame(columns=["lib", "rho", "rocauc"])

    if all_lib or len(lib_list) == 0:
        df_csvs = sorted(glob(f"{os.path.normpath(data_dir)}/*/scale2{scalefit}/*.csv"))
    else:
        df_csvs = [
            f"{os.path.normpath(data_dir)}/{lib}/scale2{scalefit}/{lib}.csv"
            for lib in lib_list
        ]

    for df_csv in df_csvs:
        print(f"Processing {df_csv} ...")
        rho, roc_aud = get_hd_avg_metric(df_csv, hd_dir)

        hd_avg_metric = hd_avg_metric._append(
            {
                "lib": get_file_name(df_csv),
                "rho": rho,
                "rocauc": roc_aud,
            },
            ignore_index=True,
        )

    checkNgen_folder(hd_dir)
    hd_avg_metric.to_csv(os.path.join(hd_dir, "hd_avg_metric.csv"))

    return hd_avg_metric


def plot_hd_avg_fit(
    figname: str,
    hd_fit_dir: str = "results4upload/hd/hd_fit",
    fit_dir: str = "data4upload",
    fitscale: str = "scale2max",
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    all_dfs = []
    wt_mean = {}
    full_mean = {}

    for lib, lib_dict in LIB_INFO_DICT.items():

        df = pd.read_csv(os.path.join(hd_fit_dir, f"{lib}.csv"))
        df["lib"] = lib
        all_dfs.append(df)

        wt_mean[lib] = df[df["AAs"] == "".join(lib_dict["AAs"].values())][
            "mean"
        ].values[0]

        fit_df = pd.read_csv(
            os.path.join(fit_dir, lib2prot(lib), fitscale, f"{lib}.csv")
        )
        full_mean[lib] = fit_df["fitness"].mean()

    all_df = pd.concat(all_dfs)

    fig = plt.figure(figsize=(16, 8))
    ax = sns.violinplot(
        x="lib", y="mean", data=all_df, hue="lib", palette=LIB_COLORS_GLASBEY
    )

    # Set the alpha value of the facecolor automatically to 0.8
    for violin in ax.collections[:]:  # Access only the violin bodies
        facecolor = violin.get_facecolor().flatten()  # Get the current facecolor
        violin.set_facecolor(mcolors.to_rgba(facecolor, alpha=0.4))  # Set new facecolor

    for lib in LIB_INFO_DICT.keys():

        # Find the position of the violin to add the line to
        position = all_df["lib"].unique().tolist().index(lib)

        # Overlay the mean as a scatter plot
        ax.axhline(
            all_df[all_df["lib"] == lib]["mean"].mean(),
            color=FZL_PALETTE["light_gray"],
            linestyle="solid",
            marker="x",
            linewidth=2,
            xmin=position / len(LIB_INFO_DICT) + 0.03125,
            xmax=(position + 1) / len(LIB_INFO_DICT) - 0.03125,
        )
        ax.axhline(
            wt_mean[lib],
            color=LIB_COLORS_GLASBEY[lib],
            linestyle="--",
            linewidth=2,
            xmin=position / len(LIB_INFO_DICT),
            xmax=(position + 1) / len(LIB_INFO_DICT),
        )
        ax.axhline(
            full_mean[lib],
            color=LIB_COLORS_GLASBEY[lib],
            linestyle="dotted",
            linewidth=2,
            xmin=position / len(LIB_INFO_DICT),
            xmax=(position + 1) / len(LIB_INFO_DICT),
        )

    lines = [
        Line2D(
            [0],
            [0],
            color=FZL_PALETTE["light_gray"],
            linestyle="none",
            lw=2,
            marker="x",
        ),
        Line2D([0], [0], color="black", linestyle="--", lw=2),
        Line2D([0], [0], color="black", linestyle="dotted", lw=2),
    ]
    labels = [
        "Mean of the mean variant fitness of double-site library\nconstructed with any active variant",
        "Mean variant fitness of double-site library\nconstruscted with the landscape parent",
        "Mean of all variants",
    ]

    ax.legend(lines, labels, loc="upper left", bbox_to_anchor=(1, 1))

    ax.set_xlabel("Landscapes")
    ax.set_ylabel(
        "Mean variant fitness of double-site library constructed with an active variant"
    )

    if ifsave:
        save_svg(fig, figname, fig_dir)


def plot_hd_corr(
    metric: str,
    figname: str,
    hd_corr_dir: str = "results4upload/hd/hd_corr",
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    all_dfs = []
    wt_mean = {}

    for lib, lib_dict in LIB_INFO_DICT.items():

        df = pd.read_csv(os.path.join(hd_corr_dir, f"{lib}.csv"))
        df["lib"] = lib
        all_dfs.append(df)

        wt_mean[lib] = df[df["AAs"] == "".join(lib_dict["AAs"].values())][
            metric
        ].values[0]

    all_df = pd.concat(all_dfs)

    fig = plt.figure(figsize=(16, 8))
    ax = sns.violinplot(
        x="lib", y=metric, data=all_df, hue="lib", palette=LIB_COLORS_GLASBEY
    )

    # Set the alpha value of the facecolor
    for violin in ax.collections[:]:  # Access only the violin bodies
        facecolor = violin.get_facecolor().flatten()  # Get the current facecolor
        violin.set_facecolor(mcolors.to_rgba(facecolor, alpha=0.4))  # Set new facecolor

    for lib in LIB_INFO_DICT.keys():

        # Find the position of the violin to add the line to
        position = all_df["lib"].unique().tolist().index(lib)

        # Overlay the mean
        ax.axhline(
            all_df[all_df["lib"] == lib][metric].mean(),
            color=FZL_PALETTE["light_gray"],
            linestyle="solid",
            marker="x",
            linewidth=2,
            xmin=position / len(LIB_INFO_DICT) + 0.03125,
            xmax=(position + 1) / len(LIB_INFO_DICT) - 0.03125,
        )
        ax.axhline(
            wt_mean[lib],
            color=LIB_COLORS_GLASBEY[lib],
            linestyle="--",
            linewidth=2,
            xmin=position / len(LIB_INFO_DICT),
            xmax=(position + 1) / len(LIB_INFO_DICT),
        )

    lines = [
        Line2D(
            [0],
            [0],
            color=FZL_PALETTE["light_gray"],
            linestyle="none",
            lw=2,
            marker="x",
        ),
        Line2D([0], [0], color="black", linestyle="--", lw=2),
        Line2D([0], [0], color="black", linestyle="dotted", lw=2),
    ]
    labels = [
        f"Mean {ZS_METRIC_MAP_LABEL[metric]}\nfrom any active variant",
        "From the landscape parent",
    ]
    ax.axhline(
        ZS_METRIC_BASELINE[metric],
        color=FZL_PALETTE["light_gray"],
        linestyle="dotted",
        linewidth=2,
    )
    ax.legend(lines, labels, loc="upper left", bbox_to_anchor=(1, 1))

    ax.set_xlabel("Landscapes")
    y_dets = (
        ZS_METRIC_MAP_TITLE[metric]
        .replace("\n", " ")
        .replace("F", "f")
        .replace("A", "a")
    )
    ax.set_ylabel(f"Hamming distance {y_dets}")

    if ifsave:
        save_svg(fig, figname, fig_dir)