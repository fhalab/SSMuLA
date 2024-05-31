"""
A script for handling the calculation of the hamming distance cutoff fitness
"""

import os
from glob import glob

import pandas as pd
import numpy as np
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


# Call main function with your DataFrame
# result_dict = main(df)


def get_hd_avg_fit(
    df_csv: str,
    hd_dir: str = "results/hd",
    num_processes: None | int = None,
):

    df = pd.read_csv(df_csv)

    # no stop codons
    df = df[~df["AAs"].str.contains("\*")].copy()

    all_aas = df["AAs"].tolist()
    all_fitnesses = df.loc[df["AAs"].isin(all_aas), "fitness"].tolist()

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
    data_dir: str = "data", scalefit: str = "max", num_processes: None | int = None
):
    for df_csv in sorted(
        glob(f"{os.path.normpath(data_dir)}/*/scale2{scalefit}/*.csv")
    ):
        print(f"Processing {df_csv} ...")
        df = get_hd_avg_fit(df_csv)

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