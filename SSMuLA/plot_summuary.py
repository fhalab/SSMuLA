"""A script for plotting summary plots"""

import pandas as pd
import matplotlib.pyplot as plt

from SSMuLA.landscape_global import LIB_INFO_DICT
from SSMuLA.vis import LIB_COLORS
from SSMuLA.util import ecdf_transform, checkNgen_folder, save_plt


ten_colors = list(LIB_COLORS.values())

# combine all in one
# Create the figure and subplots

plot_folder = checkNgen_folder("results/de_vs_mlde/onehot/collage/n_samples")
mlde_all = pd.read_csv("results/mlde/vis_3/all_df.csv").copy()

for zs, mlde_title in zip(
    ["Triad_score", "esm_score", "ev_score", "none"],
    ["Triad-ftMLDE", "ESM-ftMLDE", "EVmutation-ftMLDE", "MLDE"]
):

    sup_title = f"{mlde_title} vs DE"

    fig, axs = plt.subplots(3, 4, figsize=(28, 16))

    for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):

        ss_df_all = pd.read_csv(
            f"results/simulations/DE-active/scale2max/{lib}-single_step_DE.csv"
        )
        recomb_df_all = pd.read_csv(
            f"results/simulations/DE-active/scale2max/{lib}-recomb_SSM.csv"
        )

        ss_df = ss_df_all.copy()
        recomb_df = recomb_df_all.copy()

        mlde_df = mlde_all[
            (mlde_all["lib"] == lib)
            & (mlde_all["n_mut_cutoff"] == "all")
            & (mlde_all["zs"] == zs)
            & (mlde_all["encoding"] == "one-hot")
        ].copy()

        ax.plot(
            ss_df["final_fitness"],
            ecdf_transform(ss_df["final_fitness"]),
            ".",
            label="DE - single-step",
            color=ten_colors[0],
        )
        ax.plot(
            recomb_df["final_fitness"],
            ecdf_transform(recomb_df["final_fitness"]),
            ".",
            label="DE - recombination",
            color=ten_colors[1],
        )

        for n, n_samples in enumerate([24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]):
            # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
            mlde_df_n = mlde_df[mlde_df["n_sample"] == n_samples]["top_maxes"]
            ax.plot(
                mlde_df_n,
                ecdf_transform(mlde_df_n),
                ".",
                label=f"{mlde_title} - {str(n_samples)}",
                color=ten_colors[n + 2],
            )

        if i == 3:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        ax.set_title(lib)
        ax.set_xlabel("Max fitness achieved")
        ax.set_ylabel("ECDF")

    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )