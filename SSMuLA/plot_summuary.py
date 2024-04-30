"""A script for plotting summary plots"""

import pandas as pd
import matplotlib.pyplot as plt

from SSMuLA.landscape_global import LIB_INFO_DICT
from SSMuLA.vis import LIB_COLORS, PRESENTATION_PALETTE_SATURATE
from SSMuLA.util import ecdf_transform, checkNgen_folder, save_plt


TEN_COLORS = list(LIB_COLORS.values())
N_SAMPLE_LIST = [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]


def plot_de_v_mlde(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_samples", 
    mlde_csv: str = "results/mlde/vis_3/all_df.csv",
    de_folder: str = "results/simulations/DE-active/scale2max",
    ) -> None:

    """
    A function for plotting DE vs MLDE in a collage plot
    loop over all ZS scores for all libraries in each

    Args:
    - plot_folder: str, path to the folder where the plots will be saved
    - mlde_csv: str, path to the MLDE csv file
    """

    plot_folder = checkNgen_folder(plot_folder)
    mlde_all = pd.read_csv(mlde_csv).copy()

    for zs, mlde_title in zip(
        ["Triad_score", "esm_score", "ev_score", "none"],
        ["Triad-ftMLDE", "ESM-ftMLDE", "EVmutation-ftMLDE", "MLDE"]
    ):

        sup_title = f"{mlde_title} vs DE"

        fig, axs = plt.subplots(3, 4, figsize=(28, 16))

        for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):

            ss_df_all = pd.read_csv(
                f"{de_folder}/{lib}-single_step_DE.csv"
            )
            recomb_df_all = pd.read_csv(
                f"{de_folder}/{lib}-recomb_SSM.csv"
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
                color=TEN_COLORS[0],
            )
            ax.plot(
                recomb_df["final_fitness"],
                ecdf_transform(recomb_df["final_fitness"]),
                ".",
                label="DE - recombination",
                color=TEN_COLORS[1],
            )

            for n, n_samples in enumerate(N_SAMPLE_LIST):
                # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                mlde_df_n = mlde_df[mlde_df["n_sample"] == n_samples]["top_maxes"]
                ax.plot(
                    mlde_df_n,
                    ecdf_transform(mlde_df_n),
                    ".",
                    label=f"{mlde_title} - {str(n_samples)}",
                    color=TEN_COLORS[n + 2],
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



def plot_n_ftmlde(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/ftMLDE",
    mlde_csv: str = "results/mlde/vis_3/all_df.csv",
    de_folder: str = "results/simulations/DE-active/scale2max",
) -> None:

    """
    
    """

    plot_folder = checkNgen_folder(plot_folder)
    mlde_all = pd.read_csv(mlde_csv).copy()

    for n in N_SAMPLE_LIST:

        sup_title = f"{str(n)} MLDE sample 12.5% ft library vs DE"

        fig, axs = plt.subplots(3, 4, figsize=(28, 16))

        for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):

            ss_df_all = pd.read_csv(
                f"{de_folder}/{lib}-single_step_DE.csv"
            )
            recomb_df_all = pd.read_csv(
                f"{de_folder}/{lib}-recomb_SSM.csv"
            )

            ss_df = ss_df_all.copy()
            recomb_df = recomb_df_all.copy()

            mlde_df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == "all")
                & (mlde_all["n_sample"] == n)
                & (mlde_all["encoding"] == "one-hot")
            ].copy()

            ax.plot(
                ss_df["final_fitness"],
                ecdf_transform(ss_df["final_fitness"]),
                ".",
                label="DE - single-step",
                color=TEN_COLORS[0],
            )
            ax.plot(
                recomb_df["final_fitness"],
                ecdf_transform(recomb_df["final_fitness"]),
                ".",
                label="DE - recombination",
                color=TEN_COLORS[1],
            )

            for zs_label, zs_color, zs in zip(
                ["MLDE", "Triad ftMLDE", "ESM ftMLDE", "EVmutation ftMLDE"],
                ["gray", "blue", "purple", "green"],
                ["none", "Triad_score", "esm_score", "ev_score"],
            ):  
                if zs == "none":
                    mlde_df_n = mlde_df[
                        (mlde_df["zs"] == zs)
                    ]["top_maxes"]

                else:
                    mlde_df_n = mlde_df[
                        (mlde_df["zs"] == zs) & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                    ]["top_maxes"]

                ax.plot(
                    mlde_df_n,
                    ecdf_transform(mlde_df_n),
                    ".",
                    label=zs_label,
                    color=PRESENTATION_PALETTE_SATURATE[zs_color],
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