"""Test the triad pre and post processing."""

import sys
import os

from datetime import datetime

from SSMuLA.vis_summary import *
# plot_de_mlde_ft_v_n
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/vis_summary"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    # DESumVis(
    #     input_csv = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
    #     output_folder = "results/summary/DE")

    # ZSSSumVis(
    #     input_csv="results/zs_sum_2/none/zs_stat_scale2max.csv",
    #     output_folder="results/summary/ZS"
    # )

    # plot_de_v_mlde(
    #     plot_folder = "results/de_vs_mlde/onehot/collage2_wcombzs/n_samples", 
    #     mlde_csv = "results/mlde/vis_4/all_df.csv",
    #     de_folder = "results/de/DE-active/scale2max",
    #     nsite = "double"
    # )

    # plot_n_ftmlde(
    #     plot_folder = "results/de_vs_mlde/onehot/collage2_wcombzs/ftMLDE",
    #     mlde_csv = "results/mlde/vis_4/all_df.csv",
    #     de_folder = "results/de/DE-active/scale2max",
    #     nsite = "double"
    # )

    # plot_de_mlde_ft_v_n(
    #     plot_folder = "results/de_vs_mlde/onehot/collage/n_mean_frac",
    #     mlde_csv = "results/mlde/vis_4/all_df.csv",
    #     de_folder = "results/de/DE-active/scale2max",
    #     nsite = "double"
    # )

    for simplezs in [True, False]:

        plot_de_mlde_ft_mean_v_n(
            plot_folder = "results/de_vs_mlde/onehot/collage/n_mean_frac",
            # mlde_csv: str = "results/mlde/vis_4/all_df.csv",
            # de_folder: str = "results/de/DE-active/scale2max",
            simplezs=simplezs,
            nsite = "double"
        )
        plot_de_mlde_ft_percent_v_n(
            plot_folder = "results/de_vs_mlde/onehot/collage/n_mean_frac",
            # mlde_csv: str = "results/mlde/vis_4/all_df.csv",
            # de_folder: str = "results/de/DE-active/scale2max",
            simplezs=simplezs,
            nsite = "double"
        )

    for meanorfrac in ["mean", "frac"]:
        for simplezs in [True, False]:
            for min_ftlib in [True, False]:
                plot_de_mlde_ft_meancount_v_n(
                    plot_folder = "results/de_vs_mlde/onehot/collage/n_mean_count",
                    mlde_csv = "results/mlde/vis_4/all_df.csv",
                    de_folder = "results/de/DE-active/scale2max",
                    meanorfrac = meanorfrac,
                    simplezs = simplezs,
                    min_ftlib = min_ftlib,
                    nsite = "double"
                )

    """
    DESumVis(SumVis):
        input_csv: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        output_folder: str = ""

    ZSSSumVis(SumVis):
        input_csv="results/zs_sum/none/zs_stat_scale2max.csv",
        output_folder="",

    plot_de_v_mlde(
        plot_folder: str = "results/de_vs_mlde/onehot/collage/n_samples", 
        mlde_csv: str = "results/mlde/vis_3/all_df.csv",
        de_folder: str = "results/de/DE-active/scale2max",
    )

    plot_n_ftmlde(
        plot_folder: str = "results/de_vs_mlde/onehot/collage/ftMLDE",
        mlde_csv: str = "results/mlde/vis_3/all_df.csv",
        de_folder: str = "results/de/DE-active/scale2max",
    )
    """
    
    f.close()