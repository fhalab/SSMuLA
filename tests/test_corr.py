"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.get_corr import MergeLandscapeAttributes, MergeMLDEAttributes, perfom_corr
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/corr")

    # log outputs
    f = open(
        os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"),
        "w",
    )
    sys.stdout = f

    # MergeMLDEAttributes(mlde_path = "results4upload/mlde/all_results.csv", merge_dir = "results/merged_2", models=["boosting"])

    # MergeLandscapeAttributes(
    #     lib_stat_path="results/fitness_distribution/max/all_lib_stats.csv",
    #     loc_opt_path="results/local_optima/scale2max.csv",
    #     pwe_path="results/pairwise_epistasis_vis/none/scale2max.csv",
    #     zs_path="results/zs_sum_5/none/zs_stat_scale2max.csv",
    #     de_path="results/de/DE-active/scale2max/all_landscape_de_summary.csv",
    #     merge_dir="results/merged",
    # )

    perfom_corr(
        n_mut_cutoff=0,
        n_list=[384],
        zs_path="results/zs_sum_6/none/zs_stat_scale2max.csv",
        mlde_path="results/mlde/all_df_comb_onehot_5.csv",
        corr_dir="results/corr_4",
        ifplot=False,
    )

    # perfom_corr(
    #     n_mut_cutoff=0,
    #     n_list=[384],
    #     zs_path="results/zs_sum_5/none/zs_stat_scale2max.csv",
    #     mlde_path="results/mlde/all_df_comb_onehot_2.csv",
    #     corr_dir="results/corr_3",
    #     ifplot=False,
    # )

    # perfom_corr(
    #     n_mut_cutoff=0,
    #     n_list=[384],
    #     filter_active=0,
    #     zs_path="results/zs_sum_5/none/zs_stat_scale2max.csv",
    #     mlde_path="results/mlde/all_df_comb_onehot_2.csv",
    #     corr_dir="results/corr_3",
    #     ifplot=False,
    # )

    # perfom_corr(
    #     n_mut_cutoff=2,
    #     n_list=[384],
    #     zs_path="results/zs_sum_5/none/zs_stat_scale2max.csv",
    #     mlde_path="results/mlde/all_df_comb_onehot_2.csv",
    #     corr_dir="results/corr_3",
    #     ifplot=False,
    # )

    """

    MergeLandscapeAttributes:

    def __init__(
        self,
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum_5/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        merge_dir: str = "results/merged",
    )

    MergeMLDEAttributes(MergeLandscapeAttributes):

    def __init__(
        self,
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/all_df_comb_onehot_2.csv",
        merge_dir: str = "results/merged",
        n_mut_cutoff: int = 0,
        n_sample: int = 384,
        n_top: int = 96,
        filter_active: float = 1,
        ft_frac=0.125,
        models: list[str] = ["boosting", "ridge"],
        ifplot: bool = True,
    )

    perfom_corr(
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/all_df_comb_onehot_2.csv",
        corr_dir: str = "results/corr",
        n_mut_cutoff: int = 0,
        filter_active: float = 1,
        ft_frac: float = 0.125,
        n_top_list: list[int] = [96, 384],
        n_list: list[int] = N_SAMPLE_LIST,
        models_list: list[list[str]] = [["boosting", "ridge"], ["boosting"], ["ridge"]],
        ifplot: bool = True,
    )
    """

    f.close()