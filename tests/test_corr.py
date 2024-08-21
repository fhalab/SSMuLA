"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.get_corr import perfom_corr
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/corr")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    perfom_corr(n_mut_cuttoff=0, n_list=[384], zs_path="results/zs_sum_4/none/zs_stat_scale2max.csv", mlde_path="results/mlde/all_df_comb_onehot_2.csv", corr_dir="results/corr_3", ifplot=False)
    perfom_corr(n_mut_cuttoff=2, n_list=[384], zs_path="results/zs_sum_4/none/zs_stat_scale2max.csv", mlde_path="results/mlde/all_df_comb_onehot_2.csv", corr_dir="results/corr_3", ifplot=False)

    """
    perfom_corr(
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/all_df_comb_onehot_2.csv",
        corr_dir: str = "results/corr",
        n_mut_cuttoff: int = 0,
        filter_active: float = 1,
        n_samples: list[int] = [384],
        n_top_list: list[int] = [96, 384],
        n_list: list[int] = N_SAMPLE_LIST,
        models_list: list[list[str]] = [["boosting"], ["ridge"], ["boosting", "ridge"]]),
        ifplot: bool = True,
    """
    
    f.close()