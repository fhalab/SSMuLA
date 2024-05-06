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

    perfom_corr()

    """
    perfom_corr(
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/vis_3/all_df.csv",
        corr_dir: str = "results/corr",
        n_mut_cuttoff: int = 0,
        n_samples: list[int] = [384],
        n_top_list: list[int] = [96, 384],
        models_list: list[list[str]] = [["boosting"], ["ridge"], ["boosting", "ridge"]])
    """
    
    f.close()