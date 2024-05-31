"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.get_factor import get_lib_stat, get_corr_heatmap, get_importance_heatmap, get_zs_corr, get_zs_corr_ls, get_zs_zs_corr
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/factor")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    get_lib_stat(n_mut="double")

    # for metric in ["rho", "rocauc"]:
    #     get_zs_zs_corr(n_mut="double", metric=metric)
    #     get_zs_zs_corr(n_mut="all", metric=metric)

    # for metric in ["rho", "rocauc"]:
    #     get_zs_corr_ls(n_mut="double", metric=metric)
    #     get_zs_corr_ls(n_mut="all", metric=metric)

    # for de_calc in ["mean_all", "fraction_max"]:
        # get_zs_corr(de_calc=de_calc, n_mut="double", deorls="ls")
        # get_zs_corr(de_calc=de_calc, n_mut="all", deorls="ls")
        # get_corr_heatmap(de_calc=de_calc)
        # get_importance_heatmap(de_calc=de_calc)

    f.close()