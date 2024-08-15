"""
A script to test the pairwise epistasis calculation
"""

from __future__ import annotations

import sys
import os

from datetime import datetime


from SSMuLA.pairwise_epistasis import calc_all_pairwise_epistasis, plot_pairwise_epistasis
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/pairwise_epistasis")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f
    
    fitness_process_type = "scale2max"

    calc_all_pairwise_epistasis(
        fitness_process_type=fitness_process_type,
        ifall=False,
        lib_list=["T7", "TEV"],
        output_folder="results/pairwise_epistasis",
        n_jobs=128,
    )

    """
    calc_all_pairwise_epistasis(
    input_folder: str = "data",
    fitness_process_type: str = "scale2max",
    activestart: bool = True,
    ifall: bool = True,
    lib_list: list[str] = [],
    output_folder: str = "results/pairwise_epistasis",
    n_jobs: int = 128,
    """
    
    pos_calc_filter_min = "none"

    plot_pairwise_epistasis(
        fitness_process_type=fitness_process_type,
        pos_calc_filter_min=pos_calc_filter_min,
        input_folder="results/pairwise_epistasis",
        output_folder="results/pairwise_epistasis_vis",
        dets_folder="results/pairwise_epistasis_dets",
    )

    f.close()