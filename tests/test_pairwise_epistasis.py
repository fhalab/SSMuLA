"""
A script to test the pairwise epistasis calculation
"""

from __future__ import annotations

import sys
import os

from datetime import datetime


from SSMuLA.pairwise_epistasis import run_pairwise_epistasis, plot_pairwise_epistasis
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/pairwise_epistasis")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f
    # for filter_min_by in ["none", "0", "active_min"]:
    for filter_min_by in ["none"]:
        # for fitness_process_type in ["scale2max", "scale2parent"]:
        for fitness_process_type in ["scale2max"]:

            # run_pairwise_epistasis(
            #     input_folder="data",
            #     fitness_process_type=fitness_process_type,
            #     filter_min_by=filter_min_by,
            #     output_folder="results/pairwise_epistasis",
            #     n_jobs = 256,
            # )

            plot_pairwise_epistasis(
                fitness_process_type=fitness_process_type,
                filter_min_by=filter_min_by,
                input_folder="results/pairwise_epistasis",
                output_folder="results/pairwise_epistasis_vis",
            )

    f.close()