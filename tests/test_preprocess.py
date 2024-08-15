"""Test the preprocess module."""

import sys
import os

from datetime import datetime

from SSMuLA.fitness_process_vis import process_all, ProcessT7, ProcessTEV, PlotParD, sum_ks, get_all_sda, get_all_lib_stats
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/fitness_process_vis")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    # process_all(scale_fit="max")
    ProcessT7(scale_fit="max")
    ProcessTEV(scale_fit="max")

    sum_ks(process_types = {
        # "fitness_landscape": "preprocessed",
        # "processed": "exp_log",
        "scale2max": "scaled_to_max",
        # "scale2parent": "scaled_to_parent",
    })
    
    get_all_sda()
    get_all_lib_stats()

    f.close()