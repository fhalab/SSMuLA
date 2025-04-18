"""Test the preprocess module."""

import sys
import os

from datetime import datetime

from SSMuLA.fitness_process_vis import process_all, get_all_lib_stats
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/fitness_process_vis")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    process_all(scale_fit="max")
    get_all_lib_stats()

    f.close()