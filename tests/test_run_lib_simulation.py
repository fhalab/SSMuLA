"""A script for running all simulations for each landscape."""

import os
import sys

from datetime import datetime

from SSMuLA.run_simulations import run_all_lib_de_simulations
from SSMuLA.util import checkNgen_folder



if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/de_simulations")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    for cutoff in ["active", "no_stop_codons", "all"]:
        run_all_lib_de_simulations(cutoff=cutoff)

    f.close()