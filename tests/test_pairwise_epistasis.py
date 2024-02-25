"""
A script to test the pairwise epistasis calculation
"""

from __future__ import annotations

import sys
import os

from datetime import datetime


from SSMuLA.pairwise_epistasis import run_pairwise_epistasis
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/pairwise_epistasis_calc")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    # run_pairwise_epistasis()
    run_pairwise_epistasis(fitness_process_type="scale2parent")
    
    f.close()