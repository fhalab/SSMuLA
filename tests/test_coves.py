"""
A script for testing coves
Use coves environment
"""

import sys
import os

from datetime import datetime

from SSMuLA.run_coves import run_all_coves
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/coves")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    run_all_coves(patern="coves_data/*/*", n_ave=100)

    f.close()