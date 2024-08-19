"""
A script for testing coves
Use coves environment
"""

import sys
import os

from datetime import datetime

from SSMuLA.run_coves import run_all_coves, append_all_coves_scores
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/coves")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    # run_all_coves(patern=["coves_data/DHFR_trans/DHFR_trans.pdb"], n_ave=100)
    append_all_coves_scores()

    f.close()

    """
    append_all_coves_scores(
    libs: list|str = "ev_esm2/*", 
    ev_esm_dir: str = "ev_esm2", 
    coves_dir: str = "coves/100", 
    t: float = 0.1
    """