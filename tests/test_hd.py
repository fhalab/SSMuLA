"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.calc_hd import run_hd_avg_fit, plot_all_hd2
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/hd")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f
    
    # run_hd_avg_fit(num_processes=256)
    plot_all_hd2(hd_dir = "results/hd")

    """
    run_hd_avg_fit(data_dir: str = 'data', scalefit: str = "max", num_processes: None|int = None)
    """
    
    f.close()