"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.get_factor import get_lib_stat, get_corr_heatmap, get_importance_heatmap
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/factor")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    # get_lib_stat()
    get_corr_heatmap()
    get_importance_heatmap()

    f.close()