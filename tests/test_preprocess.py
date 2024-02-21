"""Test the preprocess module."""

import sys
import os

from datetime import datetime

from SSMuLA.fitness_process_vis import ProcessDHFR, ProcessGB1, PlotTrpB
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/fitness_process_vis")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    for scale_fit in ["parent", "max"]:
        ProcessDHFR(scale_fit=scale_fit)
        ProcessGB1(scale_fit=scale_fit)
        PlotTrpB(scale_fit=scale_fit)

    f.close()