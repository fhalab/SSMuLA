"""Test the triad pre and post processing."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.zs_analysis import run_zs_analysis
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/zs"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    run_zs_analysis(scale_types=["max"], filter_min_by="none", ev_esm_folder = "ev_esm2")
    run_zs_analysis(scale_types=["max"], filter_min_by="min0", ev_esm_folder = "ev_esm2")

    f.close()