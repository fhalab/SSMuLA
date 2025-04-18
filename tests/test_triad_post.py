"""Test the triad pre and post processing."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.triad_prepost import run_parse_triad_results
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/triad/post"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    run_parse_triad_results(all_lib = False, lib_list = ["T7", "TEV"])

    f.close()