"""Test the triad pre and post processing."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.triad_prepost import run_traid_gen_mut_file
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/triad/pre"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    run_traid_gen_mut_file(lib_list = ["T7", "TEV"])

    f.close()