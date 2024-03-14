"""Test MLDE."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.mlde_lite import run_all_mlde
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/mlde"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    run_all_mlde(n_mut_cutoffs=[1, 2, 0])

    f.close()