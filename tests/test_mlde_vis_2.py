"""Test MLDE."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.mlde_analysis import MLDEVis
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/mlde_vis"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    # MLDEVis(
    #     mlde_results_dir = "results/mlde/saved",
    #     mlde_vis_dir = "results/mlde/vis"
    # )

    MLDEVis(
        mlde_results_dir = "results/mlde_old/saved",
        mlde_vis_dir = "results/mlde_old/vis"
    )

    f.close()