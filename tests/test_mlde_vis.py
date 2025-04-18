"""Test MLDE."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.mlde_analysis import MLDESum
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


    # MLDESum(
    #     mlde_results_dir = "results_rev/mlde_coves_ens/saved",
    #     mlde_vis_dir = "results_rev/mlde_coves_ens/vis"
    # )
    MLDESum(
        mlde_results_dir = "results_rev/mlde_lown/saved",
        mlde_vis_dir = "results_rev/mlde_lown/vis"
    )

    """
    MLDESum:

    def __init__(
        self,
        mlde_results_dir: str = "results/mlde/saved",
        mlde_vis_dir: str = "results/mlde/vis",
        all_encoding: bool = True,
        encoding_lists: list[str] = [],
        ifvis: bool = False,
    ) -> None:
    """
    f.close()