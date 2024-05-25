"""Test MLDE."""

import sys
import os

from datetime import datetime

from SSMuLA.mlde_analysis import MLDESum, comb_mlde_dfs
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

    # MLDESum(
    #     mlde_results_dir = "results/mlde_hm2zs/saved",
    #     mlde_vis_dir = "results/mlde_hm2zs/vis",
    #     all_encoding = True,
    #     # encoding_lists = ["one-hot"],
    # )

    comb_mlde_dfs(
        mlde_csv = "results/mlde/vis_5/all_df.csv",
        mlde_csv2 = "results/mlde_hm2zs/vis/all_df.csv", 
        save_path = "results/mlde/all_df_comb.csv",
        onehotonly = True
    )

    f.close()