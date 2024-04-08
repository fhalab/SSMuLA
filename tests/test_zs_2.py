"""Test the triad pre and post processing."""

import sys
import os

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

    run_zs_analysis(
        scale_types=["max"], 
        ev_esm_folder="zs", 
        filter_min_by="none", 
        zs_comb_dir = "results/zs_comb_2",
        zs_vis_dir = "results/zs_vis_2",
        zs_sum_dir = "results/zs_sum_2",
        )

    run_zs_analysis(
        scale_types=["max"], 
        ev_esm_folder="zs", 
        filter_min_by="min0", 
        zs_comb_dir = "results/zs_comb_2",
        zs_vis_dir = "results/zs_vis_2",
        zs_sum_dir = "results/zs_sum_2",
        )

    f.close()