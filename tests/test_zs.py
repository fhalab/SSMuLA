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

    # run_zs_analysis(
    #     scale_types=["max"], 
    #     filter_min_by="none",
    #     ev_esm_folder = "ev_esm2",
    #     zs_comb_dir = "results/zs_comb_2",
    #     zs_vis_dir = "results/zs_vis_2",
    #     zs_sum_dir = "results/zs_sum_2",
    # )
    
    # run_zs_analysis(
    #     scale_types=["max"], 
    #     filter_min_by="min0", 
    #     ev_esm_folder = "ev_esm2",
    #     zs_comb_dir = "results/zs_comb_2",
    #     zs_vis_dir = "results/zs_vis_2",
    #     zs_sum_dir = "results/zs_sum_2",
    # )

    run_zs_analysis(
        scale_types=["max"], 
        filter_min_by="none", 
        ev_esm_folder = "ev_esm2",
        zs_comb_dir = "results/zs_comb_5",
        zs_vis_dir = "results/zs_vis_5",
        zs_sum_dir = "results/zs_sum_5",
    )
    
    f.close()

    """
    run_zs_analysis(
        scale_types: list = ["max", "parent"],
        data_folder: str = "data",
        ev_esm_folder: str = "ev_esm",
        triad_folder: str = "triad",
        esmif_folder: str = "esmif",
        filter_min_by: str = "none",
        n_mut_cutoff_list: list[int] = [0, 1, 2],
        zs_comb_dir: str = "results/zs_comb",
        zs_vis_dir: str = "results/zs_vis",
        zs_sum_dir: str = "results/zs_sum",
        )
    """