"""Test the triad pre and post processing."""

import sys
import os

from datetime import datetime

from SSMuLA.vis_summary import DESumVis, ZSSSumVis
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/vis_summary"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    DESumVis(
        input_csv = "results/simulations/DE-active/scale2max/all_landscape_de_summary.csv",
        output_folder = "results/summary/DE")

    ZSSSumVis(
        input_csv="results/zs_sum/none/zs_stat_scale2max.csv",
        output_folder="results/summary/ZS"
    )

    """
    DESumVis(SumVis):

    def __init__(
        self,
        input_csv: str = "results/simulations/DE-active/scale2max/all_landscape_de_summary.csv",
        output_folder: str = ""

    ZSSSumVis(SumVis):

    def __init__(
        self,
        input_csv="results/zs_sum/none/zs_stat_scale2max.csv",
        output_folder="",
    """



    

    f.close()