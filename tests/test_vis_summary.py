"""Test the triad pre and post processing."""

import sys
import os

from datetime import datetime

from SSMuLA.vis_summary import DESumVis, ZSSSumVis, vis_sum_de_mlde
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

    # DESumVis(
    #     input_csv = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
    #     output_folder = "results/summary/DE")

    # ZSSSumVis(
    #     input_csv="results/zs_sum_2/min0/zs_stat_scale2max.csv",
    #     output_folder="results/summary/ZS_2"
    # )

    ZSSSumVis(
        input_csv="results/zs_sum_4/none/zs_stat_scale2max.csv",
        output_folder="results/summary/ZS_4"
    )

    # vis_sum_de_mlde(simplezs=True)
    # vis_sum_de_mlde(simplezs=False)
        
    """
    DESumVis(SumVis):
        input_csv: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        output_folder: str = ""

    ZSSSumVis(SumVis):
        input_csv="results/zs_sum/none/zs_stat_scale2max.csv",
        output_folder="",

    plot_de_v_mlde(
        plot_folder: str = "results/de_vs_mlde/onehot/collage/n_samples", 
        mlde_csv: str = "results/mlde/vis_3/all_df.csv",
        de_folder: str = "results/de/DE-active/scale2max",
    )

    plot_n_ftmlde(
        plot_folder: str = "results/de_vs_mlde/onehot/collage/ftMLDE",
        mlde_csv: str = "results/mlde/vis_3/all_df.csv",
        de_folder: str = "results/de/DE-active/scale2max",
    )
    """
    
    f.close()