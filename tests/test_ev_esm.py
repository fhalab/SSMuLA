"""Test ems inverse folding zs"""

import sys
import os


from datetime import datetime

# from SSMuLA.zs_analysis import run_zs_analysis
from SSMuLA.zs_calc import calc_all_zs
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

    calc_all_zs(
        landscape_folder="data",
        output_folder="results/zs",
        ev_model_folder="data",
        regen_esm = False,
        rerun_zs = False
    )

    f.close()

    """
    calc_all_zs(
        landscape_folder: str = "data/processed",
        dataset_list: list[str] = [],
        output_folder: str = "results/zs",
        zs_model_names: str = "all",
        ev_model_folder: str = "data/evmodels",
        regen_esm: str = False,
        rerun_zs: str = False,
        )
    """