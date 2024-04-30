"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.de_simulations import run_plot_de
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/plot_de")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    run_plot_de(
        scale_types = ["scale2max"],
        de_opts = ["DE-active"], # "DE-0"
        all_lib = False,
        lib_list = ["ParD2", "ParD3"],
    )

    """
    def run_plot_de(
        scale_types: list = ["scale2max", "scale2parent"],
        de_opts: list = ["DE-active"],
        sim_folder: str = "results/de",
        vis_folder: str = "results/de_vis",
        v_width: int = 400,
        all_lib: bool = True,
        lib_list: list[str] = [],
    ):
    """
    
    f.close()