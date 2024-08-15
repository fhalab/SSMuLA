"""A script for testing plotting for de"""

import sys
import os

from datetime import datetime

from SSMuLA.de_simulations import run_all_lib_de_simulations
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/de")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    run_all_lib_de_simulations(
        scale_types = ["scale2max"],
        de_opts = ["DE-active", "DE-0", "DE-all"],
        all_lib = False,
        lib_list = ["T7", "TEV"],
        rerun = False
    )

    """
    run_all_lib_de_simulations(
    scale_types: list = ["scale2max", "scale2parent"],
    de_opts: list = ["DE-active", "DE-0", "DE-all"],
    save_dir: str = "results/de",
    all_lib: bool = True,
    lib_list: list[str] = [],
    rerun: bool = False,
    )
    """
    
    f.close()