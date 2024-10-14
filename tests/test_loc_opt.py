"""A script for testing the local optima"""

import sys
import os

from datetime import datetime

from SSMuLA.landscape_optima import run_loc_opt
from SSMuLA.util import checkNgen_folder


if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/loc_opt")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    run_loc_opt(input_folder = "data4upload", 
                fitness_process_type = "scale2max",
                output_folder = "results/local_optima", 
                n_jobs = 16,
                if_append_escape = True,
                rerun = True,
                )
    
    f.close()

    """
    run_loc_opt(
        input_folder: str = "data",
        fitness_process_type: str = "scale2max",
        output_folder: str = "results/local_optima",
        n_jobs: int = 16,
        rerun: bool = False,
    ) -> None
    """


