"""Test ems inverse folding zs"""

import sys
import os


from datetime import datetime

# from SSMuLA.zs_analysis import run_zs_analysis
from SSMuLA.zs_data import get_all_mutfasta
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

    get_all_mutfasta(
        ev_esm_dir="ev_esm2",
        all_libs=False,
        lib_list=["T7"],
    )

    f.close()

    """
    get_all_mutfasta(
        ev_esm_dir: str = "ev_esm2", 
        all_libs: bool = True,
        lib_list: list[str] = []
        )
    """