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

    get_all_mutfasta()

    f.close()