"""
A script for testing coves
Use coves environment
"""

import sys
import os
import re

from glob import glob

from datetime import datetime

from SSMuLA.run_coves import run_all_coves
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/coves")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    pattern = re.compile(r'^\d')
    print([s for s in glob("coves_data/*") if pattern.match(os.path.basename(s))])

    run_all_coves(patern=[s for s in glob("coves_data/*") if pattern.match(os.path.basename(s))], n_ave=100)

    f.close()