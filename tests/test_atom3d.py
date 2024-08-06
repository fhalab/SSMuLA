"""A script for testing plotting for atom3d"""

import sys
import os

from datetime import datetime

from SSMuLA.gen_atom3d import gen_all_lmdb
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    log_folder = checkNgen_folder("logs/atom3d")

    # log outputs
    f = open(os.path.join(log_folder, f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out"), 'w')
    sys.stdout = f

    gen_all_lmdb(pdb_pattern = "data/*/*.pdb", lmdb_dir = "lmdb")

    f.close()