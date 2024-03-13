"""A script to test the learned embedding generation"""


"""Test the triad pre and post processing."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.gen_learned_emb import gen_all_learned_emb
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/emb"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    gen_all_learned_emb()

    f.close()