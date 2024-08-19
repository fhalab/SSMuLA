"""
A script to test the learned embedding generation
Test the triad pre and post processing.
"""

import sys
import os

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

    gen_all_learned_emb(
        input_folder = "results/zs_comb_4/none/scale2max/all",
        all_libs = False,
        lib_list = ["T7", "TEV"],
    )

    f.close()

    """
    input_folder: str = "results/zs_comb/none/scale2max",
    encoder_name: str = DEFAULT_ESM,
    batch_size: int = 128,
    regen: bool = False,
    emb_folder: str = "learned_emb",
    all_libs: bool = True,
    lib_list: list[str] = [],
    """