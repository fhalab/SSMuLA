
"""Test alde file comb"""

import sys
import os


from datetime import datetime

from SSMuLA.alde_analysis import aggregate_alde_df
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/alde_comb"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    aggregate_alde_df(
        eq_ns = [2, 3, 4], 
        alde_dir = "results/alde",
        alde_df_path = "results/alde/alde_all.csv",

    )

    f.close()

    """
    aggregate_alde_df(
        eq_ns: list[int] = [1, 2, 3, 4],
        zs_opts: list[str] = ["esmif", "ev", "coves", "ed", "esm", "Triad", ""],
        alde_model: str = "Boosting Ensemble",
        alde_encoding: str = "onehot",
        alde_acq: str = "GREEDY",
        alde_dir: str = "/disk2/fli/alde4ssmula",
        alde_df_path: str = "results/alde/alde_all.csv",
    )
    """

