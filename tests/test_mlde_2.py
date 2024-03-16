"""Test MLDE."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.mlde_lite import run_all_mlde
from SSMuLA.util import checkNgen_folder
from SSMuLA.aa_global import DEFAULT_LEARNED_EMB_COMBO

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/mlde"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    run_all_mlde(
        # zs_folder: str = "results/zs_comb",
        # filter_min_by: str = "none",
        # n_mut_cutoffs: list[int] = [0, 1, 2],
        # scale_type: str = "scale2max",
        # zs_predictors: list[str] = ["none", "Triad", "ev", "esm"],
        # ft_lib_fracs: list[float] = [0.5, 0.25, 0.125],
        # encodings: list[str] = ["one-hot"],
        # model_classes: list[str] = ["boosting", "ridge"],
        # n_samples: list[int] = [384],
        # n_split: int = 5,
        # n_replicate: int = 100,
        # n_tops: list[int] = [96, 384],
        # boosting_n_worker: int = 1,
        # global_seed: int = 42,
        # verbose: bool = False,
        # save_model: bool = False,
        # mlde_folder: str = "results/mlde",
        n_mut_cutoffs = [2],
        zs_predictors = [ "Triad", "ev", "esm"],
        encodings = DEFAULT_LEARNED_EMB_COMBO,
        n_replicate=5
    )

    f.close()