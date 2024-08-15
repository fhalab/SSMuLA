"""Test MLDE."""

import sys
import os

from datetime import datetime

from SSMuLA.mlde_lite import run_all_mlde_parallelized
from SSMuLA.aa_global import DEFAULT_LEARNED_EMB_COMBO
from SSMuLA.util import checkNgen_folder

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

    # zs_folder = "results/zs_comb_2"
    # mlde_folder = "results/mlde_emb"
    mlde_folder = "results/mlde"

    run_all_mlde_parallelized(
        encodings=["one-hot"],
        n_mut_cutoffs=[0, 2],
        zs_predictors=["none"],
        n_samples=[96, 384, 24, 48, 192, 288, 480, 576, 960, 1920],
        n_replicate=50,
        n_job=256,
        all_libs=False,
        ft_first=False,
        lib_list=["T7", "TEV"],
    )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2],
    #     zs_predictors=["none", "ev", "Triad", "esm", "esmif"],
    #     n_samples=[96, 384],
    #     n_replicate=50,
    #     n_job=256,
    #     ft_lib_fracs=[0.015625],
    #     all_libs=True,
    #     ft_first=False,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["none", "ev", "Triad", "esm", "esmif"],
    #     n_samples=[96, 384],
    #     n_replicate=50,
    #     n_job=256,
    #     ft_lib_fracs=[0.015625],
    #     all_libs=True,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )


    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["Triad-ev", "Triad-esm"],
    #     n_samples=[384],
    #     n_replicate=50,
    #     n_job=128,
    #     all_libs=True,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["Triad-ev", "Triad-esm"],
    #     n_samples=[24, 48, 96],
    #     n_replicate=50,
    #     n_job=128,
    #     all_libs=True,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["Triad-ev", "Triad-esm"],
    #     n_samples=[192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=128,
    #     all_libs=True,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["Triad-ev", "Triad-esm"],
    #     n_samples=[960, 1920],
    #     n_replicate=50,
    #     n_job=128,
    #     all_libs=True,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )
    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     # n_samples=[960, 1920],
    #     n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=96,
    #     all_libs=False,
    #     lib_list=["ParD2", "ParD3"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     # n_samples=[960, 1920],
    #     n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=96,
    #     all_libs=False,
    #     lib_list=["ParD2", "ParD3"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=96,
    #     all_libs=False,
    #     lib_list=["ParD2", "ParD3"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["esmif"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=96
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["esm"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=False,
    #     lib_list=["DHFR", "GB1", "TrpB3A", "TrpB3B", "TrpB3C", "TrpB3D", "TrpB3E", "TrpB3F", "TrpB3G", "TrpB3H", "TrpB3I", "TrpB4"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2, 1],
    #     zs_predictors=["esm"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=False,
    #     lib_list=["DHFR", "GB1", "TrpB3A", "TrpB3B", "TrpB3C", "TrpB3D", "TrpB3E", "TrpB3F", "TrpB3G", "TrpB3H", "TrpB3I", "TrpB4"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     n_samples=[960, 1920],
    #     # n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=96,
    #     all_libs=False,
    #     lib_list=["ParD2", "ParD3"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_samples=[960, 1920],
    #     # n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=96,
    #     all_libs=False,
    #     lib_list=["ParD2", "ParD3"],
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=96,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=96,
    # )


    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["none", "ev", "Triad"],
    #     n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=256,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["none", "ev", "Triad"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=False,
    #     lib_list=["ParD2", "ParD3"],
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0],
    #     zs_predictors=["none", "ev", "Triad"],
    #     n_samples=[960, 1920],
    #     n_replicate=50,
    #     n_job=256,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0,],
    #     zs_predictors=["esmif"],
    #     n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=256,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0,],
    #     zs_predictors=["esmif"],
    #     n_samples=[96, 384],
    #     n_replicate=50,
    #     n_job=256,
    #     ft_first=True,
    #     zs_folder=zs_folder,
    #     mlde_folder=mlde_folder,
    # )

    run_all_mlde_parallelized(
        encodings=DEFAULT_LEARNED_EMB_COMBO,
        n_mut_cutoffs=[0,],
        zs_predictors=["esmif"],
        n_samples=[192, 288],
        n_replicate=50,
        n_job=256,
        ft_first=True,
        zs_folder=zs_folder,
        mlde_folder=mlde_folder,
        all_libs=False,
        lib_list=["GB1", "TrpB4"],
    )

    run_all_mlde_parallelized(
        encodings=DEFAULT_LEARNED_EMB_COMBO,
        n_mut_cutoffs=[0,],
        zs_predictors=["esmif"],
        n_samples=[480, 576],
        n_replicate=50,
        n_job=256,
        ft_first=True,
        zs_folder=zs_folder,
        mlde_folder=mlde_folder,
        all_libs=False,
        lib_list=["GB1", "TrpB4"],
    )

    run_all_mlde_parallelized(
        encodings=DEFAULT_LEARNED_EMB_COMBO,
        n_mut_cutoffs=[0],
        zs_predictors=["esmif"],
        n_samples=[960, 1920],
        n_replicate=50,
        n_job=256,
        ft_first=True,
        zs_folder=zs_folder,
        mlde_folder=mlde_folder,
    )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=256,
    # )



    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     n_replicate=100,
    #     n_job=96,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_replicate=100,
    #     n_job=96,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     n_replicate=50,
    #     n_job=256,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_replicate=50,
    #     n_job=256,
    # )

    f.close()

    """
    def run_all_mlde_parallelized(
        zs_folder: str = "results/zs_comb",
        filter_min_by: str = "none",
        n_mut_cutoffs: list[int] = [0, 1, 2],
        scale_type: str = "scale2max",
        zs_predictors: list[str] = ["none", "Triad", "ev", "esm"],
        ft_lib_fracs: list[float] = [0.5, 0.25, 0.125],
        encodings: list[str] = ["one-hot"] + DEFAULT_LEARNED_EMB_COMBO,
        model_classes: list[str] = ["boosting", "ridge"],
        n_samples: list[int] = [384],
        n_split: int = 5,
        n_replicate: int = 100,
        n_tops: list[int] = [96, 384],
        boosting_n_worker: int = 1,
        n_job: int = 128,
        global_seed: int = 42,
        verbose: bool = False,
        save_model: bool = False,
        mlde_folder: str = "results/mlde",
        all_libs: bool = True,
        lib_list: list[str] = [],
    ):
    """