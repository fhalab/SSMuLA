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

    mlde_folder = "results/mlde_hm2zs"

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2],
    #     zs_predictors=["Triad", "ev", "esm", "esmif"],
    #     n_samples=[384],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=True,
    #     ft_first=False,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2],
    #     zs_predictors=["struc-comb", "msanoif-comb", "msa-comb", "structnmsa-comb"],
    #     n_samples=[384],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=True,
    #     ft_first=False,
    #     mlde_folder=mlde_folder,
    # )


    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2],
    #     zs_predictors=["Triad", "ev", "esm", "esmif"],
    #     n_samples=[24, 48],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=True,
    #     ft_first=False,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2],
    #     zs_predictors=["Triad", "ev", "esm", "esmif"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=True,
    #     ft_first=False,
    #     mlde_folder=mlde_folder,
    # )

    # run_all_mlde_parallelized(
    #     encodings=["one-hot"],
    #     n_mut_cutoffs=[2],
    #     zs_predictors=["Triad", "ev", "esm", "esmif"],
    #     n_samples=[960, 1920],
    #     n_replicate=50,
    #     n_job=256,
    #     all_libs=True,
    #     ft_first=False,
    #     mlde_folder=mlde_folder,
    # )

    run_all_mlde_parallelized(
        encodings=["one-hot"],
        n_mut_cutoffs=[2],
        zs_predictors=["struc-comb", "msanoif-comb", "msa-comb", "structnmsa-comb"],
        n_samples=[24, 48, 96],
        n_replicate=50,
        n_job=96,
        all_libs=True,
        ft_first=False,
        mlde_folder=mlde_folder,
    )

    run_all_mlde_parallelized(
        encodings=["one-hot"],
        n_mut_cutoffs=[2],
        zs_predictors=["struc-comb", "msanoif-comb", "msa-comb", "structnmsa-comb"],
        n_samples=[192, 288, 480, 576],
        n_replicate=50,
        n_job=96,
        all_libs=True,
        ft_first=False,
        mlde_folder=mlde_folder,
    )

    run_all_mlde_parallelized(
        encodings=["one-hot"],
        n_mut_cutoffs=[2],
        zs_predictors=["struc-comb", "msanoif-comb", "msa-comb", "structnmsa-comb"],
        n_samples=[960, 1920],
        n_replicate=50,
        n_job=96,
        all_libs=True,
        ft_first=False,
        mlde_folder=mlde_folder,
    )


    # zs_predictors=["struc-comb", "msanoif-comb", "msa-comb", "structnmsa-comb"],
    
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
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=96,
    # )



    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["none"],
    #     n_samples=[96, 192, 288, 480, 576],
    #     n_replicate=50,
    #     n_job=256,
    # )

    # run_all_mlde_parallelized(
    #     encodings=DEFAULT_LEARNED_EMB_COMBO,
    #     n_mut_cutoffs=[0, 2, 1],
    #     zs_predictors=["Triad", "ev", "esm"],
    #     n_samples=[96, 192, 288, 480, 576],
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