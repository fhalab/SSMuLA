"""A script for running all simulations for each landscape."""

from __future__ import annotations

from glob import glob

import pandas as pd

from SSMuLA.simulations import run_all_de_simulations
from SSMuLA.landscape_global import LIB_INFO_DICT
from SSMuLA.util import get_file_name


# Run simulations for each library
def run_all_lib_de_simulations(cutoff: str):
    """
    Run all simulations for each library.

    Args:
    - cutoff, str: the cutoff for the fitness to be considered,
        'active' or 'no_stop_codons'
    """
    for scale_type in ["scale2parent", "scale2max"]:
        # Run simulations for each library
        for lib in glob(f"data/*/{scale_type}/*.csv"):

            lib_name = get_file_name(lib)
            n_sites = len(LIB_INFO_DICT[lib_name]["positions"])
            
            df = pd.read_csv(lib).copy()

            if cutoff == "active":
                selcted_df = df[df["active"] == True]
            elif cutoff == "no_stop_codons":
                selcted_df = df[~df["AAs"].str.contains("\*")]
            else:
                cutoff = "all"
                selcted_df = df
            
            print(f"Running simulations for {lib_name} over {n_sites} with {cutoff}...")

            run_all_de_simulations(
                df=selcted_df, 
                seq_col="AAs",
                fitness_col="fitness",
                lib_name=lib_name,
                save_dir=f"results/simulations/DE-{cutoff}/{scale_type}",
                n_sites=n_sites, 
                N=96, 
                max_samples=None,
                n_jobs=256)
