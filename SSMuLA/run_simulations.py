"""A script for running all simulations for each landscape."""

from __future__ import annotations

from glob import glob

import pandas as pd

from SSMuLA.simulations import run_all_de_simulations
from SSMuLA.landscape_global import LIB_INFO_DICT
from SSMuLA.util import get_file_name


# TODO TAKE OUT STOP CODONS

# Run simulations for each library
def run_all_lib_de_simulations():
    """
    Run all simulations for each library.
    """
    for scale_type in ["scale2parent", "scale2max"]:
        # Run simulations for each library
        for lib in glob(f"data/*/{scale_type}/*.csv"):

            lib_name = get_file_name(lib)
            n_sites = len(LIB_INFO_DICT[lib_name]["positions"])

            print(f"Running simulations for {lib_name} over {n_sites}...")
            
            df = pd.read_csv(lib).copy()

            run_all_de_simulations(
                df=df[~df["AAs"].str.contains("\*")], 
                seq_col="AAs", 
                fitness_col="fitness",
                lib_name=lib_name,
                save_dir=f"results/simulations/DE-no-stop-codons/{scale_type}",
                n_sites=n_sites, 
                N=96, 
                max_samples=None,
                n_jobs=256)