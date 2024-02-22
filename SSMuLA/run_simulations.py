"""A script for running all simulations for each landscape."""

from __future__ import annotations

from glob import glob

import pandas as pd

from SSMuLA.de_simulations import run_all_de_simulations
from SSMuLA.landscape_global import LIB_INFO_DICT
from SSMuLA.util import get_file_name


# Run simulations for each library
def run_all_lib_de_simulations(de_opts: list = ["DE-active", "DE-no_stop_codons", "DE-all"]):
    """
    Run all simulations for each library.
    """
    for scale_type in ["scale2parent", "scale2max"]:
        # Run simulations for each library
        for lib in glob(f"data/*/{scale_type}/*.csv"):

            lib_name = get_file_name(lib)
            n_sites = len(LIB_INFO_DICT[lib_name]["positions"])
            
            df = pd.read_csv(lib).copy()

            for de_det in de_opts:

                print(f"Running {de_det} simulations for {lib_name} over {n_sites}...")

                if de_det == "DE-all":
                    select_df = df.copy()
                elif de_det == "DE-active":
                    select_df = df[df["active"] == True].copy()
                elif de_det == "DE-no_stop_codons":
                    select_df = df[~df["AAs"].str.contains("\*")].copy()

                run_all_de_simulations(
                    df=select_df, 
                    seq_col="AAs", 
                    fitness_col="fitness",
                    lib_name=lib_name,
                    save_dir=f"results/simulations/{de_det}/{scale_type}",
                    n_sites=n_sites, 
                    N=96, 
                    max_samples=None,
                    n_jobs=256)
