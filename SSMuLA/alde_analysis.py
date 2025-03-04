"""
A script for combining and analyzing the results of the ALDE analysis.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np


from SSMuLA.landscape_global import N_SAMPLE_LIST


def avg_alde_df(
    eq_n: int,
    lib_list: list,
    zs: str = "",
    alde_model: str = "Boosting Ensemble",
    alde_encoding: str = "onehot",
    alde_acq: str = "GREEDY",
    alde_dir: str = "/disk2/fli/alde4ssmula",
) -> pd.DataFrame:

    """
    Average ALDE results for a given list of libraries and equal n.

    Args:
    - eq_n (int): Equal n for the libraries.
    - lib_list (list): List of libraries to aggregate.

    Returns:
    - df (pd.DataFrame): Aggregated ALDE results.
    """

    df = pd.DataFrame(
        columns=[
            "n_sample",
            "top_maxes_mean",
            "top_maxes_std",
            "if_truemaxs_mean",
            "if_truemaxs_std",
        ]
    )

    for n in N_SAMPLE_LIST:

        if zs != "":
            zs_append = f"{zs}_"
        else:
            zs_append = ""

        if eq_n == 1:
            csv_path = f"{alde_dir}/results/{zs_append}all_{str(n)}+96/all_results.csv"

        else:
            csv_path = f"{alde_dir}/results/{zs_append}{str(eq_n)}eq_{str(int((n+96)/eq_n))}/all_results.csv"

        if os.path.exists(csv_path):
            a_df = pd.read_csv(csv_path)

            # Get the max Timestep for each Protein
            max_timesteps = a_df.groupby("Protein")["Timestep"].transform("max")
            # DNN Ensemble
            # Boosting Ensemble
            slice_df = a_df[
                (a_df["Encoding"] == alde_encoding)
                & (a_df["Acquisition"] == alde_acq)
                & (a_df["Model"] == alde_model)
                & (a_df["Protein"].isin(lib_list))
                & (a_df["Timestep"] == max_timesteps)
            ]
            # for each Protein take the max of the timestep

            if len(lib_list) == 1:
                top_maxes_std = slice_df["Std"].mean()
                if_truemaxs_std = 0
            else:
                top_maxes_std = slice_df["Mean"].std()
                if_truemaxs_std = slice_df["Frac"].std()

            df = df._append(
                {
                    "n_sample": n,
                    "top_maxes_mean": slice_df["Mean"].mean(),
                    "top_maxes_std": top_maxes_std,
                    "if_truemaxs_mean": slice_df["Frac"].mean(),
                    "if_truemaxs_std": if_truemaxs_std,
                },
                ignore_index=True,
            )
        elif"ds-ed" in csv_path:
            continue
        else:
            print(f"File not found: {csv_path}")

            df = df._append(
                {
                    "n_sample": n,
                    "top_maxes_mean": np.nan,
                    "top_maxes_std": np.nan,
                    "if_truemaxs_mean": np.nan,
                    "if_truemaxs_std": np.nan,
                },
                ignore_index=True,
            )

    return df.set_index("n_sample")


def aggregate_alde_df(
    eq_ns: list[int] = [1, 2, 3, 4],
    zs_opts: list[str] = ["esmif", "ev", "coves", "ed", "esm", "Triad", ""],
    alde_dir: str = "/disk2/fli/alde4ssmula",
    alde_df_path: str = "results/alde/alde_all.csv",
) -> pd.DataFrame:

    """
    Aggregate ALDE results for a given list of libraries and equal n.

    Args:
    - eq_ns (list): List of equal n values.
    - zs_opts (list): List of zero-shot options.
    - alde_dir (str): Directory containing ALDE results.
    - alde_df_path (str): Path to save the aggregated ALDE results.

    Returns:
    - df (pd.DataFrame): Aggregated ALDE results.
    """

    # initialize the dataframe
    alde_all = pd.DataFrame(
        columns=[
            "n_mut_cutoff",
            "zs",
            "rounds",
            "n_samples",
            "Protein",
            "Encoding",
            "Model",
            "Acquisition",
            "Timestep",
            "Mean",
            "Std",
            "Frac",
        ]
    )

    for eq_n in eq_ns:

        for zs in zs_opts + ["ds-" + z for z in zs_opts if z != ""]:

            if "ds-" in zs:
                n_mut = "double"
            else:
                n_mut = "all"

            for n in N_SAMPLE_LIST:

                if zs != "":
                    zs_append = f"{zs}_"
                else:
                    zs_append = ""

                if eq_n == 1:
                    csv_path = f"{alde_dir}/results/{zs_append}all_{str(n)}+96/all_results.csv"

                else:
                    csv_path = f"{alde_dir}/results/{zs_append}{str(eq_n)}eq_{str(int((n+96)/eq_n))}/all_results.csv"

                if os.path.exists(csv_path):
                    print(f"Reading {csv_path}...")
                    a_df = pd.read_csv(csv_path)

                    max_timesteps = a_df.groupby("Protein")["Timestep"].transform("max")
                    slice_df = a_df[a_df["Timestep"] == max_timesteps].copy()

                    slice_df["n_mut_cutoff"] = n_mut
                    slice_df["zs"] = zs
                    slice_df["rounds"] = eq_n
                    slice_df["n_samples"] = n + 96

                    # replace T7_2 with T7
                    # slice_df = slice_df.replace("T7_2", "T7")

                    alde_all = alde_all._append(slice_df, ignore_index=True)

                else:
                    print(f"File not found: {csv_path}")

                    alde_all = alde_all._append(
                        {
                            "n_mut_cutoff": n_mut,
                            "zs": zs,
                            "rounds": eq_n,
                            "n_samples": n + 96,
                            "Protein": np.nan,
                            "Encoding": np.nan,
                            "Model": np.nan,
                            "Acquisition": np.nan,
                            "Timestep": np.nan,
                            "Mean": np.nan,
                            "Std": np.nan,
                            "Frac": np.nan,
                        },
                        ignore_index=True,
                    )

    
    alde_all = alde_all.dropna(subset=["Protein"])

    alde_all.to_csv(alde_df_path, index=False)

    return alde_all


def clean_alde_df(
    agg_alde_df_path: str = "results/alde/alde_all.csv",
    clean_alde_df_path: str = "results/alde/alde_results.csv",
):
    """
    A function to clean up the aggregated ALDE results.
    """

    alde_df = pd.read_csv(agg_alde_df_path)
    alde_df[
        (alde_df["rounds"].isin([2, 3, 4]))
        & (alde_df["Model"].isin(["Boosting Ensemble", "DNN Ensemble"]))
        & (alde_df["Acquisition"] == "GREEDY")
    ].rename(
        columns={
            "Protein": "lib",
            "Mean": "top_maxes_mean",
            "Std": "top_maxes_std",
            "Frac": "if_truemaxs_mean",
            "Encoding": "encoding",
            "Model": "model",
            "n_samples": "n_sample",
        }
    )[
        [
            "encoding",
            "model",
            "n_sample",
            "top_maxes_mean",
            "top_maxes_std",
            "if_truemaxs_mean",
            "n_mut_cutoff",
            "lib",
            "zs",
            "rounds",
        ]
    ].reset_index(drop=True).to_csv(clean_alde_df_path, index=False)