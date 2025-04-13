"""
A script for combining and analyzing the results of the ALDE analysis.
"""

from __future__ import annotations

import os
import pandas as pd
import numpy as np


from SSMuLA.landscape_global import N_SAMPLE_LIST, LOWN_DICT
from SSMuLA.zs_analysis import ZS_OPTS, map_zs_labels


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
        elif "ds-ed" in csv_path:
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
    n_list: list[int] = N_SAMPLE_LIST,
    zs_opts: list[str] = ["esmif", "ev", "coves", "ed", "esm", "Triad", ""],
    alde_dir: str = "/disk2/fli/alde4ssmula",
    alde_res_folder: str = "results",
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

            for n in n_list:

                if isinstance(n, int):
                    dir_det = str(int((n + 96) / eq_n))
                    n_sample = n + 96
                else:
                    dir_det = n
                    n_sample = LOWN_DICT[n]

                if zs != "":
                    zs_append = f"{zs}_"
                else:
                    zs_append = ""

                if eq_n == 1:
                    csv_path = f"{alde_dir}/{alde_res_folder}/{zs_append}all_{str(n)}+96/all_results.csv"

                else:
                    csv_path = f"{alde_dir}/{alde_res_folder}/{zs_append}{str(eq_n)}eq_{dir_det}/all_results.csv"

                if os.path.exists(csv_path):
                    print(f"Reading {csv_path}...")
                    a_df = pd.read_csv(csv_path)

                    max_timesteps = a_df.groupby("Protein")["Timestep"].transform("max")
                    slice_df = a_df[a_df["Timestep"] == max_timesteps].copy()

                    slice_df["n_mut_cutoff"] = n_mut
                    slice_df["zs"] = zs
                    slice_df["rounds"] = eq_n
                    slice_df["n_samples"] = n_sample

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
                            "n_samples": n_sample,
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


def get_ftalde_libavg(
    alde_csv: str,
    lib_list: list,
    n_total: int,
    n_round: int,
    models: list = ["Boosting Ensemble"],
    acquisition: list = ["GREEDY"],
):

    """
    Get the FT-ALDE data for each of the library, number of rounds, models, and acquisition method.
    Args:
        alde_csv (str): Path to the FT-ALDE CSV file.
        lib_list (list): List of libraries to filter.
        n_total (int): Total number of samples.
        n_round (int): Number of rounds.
        models (list): List of models to filter.
        acquisition (list): List of acquisition methods to filter.
    Returns:
        pd.DataFrame: Filtered DataFrame containing FT-ALDE data.
    """
    # have all none zs and zs opt for MLDE, ALDE different rounds
    alde_all = pd.read_csv(alde_csv)
    # Replace NaN values in column 'zs' with the string "none"
    alde_all["zs"] = alde_all["zs"].fillna("none")

    slice_df = alde_all[
        (alde_all["rounds"] == n_round)
        & (alde_all["Encoding"] == "onehot")
        & (alde_all["Model"].isin(models))
        & (alde_all["Acquisition"].isin(acquisition))
        & (alde_all["n_samples"] == n_total)
        & (alde_all["Protein"].isin(lib_list))
        # & (alde_all["n_mut_cutoff"] == "all")
    ].copy()

    # Convert 'Category' column to categorical with defined order
    slice_df["zs"] = pd.Categorical(
        slice_df["zs"],
        categories=["none"]
        + [o.replace("_score", "") for o in ZS_OPTS]
        + [
            "ds-esmif",
            "ds-ev",
            "ds-coves",
            "ds-Triad",
            "ds-esm",
        ],
        ordered=True,
    )

    slice_df = slice_df.sort_values(by=["zs", "Protein"])

    slice_df["zs"] = slice_df["zs"].apply(map_zs_labels)

    return (
        slice_df[["Protein", "zs", "Mean", "Frac"]]
        .rename(columns={"Protein": "lib", "Mean": "top_maxes", "Frac": "if_truemaxs"})
        .copy()
    )


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
    ].reset_index(
        drop=True
    ).to_csv(
        clean_alde_df_path, index=False
    )