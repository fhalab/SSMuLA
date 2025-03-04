"""
A script for combining and analyzing the results of the LoRA fine-tuning analysis.
"""

from __future__ import annotations

from glob import glob
import pandas as pd

from SSMuLA.landscape_global import N_SAMPLE_LIST
from SSMuLA.util import get_file_name


def parse_finetune_df(
    finetune_dir: str,  # ie results/finetuning/ev or none
    lib_list: list,
    n_top: int = 96,
) -> pd.DataFrame:

    """
    Parse the finetune dataframe and return a summary dataframe.

    The finetune_dir should contain the results of the finetuning analysis,
    where each landscape is in a separate folder, and each folder contains
    the results of the finetuning analysis for each landscape
    with the format <landscape>_<n_sample>_<rep>.csv.

    Args:
        finetune_dir (str): The directory containing the finetuning results.
        lib_list (list): A list of libraries to include in the analysis.
        n_top (int): The number of top variants to consider.
    """

    sum_df_list = []

    for df_path in sorted(glob(f"{finetune_dir}/*/*.csv")):

        landscape, n_sample, rep = get_file_name(df_path).split("_")

        if landscape not in lib_list:
            continue

        df = pd.read_csv(df_path)
        max_fit_seq = df.loc[df["fitness"].idxmax()]["seq"]

        # get top 96 maxes
        top_df = (
            df.sort_values(by="predictions", ascending=False)
            .reset_index(drop=True)
            .iloc[:n_top, :]
        )
        top_seqs = top_df["seq"].astype(str).values

        # write to sum_df
        sum_df_list.append(
            {
                "landscape": landscape,
                "n_sample": int(n_sample),
                "rep": int(rep),
                "top_maxes": top_df["fitness"].max(),
                "if_truemaxs": int(max_fit_seq in top_seqs),
            }
        )

    return (
        pd.DataFrame(sum_df_list)
        .sort_values(by=["n_sample", "landscape", "rep"])
        .reset_index(drop=True)
        .copy()
    )


def avg_finetune_df(
    finetune_df: pd.DataFrame,
    n_sample_list: list = N_SAMPLE_LIST,
) -> pd.DataFrame:

    """
    Average the finetune dataframe over the number of samples and repetitions.

    Args:
        finetune_df (pd.DataFrame): The dataframe containing the finetuning results.
        n_sample_list (list): A list of the number of samples to consider.
    """

    avg_sum_df = (
        finetune_df[["n_sample", "top_maxes", "if_truemaxs"]]
        .groupby("n_sample")
        .agg(["mean", "std"])
        .reset_index()
    )
    avg_sum_df.columns = ["{}_{}".format(i, j) for i, j in avg_sum_df.columns]
    return (
        avg_sum_df.rename(columns={"n_sample_": "n_sample"})
        .set_index("n_sample")
        .copy()
    )