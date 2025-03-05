"""
A script for analyzing Elo ratings of different strategies and ZS.

Gratefully modified from https://lmsys.org/blog/2023-05-03-arena/
https://colab.research.google.com/drive/1RAWb22-PFNI-X1gPVzc927SGUdfr6nsR?usp=sharing
"""

from __future__ import annotations

from tqdm import tqdm
from copy import deepcopy

import random
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from SSMuLA.mlde_analysis import (
    get_demlal_libavg,
    get_ftmlal_libavg,
    PLOT_MLDE_METRICS,
    PLOT_LINE_PERFORMANCE_YAXIS,
)
from SSMuLA.vis import save_svg


RAND_SEED = 42

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)

# Function to generate a list of seeded random integers
def get_random_states(size=1000, seed=42):
    """
    Generate a list of random integers with a fixed seed.

    Args:
        size (int): Number of random integers to generate.
        seed (int): Seed value for reproducibility.
    Returns:
        list: List of random integers.
    """
    random.seed(seed)
    return deepcopy([random.randint(0, 1000000) for _ in range(size)])


def process_landscape_method_comparisons(
    df: pd.DataFrame, libs: list, methods: list, method_col: str, metric: str
):
    """
    Process the landscape method comparisons.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        libs (list): List of libraries to compare.
        methods (list): List of methods to compare.
        method_col (str): Column name for the method.
        metric (str): Metric to use for comparison.

    Returns:
        pd.DataFrame: DataFrame containing the results of the comparisons.
    """

    battles = []
    for lib in libs:
        df_subset = df[df["lib"] == lib]
        for i, method_a in enumerate(methods):
            for j, method_b in enumerate(methods):
                if i >= j:
                    continue
                score_a = df_subset[df_subset[method_col] == method_a][metric].mean()
                score_b = df_subset[df_subset[method_col] == method_b][metric].mean()
                if score_a > score_b:
                    winner = "method_a"
                elif score_b > score_a:
                    winner = "method_b"
                else:
                    winner = "tie"
                battles.append([method_a, method_b, winner])
    return pd.DataFrame(battles, columns=["method_a", "method_b", "winner"])


# Elo rating computation function
def compute_elo(
    battles: pd.DataFrame,
    elo_k: int = 4,
    elo_scale=400,
    elo_base=10,
    elo_init_rating=1000,
):

    """
    Compute Elo ratings based on battle results.

    Args:
        battles (pd.DataFrame): DataFrame containing the battles.
        elo_k (int): K-factor for Elo rating calculation.
        elo_scale (int): Scale factor for Elo rating calculation.
        elo_base (int): Base for Elo rating calculation.
        elo_init_rating (int): Initial rating for each method.

    Returns:
        dict: Dictionary containing the Elo ratings for each method.
    """

    rating = defaultdict(lambda: elo_init_rating)

    for rd, method_a, method_b, winner in battles[
        ["method_a", "method_b", "winner"]
    ].itertuples():
        ra = rating[method_a]
        rb = rating[method_b]
        ea = 1 / (1 + elo_base ** ((rb - ra) / elo_scale))
        eb = 1 / (1 + elo_base ** ((ra - rb) / elo_scale))
        if winner == "method_a":
            sa = 1
        elif winner == "method_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[method_a] += elo_k * (sa - ea)
        rating[method_b] += elo_k * (1 - sa - eb)

    return rating


# Bootstrapping function
def get_bootstrap_elo(
    battles: pd.DataFrame, compute_elo: callable, bootstrap_round: int = 1000
):

    """
    Perform bootstrapping to compute Elo ratings.

    Args:
        battles (pd.DataFrame): DataFrame containing the battles.
        compute_elo (function): Function to compute Elo ratings.
        bootstrap_round (int): Number of bootstrap rounds.
    Returns:
        pd.DataFrame: DataFrame containing the bootstrapped Elo ratings.
    """

    random_states = get_random_states(size=bootstrap_round, seed=RAND_SEED)

    rows = []
    for i in tqdm(range(bootstrap_round), desc="bootstrap"):
        rows.append(
            compute_elo(
                battles.sample(frac=1.0, replace=True, random_state=random_states[i])
            )
        )
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


# Stratified Bootstrapping function
def get_stratified_bootstrap_elo(
    battles: pd.DataFrame, compute_elo: callable, bootstrap_round=1000
):

    """
    Perform stratified bootstrapping to balance methods with fewer entries.

    Args:
        battles (pd.DataFrame): DataFrame containing the battles.
        compute_elo (function): Function to compute Elo ratings.
        bootstrap_round (int): Number of bootstrap rounds.
    Returns:
        pd.DataFrame: DataFrame containing the bootstrapped Elo ratings.
    """

    random_states = get_random_states(size=bootstrap_round, seed=RAND_SEED)

    rows = []
    for i in tqdm(range(bootstrap_round), desc="Stratified Bootstrapping"):
        # Ensure each method is fairly represented in bootstrapped samples
        sampled_battles = battles.groupby("method_a", group_keys=False).apply(
            lambda x: x.sample(frac=1.0, replace=True, random_state=random_states[i])
            if len(x) > 0
            else x
        )
        sampled_battles = sampled_battles.groupby("method_b", group_keys=False).apply(
            lambda x: x.sample(frac=1.0, replace=True, random_state=random_states[i])
            if len(x) > 0
            else x
        )
        rows.append(compute_elo(sampled_battles))

    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


# Visualization function (Split Violin Plot)
def visualize_split_violin(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    fig_name: str,
    iftitle: bool = False,
    if_save: bool = False,
    fig_dir: str = "figs",
):

    """
    Visualize the results using a split violin plot.

    Args:
        df1 (pd.DataFrame): DataFrame containing the first set of results.
        df2 (pd.DataFrame): DataFrame containing the second set of results.
        fig_name (str): Title for the plot.
        iftitle (bool): Whether to add a title to the plot.
        if_save (bool): Whether to save the plot.
        fig_dir (str): Directory to save the plot.
    """

    df1 = df1.melt(var_name="Strategy", value_name="Elo rating")
    df2 = df2.melt(var_name="Strategy", value_name="Elo rating")
    df1["Metric"] = PLOT_LINE_PERFORMANCE_YAXIS[0]
    df2["Metric"] = PLOT_LINE_PERFORMANCE_YAXIS[1]
    df_combined = pd.concat([df1, df2])

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.violinplot(
        data=df_combined,
        x="Strategy",
        y="Elo rating",
        hue="Metric",
        split=True,
        inner="quartile",
        ax=ax,
    )
    ax.set_xticklabels(df_combined["Strategy"].unique(), rotation=90, ha="center")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Elo rating")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if iftitle:
        ax.set_title(fig_name)
    plt.legend(title="Metric", loc="lower left")

    if if_save:
        save_svg(fig, plot_title=fig_name, path2folder=fig_dir)


def predict_win_rate(
    bootstrap_elo_lu: pd.DataFrame,
    quantile: float = 0.5,
    elo_scale: int = 400,
    elo_base: int = 10,
    elo_init_rating: int = 1000,
):

    """
    Predict win rates based on Elo ratings.

    Args:
        bootstrap_elo_lu (pd.DataFrame): DataFrame containing the bootstrapped Elo ratings.
        quantile (float): Quantile to use for predictions.
        elo_scale (int): Scale factor for Elo rating calculation.
        elo_base (int): Base for Elo rating calculation.
        INIT_RATING (int): Initial rating for each method.

    Returns:
        pd.DataFrame: DataFrame containing the predicted win rates.
    """

    elo_ratings = dict(bootstrap_elo_lu.quantile(quantile))
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + elo_base ** ((elo_ratings[b] - elo_ratings[a]) / elo_scale))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {a: [wins[a][b] if a != b else np.nan for b in names] for a in names}

    df = pd.DataFrame(data, index=names)
    df.index.name = "method_a"
    df.columns.name = "method_b"
    return df.T


def plot_win_rate(
    bootstrap_elo_lu: pd.DataFrame,
    fig_name: str,
    iftitle: bool = False,
    if_save: bool = False,
    fig_dir: str = "figs",
):

    """
    Plot the predicted win rates based on Elo ratings.

    Args:
        bootstrap_elo_lu (pd.DataFrame): DataFrame containing the bootstrapped Elo ratings.
        title (str): Title for the plot.
    """
    win_rate = predict_win_rate(dict(bootstrap_elo_lu.quantile(0.5)))
    ordered_methods = win_rate.mean(axis=1).sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(win_rate.loc[ordered_methods, ordered_methods], cmap="vlag")
    # plt.colorbar(cax)
    # add colorbar title
    cbar = plt.colorbar(cax)
    cbar.set_label("Win Rate", rotation=270, labelpad=15)

    ax.set_xticks(range(len(ordered_methods)))
    ax.set_yticks(range(len(ordered_methods)))
    ax.set_xticklabels(ordered_methods, rotation=90)
    ax.set_yticklabels(ordered_methods)
    ax.set_xlabel("Strategy B")
    ax.set_ylabel("Strategy A")

    if iftitle:
        ax.set_title(fig_name)

    if if_save:
        save_svg(fig, plot_title=fig_name, path2folder=fig_dir)


def plot_demlal_elo(
    mlde_csv: str,
    alde_csv: str,
    de_csv: str,
    fig_name: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_sample: int = 384,
    mlde_models: list = ["boosting"],
    bootstrap_round: int = 1000,
    if_save: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot the Elo ratings for different strategies.

    Args:
        mlde_csv (str): Path to the MLDE CSV file.
        alde_csv (str): Path to the ALDE CSV file.
        de_csv (str): Path to the DE CSV file.
        lib_list (list): List of libraries to compare.
        n_top (int): Number of top strategies to consider.
        n_sample (int): Number of samples to consider.
        mlde_models (list): List of MLDE models to consider.
        bootstrap_round (int): Number of bootstrap rounds.
        if_save (bool): Whether to save the plot.
        fig_dir (str): Directory to save the plot.
    """

    # get the data
    merge_demlal_avg = get_demlal_libavg(
        mlde_csv=mlde_csv,
        alde_csv=alde_csv,
        de_csv=de_csv,
        lib_list=lib_list,
        n_sample=n_sample,
        models=mlde_models,
        n_top=n_top,
        ifmapde=True,
    )

    # get the bootstrap result
    elo_df_dict = {}

    for metric in PLOT_MLDE_METRICS:
        elo_df_dict[metric] = get_bootstrap_elo(
            process_landscape_method_comparisons(
                df=merge_demlal_avg,
                libs=merge_demlal_avg.lib.unique().tolist(),
                methods=merge_demlal_avg.method.unique().tolist(),
                method_col="method",
                metric=metric,
            ),
            compute_elo,
            bootstrap_round,
        )

    visualize_split_violin(
        elo_df_dict[PLOT_MLDE_METRICS[0]],
        elo_df_dict[PLOT_MLDE_METRICS[1]],
        fig_name=fig_name,
        iftitle=False,
        if_save=if_save,
        fig_dir=fig_dir,
    )


def plot_ftmlal_elo(
    mlde_csv: str,
    alde_csv: str,
    fig_name: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_sample: int = 384,
    mlde_models: list = ["boosting"],
    alde_models: list = ["Boosting Ensemble"],
    alde_acquisition: list = ["GREEDY"],
    bootstrap_round: int = 1000,
    add_ensemble=True,
    if_save: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot the Elo ratings for different strategies.

    Args:
        mlde_csv (str): Path to the MLDE CSV file.
        alde_csv (str): Path to the ALDE CSV file.
        de_csv (str): Path to the DE CSV file.
        lib_list (list): List of libraries to compare.
        n_top (int): Number of top strategies to consider.
        n_sample (int): Number of samples to consider.
        mlde_models (list): List of MLDE models to consider.
        bootstrap_round (int): Number of bootstrap rounds.
        if_save (bool): Whether to save the plot.
        fig_dir (str): Directory to save the plot.
    """

    # get the data
    ftmlde_libdf = get_ftmlal_libavg(
        mlde_csv=mlde_csv,
        alde_csv=alde_csv,
        lib_list=lib_list,
        n_sample=n_sample,
        mlde_models=mlde_models,
        alde_models=alde_models,
        alde_acquisition=alde_acquisition,
        n_top=n_top,
        add_ensemble=add_ensemble,
    )

    if add_ensemble:
        booststrap_func = get_stratified_bootstrap_elo
    else:
        booststrap_func = get_bootstrap_elo

    # get the bootstrap result
    elo_df_dict = {}

    for metric in PLOT_MLDE_METRICS:
        elo_df_dict[metric] = booststrap_func(
            process_landscape_method_comparisons(
                df=ftmlde_libdf,
                libs=ftmlde_libdf["lib"].unique().tolist(),
                methods=ftmlde_libdf["zs"].unique().tolist(),
                method_col="zs",
                metric=metric,
            ),
            compute_elo,
            bootstrap_round,
        )

    visualize_split_violin(
        elo_df_dict[PLOT_MLDE_METRICS[0]],
        elo_df_dict[PLOT_MLDE_METRICS[1]],
        fig_name=fig_name,
        iftitle=False,
        if_save=if_save,
        fig_dir=fig_dir,
    )