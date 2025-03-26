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

from scipy.interpolate import interp1d

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

from SSMuLA.landscape_global import N_SAMPLE_LIST
from SSMuLA.de_simulations import DE_N_TEST, DE_TYPES, DE_LEGEND_MAP
from SSMuLA.mlde_analysis import (
    get_demlal_libavg,
    get_ftmlal_libavg,
    PLOT_MLDE_METRICS,
    PLOT_LINE_PERFORMANCE_YAXIS,
    PERFORMANCE_YAXIS_DICT,
    MLDE_ALDE_COLORS,
    FTMLDE_COLOR_LIST,
    N_TICK_LIST,
    TOTAL_N_LIST,
    DE_MLAL_ORDER,
    ALL_FT_ORDER
)
from SSMuLA.vis import (
    FZL_PALETTE,
    GRAY_COLORS
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


def estimate_y(x_vals, y_vals, new_x, method="linear"):
    """
    Estimate y-value for a given x using interpolation/extrapolation.

    Parameters:
    - x_vals (list or np.array): List of x-values.
    - y_vals (list or np.array): List of corresponding y-values.
    - new_x (float or list): The x-value(s) to estimate y for.
    - method (str): Interpolation method ('linear', 'quadratic', 'cubic').

    Returns:
    - Estimated y-value(s) for new_x.
    """
    # Convert lists to numpy arrays
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Ensure x-values are sorted (required for interpolation)
    sorted_indices = np.argsort(x_vals)
    x_vals = x_vals[sorted_indices]
    y_vals = y_vals[sorted_indices]

    # Create interpolation function
    interp_func = interp1d(x_vals, y_vals, kind=method, fill_value="extrapolate")

    # Estimate y-values
    estimated_y = interp_func(new_x)

    return estimated_y


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
    battles: pd.DataFrame,
    compute_elo: callable,
    bootstrap_round: int = 1000,
    ifsort: bool = True,
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

    if ifsort:
        return df[df.median().sort_values(ascending=False).index]
    else:
        return df


# Stratified Bootstrapping function
def get_stratified_bootstrap_elo(
    battles: pd.DataFrame,
    compute_elo: callable,
    bootstrap_round: int = 1000,
    ifsort: bool = True,
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

    if ifsort:
        return df[df.median().sort_values(ascending=False).index]
    else:
        return df


def get_demlal_n_elo(
    mlde_csv: str,
    alde_csv: str,
    de_csv: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_sample_list: list = N_SAMPLE_LIST,
    mlde_models: list = ["boosting"],
    bootstrap_round: int = 1000,
) -> dict:

    """
    Get the Elo ratings for different numbers of samples.
    Returns:
        dict: Dictionary containing the Elo ratings for different numbers of samples.
    """

    elo_df_dict = {}

    for metric in PLOT_MLDE_METRICS:
        elo_df_dict[metric] = {}

        for n_sample in n_sample_list:

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
            elo_df_dict[metric][n_sample + n_top] = get_bootstrap_elo(
                process_landscape_method_comparisons(
                    df=merge_demlal_avg,
                    libs=merge_demlal_avg.lib.unique().tolist(),
                    methods=merge_demlal_avg.method.unique().tolist(),
                    method_col="method",
                    metric=metric,
                ),
                compute_elo,
                bootstrap_round,
                ifsort=False,
            )

    return elo_df_dict


def get_ftmlal_n_elo(
    mlde_csv: str,
    alde_csv: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_sample_list: list = N_SAMPLE_LIST,
    mlde_models: list = ["boosting"],
    alde_models: list = ["Boosting Ensemble"],
    alde_acquisition: list = ["GREEDY"],
    bootstrap_round: int = 1000,
    add_ensemble: bool=True,
) -> dict:
    """
    Get Elo ratings for different numbers of samples for ftMLDE and ftALDE double-site predictors.

    Returns:
        dict: Dictionary containing the Elo ratings for different numbers of samples.
    """

    elo_df_dict = {}

    for metric in PLOT_MLDE_METRICS:
        elo_df_dict[metric] = {}

        for n_sample in n_sample_list:
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
            elo_df_dict[metric][n_sample] = booststrap_func(
                process_landscape_method_comparisons(
                    df=ftmlde_libdf,
                    libs=ftmlde_libdf["lib"].unique().tolist(),
                    methods=ftmlde_libdf["zs"].unique().tolist(),
                    method_col="zs",
                    metric=metric,
                ),
                compute_elo,
                bootstrap_round,
                ifsort=False,
            )

    return elo_df_dict


def process_n_elo_df(elo_df_dict: dict, metric: str) -> pd.DataFrame:
    """
    Process the Elo ratings DataFrame.

    Args:
        elo_df_dict (dict): Dictionary containing the Elo ratings for different numbers of samples.
        metric (str): Metric to process.

    Returns:
        pd.DataFrame: DataFrame containing the processed Elo ratings.
    """

    data_dict = elo_df_dict[metric]

    # Create an empty list to store processed data
    processed_data = []

    # Iterate through the dictionary
    for key, df in data_dict.items():
        # Compute the median of each column
        median_values = df.median().to_dict()
        # Add the key as an index
        median_values["Total number of variants"] = key
        # Append to the list
        processed_data.append(median_values)

    # Convert list to DataFrame
    return pd.DataFrame(processed_data).set_index("Total number of variants")



def vis_demlal_n_elo(
    elo_df_dict: dict,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifsave: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot Elo ratings vs. number of screened variants, maintaining style consistency with plot_de_vs_mlde.

    Args:
    - elo_df_dict: dict, Elo ratings for different numbers of samples.
    - fig_name: str, Name of the figure to save.
    - ifsave: bool, Whether to save the figure.
    - fig_dir: str, Directory to save the figure.

    Returns:
    - None
    """

    fig, axes = plt.subplots(1, len(PLOT_MLDE_METRICS), figsize=(5 * len(PLOT_MLDE_METRICS), 4))
    # plt.subplots(1, 2, figsize=(10, 4))
    if len(PLOT_MLDE_METRICS) == 1:
        axes = [axes]

    de_colors = sns.color_palette("Greys", 4)
    de_ls = ["dotted", "dashed", "dashdot"]
    
    line_styles = [
        "dashed",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
    ] + de_ls

    demlal_colros = MLDE_ALDE_COLORS + de_colors

    for ax, metric, y_label in zip(axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS):
        elo_df = process_n_elo_df(elo_df_dict, metric)

        # replace " x 2" with ""
        elo_df.columns = elo_df.columns.str.replace(" x 2", "")
        # add DE: for DE methods
        elo_df.columns = elo_df.columns.str.replace("Recomb", "DE: Recomb")
        elo_df.columns = elo_df.columns.str.replace("Single step", "DE: Single step")
        elo_df.columns = elo_df.columns.str.replace("Top96 recomb", "DE: Top96 recomb")

        # sort column
        elo_df = elo_df[DE_MLAL_ORDER]
      
        # Plot Elo rating for each strategy
        for i, strategy in enumerate(elo_df.columns):

            # do not have marker if de in it
            if "DE: " in strategy:
                marker = None
            else:
                marker = "o"

            ax.plot(
                elo_df.index,
                elo_df[strategy],
                label=strategy.replace("Average ", ""),
                marker=marker,
                linestyle=line_styles[i],
                linewidth=2,
                color=demlal_colros[i],
            )

            # Add shaded standard deviation
            ax.fill_between(
                elo_df.index,
                elo_df[strategy] - elo_df[strategy].std(),
                elo_df[strategy] + elo_df[strategy].std(),
                color=demlal_colros[i],
                alpha=0.05,
            )

        # Add vertical lines at 192 and 480
        ax.axvline(n_corr+n_top, color="gray", linestyle="dotted", linewidth=1.2)

        # Add triangle and diamond markers for 3-site and 4-site full-coverage and unique variants
        for d, de in enumerate(DE_TYPES):

            de_xs = elo_df.index.tolist()
            de_ys = elo_df["DE: " + DE_LEGEND_MAP[de]].values.tolist()

            # annotate full coverage screen
            ax.scatter(
                3 * 96 + DE_N_TEST[de],
                estimate_y(x_vals=de_xs, y_vals=de_ys, new_x=3 * 96 + DE_N_TEST[de]),
                marker="^",
                color=de_colors[d + 1],
                s=40,
                # label="3-site full-coverage",
            )
            ax.scatter(
                4 * 96 + DE_N_TEST[de],
                estimate_y(x_vals=de_xs, y_vals=de_ys, new_x=4 * 96 + DE_N_TEST[de]),
                marker="d",
                color=de_colors[d + 1],
                s=40,
                # label="4-site full-coverage",
            )

            # annotate
            if de == "top96_SSM":
                # add "^" at x = 19*3 + 96 and the correpsonding y
                # no line just one dot
                ax.scatter(
                    19 * 3 + n_top,
                    estimate_y(x_vals=de_xs, y_vals=de_ys, new_x=19 * 3 + n_top),
                    marker="^",
                    facecolors="none",
                    edgecolors=de_colors[d + 1],
                    linewidth=1.2,
                    color=de_colors[d + 1],
                    s=36,
                    # label="3-site unique",
                )
                # add "D" at x = 19*4 + 96 and the correpsonding y
                ax.scatter(
                    19 * 4 + n_top,
                    estimate_y(x_vals=de_xs, y_vals=de_ys, new_x=19 * 4 + n_top),
                    marker="d",
                    facecolors="none",
                    edgecolors=de_colors[d + 1],
                    linewidth=1.2,
                    color=de_colors[d + 1],
                    s=36,
                    # label="4-site unique",
                )

        # Formatting
        ax.set_xscale("log")
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(f"Elo rating for {y_label.lower()}")
        # make the ylable a bit lower
        ax.yaxis.set_label_coords(-0.175, 0.495)
        # set y range
        ax.set_ylim(750, 1250)
        ax.set_xticks(N_TICK_LIST)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())

        if ax == axes[-1]:  # Ensure the legend is correctly ordered and placed in the last subplot
            handles, labels = ax.get_legend_handles_labels()

            # Manually create legend handles with assigned colors
            handles.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="none",
                        markeredgecolor="black",
                        markerfacecolor="none",
                        markersize=6,
                        label="3-site unique",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="d",
                        color="none",
                        markeredgecolor="black",
                        markerfacecolor="none",
                        markersize=6,
                        label="4-site unique",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="none",
                        markerfacecolor="black",
                        markersize=6,
                        label="3-site full-coverage",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="d",
                        color="none",
                        markerfacecolor="black",
                        markersize=6,
                        label="4-site full-coverage",
                    ),
                ]
            )

            # Manually create legend labels
            labels.extend(
                [
                    "3-site unique",
                    "4-site unique",
                    "3-site full-coverage",
                    "4-site full-coverage",
                ]
            )

            # fix the dash handle
            [
                handles[index].set_dashes([6, 2])
                for index, value in enumerate(line_styles)
                if value == "dashed"
            ]

            # Reorder legend handles
            # ordered_handles = [handles[labels.index(l)] for l in legend_order if l in labels]
            # ordered_labels = [l for l in legend_order if l in labels]

            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1.025))

    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def vis_dsmlal_n_elo(
    elo_df_dict: dict,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifsave: bool = True,
    fig_dir: str = "figs",
):
    """
    Visualize Elo ratings across different sample sizes for ftMLDE double-site predictors.

    Args:
    - elo_df_dict: dict, Elo ratings for different numbers of samples.
    - fig_name: str, Name of the figure to save.
    - ifsave: bool, Whether to save the figure.
    - fig_dir: str, Directory to save the figure.

    Returns:
    - None
    """

    ds_cols = ALL_FT_ORDER[:12]

    fig, axes = plt.subplots(1, len(PLOT_MLDE_METRICS), figsize=(11.2, 4))

    if len(PLOT_MLDE_METRICS) == 1:
        axes = [axes]
    
    for ax, metric, y_label in zip(axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS):
        elo_df = process_n_elo_df(elo_df_dict, metric)

        # replace "\n" with ""
        elo_df.columns = elo_df.columns.str.replace("\n", " ")

        # Sort column
        elo_df = elo_df[ds_cols]

        # Plot Elo rating for each strategy
        for i, strategy in enumerate(ds_cols):

            # doulbes 
            
            if i <= 6:
                color = FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]]
                linestyle = "solid"
            else:
                color = GRAY_COLORS["gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)+2]]
                linestyle = "dashed"
                
            ax.plot(
                elo_df.index,
                elo_df[strategy],
                label=strategy,
                marker="o",
                linestyle=linestyle,
                linewidth=2,
                color=color,
            )

            # Add shaded standard deviation
            ax.fill_between(
                elo_df.index,
                elo_df[strategy] - elo_df[strategy].std(),
                elo_df[strategy] + elo_df[strategy].std(),
                color=color,
                alpha=0.05,
            )


        # Add vertical lines
        ax.axvline(n_corr + n_top, color="gray", linestyle="dotted", linewidth=1.2)

        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(700, 1350)
        ax.set_xscale("log")
        ax.set_xticks(N_TICK_LIST)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(f"Elo rating for {y_label.lower()}")
        # make the ylable a bit lower
        ax.yaxis.set_label_coords(-0.175, 0.495)

        if ax == axes[-1]:  # Ensure the legend is correctly placed in the last subplot
            
            # fix dash
            handles, labels = ax.get_legend_handles_labels()
            # fix the dash handle
            [
                handles[index].set_dashes([6, 2])
                for index in range(len(handles))
                if index > 6
            ]
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1.025))

    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def vis_ensemblemlal_n_elo(
    elo_df_dict: dict,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifsave: bool = True,
    fig_dir: str = "figs",
):
    """
    Visualize Elo ratings across different sample sizes for ftMLDE double-site predictors.

    Args:
    - elo_df_dict: dict, Elo ratings for different numbers of samples.
    - fig_name: str, Name of the figure to save.
    - ifsave: bool, Whether to save the figure.
    - fig_dir: str, Directory to save the figure.

    Returns:
    - None
    """

    mlde_color_list = ["gray", "green", "purple", "yellow", "brown", "orange", "blue"]

    ft_cols = ALL_FT_ORDER[:7] + ALL_FT_ORDER[-6:]
    fig, axes = plt.subplots(1, len(PLOT_MLDE_METRICS), figsize=(10.5, 4))

    if len(PLOT_MLDE_METRICS) == 1:
        axes = [axes]
    
    for ax, metric, y_label in zip(axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS):
        elo_df = process_n_elo_df(elo_df_dict, metric)

        # replace "\n" with ""
        elo_df.columns = elo_df.columns.str.replace("\n", " ")

        # Sort column
        elo_df = elo_df[ft_cols]

        # Plot Elo rating for each strategy
        for i, strategy in enumerate(ft_cols):

            # doulbes 
            
            if i <= 6:
                linestyle = "solid"
                color = FZL_PALETTE[mlde_color_list[i]]
            else:
                linestyle = "dashed"
                color = GRAY_COLORS["gray-" + mlde_color_list[i-6]]
                
            ax.plot(
                elo_df.index,
                elo_df[strategy],
                label=strategy,
                marker="o",
                linestyle=linestyle,
                linewidth=2,
                color=color,
            )

            # Add shaded standard deviation
            ax.fill_between(
                elo_df.index,
                elo_df[strategy] - elo_df[strategy].std(),
                elo_df[strategy] + elo_df[strategy].std(),
                color=color,
                alpha=0.05,
            )


        # Add vertical lines
        ax.axvline(n_corr + n_top, color="gray", linestyle="dotted", linewidth=1.2)

        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(700, 1350)
        ax.set_xscale("log")
        ax.set_xticks(N_TICK_LIST)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(f"Elo rating for {y_label.lower()}")
        # make the ylable a bit lower
        ax.yaxis.set_label_coords(-0.175, 0.495)

        if ax == axes[-1]:  # Ensure the legend is correctly placed in the last subplot
            
            # fix dash
            handles, labels = ax.get_legend_handles_labels()
            # fix the dash handle
            [
                handles[index].set_dashes([6, 2])
                for index in range(len(handles))
                if index > 6
            ]
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1.025))

    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def vis_n_elo_heatmap(
    elo_df_dict: dict,
    fig_name: str,
    n_cutoff: int = 480,
    y_order=None,
    add_cutoff: bool = False,
    iftitle: bool = False,
    if_save: bool = False,
    fig_dir: str = "figs",
):

    """
    Visualize the Elo ratings for different numbers of samples.
    Two subplots for two different metrics.
    xaxis: number of samples
    yaxis: methods

    Args:
        elo_df_dict (dict): Dictionary containing the Elo ratings for different numbers of samples.
        fig_name (str): Title for the plot.
    """

    fig, axs = plt.subplots(1, 2, figsize=(14.8, 4))

    for i, metric in enumerate(PLOT_MLDE_METRICS):

        df = process_n_elo_df(elo_df_dict, metric)

        # make sure the index is sorted
        df = df.sort_index(axis=1)

        if add_cutoff:
            # get row index value for n_cutoff
            n_cutoff_idx = df.index.get_loc(n_cutoff)

            # add a low N average meaning index value <= n_cutoff,
            # and a high N average meaning index value > n_cutoff
            df.loc[r"Avg low n$_{total}$"] = df.iloc[:n_cutoff_idx].mean()
            df.loc[r"Avg high n$_{total}$"] = df.iloc[n_cutoff_idx:].mean()
            df.loc[r"Avg n$_{total}$"] = df.mean()

            # sort columns by the All N average
            df = df.T.sort_values(by=r"Avg n$_{total}$", ascending=False)
        else:
            df = df.T
        # if "\n" in the index, remove it
        df.index = df.index.str.replace("\n", " ").str.replace(" x 2", "")

        # sort index by input order
        if y_order:
            df = df.reindex(y_order)

        sns.heatmap(
            df,
            cmap="vlag",
            ax=axs[i],
            cbar_kws={"label": "Elo rating"},
            annot=False,
            fmt=".1f",
            square=True,
        )
        axs[i].set_title(PERFORMANCE_YAXIS_DICT[metric])
        axs[i].set_xlabel("Total number of variants")
        axs[i].set_ylabel("Strategy")
        axs[i].set_xticklabels(df.columns)
        axs[i].set_yticklabels(df.index)

    if iftitle:
        fig.suptitle(fig_name)

    if if_save:
        save_svg(fig, plot_title=fig_name, path2folder=fig_dir)


# Visualization function (Split Violin Plot)
def visualize_split_violin(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    fig_name: str,
    xaxis_name: str,
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
    # replace "x 2" with ""
    x_label = df_combined["Strategy"].unique()
    x_label = [x.replace(" x 2", "") for x in x_label]
    ax.set_xticklabels(x_label, rotation=90, ha="center")
    ax.set_xlabel(xaxis_name)
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
        xaxis_name="Strategy",
        iftitle=False,
        if_save=if_save,
        fig_dir=fig_dir,
    )


def plot_demlal_n_elo(
    mlde_csv: str,
    alde_csv: str,
    de_csv: str,
    fig_name: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_corr = 384,
    n_sample_list: list = N_SAMPLE_LIST,
    mlde_models = ["boosting"],
    bootstrap_round = 1000,
    if_save: bool = True,
    fig_dir: str = "figs",
):

    elo_df_dict = get_demlal_n_elo(
        mlde_csv = mlde_csv,
        alde_csv = alde_csv,
        de_csv = de_csv,
        lib_list = lib_list,
        n_top = n_top,
        n_sample_list = n_sample_list,
        mlde_models = mlde_models,
        bootstrap_round = bootstrap_round,
    )

    vis_demlal_n_elo(
        elo_df_dict = elo_df_dict,
        fig_name = fig_name,
        n_top = n_top,
        n_corr = n_corr,
        ifsave = if_save,
        fig_dir = fig_dir,
    )


def plot_demlal_n_elo_heatmap(
    mlde_csv: str,
    alde_csv: str,
    de_csv: str,
    fig_name: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_sample_list: list = N_SAMPLE_LIST,
    y_order: list
    | None = [
        "MLDE",
        "ftMLDE",
        "ALDE",
        "ftALDE",
        "ALDE x 3",
        "ftALDE x 3",
        "ALDE x 4",
        "ftALDE x 4",
        "Recomb",
        "Single step",
        "Top96 recomb",
    ],
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
        n_sample_list (list): List of number of samples to consider.
        mlde_models (list): List of MLDE models to consider.
        bootstrap_round (int): Number of bootstrap rounds.
        if_save (bool): Whether to save the plot.
        fig_dir (str): Directory to save the plot.
    """

    elo_df_dict = get_demlal_n_elo(
        mlde_csv=mlde_csv,
        alde_csv=alde_csv,
        de_csv=de_csv,
        lib_list=lib_list,
        n_top=n_top,
        n_sample_list=n_sample_list,
        mlde_models=mlde_models,
        bootstrap_round=bootstrap_round,
    )

    vis_n_elo_heatmap(
        elo_df_dict=elo_df_dict,
        fig_name=fig_name,
        iftitle=False,
        y_order=y_order,
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
        xaxis_name="ZS predictor",
        iftitle=False,
        if_save=if_save,
        fig_dir=fig_dir,
    )


def plot_ftmlde_n_elo(
    mlde_csv: str,
    alde_csv: str,
    fig_names: list,
    lib_list: list | None = None,
    n_top: int = 96,
    n_corr: int = 384,
    n_sample_list: list = N_SAMPLE_LIST,
    mlde_models: list = ["boosting"],
    alde_models: list = ["Boosting Ensemble"],
    alde_acquisition: list = ["GREEDY"],
    bootstrap_round: int = 1000,
    add_ensemble: bool = True,
    if_save: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot Elo ratings for ftMLDE double-site predictors.

    Args:
        mlde_csv (str): Path to the MLDE CSV file.
        fig_name (str): Title for the plot.
    """

    elo_df_dict = get_ftmlal_n_elo(
        mlde_csv=mlde_csv,
        alde_csv=alde_csv,
        lib_list=lib_list,
        n_top=n_top,
        n_sample_list=n_sample_list,
        mlde_models=mlde_models,
        alde_models=alde_models,
        alde_acquisition=alde_acquisition,
        bootstrap_round=bootstrap_round,
        add_ensemble=add_ensemble,
    )

    vis_dsmlal_n_elo(
        elo_df_dict=elo_df_dict,
        fig_name=fig_names[0],
        n_top=n_top,
        n_corr=n_corr,
        ifsave=if_save,
        fig_dir=fig_dir,
    )

    vis_ensemblemlal_n_elo(
        elo_df_dict=elo_df_dict,
        fig_name=fig_names[1],
        n_top=n_top,
        n_corr=n_corr,
        ifsave=if_save,
        fig_dir=fig_dir,
    )



def plot_ftmlal_n_elo_heatmap(
    mlde_csv: str,
    alde_csv: str,
    fig_name: str,
    lib_list: list | None = None,
    n_top: int = 96,
    n_sample_list: list = N_SAMPLE_LIST,
    y_order: list| None = ALL_FT_ORDER,
    mlde_models: list = ["boosting"],
    alde_models: list = ["Boosting Ensemble"],
    alde_acquisition: list = ["GREEDY"],
    bootstrap_round: int = 1000,
    add_ensemble: bool = True,
    if_save: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot the Elo ratings for different strategies across different numbers of samples.

    Args:
        mlde_csv (str): Path to the MLDE CSV file.
        alde_csv (str): Path to the ALDE CSV file.
        fig_name (str): Title for the plot.
        lib_list (list, optional): List of libraries to compare.
        n_top (int): Number of top strategies to consider.
        n_sample_list (list): List of numbers of samples to evaluate.
        mlde_models (list): List of MLDE models to consider.
        alde_models (list): List of ALDE models to consider.
        alde_acquisition (list): List of ALDE acquisition functions.
        bootstrap_round (int): Number of bootstrap rounds for Elo scoring.
        add_ensemble (bool): Whether to include an ensemble model.
        if_save (bool): Whether to save the plot.
        fig_dir (str): Directory to save the plot.
    """

    # Select the bootstrap function
    bootstrap_func = get_stratified_bootstrap_elo if add_ensemble else get_bootstrap_elo

    # Initialize Elo results dictionary
    elo_df_dict = {}

    for metric in PLOT_MLDE_METRICS:
        elo_df_dict[metric] = {}

        for n_sample in n_sample_list:
            # Get processed data for different sample sizes
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

            # Compute Elo ratings using bootstrapping
            elo_df_dict[metric][n_sample + n_top] = bootstrap_func(
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

    # Generate heatmap visualization
    vis_n_elo_heatmap(
        elo_df_dict=elo_df_dict,
        fig_name=fig_name,
        iftitle=False,
        y_order=y_order,
        if_save=if_save,
        fig_dir=fig_dir,
    )