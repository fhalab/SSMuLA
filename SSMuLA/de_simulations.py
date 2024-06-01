"""
A script for simluating DE, and SSM recombination and plotting
"""

from __future__ import annotations

import os
import itertools
from glob import glob
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import pandas as pd

from multiprocessing import Pool

import warnings

import holoviews as hv
from holoviews import dim


from SSMuLA.aa_global import ALL_AAS, ALL_AA_STR
from SSMuLA.landscape_global import (
    LIB_INFO_DICT,
    LIB_NAMES,
    TrpB_names,
    n_mut_cutoff_dict,
    make_new_sequence,
    hamming,
)
from SSMuLA.vis import (
    save_bokeh_hv,
    JSON_THEME,
    LIB_COLORS,
    PRESENTATION_PALETTE_SATURATE,
    one_decimal_x,
    one_decimal_y,
    fixmargins,
)
from SSMuLA.util import checkNgen_folder, get_file_name, ecdf_transform


hv.extension("bokeh")
hv.renderer("bokeh").theme = JSON_THEME

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


# order of de simluation from simple to complex
DE_TYPES = deepcopy(["recomb_SSM", "single_step_DE", "top96_SSM"])

DE_COLORS = deepcopy(
    {
        "recomb_SSM": PRESENTATION_PALETTE_SATURATE["light_brown"],
        "single_step_DE": PRESENTATION_PALETTE_SATURATE["light_orange"],
        "top96_SSM": PRESENTATION_PALETTE_SATURATE["brown"],
    }
)

DE_LINE_STYLES = deepcopy(
    {
        "recomb_SSM": "solid",
        "single_step_DE": "dashed",
        "top96_SSM": "dotted",
    }
)

DE_LEGEND_MAP = deepcopy(
    {
        "recomb_SSM": "Recomb",
        "single_step_DE": "Single step",
        "top96_SSM": "Top96 recomb",
    }
)

def simulate_single_step_DE(
    df: pd.DataFrame, seq_col: str, fitness_col: str, n_sites: int = 4
):

    """
    Simulate a single step directed evolution experiment

    Args:
    - df: pd.DataFrame, df of sequence and fitness (without stop codons!)
        - sequence column name
        - fitness column name
    - seq_col: str, the sequence column name
    - fitness_col: str, the fitness column name
    - n_sites: int, number of sites to simulate

    Returns:
    - np.array, the fitness values
    - pd.DataFrame, the results of the simulation
        - start sequence
        - end sequence
        - start fitness
        - end fitness
        - optional: order of steps taken (order that positions were targeted)
    """

    # take out stop codons if any left
    df = df[~df[seq_col].str.contains("\*")].copy()

    df[seq_col] = df[seq_col].apply(lambda x: "".join(x.split("_")))

    # to make things faster
    df_dict = deepcopy(dict(zip(df[seq_col].values, df[fitness_col].values)))

    AAs = df[seq_col].values

    min_fit = df[fitness_col].min()

    position_orders = list(itertools.permutations(range(n_sites)))
    fitness_array = np.empty(len(AAs) * len(position_orders))
    fitness_dict = {}

    for i, start_seq in tqdm(enumerate(AAs)):

        # Draw an initial variant
        start_fitness = df_dict[start_seq]

        # Loop through all possible orders of positions
        for j, temp_order in enumerate(position_orders):

            best_seq = start_seq
            best_fitness = start_fitness

            # Loop through the positions
            for pos in temp_order:

                # Try all possible mutations at the position
                for AA in ALL_AAS:
                    temp_seq = make_new_sequence(best_seq, AA, pos)

                    # In case the AA combo doesn't exist in the dataframe
                    if temp_seq in df_dict.keys():
                        temp_fitness = df_dict[temp_seq]
                    else:
                        temp_fitness = min_fit

                    # If this sequence is better than any previous then keep it
                    if temp_fitness > best_fitness:
                        best_seq = temp_seq
                        best_fitness = temp_fitness
                    else:
                        pass

            fitness_array[len(position_orders) * i + j] = best_fitness
            fitness_dict[(start_seq, temp_order)] = [
                start_fitness,
                best_seq,
                best_fitness,
            ]

    output_df = (
        pd.DataFrame(fitness_dict)
        .T.reset_index()
        .rename(
            columns={
                "level_0": "start_seq",
                "level_1": "order",
                0: "start_fitness",
                1: "final_seq",
                2: "final_fitness",
            }
        )
    )

    output_df["final_fitness ECDF"] = (
        output_df["final_fitness"].transform(ecdf_transform).values
    )

    return fitness_array, output_df


def simulate_simple_recomb_SSM_DE(
    df: pd.DataFrame, seq_col: str, fitness_col: str, n_sites: int = 4
) -> pd.DataFrame:

    """
    Simulate a simple SSM recombination experiment

    Args:
    - df: pd.DataFrame, df of sequence and fitness (without stop codons!)
        - sequence column name
        - fitness column name
    - seq_col: str, the sequence column name
    - fitness_col: str, the fitness column name
    - n_sites: int, number of sites to simulate

    Returns:
    - pd.DataFrame, the results of the simulation
        - start sequence
        - end sequence
        - start fitness
        - end fitness
        - optional: order of steps taken (order that positions were targeted)
    """

    # take out stop codons if any left
    df = df[~df[seq_col].str.contains("\*")].copy()

    df[seq_col] = df[seq_col].apply(lambda x: "".join(x.split("_")))

    df_dict = deepcopy(dict(zip(df[seq_col].values, df[fitness_col].values)))

    AAs = df[seq_col].values

    min_fit = df[fitness_col].min()

    fitness_dict = {}

    for start_seq in tqdm(AAs):

        # Draw an initial variant
        start_fitness = df_dict[start_seq]

        top_SSM_variants = {}

        # Loop through the positions
        for pos in range(n_sites):

            best_seq = start_seq
            best_fitness = start_fitness

            # Try all possible mutations at the position to find the best
            for AA in ALL_AAS:
                temp_seq = make_new_sequence(start_seq, AA, pos)

                # In case the AA combo doesn't exist in the dataframe
                if temp_seq in df_dict.keys():
                    temp_fitness = df_dict[temp_seq]
                else:
                    temp_fitness = min_fit

                # If this sequence is better than any previous then keep it
                if temp_fitness > best_fitness:
                    best_seq = temp_seq
                    best_fitness = temp_fitness
                else:
                    pass

            # collect the best variant for this position
            top_SSM_variants[pos] = best_seq

        # simple recombination
        recomb_seq = "".join([top_SSM_variants[pos][pos] for pos in range(n_sites)])

        if recomb_seq in df_dict.keys():
            recomb_fitness = df_dict[recomb_seq]
        else:
            recomb_fitness = min_fit

        best_seq = start_seq
        best_fitness = start_fitness

        # check if the recombined sequence is better than
        # the starting sequence as well as all of the SSM variants.
        # Return the best one of these.
        if recomb_fitness > best_fitness:
            best_seq = recomb_seq
            best_fitness = recomb_fitness

        for SSM_seq in top_SSM_variants.values():
            SSM_fit = df_dict[SSM_seq]
            if SSM_fit > best_fitness:
                best_seq = SSM_seq
                best_fitness = SSM_fit

        fitness_dict[start_seq] = [
            start_fitness,
            tuple(top_SSM_variants.values()),
            best_seq,
            best_fitness,
        ]

    output_df = (
        pd.DataFrame(fitness_dict)
        .T.reset_index()
        .rename(
            columns={
                "index": "start_seq",
                0: "start_fitness",
                1: "top_SSM_variants",
                2: "final_seq",
                3: "final_fitness",
            }
        )
    )

    output_df["final_fitness ECDF"] = (
        output_df["final_fitness"].transform(ecdf_transform).values
    )

    return output_df


def try_start_seq(start_seq: str, df_dict: dict, ALL_AAS: list, n_sites: int, N: int):

    """
    Try a starting sequence for the SSM predict top N experiment

    Args:
    - start_seq: str, the starting sequence
    - df_dict: dict, the fitness values
    - ALL_AAS: list, all amino acids
    - n_sites: int, number of sites to simulate
    - N: int, the number of top sequences to predict

    Returns:
    - tuple, the starting fitness, the best sequence, and the best fitness
    """

    # Draw an initial variant
    start_fitness = df_dict[start_seq]

    min_fit = min(df_dict.values())

    SSM_data = {}
    SSM_to_compare = {}

    # Loop through the positions to collect SSM data
    for pos in range(n_sites):

        SSM_data[pos] = {}
        SSM_to_compare[pos] = {}

        # Try all possible mutations at the position to find the best
        for AA in ALL_AAS:
            temp_seq = make_new_sequence(start_seq, AA, pos)

            # Check in case the AA combo doesn't exist in the dataframe
            if temp_seq in df_dict.keys():
                temp_fitness = df_dict[temp_seq]
            else:
                temp_fitness = min_fit

            SSM_data[pos][AA] = temp_fitness
            SSM_to_compare[pos][temp_seq] = temp_fitness

    all_possible_combos = [
        "".join(x) for x in list(itertools.product(ALL_AA_STR, repeat=n_sites))
    ]

    calculated_improvement = {}

    for combo in all_possible_combos:
        calculated_improvement[combo] = np.product(
            [SSM_data[i][combo[i]] / start_fitness for i in range(n_sites)]
        )

    top_predicted = (
        pd.DataFrame(
            calculated_improvement.items(), columns=["AAs", "calculated improvement"]
        )
        .sort_values("calculated improvement", ascending=False)
        .head(N)["AAs"]
        .values
    )

    best_seq = start_seq
    best_fitness = start_fitness

    for variant_seq in top_predicted:

        if variant_seq in df_dict.keys():
            variant_fit = df_dict[variant_seq]
        else:
            variant_fit = min_fit

        if variant_fit > best_fitness:
            best_seq = variant_seq
            best_fitness = variant_fit

    # add a step where I also look at all the SSM variants and 
    # see if any of them are better than the top predicted
    for pos, temp_fit_dict in SSM_data.items():
        for SSM_seq, SSM_fit in temp_fit_dict.items():

            if SSM_fit > best_fitness:
                best_seq = SSM_seq
                best_fitness = SSM_fit

    return (start_fitness, best_seq, best_fitness)


def sample_SSM_test_top_N(
    df: pd.DataFrame,
    seq_col: str,
    fitness_col: str,
    n_sites: int = 4,
    N: int = 96,
    max_samples: int | None = None,
    n_jobs: int = 1,
):

    """
    Simulate the SSM predict top N experiment

    Args:
    - df: pd.DataFrame, df of sequence and fitness (without stop codons!)
        - sequence column name
        - fitness column name
    - seq_col: str, the sequence column name
    - fitness_col: str, the fitness column name
    - n_sites: int, number of sites to simulate
    - N: int, the number of top sequences to predict
    - max_samples: int, the maximum number of samples to simulate
    - n_jobs: int, the number of jobs to use for multiprocessing

    Returns:
    - pd.DataFrame, the results of the simulation
        - start sequence
        - end sequence
        - start fitness
        - end fitness
        - optional: order of steps taken (order that positions were targeted)
    """

    # take out stop codons if any left
    df = df[~df[seq_col].str.contains("\*")].copy()

    df[seq_col] = df[seq_col].apply(lambda x: "".join(x.split("_")))

    df_dict = deepcopy(dict(zip(df[seq_col].values, df[fitness_col].values)))

    AAs = df.sample(frac=1)["AAs"].values

    if max_samples is not None and isinstance(max_samples, int):
        AAs = AAs[:max_samples]

    fitness_dict = {}

    # Get the multiprocessing args
    pool_args = [(start_seq, df_dict, ALL_AAS, n_sites, N) for start_seq in AAs]

    with Pool(n_jobs) as pool:
        results = pool.starmap(try_start_seq, tqdm(pool_args))

    fitness_dict = {AAs[i]: results[i] for i in range(len(AAs))}

    output_df = (
        pd.DataFrame(fitness_dict)
        .T.reset_index()
        .rename(
            columns={
                "index": "start_seq",
                0: "start_fitness",
                1: "final_seq",
                2: "final_fitness",
            }
        )
    )

    output_df["final_fitness ECDF"] = (
        output_df["final_fitness"].transform(ecdf_transform).values
    )

    return output_df


def calc_char(
    df: pd.DataFrame, col_name: str = "final_fitness", topns: list = [96, 384]
) -> dict:
    """
    Calculate the mean, median for all and for topn, and fraction reaching max fitness

    Args:
    - df: pd.DataFrame, the dataframe to calculate the characteristics for
    - col_name: str, the column name to calculate the characteristics for
    - topns: list, the top N values to calculate the fraction reaching max fitness for

    Returns:
    - dict, the characteristics, with keys:
        - mean_all
        - median_all
        - mean_topN
        - median_topN
        - fraction_max
    """

    characteristics = {}

    characteristics["max"] = np.max(df[col_name].values)
    characteristics["mean_all"] = np.mean(df[col_name].values)
    characteristics["median_all"] = np.median(df[col_name].values)

    for topn in topns:
        if topn <= len(df):
            characteristics[f"mean_top{topn}"] = np.mean(
                df[col_name].sort_values(ascending=False).head(topn).values
            )
            characteristics[f"median_top{topn}"] = np.median(
                df[col_name].sort_values(ascending=False).head(topn).values
            )
        else:
            characteristics[f"mean_top{topn}"] = np.nan
            characteristics[f"median_top{topn}"] = np.nan

    characteristics["fraction_max"] = sum(df[col_name].values == 1) / len(
        df[col_name].values
    )
    characteristics["numb_path"] = len(df[col_name].values)

    print("Output dict:")
    for key, value in characteristics.items():
        print(f"{key}: {value}")

    return characteristics


def run_all_de_simulations(
    df: pd.DataFrame,
    lib_name: str,
    save_dir: str = "results/de/DE",
    seq_col: str = "AAs",
    fitness_col: str = "fitness",
    n_sites: int = 4,
    N: int = 96,
    topns: list = [96, 384],
    max_samples: int | None = None,
    n_jobs: int = 256,
    rerun: bool = False,
):

    """
    Run all DE simulations and save results to a csv

    Args:
    - df: pd.DataFrame, df of sequence and fitness (without stop codons!)
        - sequence column name
        - fitness column name
    - lib_name: str, the library name
    - save_dir: str, the directory to save the results to
    - seq_col: str, the sequence column name
    - fitness_col: str, the fitness column name
    - n_sites: int, number of sites to simulate
    - N: int, the number of top sequences to predict
    - topns: list, the top N values to calculate the characteristics
    - max_samples: int, the maximum number of samples to simulate
    - n_jobs: int, the number of jobs to use for multiprocessing
    - rerun: bool, rerun the simulations

    Returns:
    - dict, the results of the simulations
    - pd.DataFrame, the summary characteristics of the simulations
    """

    save_dir = checkNgen_folder(save_dir)

    if rerun:

        ######## Simulate a single step directed evolution walk ########
        print("Simulate a single step directed evolution walk")
        fitness_array, single_step_DE = simulate_single_step_DE(
            df=df,
            seq_col=seq_col,
            fitness_col=fitness_col,
            n_sites=n_sites,
        )

        # get the characteristics of the simulation
        single_step_DE_char = calc_char(
            single_step_DE, col_name="final_fitness", topns=topns
        )

        # save reults to csv
        single_step_DE.to_csv(
            os.path.join(save_dir, f"{lib_name}-single_step_DE.csv"), index=False
        )

        ######## Simulate a simple SSM recombination ########
        print("\nSimulate a simple SSM recombination")
        recomb_SSM = simulate_simple_recomb_SSM_DE(
            df=df,
            seq_col=seq_col,
            fitness_col=fitness_col,
            n_sites=n_sites,
        )

        # get the characteristics of the simulation
        recomb_SSM_char = calc_char(recomb_SSM, col_name="final_fitness", topns=topns)

        # save reults to csv
        recomb_SSM.to_csv(os.path.join(save_dir, f"{lib_name}-recomb_SSM.csv"), index=False)

        ######## Simulate SSM predict top N ########
        print(f"\nSimulate SSM predict top {N}")
        top96_SSM = sample_SSM_test_top_N(
            df=df,
            seq_col=seq_col,
            fitness_col=fitness_col,
            n_sites=n_sites,
            N=N,
            max_samples=max_samples,
            n_jobs=n_jobs,
        )

        # get the characteristics of the simulation
        top96_SSM_char = calc_char(top96_SSM, col_name="final_fitness", topns=topns)

        # save reults to csv
        top96_SSM.to_csv(os.path.join(save_dir, f"{lib_name}-top{N}_SSM.csv"), index=False)
    
    else:
        print("Results already exist, loading from csv...")

        single_step_DE = pd.read_csv(
            os.path.join(save_dir, f"{lib_name}-single_step_DE.csv")
        )
        recomb_SSM = pd.read_csv(os.path.join(save_dir, f"{lib_name}-recomb_SSM.csv"))
        top96_SSM = pd.read_csv(os.path.join(save_dir, f"{lib_name}-top{N}_SSM.csv"))

        single_step_DE_char = calc_char(
            single_step_DE, col_name="final_fitness", topns=topns
        )
        recomb_SSM_char = calc_char(recomb_SSM, col_name="final_fitness", topns=topns)
        top96_SSM_char = calc_char(top96_SSM, col_name="final_fitness", topns=topns)

    # Create an initially empty DataFrame
    char_sum_df = pd.DataFrame()

    for de_sim_name, de_sim_char in zip(
        ["single_step_DE", "recomb_SSM", f"top{N}_SSM"],
        [single_step_DE_char, recomb_SSM_char, top96_SSM_char],
    ):
        print(
            {
                "lib": lib_name,
                "de_type": de_sim_name,
                **de_sim_char,
            }
        )

        char_sum_df = char_sum_df._append(
            {
                "lib": lib_name,
                "de_type": de_sim_name,
                **de_sim_char,
            },
            ignore_index=True,
        )

    return {
        "single_step_DE": single_step_DE,
        "recomb_SSM": recomb_SSM,
        f"top{N}_SSM": top96_SSM,
    }, char_sum_df.copy()


# Run simulations for each library
def run_all_lib_de_simulations(
    scale_types: list = ["scale2max", "scale2parent"],
    de_opts: list = ["DE-active", "DE-0", "DE-all"],
    save_dir: str = "results/de",
    all_lib: bool = True,
    lib_list: list[str] = [],
    rerun: bool = False,
):
    """
    Run all simulations for each library.

    Args:
    - scale_types: list, the scale types of fitness to simulate
    - de_opts: list, the DE options to simulate
    - save_dir: str, the directory to save the results to
    - all_lib: bool, run all libraries
    - lib_list: list, the list of libraries to simulate
    - rerun: bool, rerun the simulations
    """
    for scale_type in scale_types:
        for de_det in de_opts:

            all_char_sum_df_list = []

            if all_lib or len(lib_list) == 0:
                lib_csv_list = sorted(glob(f"data/*/{scale_type}/*.csv"))
            else:
                lib_csv_list = sorted([
                        f"data/{lib}/{scale_type}/{lib}.csv" for lib in lib_list
                ])

            # Run simulations for each library
            for lib in lib_csv_list:

                lib_name = get_file_name(lib)
                n_sites = len(LIB_INFO_DICT[lib_name]["positions"])

                print(
                    "Running {} simulations for {} over {} with fitness {}...".format(
                        de_det, lib_name, n_sites, scale_type
                    )
                )

                df = pd.read_csv(lib).copy()

                # take out stop codons
                df = df[~df["AAs"].str.contains("\*")].copy()

                if de_det == "DE-all":
                    select_df = df.copy()
                elif de_det == "DE-0":
                    select_df = df[df["fitness"] >= 0].copy()
                elif de_det == "DE-active":
                    select_df = df[df["active"]].copy()

                save_subdir = f"{save_dir}/{de_det}/{scale_type}"

                _, char_sum_df = run_all_de_simulations(
                    df=select_df,
                    seq_col="AAs",
                    fitness_col="fitness",
                    lib_name=lib_name,
                    save_dir=save_subdir,
                    n_sites=n_sites,
                    N=96,
                    topns=[96, 384],
                    max_samples=None,
                    n_jobs=256,
                    rerun=rerun,
                )

                all_char_sum_df_list.append(char_sum_df)

            all_char_sum_df = pd.concat(all_char_sum_df_list, ignore_index=True)
            all_char_sum_df.to_csv(
                f"{save_subdir}/all_landscape_de_summary.csv", index=False
            )


def de_violin(
    slice_df: pd.DataFrame,
    lib_name: str,
    plot_name: str,
    plot_folder: str,
    v_width: int = 400,
):

    """
    A function to plot a violin plot of the DE simulation results

    Args:
    - slice_df (pd.DataFrame): A dataframe containing the DE simulation results
    - lib_name (str): The name of the library to plot
    - plot_name (str): The name of the plot
    - plot_folder (str): The folder to save the plot
    - v_width (int): Width of the violin plot
    """

    print("Plotting DE max fitness achieved violin...")

    if lib_name == "TrpB":
        v_width = 1280
        cmap = [LIB_COLORS[lib_name] for lib_name in TrpB_names]
        xrotation = 45
    else:
        v_width = 400
        cmap = [LIB_COLORS[lib_name]]
        xrotation = 0

    violin = hv.Violin(
        slice_df,
        kdims=["simulation", "lib"],
        vdims=["final_fitness"],
    ).opts(
        violin_color=dim("lib").str(),
        cmap=cmap,
        width=v_width,
        height=300,
        violin_width=0.8,
        title=plot_name,
        hooks=[fixmargins, one_decimal_y],
        ylabel="Max fitness achieved",
        xrotation=xrotation,
    )

    save_bokeh_hv(
        violin,
        plot_name=plot_name,
        plot_path=checkNgen_folder(os.path.join(plot_folder, "violin")),
        bokehorhv="hv",
        dpi=300,
        scale=2,
    )

    return violin


def de_ecdf(
    slice_df: pd.DataFrame,
    lib_name: str,
    plot_name: str,
    plot_folder: str,
    skippng: bool = False,
):

    """
    A function to plot an ECDF of the DE simulation results

    Args:
    - slice_df: pd.DataFrame, A dataframe containing the DE simulation results
    - lib_name: str, The name of the library to plot
    - plot_name: str, The name of the plot
    - plot_folder: str, The folder to save the plot
    - skippng: bool, Skip plotting GB1 and TrpB
    """

    print("Plotting DE max fitness achieved ECDF...")

    # Initialize an empty HoloViews Overlay container
    overlay = hv.Overlay()

    if lib_name == "TrpB":
        # Initialize a dictionary to store legend labels
        sim_legend_labels = {}
        lib_legend_labels = {}

        # Iterate over simulation types and libraries, create individual traces, and overlay them
        for sim, style in DE_LINE_STYLES.items():
            for lib_name in TrpB_names:
                selection = slice_df[
                    (slice_df["simulation"] == sim) & (slice_df["lib"] == lib_name)
                ]
                if not selection.empty:
                    curve = hv.Curve(
                        selection.sort_values(
                            ["simulation", "lib", "final_fitness", "final_fitness ECDF"]
                        ),
                        kdims="final_fitness",
                        vdims=["final_fitness ECDF", "lib", "simulation"],
                    )
                    overlay *= curve.opts(
                        line_dash=style,
                        color=LIB_COLORS[lib_name],
                        width=1200,
                        height=800,
                        title=plot_name,
                        hooks=[fixmargins, one_decimal_x, one_decimal_y],
                        xlabel="Max fitness achieved",
                        ylabel="ECDF",
                    )

                sim_legend_labels[sim] = hv.Curve([0], label=sim).opts(
                    line_dash=style, color="gray"
                )
                lib_legend_labels[lib_name] = hv.Curve([0], label=lib_name).opts(
                    line_dash="solid", color=LIB_COLORS[lib_name]
                )

        legend_labels = {**deepcopy(sim_legend_labels), **deepcopy(lib_legend_labels)}

        layout = overlay.opts(
            title=plot_name,
            hooks=[fixmargins, one_decimal_x, one_decimal_y],
            xlabel="Max fitness achieved",
            ylabel="ECDF",
        )
    else:
        # Initialize a dictionary to store legend labels
        legend_labels = {}

        # Iterate over simulation types and libraries, create individual traces, and overlay them
        for sim, style in DE_LINE_STYLES.items():
            # for lib_name, lib_color in LIB_COLORS.items():
            selection = slice_df[
                (slice_df["simulation"] == sim) & (slice_df["lib"] == lib_name)
            ]
            if not selection.empty:
                curve = hv.Curve(
                    selection.sort_values(
                        ["simulation", "lib", "final_fitness", "final_fitness ECDF"]
                    ),
                    kdims="final_fitness",
                    vdims=["final_fitness ECDF", "lib", "simulation"],
                )
                overlay *= curve.opts(
                    line_dash=style,
                    color=LIB_COLORS[lib_name],
                    width=500,
                    height=300,
                    title=plot_name,
                    hooks=[fixmargins, one_decimal_x, one_decimal_y],
                    xlabel="Max fitness achieved",
                    ylabel="ECDF",
                )
                legend_labels[sim] = hv.Curve([0], label=sim).opts(
                    line_dash=style, color=LIB_COLORS[lib_name]
                )

    layout = overlay.opts(
        title=f"{lib_name} max fitness achieved no imputed",
        hooks=[fixmargins, one_decimal_x, one_decimal_y],
        xlabel="Max fitness achieved",
        ylabel="ECDF",
    )

    if skippng:
        skippng = lib_name == "GB1" or lib_name == "TrpB"
    else:
        skippng = False

    # Add the legend to the layout
    save_bokeh_hv(
        layout
        * hv.NdOverlay(legend_labels).opts(
            legend_position="right", legend_offset=(10, 100), legend_limit=50
        ),
        plot_name=plot_name,
        plot_path=checkNgen_folder(os.path.join(plot_folder, "ecdf")),
        bokehorhv="hv",
        dpi=300,
        scale=2,
        skippng=skippng,
    )


class VisDESims:

    """
    Class for visualizing DE simulation results
    """

    def __init__(
        self,
        lib_name: str,
        append_title: str = "max fitness achieved",
        v_width: int = 400,
        sim_folder: str = "results/de",
        de_sub_folder: str = "DE-active",
        fit_scale_sub_folder: str = "scale2max",
        n_mut_cutoff: int = 0,
        vis_folder: str = "results/de_vis",
    ) -> None:

        """
        Args:
        - lib_name, str: The name of the library to plot
        - append_title, str: Additional title to append to the plot title
        - v_width, int: Width of the violin plot
        - sim_folder, str: Path to the DE simulation results
        - de_sub_folder, str: Subfolder of the DE simulation results
        - fit_scale_sub_folder, str: Subfolder of the DE simulation results
        - n_mut_cutoff, int: Number of mutations to cutoff
        - vis_folder, str: Path to save the DE simulation plots
        """

        self._lib_name = lib_name
        self._append_title = append_title
        self._v_width = v_width
        self._sim_folder = os.path.normpath(sim_folder)
        self._de_sub_folder = de_sub_folder
        self._fit_scale_sub_folder = fit_scale_sub_folder
        self._n_mut_cutoff = n_mut_cutoff
        self._vis_folder = checkNgen_folder(os.path.normpath(vis_folder))

        print(
            "Visualizing DE simulation results for {} {} {}...".format(
                lib_name, de_sub_folder, vis_folder
            )
        )
        self._plot_violin_ecdf()

    def _plot_violin_ecdf(self):

        """
        Plot a violin plot and ecdf of the DE simulation results

        Args:
        - lib_name: str, The name of the library to plot
        - append_title: str, Additional title to append to the plot title
        - v_width: int, Width of the violin plot
        """

        if self._lib_name == "TrpB":
            # all trpb
            slice_df = (
                self.all_df[self.all_df["lib"].isin(TrpB_names)]
                .sort_values(["simulation", "lib", "final_fitness"])
                .copy()
            )
        else:
            slice_df = (
                self.all_df[self.all_df["lib"] == self._lib_name]
                .sort_values(["simulation", "lib", "final_fitness"])
                .copy()
            )

        plot_name = f"{self._lib_name} {self._append_title} start from {n_mut_cutoff_dict[self._n_mut_cutoff]}"
        plot_folder = checkNgen_folder(
            os.path.join(
                self._vis_folder,
                self._de_sub_folder,
                self._fit_scale_sub_folder,
                n_mut_cutoff_dict[self._n_mut_cutoff],
            )
        )

        self._violin = de_violin(
            slice_df=slice_df,
            lib_name=self._lib_name,
            plot_name=plot_name,
            plot_folder=plot_folder,
            v_width=self._v_width,
        )

        self._ecdf = de_ecdf(
            slice_df=slice_df,
            lib_name=self._lib_name,
            plot_name=plot_name,
            plot_folder=plot_folder,
            skippng=(self._n_mut_cutoff == 0 and self._de_sub_folder == "DE-all"),
        )

    @property
    def all_df(self) -> pd.DataFrame:

        """
        Combine all DE simulation results into a single dataframe

        Returns:
        - all_df: pd.DataFrame, A dataframe containing all DE simulation results
        """

        all_de_sim_files = sorted(
            glob(f"{os.path.normpath(self.de_folder_full_path)}/*.csv")
        )

        all_df = pd.DataFrame()

        for res in all_de_sim_files:
            # do not include the summary file
            if "all_landscape" not in res:
                lib_name, sim_name = get_file_name(res).split("-")

                df = pd.read_csv(res)

                df["simulation"] = sim_name
                df["lib"] = lib_name

                # slice df based on n_mut_cutoff
                if self._n_mut_cutoff > 0:
                    df["n_mut"] = df["start_seq"].apply(
                        hamming, str2="".join(LIB_INFO_DICT[lib_name]["AAs"].values())
                    )
                    df = df[df["n_mut"] <= self._n_mut_cutoff]
                    df["final_fitness ECDF"] = (
                        df["final_fitness"].transform(ecdf_transform).values
                    )

                all_df = all_df._append(df)

        all_df = all_df.reset_index(drop=True)
        all_df["final_fitness"] = all_df["final_fitness"].astype(float)

        # sort sim to be in custom order from easy to hard
        all_df["simulation"] = pd.Categorical(
            all_df["simulation"], categories=DE_TYPES, ordered=True
        )

        return all_df.sort_values("simulation")

    @property
    def de_folder_full_path(self) -> str:
        """Return the path to the DE simulation results"""
        return os.path.join(
            self._sim_folder, self._de_sub_folder, self._fit_scale_sub_folder
        )

    @property
    def de_violin(self) -> hv.Violin:
        """Return the violin plot"""
        return self._violin

    @property
    def de_ecdf(self) -> hv.Curve:
        """Return the ecdf plot"""
        return self._ecdf


def run_plot_de(
    scale_types: list = ["scale2max", "scale2parent"],
    de_opts: list = ["DE-active"],
    sim_folder: str = "results/de",
    vis_folder: str = "results/de_vis",
    v_width: int = 400,
    all_lib: bool = True,
    lib_list: list[str] = [],
):

    """
    Run the DE simulation plotting

    Args:
    - scale_types: list, The scale types of fitness
    - de_opts: list, The DE options to plot
    - sim_folder: str, Path to the DE simulation results
    - vis_folder: str, Path to save the DE simulation plots
    - v_width: int, Width of the violin plot
    - all_lib: bool, Run all libraries
    - lib_list: list, The list of libraries to simulate
    """

    for de_sub_folder in de_opts:
        for fit_scale_sub_folder in scale_types:
            for n_mut in [0, 1, 2]:

                if all_lib or len(lib_list) == 0:
                    lib_list = LIB_NAMES + ["TrpB"]
                else:
                    lib_list = lib_list + ["TrpB"]

                for lib in tqdm(lib_list):
                    if "TrpB" in lib:
                        v_width = 1280
                    else:
                        v_width = 400

                    _ = VisDESims(
                        lib_name=lib,
                        append_title="max fitness achieved",
                        v_width=v_width,
                        sim_folder=sim_folder,
                        de_sub_folder=de_sub_folder,
                        fit_scale_sub_folder=fit_scale_sub_folder,
                        n_mut_cutoff=n_mut,
                        vis_folder=vis_folder,
                    )