"""
A script for simluating DE, and SSM recombination
"""

from __future__ import annotations

import os
from glob import glob

import pandas as pd
import numpy as np

from multiprocessing import Pool

import itertools
from tqdm import tqdm

from SSMuLA.aa_global import ALL_AAS, ALL_AA_STR
from SSMuLA.landscape_global import LIB_INFO_DICT
from SSMuLA.util import checkNgen_folder, get_file_name


def make_new_sequence(input_seq: str, new_AA: str, position: int) -> str:

    """
    Make a new sequence by replacing the amino acid at a given position

    Args:
    - input_seq: str, the input sequence
    - new_AA: str, the new amino acid to replace the old one
    - position: int, the position to replace the amino acid

    Returns:
    - str, the new sequence
    """

    seq_list = list(input_seq)
    seq_list[position] = new_AA
    return "".join(seq_list)


def ecdf_transform(data: pd.Series) -> pd.Series:

    """
    Transform a series of fitness values into an empirical cumulative distribution function

    Args:
    - data: pd.Series, the fitness values

    Returns:
    - pd.Series, the ECDF
    """

    return data.rank(method="first") / len(data)


def simulate_single_step_DE(data, seq_col, fitness_col, n_sites=4):

    """
    Simulate a single step directed evolution experiment

    #### Inputs
    - DataFrame of sequence and fitness (without stop codons!)
        - sequence column name
        - fitness column name
    - zero-fitness cutoff
    - number of samples to simulate

    #### Outputs
    - DataFrame
        - start sequence
        - end sequence
        - start fitness
        - end fitness
        - optional: order of steps taken (order that positions were targeted)
    """

    data = data.copy()
    data[seq_col] = data[seq_col].apply(lambda x: "".join(x.split("_")))

    data_dict = dict(zip(data[seq_col].values, data[fitness_col].values))

    active_AAs = data[data["active"]][seq_col].values

    position_orders = list(itertools.permutations(range(n_sites)))
    fitness_array = np.empty(len(active_AAs) * len(position_orders))
    fitness_dict = {}

    for i, start_seq in tqdm(enumerate(active_AAs)):

        # Draw an initial variant
        start_fitness = data_dict[start_seq]

        # Loop through all possible orders of positions
        for j, temp_order in enumerate(position_orders):

            best_seq = start_seq
            best_fitness = start_fitness

            # Loop through the positions
            for pos in temp_order:

                # Try all possible mutations at the position
                for AA in ALL_AAS:
                    temp_seq = make_new_sequence(best_seq, AA, pos)

                    # Use Try/Except in case the AA combo doesn't exist in the dataframe
                    try:
                        temp_fitness = data_dict[temp_seq]
                    except:
                        temp_fitness = 0

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

    return (fitness_array, output_df)


def simulate_simple_recomb_SSM_DE(data, seq_col, fitness_col, n_sites=4):

    data = data.copy()
    data[seq_col] = data[seq_col].apply(lambda x: "".join(x.split("_")))

    data_dict = dict(zip(data[seq_col].values, data[fitness_col].values))

    active_AAs = data[data["active"]][seq_col].values

    fitness_dict = {}

    for start_seq in tqdm(active_AAs):

        # Draw an initial variant
        start_fitness = data_dict[start_seq]

        top_SSM_variants = {}

        # Loop through the positions
        for pos in range(n_sites):

            best_seq = start_seq
            best_fitness = start_fitness

            # Try all possible mutations at the position to find the best
            for AA in ALL_AAS:
                temp_seq = make_new_sequence(start_seq, AA, pos)

                # Use Try/Except in case the AA combo doesn't exist in the dataframe
                try:
                    temp_fitness = data_dict[temp_seq]
                except:
                    temp_fitness = 0

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
        try:
            recomb_fitness = data_dict[recomb_seq]
        except:
            recomb_fitness = 0

        best_seq = start_seq
        best_fitness = start_fitness

        # check if the recombined sequence is better than
        # the starting sequence as well as all of the SSM variants.
        # Return the best one of these.
        if recomb_fitness > best_fitness:
            best_seq = recomb_seq
            best_fitness = recomb_fitness

        for SSM_seq in top_SSM_variants.values():
            SSM_fit = data_dict[SSM_seq]
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


# def sample_SSM_test_top_N(data, seq_col, fitness_col, n_sites=4, N=96, max_samples=1000):

#     data = data.copy()
#     data[seq_col] = data[seq_col].apply(lambda x: ''.join(x.split('_')))

#     data_dict = dict(zip(data[seq_col].values,data[fitness_col].values))

#     active_AAs = data[data['active']].sample(frac=1)['AAs'].values

#     AA_list = list('ACDEFGHIKLMNPQRSTVWY')
#     fitness_dict = {}

#     for sample_counter, start_seq in tqdm.tqdm(enumerate(active_AAs)):

#         # Draw an initial variant
#         start_fitness = data_dict[start_seq]

#         SSM_data = {}
#         SSM_to_compare = {}

#         # Loop through the positions to collect SSM data
#         for pos in range(n_sites):

#             SSM_data[pos] = {}
#             SSM_to_compare[pos] = {}

#             # Try all possible mutations at the position to find the best
#             for AA in ALL_AAS:
#                 temp_seq = make_new_sequence(start_seq, AA, pos)

#                 # Use Try/Except in case the AA combo doesn't exist in the dataframe
#                 try:
#                     temp_fitness = data_dict[temp_seq]
#                 except:
#                     temp_fitness = 0

#                 SSM_data[pos][AA] = temp_fitness
#                 SSM_to_compare[pos][temp_seq] = temp_fitness

#         all_possible_combos = [''.join(x) for x in list(itertools.product('ACDEFGHIKLMNPQRSTVWY', repeat=n_sites))]

#         calculated_improvement = {}

#         for combo in all_possible_combos:
#             calculated_improvement[combo] = np.product([SSM_data[i][combo[i]] / start_fitness for i in range(n_sites)])

#         top_predicted = pd.DataFrame(calculated_improvement.items(), columns=['AAs', 'calculated improvement']).sort_values('calculated improvement', ascending=False).head(N)['AAs'].values

#         best_seq = start_seq
#         best_fitness = start_fitness

#         for variant_seq in top_predicted:

#             try:
#                 variant_fit = data_dict[variant_seq]
#             except:
#                 variant_fit = 0

#             if variant_fit > best_fitness:
#                 best_seq = variant_seq
#                 best_fitness = variant_fit

#         # add a step where I also look at all the SSM variants and see if any of them are better than the top predicted
#         for pos,temp_fit_dict in SSM_data.items():
#             for SSM_seq,SSM_fit in temp_fit_dict.items():

#                 if SSM_fit > best_fitness:
#                     best_seq = SSM_seq
#                     best_fitness = SSM_fit

#         fitness_dict[start_seq] = [start_fitness, best_seq, best_fitness]

#         if sample_counter >= max_samples-1:
#             break

#     output_df = pd.DataFrame(fitness_dict).T.reset_index().rename(columns={'index':'start_seq', 0:'start_fitness', 1:'final_seq', 2:'final_fitness'})

#     output_df["final_fitness ECDF"] = output_df[
#         'final_fitness'
#     ].transform(ecdf_transform).values

#     return output_df


def try_start_seq(start_seq, data_dict, ALL_AAS, n_sites, N):

    # Draw an initial variant
    start_fitness = data_dict[start_seq]

    SSM_data = {}
    SSM_to_compare = {}

    # Loop through the positions to collect SSM data
    for pos in range(n_sites):

        SSM_data[pos] = {}
        SSM_to_compare[pos] = {}

        # Try all possible mutations at the position to find the best
        for AA in ALL_AAS:
            temp_seq = make_new_sequence(start_seq, AA, pos)

            # Use Try/Except in case the AA combo doesn't exist in the dataframe
            try:
                temp_fitness = data_dict[temp_seq]
            except:
                temp_fitness = 0

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

        try:
            variant_fit = data_dict[variant_seq]
        except:
            variant_fit = 0

        if variant_fit > best_fitness:
            best_seq = variant_seq
            best_fitness = variant_fit

    # add a step where I also look at all the SSM variants and see if any of them are better than the top predicted
    for pos, temp_fit_dict in SSM_data.items():
        for SSM_seq, SSM_fit in temp_fit_dict.items():

            if SSM_fit > best_fitness:
                best_seq = SSM_seq
                best_fitness = SSM_fit

    return (start_fitness, best_seq, best_fitness)


def sample_SSM_test_top_N(
    data, seq_col, fitness_col, n_sites=4, N=96, max_samples=None, n_jobs=1
):

    data = data.copy()
    data[seq_col] = data[seq_col].apply(lambda x: "".join(x.split("_")))

    data_dict = dict(zip(data[seq_col].values, data[fitness_col].values))

    active_AAs = data[data["active"]].sample(frac=1)["AAs"].values

    if max_samples is not None and type(max_samples) == int:
        active_AAs = active_AAs[:max_samples]

    fitness_dict = {}

    # Get the multiprocessing args
    pool_args = [
        (start_seq, data_dict, ALL_AAS, n_sites, N) for start_seq in active_AAs
    ]

    with Pool(n_jobs) as pool:
        results = pool.starmap(try_start_seq, tqdm(pool_args))

    fitness_dict = {active_AAs[i]: results[i] for i in range(len(active_AAs))}

    # for start_seq in tqdm.tqdm(active_AAs):
    #     fitness_dict[start_seq] = try_start_seq(start_seq, data_dict, ALL_AAS, n_sites, N)

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


def simulate_iterative_SM(data, seq_col, fitness_col, n_sites=4):

    data = data.copy()
    data[seq_col] = data[seq_col].apply(lambda x: "".join(x.split("_")))

    data_dict = dict(zip(data[seq_col].values, data[fitness_col].values))

    active_AAs = data[data["active"]]["AAs"].values

    fitness_array = np.empty(len(active_AAs) * 1)
    fitness_dict = {}

    for i, start_seq in tqdm(enumerate(active_AAs)):

        # Draw an initial variant
        start_fitness = data_dict[start_seq]

        best_seq = start_seq
        best_fitness = start_fitness

        # Loop through all possible orders of positions
        remaining_positions = list(range(n_sites))
        temp_order = []

        for j in range(n_sites):
            # Loop through the positions
            previous_best_seq = best_seq
            found_improvement = False
            for pos in remaining_positions:

                # Try all possible mutations at the position
                for AA in ALL_AAS:
                    temp_seq = make_new_sequence(previous_best_seq, AA, pos)

                    # Use Try/Except in case the AA combo doesn't exist in the dataframe
                    try:
                        temp_fitness = data_dict[temp_seq]
                    except:
                        temp_fitness = 0

                    # If this sequence is better than any previous then keep it
                    if temp_fitness > best_fitness:
                        best_seq = temp_seq
                        best_fitness = temp_fitness
                        best_site = pos
                        found_improvement = True
                    else:
                        pass

            if found_improvement:
                remaining_positions.remove(best_site)
                temp_order.append(best_site)
            else:
                # finish if there are no more beneficial mutations
                break
            # print(start_seq, best_seq, temp_order)

        # print(best_seq)
        temp_order = tuple(temp_order)
        fitness_array[i] = best_fitness
        fitness_dict[(start_seq, temp_order)] = [start_fitness, best_seq, best_fitness]

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

    return (fitness_array, output_df)


def print_characteristics(df):

    print("mean:", np.mean(df["final_fitness"].values))
    print("median:", np.median(df["final_fitness"].values))
    print(
        "fraction reaching max:",
        sum(df["final_fitness"].values == 1) / len(df["final_fitness"].values),
    )


def run_all_de_simulations(
    df: pd.DataFrame,
    lib_name: str,
    save_dir: str = "results/simulations/DE",
    seq_col: str = "AAs",
    fitness_col: str = "fitness",
    n_sites=4,
    N=96,
    max_samples=None,
    n_jobs=256,
):

    save_dir = checkNgen_folder(save_dir)

    ######## Simulate a single step directed evolution walk ########
    print(f"Simulate a single step directed evolution walk")
    fitness_array, single_step_DE = simulate_single_step_DE(
        data=df,
        seq_col=seq_col,
        fitness_col=fitness_col,
        n_sites=n_sites,
    )
    print_characteristics(single_step_DE)

    single_step_DE.to_csv(
        os.path.join(save_dir, f"{lib_name}-single_step_DE.csv"), index=False
    )

    ######## Simulate a simple SSM recombination ########
    print("\nSimulate a simple SSM recombination")
    recomb_SSM = simulate_simple_recomb_SSM_DE(
        data=df,
        seq_col=seq_col,
        fitness_col=fitness_col,
        n_sites=n_sites,
    )
    print_characteristics(recomb_SSM)

    recomb_SSM.to_csv(os.path.join(save_dir, f"{lib_name}-recomb_SSM.csv"), index=False)

    ######## Simulate SSM predict top N ########
    print(f"\nSimulate SSM predict top {N}")
    top96_SSM = sample_SSM_test_top_N(
        data=df,
        seq_col=seq_col,
        fitness_col=fitness_col,
        n_sites=n_sites,
        N=N,
        max_samples=max_samples,
        n_jobs=n_jobs,
    )
    print_characteristics(top96_SSM)

    top96_SSM.to_csv(os.path.join(save_dir, f"{lib_name}-top{N}_SSM.csv"), index=False)

    return {
        "single step SSM": single_step_DE,
        " recomb": recomb_SSM,
        "SSM predict top 96": top96_SSM,
    }


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