"""A script for getting all results into a summary"""

from glob import glob

import pandas as pd
import numpy as np

from ast import literal_eval


import bokeh
from bokeh.io import show, export_svg, export_png, output_notebook
from bokeh.plotting import show, figure
from bokeh.themes.theme import Theme


import holoviews as hv
from holoviews import opts, dim

from SSMuLA.landscape_global import get_file_name
from SSMuLA.zs_analysis import ZS_OPTS
from SSMuLA.vis import LIB_COLORS


output_notebook()
bokeh.io.output_notebook()

hv.extension("bokeh", "matplotlib")


########## lib stat ##########
lib_stat = pd.read_csv("results/fitness_distribution/max/all_lib_stats.csv", index_col=0)
lib_stat_all = lib_stat[lib_stat["n_mut_cuttoff"]== 0]

lib_df = pd.concat(
        [
            lib_stat_all["lib"],
            lib_stat_all["lib_basic_dict"].apply(literal_eval).apply(pd.Series),
            lib_stat_all["fit_basic_dict"].apply(literal_eval).apply(pd.Series),
            lib_stat_all["cauchy"].apply(literal_eval).apply(pd.Series),
            lib_stat_all["kde"].apply(literal_eval).apply(pd.Series),

        ],
        axis=1,
    )
lib_df["parent_rank_percent"] = lib_df["parent_rank"] / lib_df["numb_measured"]
# qs = pd.DataFrame(lib_df['quartiles'].tolist(), index=lib_df.index)
# qs.columns =   # Rename columns

df_expanded = lib_df['quartiles'].apply(pd.Series)
df_expanded.columns = ['Q1', 'Q2', 'Q3'] # Rename columns
df_expanded["numb_kde_peak"] = lib_df["peak_kde"].apply(len)

lib_df = pd.concat([lib_df, df_expanded], axis=1)


########## loc opt ##########

loc_opt_list = glob("results/local_optima/scale2max/*.csv")
loc_opt_df = pd.DataFrame()

for lo in loc_opt_list:
    lo_df = pd.read_csv(lo)
    loc_opt_df = loc_opt_df._append({"lib": get_file_name(lo).replace("_loc_opt_escape", ""), "n_locopt": len(lo_df)}, ignore_index=True)
    # pd.concat([loc_opt_df, pd.read_csv(lo)], ignore_index=True)

if "GB1" not in loc_opt_df["lib"].values:
    loc_opt_df = loc_opt_df._append({"lib": "GB1", "n_locopt": 181}, ignore_index=True)


########## de stat ##########

de_df = pd.read_csv("results/simulations/DE-active/scale2max/all_landscape_de_summary.csv")
# clean up the nan if not full N
# Fill NaNs in 'mean_top96' and 'median_top96' from 'mean_all' and 'median_all'
de_df["mean_top96"] = de_df["mean_top96"].fillna(de_df["mean_all"])
de_df["median_top96"] = de_df["median_top96"].fillna(de_df["median_all"])

# Fill NaNs in 'mean_top384' and 'median_top384' from 'mean_top96' and 'median_top96'
de_df["mean_top384"] = de_df["mean_top384"].fillna(de_df["mean_top96"])
de_df["median_top384"] = de_df["median_top384"].fillna(de_df["median_top96"])

de_types = de_df["de_type"].unique()

de_df_cols = list(de_df.columns)

de_df_concat = pd.DataFrame({"lib": list(de_df["lib"].unique())})

for de_type in de_types:

    sliced_de_df = de_df[de_df["de_type"] == de_type]

    for de_df_col in de_df_cols:
        if de_df_col not in ["lib", "de_type"]:
            sliced_de_df = sliced_de_df.rename(
                columns={de_df_col: f"{de_type}_{de_df_col}"}
            ).drop(["de_type"], axis=1)

    de_df_concat = pd.merge(de_df_concat, sliced_de_df, on="lib")


########## mlde stat ##########

mlde_df = pd.read_csv("results/mlde/vis_2/all_df.csv")

mlde_avg = mlde_df[
    (mlde_df["zs"] == "none")
    & (mlde_df["encoding"] == "one-hot")
    & (mlde_df["model"] == "boosting")
    & (mlde_df["n_mut_cutoff"] == "all")
    & (mlde_df["n_sample"] == 384)
    & (mlde_df["n_top"] == 96)
][["lib", "top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]].groupby("lib").mean()

zs_simpele = [zs.split("_score")[0] for zs in ZS_OPTS]

for zs in zs_simpele:

    rename_cols = {
        "top_maxes": f"maxes_{zs}",
        "top_means": f"means_{zs}",
        "ndcgs": f"ndcgs_{zs}",
        "rhos": f"rhos_{zs}",
        "if_truemaxs": f"if_truemaxs_{zs}",
    }

    mlde_avg = pd.merge(mlde_avg, (
                                    mlde_df[
                                        (mlde_df["zs"] == f"{zs}_score")
                                        & (mlde_df["encoding"] == "one-hot")
                                        & (mlde_df["model"] == "boosting")
                                        & (mlde_df["n_mut_cutoff"] == "all")
                                        & (mlde_df["n_sample"] == 384)
                                        & (mlde_df["n_top"] == 96)
                                    ][["lib", "top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]]
                                    .groupby("lib")
                                    .mean()
                                    .rename(
                                        columns=rename_cols
                                    )
                                ), on="lib")

merged_active_de = pd.merge(lib_df, de_df_concat, on="lib")
simple_merge = pd.merge(merged_active_de, mlde_avg, on="lib")
simple_merge = pd.merge(simple_merge, loc_opt_df, on="lib")

##### calc delta #####
for ft_col in [""] + zs_simpele:
    for de in de_types:
        if ft_col == "":
            simple_merge[f"mlde_{de}_delta"] = simple_merge["top_maxes"] - simple_merge[f"{de}_mean_all"]
        else:
            simple_merge[f"{ft_col}_{de}_delta"] = simple_merge[f"maxes_{ft_col}"] - simple_merge[f"{de}_mean_all"]


####### epistasis #######

df_pw = pd.read_csv("results/pairwise_epistasis_vis/none/scale2max.csv")

df_pw_s_rs = df_pw[
    (df_pw["n_mut"] == "all")
    & (df_pw["summary_type"] == "fraction")
    & (df_pw["epistasis_type"] != "magnitude")
][["lib", "value"]].groupby("lib").sum().rename(columns={"value": "fraction_non-magnitude"})

df_pw_rs = df_pw[
    (df_pw["n_mut"] == "all")
    & (df_pw["summary_type"] == "fraction")
    & (df_pw["epistasis_type"] == "reciprocal sign")
][["lib", "value"]].rename(columns={"value": "fraction_reciprocal-sign"})

df_pw = pd.merge(df_pw_s_rs, df_pw_rs, on="lib")

simple_merge_pw = pd.merge(simple_merge, df_pw, on="lib")

simple_merge_pw["norm_non-magnitude"] = simple_merge_pw["fraction_non-magnitude"] * simple_merge_pw["percent_active"]
simple_merge_pw["norm_reciprocal-sign"] = simple_merge_pw["fraction_reciprocal-sign"] * simple_merge_pw["percent_active"]