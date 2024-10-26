"""A script for getting all results into a summary"""

import os
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np

from ast import literal_eval

from scipy.stats import spearmanr

import bokeh
from bokeh.io import show, export_svg, export_png, output_notebook
from bokeh.plotting import show, figure
from bokeh.themes.theme import Theme


import holoviews as hv
from holoviews import opts, dim

from SSMuLA.fitness_process_vis import parse_lib_stat
from SSMuLA.landscape_global import get_file_name, n_mut_cutoff_dict
from SSMuLA.de_simulations import DE_TYPES
from SSMuLA.zs_analysis import ZS_OPTS, ZS_COMB_OPTS
from SSMuLA.vis_summary import N_SAMPLE_LIST, N_MUT_SUBS, ZS_METRICS, ZS_N_MUTS
from SSMuLA.vis import LIB_COLORS, save_bokeh_hv
from SSMuLA.util import checkNgen_folder


output_notebook()
bokeh.io.output_notebook()

hv.extension("bokeh", "matplotlib")

LANDSCAPE_ATTRIBUTES = [
    "n_site",
    "numb_measured",
    "percent_measured",
    "numb_active",
    "percent_active",
    "active_fit_min",
    "mean",
    "median",
    "range",
    "iqr",
    "std_dev",
    "variance",
    "skewness",
    "kurt",
    "loc",
    "scale",
    "numb_kde_peak",
    "Q1",
    "Q2",
    "Q3",
    "numb_loc_opt",
    "frac_loc_opt_total",
    "frac_loc_opt_hd2_escape_numb",
    "frac_loc_opt_hd2_cannot_escape_numb",
    "numb_loc_opt_norm_cannot_escape",
    "frac_loc_opt_norm_cannot_escape",
    "fraction_non-magnitude",
    "fraction_reciprocal-sign",
    "norm_non-magnitude",
    "norm_reciprocal-sign",
]

zs_no_score_list = [
    zs.replace("_score", "") for zs in ZS_OPTS + ZS_COMB_OPTS if zs != "ed_score"
]
zs_list = [
    f"{mut}_{opt}_{metric}"
    for mut in ZS_N_MUTS
    for opt in zs_no_score_list
    for metric in ZS_METRICS
]

de_metric = [
    f"{m}_{p}" for m in ["mean", "median"] for p in ["all", "top96", "top384"]
] + ["fraction_max"]

de_list = [f"{de}_{m}" for de in DE_TYPES for m in de_metric]

# "single_step_DE_mean_all",
# "single_step_DE_median_all",
# "single_step_DE_mean_top96",
# "single_step_DE_median_top96",
# "single_step_DE_mean_top384",
# "single_step_DE_median_top384",
# "single_step_DE_fraction_max",
# "recomb_SSM_mean_all",
# "recomb_SSM_median_all",
# "recomb_SSM_mean_top96",
# "recomb_SSM_median_top96",
# "recomb_SSM_mean_top384",
# "recomb_SSM_median_top384",
# "recomb_SSM_fraction_max",
# "top96_SSM_mean_all",
# "top96_SSM_median_all",
# "top96_SSM_mean_top96",
# "top96_SSM_median_top96",
# "top96_SSM_mean_top384",
# "top96_SSM_median_top384",
# "top96_SSM_fraction_max",

ftmlde_list = [
    f"{m}_{zs}"
    for m in ["top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]
    for zs in zs_no_score_list + ["double"]
]
# "maxes_Triad",
# "means_Triad",
# "ndcgs_Triad",
# "rhos_Triad",
# "if_truemaxs_Triad",
# "maxes_ev",
# "means_ev",
# "ndcgs_ev",
# "rhos_ev",
# "if_truemaxs_ev",
# "maxes_esm",
# "means_esm",
# "ndcgs_esm",
# "rhos_esm",
# "if_truemaxs_esm",
# "maxes_esmif",
# "means_esmif",
# "ndcgs_esmif",
# "rhos_esmif",
# "if_truemaxs_esmif",

DELTA_OPTS = ["delta", "deltafrac"]
FT_OPTS = ["ft", "ft-comb"]

ft_de_delta_list = [
    f"{ft}_{de}_{dt}"
    for dt in DELTA_OPTS
    for ft in (["mlde", "double"] + zs_no_score_list)  # instead of N_MUT_SUBS
    for de in DE_TYPES
]
# "mlde_single_step_DE_delta",
# "mlde_recomb_SSM_delta",
# "mlde_top96_SSM_delta",
# "Triad_single_step_DE_delta",
# "Triad_recomb_SSM_delta",
# "Triad_top96_SSM_delta",

delta_allft_mlde_list = [
    f"{dt}_{ft}_mlde"
    for dt in DELTA_OPTS
    for ft in zs_no_score_list + ["double"]  # N_MUT_SUBS
]
delta_bestft_mlde_list = [
    f"{dt}_{ft_des}_mlde" for dt in DELTA_OPTS for ft_des in FT_OPTS
]
delta_bestft_de_list = [
    f"{dt}_{ft_des}_{de}" for dt in DELTA_OPTS for ft_des in FT_OPTS for de in DE_TYPES
]

val_list = (
    de_list
    + ["top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]
    + ftmlde_list
    + ft_de_delta_list
    + delta_allft_mlde_list
    + delta_bestft_mlde_list
    + delta_bestft_de_list
)

DT_METRIC = {
    "delta": {"de": "mean_all", "mlde": "top_maxes"},
    "deltafrac": {"de": "fraction_max", "mlde": "if_truemaxs"},
}

FTLIB_FRAC_LIST = [0.125, 0.25, 0.5, 1]


class MergeLandscapeAttributes:
    """
    A class for merging the landscape attributes
    aka all non (ft)MLDE and (ft)ALDE results
    """

    def __init__(
        self,
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum_5/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        merge_dir: str = "results/merged",
        n_mut_cutoff: int = 0,
    ):

        self._lib_stat_path = lib_stat_path
        self._loc_opt_path = loc_opt_path
        self._pwe_path = pwe_path
        self._zs_path = zs_path
        self._de_path = de_path
        self._merge_dir = checkNgen_folder(merge_dir)
        
        self._n_mut_cutoff = n_mut_cutoff
        self._n_mut = n_mut_cutoff_dict[self._n_mut_cutoff]

        self._lib_stat_df = (
            parse_lib_stat(self._lib_stat_path, self._n_mut_cutoff)
            .reset_index(drop=True)
            .copy()
        )
        print(f"Loaded lib stat from: {self._lib_stat_path} for libs:")
        print(self._lib_stat_df["lib"].unique())

        self._loc_opt_df = self._get_loc_opt()
        print(f"Loaded loc opt from: {self._loc_opt_path} for libs:")
        print(self._loc_opt_df["lib"].unique())

        self._pwe_df = self._get_pwe()
        print(f"Loaded pwe from: {self._pwe_path} for libs:")
        print(self._pwe_df["lib"].unique())

        # self._zs_stat_df = self._get_zs_stat()
        # print(f"Loaded zs from: {self._zs_path} for libs:")
        # print(self._zs_stat_df["lib"].unique())

        self._de_stat_df, self._de_types = self._get_de_stat()
        print(f"Loaded de from: {self._de_path} for libs:")
        print(self._de_stat_df["lib"].unique())

        # now merge all
        self._merge_all_attribute_df = self._merge_all_attributes()

        self._merge_all_attribute_df.to_csv(
            f"{self._merge_dir}/merge_landscapeattributes_{self._n_mut}.csv", index=False
        )


    def _get_loc_opt(self) -> pd.DataFrame:
        """
        Get the local optima
        """

        loc_df = pd.read_csv(self._loc_opt_path)

        return loc_df[
            [col for col in loc_df.columns if "lib" in col or "loc_opt" in col]
        ].copy()

    def _get_zs_stat(self) -> pd.DataFrame:

        """
        Get all zs stats
        """

        zs_df = pd.read_csv(self._zs_path)

        zs_df_list = [zs_df[["lib", "n_mut"]]]
        # Create new columns for each score type
        for c in ZS_OPTS + ZS_COMB_OPTS:
            if c != "ed_score":
                zs_name = c.replace("_score", "")
                zs_df_list.append(
                    zs_df[f"{zs_name}_score"]
                    .str.replace(": nan", ": None")
                    .apply(literal_eval)
                    .apply(pd.Series)
                    .rename(columns={m: f"{zs_name}_{m}" for m in ZS_METRICS})
                )

        zs_df_expend = pd.concat(zs_df_list, axis=1)

        zs_mut_df_list = [zs_df_expend[zs_df_expend["n_mut"] == self._n_mut]["lib"]]

        for n_mut in ZS_N_MUTS:
            slice_df = (
                zs_df_expend[zs_df_expend["n_mut"] == n_mut]
                .drop(columns=["lib", "n_mut"])
                .reset_index(drop=True)
            )
            zs_mut_df_list.append(
                slice_df.rename(columns={c: f"{n_mut}_{c}" for c in slice_df.columns})
            )

        return pd.concat(zs_mut_df_list, axis=1)

    def _get_de_stat(self) -> pd.DataFrame:

        """
        Get the DE statistics
        """

        de_df = pd.read_csv(self._de_path)
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
                    )

            de_df_concat = pd.merge(
                de_df_concat, sliced_de_df.drop(["de_type"], axis=1), on="lib"
            )

        return de_df_concat.copy(), de_types

    def _get_pwe(self) -> pd.DataFrame:

        """
        A method to get the pairwise epistasis data
        """

        df_pw = pd.read_csv(self._pwe_path)

        df_pw_s_rs = (
            df_pw[
                (df_pw["n_mut"] == "all")
                & (df_pw["summary_type"] == "fraction")
                & (df_pw["epistasis_type"] != "magnitude")
            ][["lib", "value"]]
            .groupby("lib")
            .sum()
            .rename(columns={"value": "fraction_non-magnitude"})
        )

        df_pw_rs = df_pw[
            (df_pw["n_mut"] == "all")
            & (df_pw["summary_type"] == "fraction")
            & (df_pw["epistasis_type"] == "reciprocal sign")
        ][["lib", "value"]].rename(columns={"value": "fraction_reciprocal-sign"})

        return pd.merge(df_pw_s_rs, df_pw_rs, on="lib").copy()

    def _merge_all_attributes(self) -> pd.DataFrame:

        """
        Merge all the dataframes
        """

        merge_df = pd.merge(
            # pd.merge(
                # pd.merge(
                    pd.merge(
                        pd.merge(self._lib_stat_df, self._loc_opt_df, on="lib"),
                        self._pwe_df,
                        on="lib",
                    ),
                #     self._zs_stat_df,
                #     on="lib",
                # ),
                self._de_stat_df,
                on="lib",
            # ),
            # self._mlde_stat_df,
            # on="lib",
        )

        # numb_loc_opt
        # frac_loc_opt_total
        # frac_loc_opt_hd2_escape_numb
        # frac_loc_opt_hd2_cannot_escape_numb
        # numb_loc_opt_norm_cannot_escape
        # frac_loc_opt_norm_cannot_escape
        merge_df["numb_loc_opt_norm_cannot_escape"] = (
            merge_df["numb_loc_opt"] * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        )
        merge_df["frac_loc_opt_norm_cannot_escape"] = (
            merge_df["frac_loc_opt_total"]
            * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        )

        merge_df["norm_non-magnitude"] = (
            merge_df["fraction_non-magnitude"] * merge_df["percent_active"]
        )
        merge_df["norm_reciprocal-sign"] = (
            merge_df["fraction_reciprocal-sign"] * merge_df["percent_active"]
        )

        return merge_df.copy()

    @property
    def lib_stat_df(self) -> pd.DataFrame:
        """Return the library statistics dataframe"""
        return self._lib_stat_df

    @property
    def all_libs(self) -> list:
        """Return the list of all libraries"""
        return sorted(self._lib_stat_df["lib"].unique())

    @property
    def loc_opt_df(self) -> pd.DataFrame:
        """Return the local optima dataframe"""
        return self._loc_opt_df

    # @property
    # def zs_stat_df(self) -> pd.DataFrame:
    #     """Return the zs stat dataframe"""
    #     return self._zs_stat_df

    @property
    def de_stat_df(self) -> pd.DataFrame:
        """Return the DE statistics dataframe"""
        return self._de_stat_df

    @property
    def merge_all_attribute_df(self) -> pd.DataFrame:
        """Return the merged dataframe"""
        return self._merge_all_attribute_df


class MergeMLDEAttributes(MergeLandscapeAttributes):
    """
    A class for merging the MLDE attributes
    """

    def __init__(
        self,
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum_5/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/all_df_comb_onehot_2.csv",
        merge_dir: str = "results/merged",
        n_mut_cutoff: int = 0,
        n_sample: int = 384,
        n_top: int = 96,
        filter_active: float = 1,
        ft_frac=0.125,
        models: list[str] = ["boosting", "ridge"],
        ifplot: bool = True,
    ):

        super().__init__(lib_stat_path, loc_opt_path, pwe_path, zs_path, de_path, merge_dir, n_mut_cutoff)

        self._mlde_path = mlde_path

        self._n_sample = n_sample
        self._n_top = n_top
        self._filter_active = filter_active
        self._ft_frac = ft_frac
        self._models = models

        join_model = "|".join(models)
        dets = "_".join([
            self._n_mut,
            str(n_sample),
            f"{join_model}-top{str(n_top)}",
            f"ft-{str(self._ft_frac)}"
        ])

        self._mlde_stat_df, self._zs_list = self._get_mlde_stat()
        print(f"Loaded mlde from: {self._mlde_path} for libs:")
        print(self._mlde_stat_df["lib"].unique())

        # now merge all
        self._merge_all_df = self._get_merge_all()

        # now save
        self._merge_all_df.to_csv(
            f"{self._merge_dir}/MLDE_{dets}.csv", index=False
        )


    def _get_mlde_stat(self) -> pd.DataFrame:

        """
        Get the MLDE statistics

        Args:
        - n_top: int, the number of top samples
        - models: list[str], the list of models to consider
        """

        mlde_df = pd.read_csv(self._mlde_path)

        mlde_avg = (
            mlde_df[
                (mlde_df["zs"] == "none")
                & (mlde_df["encoding"] == "one-hot")
                & (mlde_df["model"].isin(self._models))
                & (mlde_df["n_mut_cutoff"] == self._n_mut)
                & (mlde_df["n_sample"] == self._n_sample)
                & (mlde_df["n_top"] == self._n_top)
                & (mlde_df["rep"].isin(np.arange(50)))
            ][["lib", "top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]]
            .groupby("lib")
            .mean()
        )

        print(mlde_avg.index.unique())

        zs_list = [
            zs.split("_score")[0] for zs in mlde_df["zs"].unique() if "score" in zs
        ]

        # append the singles and doubles
        # if self._n_mut == "all":
        zs_list += ["double"]  # N_MUT_SUBS

        for zs in zs_list:
            rename_cols = {
                "top_maxes": f"top_maxes_{zs}",
                "top_means": f"top_means_{zs}",
                "ndcgs": f"ndcgs_{zs}",
                "rhos": f"rhos_{zs}",
                "if_truemaxs": f"if_truemaxs_{zs}",
            }

            if zs not in N_MUT_SUBS:

                slice_mlde = mlde_df[
                    (mlde_df["zs"] == f"{zs}_score")
                    & (mlde_df["encoding"] == "one-hot")
                    & (mlde_df["model"].isin(self._models))
                    & (mlde_df["n_mut_cutoff"] == self._n_mut)
                    & (mlde_df["n_sample"] == self._n_sample)
                    & (mlde_df["n_top"] == self._n_top)
                ]

                lib_dfs = []
                for lib in self.all_libs:
                    lib_df = slice_mlde[slice_mlde["lib"] == lib].copy()
                    lib_df["ft_lib_size"] = lib_df["ft_lib"].map(
                        {
                            numb: frac
                            for numb, frac in zip(
                                sorted(lib_df["ft_lib"].unique()), FTLIB_FRAC_LIST
                            )
                        }
                    )
                    lib_dfs.append(lib_df[lib_df["ft_lib_size"] == self._ft_frac])
                slice_ftmlde = pd.concat(lib_dfs)

                mlde_avg = pd.merge(
                    mlde_avg,
                    (
                        slice_ftmlde[
                            [
                                "lib",
                                "top_maxes",
                                "top_means",
                                "ndcgs",
                                "rhos",
                                "if_truemaxs",
                            ]
                        ]
                        .groupby("lib")
                        .mean()
                        .rename(columns=rename_cols)
                    ),
                    on="lib",
                    how="outer",
                )

            else:

                mlde_avg = pd.merge(
                    mlde_avg,
                    (
                        mlde_df[
                            (mlde_df["zs"] == "none")
                            & (mlde_df["encoding"] == "one-hot")
                            & (mlde_df["model"].isin(self._models))
                            & (mlde_df["n_mut_cutoff"] == zs)
                            & (mlde_df["n_sample"] == self._n_sample)
                            & (mlde_df["n_top"] == self._n_top)
                        ][
                            [
                                "lib",
                                "top_maxes",
                                "top_means",
                                "ndcgs",
                                "rhos",
                                "if_truemaxs",
                            ]
                        ]
                        .groupby("lib")
                        .mean()
                        .rename(columns=rename_cols)
                    ),
                    on="lib",
                    how="outer",
                )

        return mlde_avg.reset_index().copy(), zs_list


    def _get_merge_all(self) -> pd.DataFrame:
        """
        Merge all the dataframes
        """

        merge_df = pd.merge(
            self.merge_all_attribute_df,
            self._mlde_stat_df,
            on="lib",
        )

        # append delta
        for ft_col in [""] + self._zs_list:

            for de in self._de_types:
                for dt in DELTA_OPTS:

                    de_col_name = f"{de}_{DT_METRIC[dt]['de']}"

                    if ft_col == "":
                        merge_df[f"mlde_{de}_{dt}"] = (
                            merge_df[DT_METRIC[dt]["mlde"]] - merge_df[de_col_name]
                        )

                    else:
                        mlde_name = DT_METRIC[dt]["mlde"]
                        merge_df[f"{ft_col}_{de}_{dt}"] = (
                            merge_df[f"{mlde_name}_{ft_col}"] - merge_df[de_col_name]
                        )

        merge_df["numb_loc_opt_norm_cannot_escape"] = (
            merge_df["numb_loc_opt"] * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        )
        merge_df["frac_loc_opt_norm_cannot_escape"] = (
            merge_df["frac_loc_opt_total"]
            * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        )

        merge_df["norm_non-magnitude"] = (
            merge_df["fraction_non-magnitude"] * merge_df["percent_active"]
        )
        merge_df["norm_reciprocal-sign"] = (
            merge_df["fraction_reciprocal-sign"] * merge_df["percent_active"]
        )

        for dt in DELTA_OPTS:

            mlde_name = DT_METRIC[dt]["mlde"]
            de_name = DT_METRIC[dt]["de"]

            # add double
            merge_df[f"{dt}_hd2_mlde"] = (
                merge_df[f"{mlde_name}_double"] - merge_df[mlde_name]
            )

            for zs in zs_no_score_list + ["double"]:  # N_MUT_SUBS:
                if f"{mlde_name}_{zs}" in merge_df.columns:
                    merge_df[f"{dt}_{zs}_mlde"] = (
                        merge_df[f"{mlde_name}_{zs}"] - merge_df[mlde_name]
                    )
                else:
                    merge_df[f"{dt}_{zs}_mlde"] = np.nan
                    merge_df[f"{mlde_name}_{zs}"] = np.nan

            best_ft = merge_df[
                [
                    "".join([f"{mlde_name}_", zs.replace("_score", "")])
                    for zs in ZS_OPTS
                    if zs != "ed_score"
                ]
                + [f"{mlde_name}_double"]
            ].max(axis=1)

            best_ftcomb = merge_df[
                [
                    "".join([f"{mlde_name}_", zs])
                    for zs in zs_no_score_list
                    if zs != "ed"
                ]
                + [f"{mlde_name}_double"]
            ].max(axis=1)

            for ft_des, ft_df in zip(FT_OPTS, [best_ft, best_ftcomb]):

                # add vs mlde
                merge_df[f"{dt}_{ft_des}_mlde"] = ft_df - merge_df[mlde_name]
                # add vs de
                for de in DE_TYPES:
                    merge_df[f"{dt}_{ft_des}_{de}"] = (
                        ft_df - merge_df[f"{de}_{de_name}"]
                    )

        return merge_df.copy()

    @property
    def mlde_stat_df(self) -> pd.DataFrame:
        """Return the MLDE statistics dataframe"""
        return self._mlde_stat_df

    @property
    def merge_all_df(self) -> pd.DataFrame:
        """Return the merged dataframe"""
        return self._merge_all_df


class CorrPerfomanceCharacter(MergeLandscapeAttributes):
    """
    A class for getting the correlation between the performance and the characteristics of the library

    TODO:
    Add model opts
    Add DE opts
    """

    def __init__(
        self,
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        zs_path: str = "results/zs_sum_5/none/zs_stat_scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/all_df_comb_onehot_2.csv",
        merge_dir: str = "results/merged",
        corr_dir: str = "results/corr",
        n_mut_cutoff: int = 0,
        n_sample: int = 384,
        n_top: int = 96,
        filter_active: float = 1,
        ft_frac=0.125,
        models: list[str] = ["boosting", "ridge"],
        ifplot: bool = True,
    ):

        super().__init__(lib_stat_path, loc_opt_path, pwe_path, zs_path, de_path, merge_dir, n_mut_cutoff)

        self._mlde_path = mlde_path
        self._corr_dir = checkNgen_folder(corr_dir)

        self._n_sample = n_sample
        self._n_top = n_top
        self._filter_active = filter_active
        self._ft_frac = ft_frac
        self._models = models

        join_model = "|".join(models)
        self._corr_subdir = checkNgen_folder(
            os.path.join(
                "_".join([self._corr_dir, self._n_mut]),
                str(n_sample),
                f"{join_model}-top{str(n_top)}",
                f"actcut-{str(filter_active)}",
            )
        )

        print(f"Correlation directory: {self._corr_subdir}")

        self._mlde_stat_df, self._zs_list = self._get_mlde_stat()
        print(f"Loaded mlde from: {self._mlde_path} for libs:")
        print(self._mlde_stat_df["lib"].unique())

        # now merge all
        self._merge_all_df = self._get_merge_all()

        # remove super dead libs 
        self._active_lib_list = self._lib_stat_df[
            (
                self._lib_stat_df["percent_active"] >= self._filter_active
            )  # & (self._lib_stat_df["percent_measured"] >= 90)
        ]["lib"].tolist()

        print(
            f"For {self._filter_active} percent_active cutoff, active lib list: {self._active_lib_list}"
        )

        self._actcutt_df = self._merge_all_df[
            self._merge_all_df["lib"].isin(self._active_lib_list)
        ]
        self._actcutt_df.to_csv(
            f"{self._corr_subdir}/merge_all_{str(self._ft_frac)}.csv", index=False
        )

        self._corr_df = self._get_corr_df()
        self._corr_df.to_csv(
            f"{self._corr_subdir}/corr_{str(self._ft_frac)}.csv", index=False
        )

        # now plot
        if ifplot:
            self._plot_corr()


    def _get_mlde_stat(self) -> pd.DataFrame:

        """
        Get the MLDE statistics

        Args:
        - n_top: int, the number of top samples
        - models: list[str], the list of models to consider
        """

        mlde_df = pd.read_csv(self._mlde_path)

        mlde_avg = (
            mlde_df[
                (mlde_df["zs"] == "none")
                & (mlde_df["encoding"] == "one-hot")
                & (mlde_df["model"].isin(self._models))
                & (mlde_df["n_mut_cutoff"] == self._n_mut)
                & (mlde_df["n_sample"] == self._n_sample)
                & (mlde_df["n_top"] == self._n_top)
                & (mlde_df["rep"].isin(np.arange(50)))
            ][["lib", "top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]]
            .groupby("lib")
            .mean()
        )

        print(mlde_avg.index.unique())

        zs_list = [
            zs.split("_score")[0] for zs in mlde_df["zs"].unique() if "score" in zs
        ]

        # append the singles and doubles
        # if self._n_mut == "all":
        zs_list += ["double"]  # N_MUT_SUBS

        for zs in zs_list:
            rename_cols = {
                "top_maxes": f"top_maxes_{zs}",
                "top_means": f"top_means_{zs}",
                "ndcgs": f"ndcgs_{zs}",
                "rhos": f"rhos_{zs}",
                "if_truemaxs": f"if_truemaxs_{zs}",
            }

            if zs not in N_MUT_SUBS:

                slice_mlde = mlde_df[
                    (mlde_df["zs"] == f"{zs}_score")
                    & (mlde_df["encoding"] == "one-hot")
                    & (mlde_df["model"].isin(self._models))
                    & (mlde_df["n_mut_cutoff"] == self._n_mut)
                    & (mlde_df["n_sample"] == self._n_sample)
                    & (mlde_df["n_top"] == self._n_top)
                ]

                lib_dfs = []
                for lib in self.all_libs:
                    lib_df = slice_mlde[slice_mlde["lib"] == lib].copy()
                    lib_df["ft_lib_size"] = lib_df["ft_lib"].map(
                        {
                            numb: frac
                            for numb, frac in zip(
                                sorted(lib_df["ft_lib"].unique()), FTLIB_FRAC_LIST
                            )
                        }
                    )
                    lib_dfs.append(lib_df[lib_df["ft_lib_size"] == self._ft_frac])
                slice_ftmlde = pd.concat(lib_dfs)

                mlde_avg = pd.merge(
                    mlde_avg,
                    (
                        slice_ftmlde[
                            [
                                "lib",
                                "top_maxes",
                                "top_means",
                                "ndcgs",
                                "rhos",
                                "if_truemaxs",
                            ]
                        ]
                        .groupby("lib")
                        .mean()
                        .rename(columns=rename_cols)
                    ),
                    on="lib",
                    how="outer",
                )

            else:

                mlde_avg = pd.merge(
                    mlde_avg,
                    (
                        mlde_df[
                            (mlde_df["zs"] == "none")
                            & (mlde_df["encoding"] == "one-hot")
                            & (mlde_df["model"].isin(self._models))
                            & (mlde_df["n_mut_cutoff"] == zs)
                            & (mlde_df["n_sample"] == self._n_sample)
                            & (mlde_df["n_top"] == self._n_top)
                        ][
                            [
                                "lib",
                                "top_maxes",
                                "top_means",
                                "ndcgs",
                                "rhos",
                                "if_truemaxs",
                            ]
                        ]
                        .groupby("lib")
                        .mean()
                        .rename(columns=rename_cols)
                    ),
                    on="lib",
                    how="outer",
                )

        return mlde_avg.reset_index().copy(), zs_list


    def _get_merge_all(self) -> pd.DataFrame:
        """
        Merge all the dataframes
        """

        merge_df = pd.merge(
            self.merge_all_attribute_df,
            self._mlde_stat_df,
            on="lib",
        )

        # append delta
        for ft_col in [""] + self._zs_list:

            for de in self._de_types:
                for dt in DELTA_OPTS:

                    de_col_name = f"{de}_{DT_METRIC[dt]['de']}"

                    if ft_col == "":
                        merge_df[f"mlde_{de}_{dt}"] = (
                            merge_df[DT_METRIC[dt]["mlde"]] - merge_df[de_col_name]
                        )
                        # merge_df[f"mlde_{de}_deltafrac"] = (
                        #     merge_df["if_truemaxs"] - merge_df[f"{de}_fraction_max"]
                        # )
                    else:
                        mlde_name = DT_METRIC[dt]["mlde"]
                        merge_df[f"{ft_col}_{de}_{dt}"] = (
                            merge_df[f"{mlde_name}_{ft_col}"] - merge_df[de_col_name]
                        )
                        # merge_df[f"{ft_col}_{de}_deltafrac"] = (
                        #     merge_df[f"if_truemaxs_{ft_col}"] - merge_df[f"{de}_fraction_max"]
                        # )

        # numb_loc_opt
        # frac_loc_opt_total
        # frac_loc_opt_hd2_escape_numb
        # frac_loc_opt_hd2_cannot_escape_numb
        # numb_loc_opt_norm_cannot_escape
        # frac_loc_opt_norm_cannot_escape
        merge_df["numb_loc_opt_norm_cannot_escape"] = (
            merge_df["numb_loc_opt"] * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        )
        merge_df["frac_loc_opt_norm_cannot_escape"] = (
            merge_df["frac_loc_opt_total"]
            * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        )

        merge_df["norm_non-magnitude"] = (
            merge_df["fraction_non-magnitude"] * merge_df["percent_active"]
        )
        merge_df["norm_reciprocal-sign"] = (
            merge_df["fraction_reciprocal-sign"] * merge_df["percent_active"]
        )
        # merge_df["delta_ft_mlde"] = (
        #     np.maximum.reduce(
        #         [
        #             merge_df["maxes_Triad"],
        #             merge_df["maxes_ev"],
        #             merge_df["maxes_esm"],
        #             merge_df["maxes_esmif"],
        #         ]
        #     )
        #     - merge_df["top_maxes"]
        # )

        """
        ["top_maxes", "if_truemaxs"],
        ["mean_all", "fraction_max"],
        """

        for dt in DELTA_OPTS:

            mlde_name = DT_METRIC[dt]["mlde"]
            de_name = DT_METRIC[dt]["de"]

            # add double
            merge_df[f"{dt}_hd2_mlde"] = (
                merge_df[f"{mlde_name}_double"] - merge_df[mlde_name]
            )
            # merge_df["deltafrac_hd2_mlde"] = merge_df["if_truemaxs_double"] - merge_df["if_truemaxs"]

            # add single
            # merge_df[f"{dt}_hd1_mlde"] = (
            #     merge_df[f"{mlde_name}_single"] - merge_df[mlde_name]
            # )
            # merge_df["deltafrac_hd1_mlde"] = merge_df["if_truemaxs_single"] - merge_df["if_truemaxs"]

            # delta_allft_mlde_list = [f"{dt}_{ft}_mlde" for dt in DELTA_OPTS for ft in zs_no_score_lis]
            for zs in zs_no_score_list + ["double"]:  # N_MUT_SUBS:
                if f"{mlde_name}_{zs}" in merge_df.columns:
                    merge_df[f"{dt}_{zs}_mlde"] = (
                        merge_df[f"{mlde_name}_{zs}"] - merge_df[mlde_name]
                    )
                else:
                    merge_df[f"{dt}_{zs}_mlde"] = np.nan
                    merge_df[f"{mlde_name}_{zs}"] = np.nan

            best_ft = merge_df[
                [
                    "".join([f"{mlde_name}_", zs.replace("_score", "")])
                    for zs in ZS_OPTS
                    if zs != "ed_score"
                ]
                + [f"{mlde_name}_double"]
            ].max(axis=1)

            best_ftcomb = merge_df[
                [
                    "".join([f"{mlde_name}_", zs])
                    for zs in zs_no_score_list
                    if zs != "ed"
                ]
                + [f"{mlde_name}_double"]
            ].max(axis=1)

            for ft_des, ft_df in zip(FT_OPTS, [best_ft, best_ftcomb]):

                # add vs mlde
                merge_df[f"{dt}_{ft_des}_mlde"] = ft_df - merge_df[mlde_name]
                # add vs de
                for de in DE_TYPES:
                    merge_df[f"{dt}_{ft_des}_{de}"] = (
                        ft_df - merge_df[f"{de}_{de_name}"]
                    )

        # merge_df["delta_ft_dessm"] = (
        #     np.maximum.reduce(
        #         [
        #             merge_df["maxes_Triad"],
        #             merge_df["maxes_ev"],
        #             merge_df["maxes_esm"],
        #             merge_df["maxes_esmif"],
        #         ]
        #     )
        #     - merge_df["recomb_SSM_mean_all"]
        # )

        return merge_df.copy()

    def _get_corr_df(self) -> pd.DataFrame:

        """
        A method to get correlation dataframe
        """

        corr_df = pd.DataFrame()

        for des in LANDSCAPE_ATTRIBUTES + zs_list:

            corr_row = {"descriptor": des}

            for val in LANDSCAPE_ATTRIBUTES + zs_list + val_list:

                if des in self._actcutt_df.columns and val in self._actcutt_df.columns:
                    corr_row[val] = spearmanr(
                        self._actcutt_df[des], self._actcutt_df[val]
                    )[0]
                else:
                    corr_row[val] = np.nan

            corr_df = corr_df._append(
                corr_row,
                ignore_index=True,
            )

        return corr_df.copy()

    def _plot_corr(self) -> None:

        """
        A method to plot the correlation
        """

        for fac in tqdm(
            [
                "percent_active",
                "skewness",
                "kurt",
                "numb_kde_peak",
                "numb_loc_opt",
                "frac_loc_opt_total",
                "frac_loc_opt_hd2_escape_numb",
                "frac_loc_opt_hd2_cannot_escape_numb",
                "numb_loc_opt_norm_cannot_escape",
                "frac_loc_opt_norm_cannot_escape",
                "fraction_non-magnitude",
                "fraction_reciprocal-sign",
                "norm_non-magnitude",
                "norm_reciprocal-sign",
            ]
        ):

            for delta_type, subdir in zip(
                ["mlde_recomb_SSM_delta", "delta_ft_recomb_SSM", "delta_ft_mlde"],
                ["mlde_de", "ft_de", "ft_mlde"],
            ):

                if delta_type == "mlde_recomb_SSM_delta":
                    title = f"{fac} vs delta MLDE and DE max fitness achieved"
                elif delta_type == "delta_ft_recomb_SSM":
                    title = f"{fac} vs delta ftMLDE and DE max fitness achieved"
                elif delta_type == "delta_ft_mlde":
                    title = f"{fac} vs delta ftMLDE and MLDE max fitness achieved"
                else:
                    raise ValueError(f"delta_type {delta_type} not recognized")

                for logx in [True, False]:

                    if logx:
                        plot_path = checkNgen_folder(
                            os.path.join(self._corr_subdir, subdir, "logx")
                        )
                    else:
                        plot_path = checkNgen_folder(
                            os.path.join(self._corr_subdir, subdir)
                        )

                    save_bokeh_hv(
                        plot_obj=hv.Scatter(self._actcutt_df, fac, [delta_type, "lib"])
                        .opts(
                            marker="o", size=10, color=dim("lib").categorize(LIB_COLORS)
                        )
                        .groupby("lib")
                        .overlay()
                        .opts(
                            hv.opts.Scatter(
                                height=400,
                                width=500,
                                marker="o",
                                size=10,
                                tools=["hover"],
                                show_legend=True,
                                color=dim("lib").categorize(LIB_COLORS),
                                legend_position="right",
                                legend_offset=(5, -20),
                                title=title,
                                logx=logx,
                            )
                        ),
                        plot_name=title,
                        plot_path=plot_path,
                        bokehorhv="hv",
                        dpi=300,
                        scale=2,
                    )

    @property
    def mlde_stat_df(self) -> pd.DataFrame:
        """Return the MLDE statistics dataframe"""
        return self._mlde_stat_df

    @property
    def merge_all_df(self) -> pd.DataFrame:
        """Return the merged dataframe"""
        return self._merge_all_df

    @property
    def corr_df(self) -> pd.DataFrame:
        """Return the correlation dataframe"""
        return self._corr_df

    @property
    def active_lib_list(self) -> list[str]:
        """Return the list of active libraries"""
        return self._active_lib_list

    @property
    def actcutt_df(self) -> pd.DataFrame:
        """Return the dataframe with active libraries"""
        return self._actcutt_df


def perfom_corr(
    lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
    loc_opt_path: str = "results/local_optima/scale2max.csv",
    pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
    zs_path: str = "results/zs_sum/none/zs_stat_scale2max.csv",
    de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
    mlde_path: str = "results/mlde/all_df_comb_onehot_2.csv",
    merge_dir: str = "results/merged",
    corr_dir: str = "results/corr",
    n_mut_cutoff: int = 0,
    filter_active: float = 1,
    ft_frac: float = 0.125,
    n_top_list: list[int] = [96, 384],
    n_list: list[int] = N_SAMPLE_LIST,
    models_list: list[list[str]] = [["boosting", "ridge"], ["boosting"], ["ridge"]],
    ifplot: bool = True,
):

    for models in models_list:
        for n_top in n_top_list:
            for n in tqdm(n_list):
                CorrPerfomanceCharacter(
                    lib_stat_path=lib_stat_path,
                    loc_opt_path=loc_opt_path,
                    pwe_path=pwe_path,
                    zs_path=zs_path,
                    de_path=de_path,
                    mlde_path=mlde_path,
                    corr_dir=corr_dir,
                    n_mut_cutoff=n_mut_cutoff,
                    n_sample=n,
                    n_top=n_top,
                    filter_active=filter_active,
                    ft_frac=ft_frac,
                    models=models,
                    ifplot=ifplot,
                )