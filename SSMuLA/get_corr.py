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

from SSMuLA.landscape_global import get_file_name, n_mut_cutoff_dict
from SSMuLA.zs_analysis import ZS_OPTS
from SSMuLA.vis import LIB_COLORS, save_bokeh_hv
from SSMuLA.util import checkNgen_folder


output_notebook()
bokeh.io.output_notebook()

hv.extension("bokeh", "matplotlib")


N_SAMPLE_LIST = [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]


class CorrPerfomanceCharacter:
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
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/vis_3/all_df.csv",
        corr_dir: str = "results/corr",
        n_mut_cuttoff: int = 0,
        n_sample: int = 384,
        n_top: int = 96,
        models: list[str] = ["boosting"],
    ):

        self._lib_stat_path = lib_stat_path
        self._loc_opt_path = loc_opt_path
        self._pwe_path = pwe_path
        self._de_path = de_path
        self._mlde_path = mlde_path
        self._corr_dir = checkNgen_folder(corr_dir)

        self._n_mut_cuttoff = n_mut_cuttoff
        self._n_sample = n_sample
        self._n_top = n_top
        self._models = models

        
        join_model = "|".join(models)
        self._corr_subdir = checkNgen_folder(
            os.path.join(
                self._corr_dir, str(n_sample), f"{join_model}-top{str(n_top)}"
            )
        )

        self._lib_stat_df = self._get_lib_stat()
        self._loc_opt_df = self._get_loc_opt()
        self._pwe_df = self._get_pwe()
        self._de_stat_df, self._de_types = self._get_de_stat()
        self._mlde_stat_df, self._zs_simple = self._get_mlde_stat()

        # now merge all
        self._merge_all_df = self._get_merge_all()
        self._merge_all_df.to_csv(f"{self._corr_subdir}/merge_all.csv", index=False)

        self._corr_df = self._get_corr_df()
        self._corr_df.to_csv(f"{self._corr_subdir}/corr.csv", index=False)

        # now plot
        self._plot_corr()

    def _get_lib_stat(self) -> pd.DataFrame:
        """
        Get the library statistics
        """

        lib_stat = pd.read_csv(self._lib_stat_path, index_col=0)
        lib_stat_all = lib_stat[lib_stat["n_mut_cuttoff"] == self._n_mut_cuttoff]

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

        df_expanded = lib_df["quartiles"].apply(pd.Series)
        df_expanded.columns = ["Q1", "Q2", "Q3"]  # Rename columns
        df_expanded["numb_kde_peak"] = lib_df["peak_kde"].apply(len)

        lib_df = pd.concat([lib_df, df_expanded], axis=1)

        return lib_df.copy()

    def _get_loc_opt(self) -> pd.DataFrame:
        """
        Get the local optima
        """

        loc_df = pd.read_csv(self._loc_opt_path)

        return loc_df[
            [col for col in loc_df.columns if "lib" in col or "loc_opt" in col]
        ].copy()

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
                & (mlde_df["n_mut_cutoff"] == n_mut_cutoff_dict[self._n_mut_cuttoff])
                & (mlde_df["n_sample"] == self._n_sample)
                & (mlde_df["n_top"] == self._n_top)
            ][["lib", "top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]]
            .groupby("lib")
            .mean()
        )

        zs_simple = [
            zs.split("_score")[0] for zs in mlde_df["zs"].unique() if "score" in zs
        ]

        for zs in zs_simple:

            rename_cols = {
                "top_maxes": f"maxes_{zs}",
                "top_means": f"means_{zs}",
                "ndcgs": f"ndcgs_{zs}",
                "rhos": f"rhos_{zs}",
                "if_truemaxs": f"if_truemaxs_{zs}",
            }

            mlde_avg = pd.merge(
                mlde_avg,
                (
                    mlde_df[
                        (mlde_df["zs"] == f"{zs}_score")
                        & (mlde_df["encoding"] == "one-hot")
                        & (mlde_df["model"].isin(self._models))
                        & (
                            mlde_df["n_mut_cutoff"]
                            == n_mut_cutoff_dict[self._n_mut_cuttoff]
                        )
                        & (mlde_df["n_sample"] == self._n_sample)
                        & (mlde_df["n_top"] == self._n_top)
                    ][["lib", "top_maxes", "top_means", "ndcgs", "rhos", "if_truemaxs"]]
                    .groupby("lib")
                    .mean()
                    .rename(columns=rename_cols)
                ),
                on="lib",
            )

        return mlde_avg.copy(), zs_simple

    def _get_pwe(self) -> pd.DataFrame:

        """
        A method to get the pairwise epistasis data
        """

        df_pw = pd.read_csv(self._pwe_path)

        df_pw_s_rs = (
            df_pw[
                (df_pw["n_mut"] == n_mut_cutoff_dict[self._n_mut_cuttoff])
                & (df_pw["summary_type"] == "fraction")
                & (df_pw["epistasis_type"] != "magnitude")
            ][["lib", "value"]]
            .groupby("lib")
            .sum()
            .rename(columns={"value": "fraction_non-magnitude"})
        )

        df_pw_rs = df_pw[
            (df_pw["n_mut"] == n_mut_cutoff_dict[self._n_mut_cuttoff])
            & (df_pw["summary_type"] == "fraction")
            & (df_pw["epistasis_type"] == "reciprocal sign")
        ][["lib", "value"]].rename(columns={"value": "fraction_reciprocal-sign"})

        return pd.merge(df_pw_s_rs, df_pw_rs, on="lib").copy()

    def _get_merge_all(self) -> pd.DataFrame:
        """
        Merge all the dataframes
        """

        merge_df = pd.merge(self._lib_stat_df, self._loc_opt_df, on="lib")
        merge_df = pd.merge(merge_df, self._de_stat_df, on="lib")
        merge_df = pd.merge(merge_df, self._mlde_stat_df, on="lib")
        merge_df = pd.merge(merge_df, self._pwe_df, on="lib")

        # append delta
        for ft_col in [""] + self._zs_simple:
            for de in self._de_types:
                if ft_col == "":
                    merge_df[f"mlde_{de}_delta"] = (
                        merge_df["top_maxes"] - merge_df[f"{de}_mean_all"]
                    )
                else:
                    merge_df[f"{ft_col}_{de}_delta"] = (
                        merge_df[f"maxes_{ft_col}"] - merge_df[f"{de}_mean_all"]
                    )

        # numb_loc_opt		
        # frac_loc_opt_total	
        # frac_loc_opt_hd2_escape_numb	
        # frac_loc_opt_hd2_cannot_escape_numb
        # numb_loc_opt_norm_cannot_escape
        # frac_loc_opt_norm_cannot_escape
        merge_df["numb_loc_opt_norm_cannot_escape"] = merge_df["numb_loc_opt"] * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]
        merge_df["frac_loc_opt_norm_cannot_escape"] = merge_df["frac_loc_opt_total"] * merge_df["frac_loc_opt_hd2_cannot_escape_numb"]

        merge_df["norm_non-magnitude"] = (
            merge_df["fraction_non-magnitude"] * merge_df["percent_active"]
        )
        merge_df["norm_reciprocal-sign"] = (
            merge_df["fraction_reciprocal-sign"] * merge_df["percent_active"]
        )
        merge_df["delta_ft_mlde"] = (
            np.maximum.reduce(
                [
                    merge_df["maxes_Triad"],
                    merge_df["maxes_ev"],
                    merge_df["maxes_esm"],
                    merge_df["maxes_esmif"],
                ]
            )
            - merge_df["top_maxes"]
        )
        merge_df["delta_ft_de"] = (
            np.maximum.reduce(
                [
                    merge_df["maxes_Triad"],
                    merge_df["maxes_ev"],
                    merge_df["maxes_esm"],
                    merge_df["maxes_esmif"],
                ]
            )
            - merge_df["recomb_SSM_mean_all"]
        )

        return merge_df.copy()

    def _get_corr_df(self) -> pd.DataFrame:

        """
        A method to get correlation dataframe
        """

        corr_df = pd.DataFrame()

        des_list = [
            "numb_measured",
            "percent_measured",
            "numb_active",
            "percent_active",
            "active_fit_min",
            "parent_fit",
            "parent_rank",
            "parent_rank_percent",
            "mean",
            "std",
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

        val_list = [
                "single_step_DE_mean_all",
                "single_step_DE_median_all",
                "single_step_DE_mean_top96",
                "single_step_DE_median_top96",
                "single_step_DE_mean_top384",
                "single_step_DE_median_top384",
                "single_step_DE_fraction_max",
                "recomb_SSM_mean_all",
                "recomb_SSM_median_all",
                "recomb_SSM_mean_top96",
                "recomb_SSM_median_top96",
                "recomb_SSM_mean_top384",
                "recomb_SSM_median_top384",
                "recomb_SSM_fraction_max",
                "top96_SSM_mean_all",
                "top96_SSM_median_all",
                "top96_SSM_mean_top96",
                "top96_SSM_median_top96",
                "top96_SSM_mean_top384",
                "top96_SSM_median_top384",
                "top96_SSM_fraction_max",
                "top_maxes",
                "top_means",
                "ndcgs",
                "rhos",
                "if_truemaxs",
                "maxes_Triad",
                "means_Triad",
                "ndcgs_Triad",
                "rhos_Triad",
                "if_truemaxs_Triad",
                "maxes_ev",
                "means_ev",
                "ndcgs_ev",
                "rhos_ev",
                "if_truemaxs_ev",
                "maxes_esm",
                "means_esm",
                "ndcgs_esm",
                "rhos_esm",
                "if_truemaxs_esm",
                "maxes_esmif",
                "means_esmif",
                "ndcgs_esmif",
                "rhos_esmif",
                "if_truemaxs_esmif",
                "mlde_single_step_DE_delta",
                "mlde_recomb_SSM_delta",
                "mlde_top96_SSM_delta",
                "Triad_single_step_DE_delta",
                "Triad_recomb_SSM_delta",
                "Triad_top96_SSM_delta",
                "ev_single_step_DE_delta",
                "ev_recomb_SSM_delta",
                "ev_top96_SSM_delta",
                "esm_single_step_DE_delta",
                "esm_recomb_SSM_delta",
                "esm_top96_SSM_delta",
                "esmif_single_step_DE_delta",
                "esmif_recomb_SSM_delta",
                "esmif_top96_SSM_delta",
                "delta_ft_mlde",
                "delta_ft_de",
            ]

        for des in des_list:

            corr_row = {"descriptor": des}

            for val in des_list + val_list:

                corr_row[val] = spearmanr(
                    self._merge_all_df[des], self._merge_all_df[val]
                )[0]

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
                ["mlde_recomb_SSM_delta", "delta_ft_de", "delta_ft_mlde"],
                ["mlde_de", "ft_de", "ft_mlde"],
            ):

                if delta_type == "mlde_recomb_SSM_delta":
                    title = f"{fac} vs delta MLDE and DE max fitness achieved"
                elif delta_type == "delta_ft_de":
                    title = f"{fac} vs delta ftMLDE and DE max fitness achieved"
                elif delta_type == "delta_ft_mlde":
                    title = f"{fac} vs delta ftMLDE and MLDE max fitness achieved"
                else:
                    raise ValueError(f"delta_type {delta_type} not recognized")

                for logx in [True, False]:

                    if logx:
                        plot_path = checkNgen_folder(os.path.join(self._corr_subdir, subdir, "logx"))
                    else:
                        plot_path = checkNgen_folder(os.path.join(self._corr_subdir, subdir))

                    save_bokeh_hv(
                        plot_obj=hv.Scatter(
                            self._merge_all_df, fac, [delta_type, "lib"]
                        )
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
    def lib_stat_df(self) -> pd.DataFrame:
        """Return the library statistics dataframe"""
        return self._lib_stat_df

    @property
    def loc_opt_df(self) -> pd.DataFrame:
        """Return the local optima dataframe"""
        return self._loc_opt_df

    @property
    def de_stat_df(self) -> pd.DataFrame:
        """Return the DE statistics dataframe"""
        return self._de_stat_df

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

def perfom_corr(
        lib_stat_path: str = "results/fitness_distribution/max/all_lib_stats.csv",
        loc_opt_path: str = "results/local_optima/scale2max.csv",
        pwe_path: str = "results/pairwise_epistasis_vis/none/scale2max.csv",
        de_path: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        mlde_path: str = "results/mlde/vis_3/all_df.csv",
        corr_dir: str = "results/corr",
        n_mut_cuttoff: int = 0,
        n_top_list: list[int] = [96, 384],
        models_list: list[list[str]] = [["boosting"], ["ridge"], ["boosting", "ridge"]]):

    
    for models in models_list:
        for n_top in n_top_list:
            for n in tqdm(N_SAMPLE_LIST):
                CorrPerfomanceCharacter(
                    lib_stat_path=lib_stat_path,
                    loc_opt_path=loc_opt_path,
                    pwe_path=pwe_path,
                    de_path=de_path,
                    mlde_path=mlde_path,
                    corr_dir=corr_dir,
                    n_mut_cuttoff=n_mut_cuttoff,
                    n_sample=n,
                    n_top=n_top,
                    models=models
                )