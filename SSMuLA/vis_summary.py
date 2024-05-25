"""
A script to visualize the results cross landscapes.
"""


import os
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
from ast import literal_eval


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import holoviews as hv


from SSMuLA.de_simulations import DE_TYPES, DE_COLORS, DE_LINE_STYLES, DE_LEGEND_MAP
from SSMuLA.zs_analysis import (
    ZS_OPTS,
    ZS_COMB_OPTS,
    ZS_OPTS_LEGEND,
)
from SSMuLA.landscape_global import LIB_INFO_DICT, LIB_NAMES
from SSMuLA.vis import (
    LIB_COLORS,
    PRESENTATION_PALETTE_SATURATE,
    MLDE_COLORS,
    ZS_COLOR_MAP,
    save_plt,
    save_bokeh_hv,
    JSON_THEME,
    one_decimal_x,
    one_decimal_y,
    fixmargins,
)
from SSMuLA.util import get_file_name, checkNgen_folder, ecdf_transform


hv.extension("bokeh")
hv.renderer("bokeh").theme = JSON_THEME


DE_METRIC_MAP = {
    "mean_all": "all simulations fitness mean",
    "median_all": "all simulations fitness median",
    "mean_top96": "top 96 simulations fitness mean",
    "median_top96": "top 96 simulations fitness median",
    "mean_top384": "top 384 simulations fitness mean",
    "median_top384": "top 384 simulations fitness median",
    "fraction_max": "fraction reached max fitness",
}

N_MUT_SUBS = ["double", "single"]

N_MUT_LEGEND = {
    "double": "HD2",
    "single": "HD1",
}

N_MUT_COLOR = {
    "double": PRESENTATION_PALETTE_SATURATE["orange"],
    "single": PRESENTATION_PALETTE_SATURATE["yellow"],
}

ZS_METRICS = ["rho", "ndcg", "rocauc"]
ZS_N_MUTS = ["all", "double", "single"]
ZS_COMB_VIS_OPTS = ["both", "nocomb", "comb"]

ZS_OPTS_APPENDED = deepcopy(ZS_OPTS_LEGEND)
for n_mut_sub in N_MUT_SUBS:
    ZS_OPTS_APPENDED[n_mut_sub] = N_MUT_LEGEND[n_mut_sub]

ZS_COLOR_MAP_APPENDED = deepcopy(ZS_COLOR_MAP)
for n_mut_sub in N_MUT_SUBS:
    ZS_COLOR_MAP_APPENDED[n_mut_sub] = N_MUT_COLOR[n_mut_sub]

N_SAMPLE_LIST = [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]

FTLIB_FRAC_LIST = [0.125, 0.25, 0.5, 1]

LIB4BY4 = [
    "ParD2",
    "ParD3",
    "DHFR",
    "TrpB3I",
    "TrpB3B",
    "TrpB3C",
    "TrpB3H",
    "TrpB3A",
    "TrpB3E",
    "TrpB3G",
    "TrpB3F",
    "TrpB3D",
    "GB1",
    "TrpB4",
]

LIB3BY5 = [
    "ParD3",
    "ParD2",
    "DHFR",
    "TrpB3I",
    "TrpB3D",
    "GB1",
    "TrpB4",
    "TrpB3G",
    "TrpB3F",
    "",
    "TrpB3E",
    "TrpB3A",
    "TrpB3C",
    "TrpB3H",
    "TrpB3B",
]

"""
['ParD3', 'ParD2', 'DHFR', 'TrpB3I', 'GB1', 'TrpB3D', 'TrpB4',
'TrpB3G', 'TrpB3F', 'TrpB3E', 'TrpB3A', 'TrpB3C', 'TrpB3H',
'TrpB3B']
df["numb_active"] / df["numb_loc_opt"] / df["fraction_non-magnitude"] / np.log(df["numb_measured"])    
"""


def de_sum_hook(plot, element):
    plot.handles["plot"].x_range.factors = [
        (lib, de) for lib in LIB_NAMES for de in DE_TYPES
    ]


def get_val_frac(s: pd.Series, numb: float = 1) -> float:
    """
    Get the fraction of a column that is equal to a certain number.
    """
    return (s == numb).sum() / len(s)


class SumVis:
    """A class to visualize the summary of simulations."""

    def __init__(self, input_csv: str, output_folder: str = "") -> None:

        """
        Args:
        - input_csv: str, The input csv file path for the summary df
        - output_folder: str, The output folder for the plots
        """

        self._input_csv = input_csv
        self._output_folder = output_folder

    @property
    def input_csv(self):
        return self._input_csv

    @property
    def input_df(self):
        return pd.read_csv(self._input_csv)

    @property
    def output_folder(self):
        if self._output_folder == "":
            return checkNgen_folder(
                os.path.join(os.path.dirname(self._input_csv), "summary")
            )
        else:
            return checkNgen_folder(self._output_folder)


class DESumVis(SumVis):
    """A class to visualize the summary of DE simulations."""

    def __init__(
        self,
        input_csv: str = "results/de/DE-active/scale2max/all_landscape_de_summary.csv",
        output_folder: str = "",
    ) -> None:

        """
        Args:
        - input_csv: str, The input csv file path for the summary df
            ie 'results/de/DE-active/scale2max/all_landscape_de_summary.csv'
        - output_folder: str, The output folder for the plots
        """

        super().__init__(input_csv, output_folder)

        for metric, metric_dets in DE_METRIC_MAP.items():

            title = f"DE from active variant {metric_dets}"

            self._plot_de_sum(metric, metric_dets, title)

    def _plot_de_sum(self, metric: str, metric_dets: str, title: str):

        """
        A method to plot the summary of DE simulations.

        Args:
        - metric: str, the metric to plot, ie: 'mean_all', 'median_all'
        - metric_dets: str, the details of the metric, ie: 'all simulations fitness mean'
        - title: str, the title of the plot
        """

        save_bokeh_hv(
            hv.Bars(self.df, kdims=["lib", "de_type"], vdims=metric).opts(
                width=1200,
                height=400,
                show_legend=True,
                legend_position="top",
                legend_offset=(0, 5),
                ylabel=metric_dets.capitalize(),
                multi_level=False,
                title=title,
                xlabel="Library",
                hooks=[fixmargins, one_decimal_y, de_sum_hook],
            ),
            plot_name=title,
            plot_path=self.output_folder,
        )

    @property
    def df(self):

        df = self.input_df.copy()
        # clean up the nan if not full N
        # Fill NaNs in 'mean_top96' and 'median_top96' from 'mean_all' and 'median_all'
        df["mean_top96"] = df["mean_top96"].fillna(df["mean_all"])
        df["median_top96"] = df["median_top96"].fillna(df["median_all"])

        # Fill NaNs in 'mean_top384' and 'median_top384' from 'mean_top96' and 'median_top96'
        df["mean_top384"] = df["mean_top384"].fillna(df["mean_top96"])
        df["median_top384"] = df["median_top384"].fillna(df["median_top96"])

        return df


class ZSSSumVis(SumVis):
    """A class to visualize the summary of ZS simulations."""

    def __init__(
        self,
        input_csv="results/zs_sum/none/zs_stat_scale2max.csv",
        output_folder="",
    ) -> None:

        """
        Args:
        - input_csv: str, The input csv file path for the summary df
            ie 'results/zs_sum/none/zs_stat_scale2max.csv'
        - output_folder: str, The output folder for the plots
        """

        super().__init__(input_csv, output_folder)

        self._zs_df = self._get_zs_df()

        for n_mut in ZS_N_MUTS:
            for metric in ZS_METRICS:
                for comb_opt in ZS_COMB_VIS_OPTS:
                    self._plot_zs_sum(
                        metric=metric,
                        n_mut=n_mut,
                        comb_opt=comb_opt,
                    )

    def _get_zs_df(self):

        """
        Make the nested dict input to be flat as columns and make this a melted df

        The input df has the following columns:
            - lib
            - n_mut
            - scale_type
            - Triad_score
            - ev_score
            - esm_score
            - esmif_score
        """

        zs_sum_df = self.input_df.copy()

        zs_cols = [zs_col for zs_col in zs_sum_df.columns if "score" in zs_col]

        print(f"Consider the following ZS columns: {zs_cols}")

        # Melt the DataFrame with the following columns:
        # lib
        # zs_type
        # corr

        zs_sum_df_melt = zs_sum_df.melt(
            id_vars=["lib", "n_mut"],
            value_vars=zs_cols,
            var_name="zs_type",
            value_name="corr",
        )

        # Concatenate the expanded columns with the following columns:
        # lib
        # zs_type
        # rho
        # ndcg
        # rocauc

        df_expanded = pd.concat(
            [
                zs_sum_df_melt.drop("corr", axis=1),
                zs_sum_df_melt["corr"]
                .str.replace(": nan", ": None")
                .apply(literal_eval)
                .apply(pd.Series),
            ],
            axis=1,
        )

        # Return the melted DataFrame with the expanded columns
        # lib
        # zs_type
        # metric
        # value

        df_expanded["zs_type"] = df_expanded["zs_type"].map(ZS_OPTS_LEGEND)

        return df_expanded.melt(
            id_vars=["lib", "n_mut", "zs_type"],
            value_vars=ZS_METRICS,
            var_name="metric",
            value_name="value",
        )

    def _plot_zs_sum(self, metric: str, n_mut: str = "all", comb_opt: str = "both"):

        """
        Plot the bar plot summary of ZS simulations.

        Args:
        - metric: str, the metric to plot, ie: 'rho', 'ndcg', 'rocauc'
        - n_mut: str, the number of mutations to plot, ie: 'all', 'double', 'single'
        - comb_opt: str, the combination option to plot, ie: 'both', 'nocomb', 'comb'
        """

        assert comb_opt in ZS_COMB_VIS_OPTS, f"{comb_opt} not in {ZS_COMB_VIS_OPTS}"

        if comb_opt == "both":
            zs_opt = [ZS_OPTS_LEGEND[opt] for opt in ZS_OPTS] + [ZS_OPTS_LEGEND[opt] for opt in ZS_COMB_OPTS]
            hook = self._zs_hook
            comb_dets = "-both"
        elif comb_opt == "nocomb":
            zs_opt = [ZS_OPTS_LEGEND[opt] for opt in ZS_OPTS]
            hook = self._zs_nocombhook
            comb_dets = "-nocomb"
        else:
            zs_opt = [ZS_OPTS_LEGEND[opt] for opt in ZS_COMB_OPTS]
            hook = self._zs_combhook
            comb_dets = "-comb"

        # Create the Holoviews Bars element
        save_bokeh_hv(
            hv.Bars(
                self._zs_df[
                    (self._zs_df["metric"] == metric)
                    & (self._zs_df["n_mut"] == n_mut)
                    & (self._zs_df["zs_type"].isin(zs_opt))
                ],
                kdims=["lib", "zs_type"],
                vdims="value",
            ).opts(
                width=1200,
                height=400,
                show_legend=True,
                legend_position="top",
                legend_offset=(0, 5),
                multi_level=False,
                title=f"ZS fitness {metric} correlation for {n_mut}",
                xlabel="Library",
                ylim=(0, 1),
                ylabel=f"{metric} correlation",
                hooks=[fixmargins, one_decimal_y, hook],
            ),
            plot_name=f"{self.zs_dets}-{metric}{comb_dets}-{n_mut}",
            plot_path=self._output_folder,
        )

    def _zs_hook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in [ZS_OPTS_LEGEND[opt] for opt in ZS_OPTS] + [ZS_OPTS_LEGEND[opt] for opt in ZS_COMB_OPTS]
        ]

    def _zs_nocombhook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in [ZS_OPTS_LEGEND[opt] for opt in ZS_OPTS]
        ]

    def _zs_combhook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in [ZS_OPTS_LEGEND[opt] for opt in ZS_COMB_OPTS]
        ]

    @property
    def zs_dir(self):
        """
        Return the directory of the ZS simulations.
            ie 'results/zs_sum/none/'
        """
        return os.path.dirname(self.input_csv)

    @property
    def zs_dets(self):
        """
        Return the details of the ZS simulations.
            ie 'zs_stat_scale2max'
        """
        return get_file_name(self.input_csv)

    @property
    def zs_df(self):
        return self._zs_df



def plot_de_v_mlde(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_samples",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    n_top: int = 96,
    min_ftlib: bool = True,
    simplezs: bool = True,
    liborderby: str = "single_step_DE",
) -> None:

    """
    A function for plotting DE vs MLDE in a collage plot
    loop over all ZS scores for all libraries in each

    Args:
    - plot_folder: str, path to the folder where the plots will be saved
    - mlde_csv: str, path to the MLDE csv file
    """

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join(
            "".join([plot_folder, app_zs.replace(" ", "_")]), n_mut, str(n_top)
        )
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    for zs in tqdm(zs_opts):

        """
        "struc-comb_score": "Triad + ESM-IF",
        "msanoif-comb_score": "EVMutation + ESM",
        "msa-comb_score": "EVMutation + ESM + ESM-IF",
        "structnmsa-comb_score": "Triad + EVMutation + ESM + ESM-IF"
        """

        if min_ftlib and zs != "none":
            app_ftlib = " (12.5% ft)"
        else:
            app_ftlib = ""

        sup_title = (
            f"{ZS_OPTS_LEGEND[zs]} MLDE{app_ftlib} top{str(n_top)} vs DE"
        )

        # organize into
        # first row easy de: pard2, pard3, dhfr, trpb3i
        # second row dead: trpb3a, trpb3b, trpb3c, trpb3h
        # third row rest of 3 sites: trpb3e, trpb3f, trpb3g, trpb3d,
        # fourth row two 4 sites: GB1, TrpB4
        # lib4by4 = {0: ["ParD2", "ParD3", "DHFR", "TrpB3I"],
        #           1: ["TrpB3B", "TrpB3C", "TrpB3H", "TrpB3A"],
        #           2: ["TrpB3E", "TrpB3G", "TrpB3F", "TrpB3D"],
        #           3: ["GB1", "TrpB4"]}
        # LIB3BY5 = ["ParD3", "ParD2", "DHFR", "TrpB3I", "TrpB3D",
        #            "GB1", "TrpB4", "TrpB3G", "TrpB3F", ""
        #            "TrpB3E", "TrpB3H", "TrpB3A", "TrpB3C", "TrpB3B",
        # ]

        de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
        liborder = list(
            de_sumdf[de_sumdf["de_type"] == liborderby]
            .sort_values(by=["mean_all"])["lib"]
            .values
        )

        ncol = 5
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16))

        for i, (ax, lib) in enumerate(zip(axs.flatten(), liborder)):

            ss_df = pd.read_csv(f"{de_folder}/{lib}-single_step_DE.csv").copy()
            recomb_df = pd.read_csv(f"{de_folder}/{lib}-recomb_SSM.csv").copy()
            toprecomb_df = pd.read_csv(f"{de_folder}/{lib}-top96_SSM.csv").copy()

            mlde_df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == n_mut)
                & (mlde_all["zs"] == zs)
                & (mlde_all["n_top"] == n_top)
                & (mlde_all["encoding"] == "one-hot")
            ].copy()

            for de, de_df in zip(DE_TYPES, [ss_df, recomb_df, toprecomb_df]):

                ax.plot(
                    de_df["final_fitness"],
                    ecdf_transform(de_df["final_fitness"]),
                    ".",
                    label=f"DE: {DE_LEGEND_MAP[de]}",
                    color=DE_COLORS[de],
                )

            for n, n_samples in enumerate(N_SAMPLE_LIST):
                # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                mlde_df_n = mlde_df[mlde_df["n_sample"] == n_samples]["top_maxes"]
                ax.plot(
                    mlde_df_n,
                    ecdf_transform(mlde_df_n),
                    ".",
                    label=f"{ZS_OPTS_LEGEND[zs]} - {str(n_samples)}",
                    color=MLDE_COLORS[n + 2],
                )

            if i == ncol - 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

            ax.set_title(lib)
            ax.set_xlabel("Max fitness achieved")
            ax.set_ylabel("ECDF")

        extra_len = len(axs.flatten()) - len(liborder)

        if extra_len > 0:
            for i in range(extra_len):
                axs.flatten()[-(i + 1)].set_visible(False)

        fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

        save_plt(
            fig,
            plot_title=sup_title,
            path2folder=plot_folder,
        )


def plot_n_ftmlde(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/ftMLDE",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/simulations/DE-active/scale2max",
    simplezs: bool = True,
    min_ftlib: bool = True,
    n_mut: str = "all",
    n_top: int = 96,
    liborderby: str = "single_step_DE",
) -> None:

    """
    Plot the fitness landscape of the top maxes for ftMLDE vs DE

    Args:
    - plot_folder: str, path to the folder where the plots will be saved
    - mlde_csv: str, path to the MLDE csv file
    - de_folder: str, path to the DE folder
    """

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join(
            "".join([plot_folder, app_zs.replace(" ", "_")]), n_mut, str(n_top)
        )
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    for n in tqdm(N_SAMPLE_LIST):

        sup_title = f"{str(n)} MLDE{app_ftlib} top{str(n_top)} vs DE"

        de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
        liborder = list(
            de_sumdf[de_sumdf["de_type"] == liborderby]
            .sort_values(by=["mean_all"])["lib"]
            .values
        )

        ncol = 5
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16))

        for i, (ax, lib) in enumerate(zip(axs.flatten(), liborder)):

            ss_df = pd.read_csv(f"{de_folder}/{lib}-single_step_DE.csv").copy()
            recomb_df = pd.read_csv(f"{de_folder}/{lib}-recomb_SSM.csv").copy()
            toprecomb_df = pd.read_csv(f"{de_folder}/{lib}-top96_SSM.csv").copy()

            mlde_df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == n_mut)
                & (mlde_all["n_sample"] == n)
                & (mlde_all["n_top"] == n_top)
                & (mlde_all["encoding"] == "one-hot")
            ].copy()

            for de, de_df in zip(DE_TYPES, [ss_df, recomb_df, toprecomb_df]):

                ax.plot(
                    de_df["final_fitness"],
                    ecdf_transform(de_df["final_fitness"]),
                    ".",
                    label=f"DE: {DE_LEGEND_MAP[de]}",
                    color=DE_COLORS[de],
                )

            for zs in zs_opts:

                if zs == "none":
                    mlde_df_n = mlde_df[(mlde_df["zs"] == zs)]["top_maxes"]

                else:
                    mlde_df_n = mlde_df[
                        (mlde_df["zs"] == zs)
                        & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                    ]["top_maxes"]

                ax.plot(
                    mlde_df_n,
                    ecdf_transform(mlde_df_n),
                    ".",
                    label=ZS_OPTS_LEGEND[zs],
                    color=ZS_COLOR_MAP[zs],
                )

            # append double and single if all
            if n_mut == "all":
                for n_mut_sub in N_MUT_SUBS:
                    hm = mlde_all[
                        (mlde_all["lib"] == lib)
                        & (mlde_all["n_mut_cutoff"] == n_mut_sub)
                        & (mlde_all["n_sample"] == n)
                        & (mlde_all["n_top"] == n_top)
                        & (mlde_all["encoding"] == "one-hot")
                        & (mlde_all["zs"] == "none")
                    ]["top_maxes"]

                    ax.plot(
                        hm,
                        ecdf_transform(hm),
                        ".",
                        label=N_MUT_LEGEND[n_mut_sub],
                        color=N_MUT_COLOR[n_mut_sub],
                    )

            if i == ncol - 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

            ax.set_title(lib)
            ax.set_xlabel("Max fitness achieved")
            ax.set_ylabel("ECDF")

        extra_len = len(axs.flatten()) - len(liborder)

        if extra_len > 0:
            for i in range(extra_len):
                axs.flatten()[-(i + 1)].set_visible(False)

        fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

        save_plt(
            fig,
            plot_title=sup_title,
            path2folder=plot_folder,
        )


def plot_de_mlde_ft_v_n(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_frac",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    meanorfrac: str = "mean",
    n_mut: str = "all",
    n_top: int = 96,
    min_ftlib: bool = True,
    simplezs: bool = True,
    liborderby: str = "single_step_DE",
) -> None:
    """"""
    # combine all in one
    # Create the figure and subplots

    ncol = 5
    nrow = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16), sharex=True, sharey=True)

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    if meanorfrac == "mean":
        app_meanfrac = "mean of max fitness"
        y_label = "Mean of max fitness"
    else:
        app_meanfrac = "fraction of max fitness = 1"
        y_label = "Fraction of max fitness = 1"

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join(
            "".join([plot_folder, app_zs.replace(" ", "_")]), n_mut, str(n_top)
        )
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    sup_title = (
        f"DE vs MLDE{app_ftlib} top{str(n_top)} {app_meanfrac} over sample size"
    )

    de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
    liborder = list(
        de_sumdf[de_sumdf["de_type"] == liborderby]
        .sort_values(by=["mean_all"])["lib"]
        .values
    )

    for i, (ax, lib) in tqdm(enumerate(zip(axs.flatten(), liborder))):

        ss_df = pd.read_csv(f"{de_folder}/{lib}-single_step_DE.csv").copy()
        recomb_df = pd.read_csv(f"{de_folder}/{lib}-recomb_SSM.csv").copy()
        toprecomb_df = pd.read_csv(f"{de_folder}/{lib}-top96_SSM.csv").copy()

        if meanorfrac == "mean":
            std_dict = {}

            ss_val = ss_df["final_fitness"].mean()
            recomb_val = recomb_df["final_fitness"].mean()
            toprecomb_val = toprecomb_df["final_fitness"].mean()

        else:
            ss_val = get_val_frac(ss_df["final_fitness"], numb=1)
            recomb_val = get_val_frac(recomb_df["final_fitness"], numb=1)
            toprecomb_val = get_val_frac(toprecomb_df["final_fitness"], numb=1)

        for zs in zs_opts:

            # Common condition
            df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == n_mut)
                & (mlde_all["n_top"] == n_top)
                & (mlde_all["encoding"] == "one-hot")
                & (mlde_all["zs"] == zs)
            ]

            # Apply the additional condition if needed
            if min_ftlib and zs != "none":
                df = df[(df["ft_lib"] == df["ft_lib"].min())]

            fit_dict = {}

            for n_samples in N_SAMPLE_LIST:
                # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                mlde_df_n = df[df["n_sample"] == n_samples]["top_maxes"]

                if len(mlde_df_n) == 0:
                    print(f"Empty mlde_df_n for {lib} {zs} {n_samples} {n_mut} {n_top}")

                if meanorfrac == "mean":
                    fit_dict[n_samples] = mlde_df_n.mean()
                    std_dict[n_samples] = mlde_df_n.std()
                else:
                    fit_dict[n_samples] = get_val_frac(mlde_df_n, numb=1)

            # Plotting
            ax.plot(
                list(fit_dict.keys()),
                list(fit_dict.values()),
                color=ZS_COLOR_MAP[zs],
                marker="o",
                linestyle="solid",
                linewidth=2,
                label=ZS_OPTS_LEGEND[zs],
            )

            # add std if mean
            if meanorfrac == "mean":
                ax.fill_between(
                    list(fit_dict.keys()),
                    [fit_dict[k] - std_dict[k] for k in fit_dict.keys()],
                    [fit_dict[k] + std_dict[k] for k in fit_dict.keys()],
                    color=ZS_COLOR_MAP[zs],
                    alpha=0.05,
                )

        # append double if all
        if n_mut == "all":
            for n_mut_sub in N_MUT_SUBS:
                hm_df = mlde_all[
                    (mlde_all["lib"] == lib)
                    & (mlde_all["n_mut_cutoff"] == n_mut_sub)
                    & (mlde_all["n_top"] == n_top)
                    & (mlde_all["encoding"] == "one-hot")
                    & (mlde_all["zs"] == "none")
                ]

                fit_dict = {}

                for n_samples in N_SAMPLE_LIST:
                    # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                    mlde_df_n = hm_df[hm_df["n_sample"] == n_samples]["top_maxes"]

                    if meanorfrac == "mean":
                        fit_dict[n_samples] = mlde_df_n.mean()
                    else:
                        fit_dict[n_samples] = get_val_frac(mlde_df_n, numb=1)

                ax.plot(
                    list(fit_dict.keys()),
                    list(fit_dict.values()),
                    color=N_MUT_COLOR[n_mut_sub],
                    marker="o",
                    linestyle="solid",
                    linewidth=2,
                    label=N_MUT_LEGEND[n_mut_sub],
                )

        for de, de_val in zip(DE_TYPES, [ss_val, recomb_val, toprecomb_val]):
            ax.axhline(
                y=de_val,
                color=DE_COLORS[de],
                linestyle=DE_LINE_STYLES[de],
                linewidth=2,
                label=f"DE: {DE_LEGEND_MAP[de]}",
            )

        # Setting labels and title
        ax.set_xlabel("Number of samples")
        ax.set_ylabel(y_label)
        ax.tick_params(axis="y")

        if i == ncol - 1:
            # create labels
            ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1))

        ax.set_title(lib)
        ax.set_ylim(0, 1)

        extra_len = len(axs.flatten()) - len(liborder)

        if extra_len > 0:
            for i in range(extra_len):
                axs.flatten()[-(i + 1)].set_visible(False)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_de_mlde_ft_v_n_comb(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_frac",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    n_top: int = 96,
    min_ftlib: bool = True,
    simplezs: bool = True,
    liborderby: str = "single_step_DE",
) -> None:
    """"""
    # combine all in one
    # Create the figure and subplots

    ncol = 5
    nrow = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16), sharex=True, sharey=True)

    frac_axs = []
    for ax in axs.flatten():
        frac_axs.append(ax.twinx())

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    y1_label = "Mean of max fitness"
    y2_label = "Fraction of max fitness = 1"

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join(
            "".join([plot_folder, app_zs.replace(" ", "_")]), n_mut, str(n_top)
        )
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    sup_title = f"DE vs MLDE{app_ftlib} top{str(n_top)} over sample size"

    de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
    liborder = list(
        de_sumdf[de_sumdf["de_type"] == liborderby]
        .sort_values(by=["mean_all"])["lib"]
        .values
    )

    for i, (ax, frac_ax, lib) in tqdm(
        enumerate(zip(axs.flatten(), frac_axs, liborder))
    ):

        ss_df = pd.read_csv(f"{de_folder}/{lib}-single_step_DE.csv").copy()
        recomb_df = pd.read_csv(f"{de_folder}/{lib}-recomb_SSM.csv").copy()
        toprecomb_df = pd.read_csv(f"{de_folder}/{lib}-top96_SSM.csv").copy()

        std_dict = {}

        ss_mean = ss_df["final_fitness"].mean()
        recomb_mean = recomb_df["final_fitness"].mean()
        toprecomb_mean = toprecomb_df["final_fitness"].mean()

        ss_frac = get_val_frac(ss_df["final_fitness"], numb=1)
        recomb_frac = get_val_frac(recomb_df["final_fitness"], numb=1)
        toprecomb_frac = get_val_frac(toprecomb_df["final_fitness"], numb=1)

        for zs in zs_opts:

            # Common condition
            df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == n_mut)
                & (mlde_all["n_top"] == n_top)
                & (mlde_all["encoding"] == "one-hot")
                & (mlde_all["zs"] == zs)
            ]

            # Apply the additional condition if needed
            if min_ftlib and zs != "none":
                df = df[(df["ft_lib"] == df["ft_lib"].min())]

            mean_dict = {}
            frac_dict = {}

            for n_samples in N_SAMPLE_LIST:
                # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                mlde_df_n = df[df["n_sample"] == n_samples]["top_maxes"]

                if len(mlde_df_n) == 0:
                    print(f"Empty mlde_df_n for {lib} {zs} {n_samples} {n_mut} {n_top}")

                mean_dict[n_samples] = mlde_df_n.mean()
                std_dict[n_samples] = mlde_df_n.std()

                frac_dict[n_samples] = get_val_frac(mlde_df_n, numb=1)

            # Plotting for mean
            ax.plot(
                list(mean_dict.keys()),
                list(mean_dict.values()),
                color=ZS_COLOR_MAP[zs],
                marker="o",
                linestyle="solid",
                linewidth=2,
                # label=ZS_OPTS_LEGEND[zs],
            )

            # add std
            ax.fill_between(
                list(mean_dict.keys()),
                [mean_dict[k] - std_dict[k] for k in mean_dict.keys()],
                [mean_dict[k] + std_dict[k] for k in mean_dict.keys()],
                color=ZS_COLOR_MAP[zs],
                alpha=0.05,
            )

            # plot for frac
            frac_ax.plot(
                list(frac_dict.keys()),
                list(frac_dict.values()),
                color=ZS_COLOR_MAP[zs],
                marker="*",
                linestyle="dashed",
                linewidth=2,
            )

        # append double if all
        if n_mut == "all":
            for n_mut_sub in N_MUT_SUBS:
                hm_df = mlde_all[
                    (mlde_all["lib"] == lib)
                    & (mlde_all["n_mut_cutoff"] == n_mut_sub)
                    & (mlde_all["n_top"] == n_top)
                    & (mlde_all["encoding"] == "one-hot")
                    & (mlde_all["zs"] == "none")
                ]

                mean_dict = {}
                std_dict = {}
                frac_dict = {}

                for n_samples in N_SAMPLE_LIST:
                    # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                    mlde_df_n = hm_df[hm_df["n_sample"] == n_samples]["top_maxes"]

                    mean_dict[n_samples] = mlde_df_n.mean()
                    std_dict[n_samples] = mlde_df_n.std()

                    frac_dict[n_samples] = get_val_frac(mlde_df_n, numb=1)

                ax.plot(
                    list(mean_dict.keys()),
                    list(mean_dict.values()),
                    color=N_MUT_COLOR[n_mut_sub],
                    marker="o",
                    linestyle="solid",
                    linewidth=2,
                    # label="HD2",
                )

                ax.fill_between(
                    list(mean_dict.keys()),
                    [mean_dict[k] - std_dict[k] for k in mean_dict.keys()],
                    [mean_dict[k] + std_dict[k] for k in mean_dict.keys()],
                    color=N_MUT_COLOR[n_mut_sub],
                    alpha=0.05,
                )

                frac_ax.plot(
                    list(frac_dict.keys()),
                    list(frac_dict.values()),
                    color=N_MUT_COLOR[n_mut_sub],
                    marker="*",
                    linestyle="dashed",
                    linewidth=2,
                )

        for de, de_mean, de_frac in zip(
            DE_TYPES,
            [ss_mean, recomb_mean, toprecomb_mean],
            [ss_frac, recomb_frac, toprecomb_frac],
        ):
            ax.axhline(
                y=de_mean,
                color=DE_COLORS[de],
                linestyle="solid",
                linewidth=2,
                # label=DE_LEGEND_MAP[de],
            )

            frac_ax.axhline(
                y=de_frac,
                color=DE_COLORS[de],
                linestyle="dashed",
                linewidth=2,
            )

        # Setting labels and title
        ax.set_xlabel("Number of samples")
        ax.set_ylabel(y1_label)
        ax.set_ylim(0, 1)
        # ax.tick_params(axis="y")

        frac_ax.set_ylabel(y2_label)
        frac_ax.set_ylim(0, 1)

        if i % ncol != ncol - 1:
            frac_ax.set_yticklabels([])

        if i == ncol - 1:
            # create labels
            legend_list = []
            for zs in zs_opts:
                legend_list.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="solid",
                        color=ZS_COLOR_MAP[zs],
                        label=ZS_OPTS_LEGEND[zs],
                    )
                )

            # add double
            for n_mut_sub in N_MUT_SUBS:
                legend_list.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="solid",
                        color=N_MUT_COLOR[n_mut_sub],
                        label=N_MUT_LEGEND[n_mut_sub],
                    )
                )

            # add single step, recomb, top96
            for de in DE_TYPES:
                legend_list.append(
                    Line2D(
                        [0],
                        [0],
                        color=DE_COLORS[de],
                        marker="o",
                        linestyle="solid",
                        label=f"DE: {DE_LEGEND_MAP[de]}",
                    )
                )

            # add frac
            legend_list.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    linestyle="solid",
                    label="Mean of max fitness",
                )
            )
            legend_list.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="*",
                    linestyle="dashed",
                    label="Fraction of max fitness = 1",
                )
            )

            ax.legend(handles=legend_list, loc="upper left", bbox_to_anchor=(1.12, 1))

        ax.set_title(lib)

    extra_len = len(axs.flatten()) - len(liborder)

    if extra_len > 0:
        for i in range(extra_len):
            axs.flatten()[-(i + 1)].set_visible(False)
            frac_axs[-(i + 1)].set_visible(False)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_de_mlde_ft_count_v_n(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_count",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    n_top: int = 96,
    meanorfrac: str = "mean",
    simplezs: bool = True,
    min_ftlib: bool = False,
):
    """ """

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    if meanorfrac == "mean":
        app_meanfrac = "mean of max fitness"
    else:
        app_meanfrac = "fraction of max fitness = 1"

    if simplezs:
        nrow = 2
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        nrow = 3
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join(
            "".join([plot_folder, app_zs.replace(" ", "_")]), n_mut, str(n_top)
        )
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    sup_title = (
        f"DE vs MLDE{app_ftlib} top{str(n_top)} {app_meanfrac} over sample size"
    )

    ncol = 3

    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 12), sharex=True, sharey=True)

    for i, (ax, zs) in tqdm(enumerate(zip(axs.flatten(), zs_opts))):

        # init benefit_count to zeros
        benefit_count = {}
        for n in N_SAMPLE_LIST:
            benefit_count[n] = {}

            for de in DE_TYPES:
                benefit_count[n][de] = 0

        for lib in LIB_INFO_DICT.keys():
            for de in DE_TYPES:
                de_df = pd.read_csv(f"{de_folder}/{lib}-{de}.csv").copy()

                if meanorfrac == "mean":
                    de_val = de_df["final_fitness"].mean()
                else:
                    de_val = get_val_frac(s=de_df["final_fitness"], numb=1)

                for n in N_SAMPLE_LIST:

                    df = mlde_all[
                        (mlde_all["n_mut_cutoff"] == n_mut)
                        & (mlde_all["encoding"] == "one-hot")
                        & (mlde_all["zs"] == zs)
                        & (mlde_all["lib"] == lib)
                        & (mlde_all["n_sample"] == n)
                        & (mlde_all["n_top"] == n_top)
                    ]

                    if min_ftlib and zs != "none":
                        df = df[df["ft_lib"] == df["ft_lib"].min()]

                    mlde_df_n = df["top_maxes"]

                    if len(mlde_df_n) == 0:
                        print(f"Empty mlde_df_n for {lib} {zs} {n} {n_mut} {n_top}")

                    if meanorfrac == "mean":
                        mlde_val = mlde_df_n.mean()
                    else:
                        mlde_val = get_val_frac(s=mlde_df_n, numb=1)

                    if mlde_val > de_val:
                        benefit_count[n][de] += 1

        x = list(benefit_count.keys())
        single_step_DE = [
            benefit_count[key]["single_step_DE"] / len(LIB_INFO_DICT) for key in x
        ]
        recomb_SSM = [
            benefit_count[key]["recomb_SSM"] / len(LIB_INFO_DICT) for key in x
        ]
        top96_SSM = [benefit_count[key]["top96_SSM"] / len(LIB_INFO_DICT) for key in x]

        for de, ys in zip(DE_TYPES, [single_step_DE, recomb_SSM, top96_SSM]):
            ax.plot(
                x,
                ys,
                color=DE_COLORS[de],
                marker="o",
                linestyle=DE_LINE_STYLES[de],
                linewidth=2,
                label=f"DE: {DE_LEGEND_MAP[de]}",
            )

        if i == ncol - 1:
            # create labels
            ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1))

        # Adding labels and title
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Fraction of MLDE > DE")
        ax.tick_params(axis="y")

        ax.set_title(ZS_OPTS_LEGEND[zs])
        ax.set_ylim(0, 1)

        extra_len = len(axs.flatten()) - len(zs_opts)

        if extra_len > 0:
            for i in range(extra_len):
                axs.flatten()[-(i + 1)].set_visible(False)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_de_mlde_ft_count_v_n_comb(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_count",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    n_top: int = 96,
    meanorfrac: str = "mean",
    simplezs: bool = True,
    min_ftlib: bool = False,
):
    """ """

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    if meanorfrac == "mean":
        app_meanfrac = "mean of max fitness"
    else:
        app_meanfrac = "fraction of max fitness = 1"

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join(
            "".join([plot_folder, app_zs.replace(" ", "_")]), n_mut, str(n_top)
        )
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    sup_title = f"DE vs MLDE{app_ftlib} top{str(n_top)} {app_meanfrac} over sample size comb"

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, zs in tqdm(enumerate(zs_opts)):

        # init benefit_count to zeros
        benefit_count = {}
        for n in N_SAMPLE_LIST:
            benefit_count[n] = {}

            for de in DE_TYPES:
                benefit_count[n][de] = 0

        for lib in LIB_INFO_DICT.keys():
            for de in DE_TYPES:
                de_df = pd.read_csv(f"{de_folder}/{lib}-{de}.csv").copy()

                if meanorfrac == "mean":
                    de_val = de_df["final_fitness"].mean()
                else:
                    de_val = get_val_frac(s=de_df["final_fitness"], numb=1)

                for n in N_SAMPLE_LIST:

                    df = mlde_all[
                        (mlde_all["n_mut_cutoff"] == n_mut)
                        & (mlde_all["encoding"] == "one-hot")
                        & (mlde_all["zs"] == zs)
                        & (mlde_all["lib"] == lib)
                        & (mlde_all["n_sample"] == n)
                        & (mlde_all["n_top"] == n_top)
                    ]

                    if min_ftlib and zs != "none":
                        df = df[df["ft_lib"] == df["ft_lib"].min()]

                    mlde_df_n = df["top_maxes"]

                    if len(mlde_df_n) == 0:
                        print(f"Empty mlde_df_n for {lib} {zs} {n} {n_mut} {n_top}")

                    if meanorfrac == "mean":
                        mlde_val = mlde_df_n.mean()
                    else:
                        mlde_val = get_val_frac(s=mlde_df_n, numb=1)

                    if mlde_val > de_val:
                        benefit_count[n][de] += 1

        x = list(benefit_count.keys())
        single_step_DE = [
            benefit_count[key]["single_step_DE"] / len(LIB_INFO_DICT) for key in x
        ]
        recomb_SSM = [
            benefit_count[key]["recomb_SSM"] / len(LIB_INFO_DICT) for key in x
        ]
        top96_SSM = [benefit_count[key]["top96_SSM"] / len(LIB_INFO_DICT) for key in x]

        for de, ys in zip(DE_TYPES, [single_step_DE, recomb_SSM, top96_SSM]):
            ax.plot(
                x,
                ys,
                color=ZS_COLOR_MAP[zs],
                marker="o",
                linestyle=DE_LINE_STYLES[de],
                linewidth=2,
                # label=f"DE: {DE_LEGEND_MAP[de]}",
            )

    # add double
    if n_mut == "all":
        for n_mut_sub in N_MUT_SUBS:
            benefit_count = {}
            for n in N_SAMPLE_LIST:
                benefit_count[n] = {}

                for de in DE_TYPES:
                    benefit_count[n][de] = 0

            for lib in LIB_INFO_DICT.keys():
                for de in DE_TYPES:
                    de_df = pd.read_csv(f"{de_folder}/{lib}-{de}.csv").copy()

                    if meanorfrac == "mean":
                        de_val = de_df["final_fitness"].mean()
                    else:
                        de_val = get_val_frac(s=de_df["final_fitness"], numb=1)

                    for n in N_SAMPLE_LIST:

                        df = mlde_all[
                            (mlde_all["n_mut_cutoff"] == n_mut_sub)
                            & (mlde_all["encoding"] == "one-hot")
                            & (mlde_all["zs"] == "none")
                            & (mlde_all["lib"] == lib)
                            & (mlde_all["n_sample"] == n)
                            & (mlde_all["n_top"] == n_top)
                        ]

                        mlde_df_n = df["top_maxes"]

                        if len(mlde_df_n) == 0:
                            print(f"Empty mlde_df_n for {lib} {zs} {n} {n_mut} {n_top}")

                        if meanorfrac == "mean":
                            mlde_val = mlde_df_n.mean()
                        else:
                            mlde_val = get_val_frac(s=mlde_df_n, numb=1)

                        if mlde_val > de_val:
                            benefit_count[n][de] += 1

            x = list(benefit_count.keys())
            single_step_DE = [
                benefit_count[key]["single_step_DE"] / len(LIB_INFO_DICT) for key in x
            ]
            recomb_SSM = [
                benefit_count[key]["recomb_SSM"] / len(LIB_INFO_DICT) for key in x
            ]
            top96_SSM = [benefit_count[key]["top96_SSM"] / len(LIB_INFO_DICT) for key in x]

            for de, ys in zip(DE_TYPES, [single_step_DE, recomb_SSM, top96_SSM]):
                ax.plot(
                    x,
                    ys,
                    color=N_MUT_COLOR[n_mut_sub],
                    marker="o",
                    linestyle=DE_LINE_STYLES[de],
                    linewidth=2,
                    # label="HD2",
                )

    # manually create legend
    legend_list = []

    for zs in zs_opts:
        legend_list.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=ZS_COLOR_MAP[zs],
                label=ZS_OPTS_LEGEND[zs],
            )
        )
    # add double only
    for n_mut_sub in N_MUT_SUBS:
        legend_list.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=N_MUT_COLOR[n_mut_sub],
                label=N_MUT_LEGEND[n_mut_sub],
            )
        )

    # add single step, recomb, top96
    for de, ls in DE_LINE_STYLES.items():
        legend_list.append(
            Line2D([0], [0], color="gray", linestyle=ls, label=f"DE: {DE_LEGEND_MAP[de]}")
        )

    # Manually add lines to the legend
    ax.legend(handles=legend_list, loc="upper left", bbox_to_anchor=(1, 1))

    # Adding labels and title
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Fraction of MLDE > DE")
    ax.tick_params(axis="y")

    ax.set_title(sup_title)
    ax.set_ylim(0, 1)
    ax.set_xscale("log")

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_ftlib_v_size(
    plot_folder: str = "results/mlde_vs_ftmlde/onehot/collage/lib_size",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    n_sample: int = 384,
    n_top: int = 96,
    simplezs: bool = True,
    liborderby: str = "single_step_DE",
):
    """ """

    if simplezs:
        nrow = 2
        app_zs = ""
        zs_opts = ["none"] + ZS_OPTS
    else:
        nrow = 3
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join("".join([plot_folder, app_zs.replace(" ", "_")]), n_mut)
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    sup_title = f"Focused training library sampling {str(n_sample)} top{str(n_top)} over library size"

    de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
    liborder = list(
        de_sumdf[de_sumdf["de_type"] == liborderby]
        .sort_values(by=["mean_all"])["lib"]
        .values
    )

    mlde_df = mlde_all[
        (mlde_all["encoding"] == "one-hot")
        & (mlde_all["n_sample"] == n_sample)
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["zs"].isin(zs_opts))
    ].copy()

    ncol = 5
    nrow = 3

    fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16), sharex=True, sharey=True)
    frac_axs = []

    for ax in axs.flatten():
        frac_axs.append(ax.twinx())

    for i, (ax, frac_ax, lib) in tqdm(
        enumerate(zip(axs.flatten(), frac_axs, liborder))
    ):
        # each subplot is a library with x axis as ft_lib and y axis as mean of top_maxes for each zs
        lib_df = mlde_df[
            (mlde_df["lib"] == lib) & (mlde_df["n_mut_cutoff"] == "all")
        ].copy()
        lib_df["ft_lib"] = lib_df["ft_lib"].map(
            {
                numb: frac
                for numb, frac in zip(
                    sorted(lib_df["ft_lib"].unique()), FTLIB_FRAC_LIST
                )
            }
        )
        # plot all none zs as a line
        rand_df = lib_df[lib_df["zs"] == "none"].copy()
        rand_mean = [rand_df["top_maxes"].mean()] * len(FTLIB_FRAC_LIST)
        rand_std = [rand_df["top_maxes"].std()] * len(FTLIB_FRAC_LIST)
        rand_frac = [get_val_frac(rand_df["top_maxes"])] * len(FTLIB_FRAC_LIST)
        ax.plot(
            FTLIB_FRAC_LIST,
            rand_mean,
            color=PRESENTATION_PALETTE_SATURATE["gray"],
            linestyle="solid",
            linewidth=2,
        )
        ax.fill_between(
            FTLIB_FRAC_LIST,
            np.array(rand_mean) - np.array(rand_std),
            np.array(rand_mean) + np.array(rand_std),
            color=PRESENTATION_PALETTE_SATURATE["gray"],
            alpha=0.05,
        )
        frac_ax.plot(
            FTLIB_FRAC_LIST,
            rand_frac,
            color=PRESENTATION_PALETTE_SATURATE["gray"],
            linestyle="dotted",
            linewidth=2,
        )

        # plot double none zs as a line
        for n_mut_sub in N_MUT_SUBS:
            hm_df = mlde_df[
                (mlde_df["lib"] == lib)
                & (mlde_df["n_mut_cutoff"] == n_mut_sub)
                & (mlde_df["zs"] == "none")
            ]
            hm_mean = [hm_df["top_maxes"].mean()] * len(FTLIB_FRAC_LIST)
            hm_std = [hm_df["top_maxes"].std()] * len(FTLIB_FRAC_LIST)
            hm_frac = [get_val_frac(hm_df["top_maxes"])] * len(FTLIB_FRAC_LIST)
            ax.plot(
                FTLIB_FRAC_LIST,
                hm_mean,
                color=N_MUT_COLOR[n_mut_sub],
                linestyle="solid",
                linewidth=2,
            )
            ax.fill_between(
                FTLIB_FRAC_LIST,
                np.array(hm_mean) - np.array(hm_std),
                np.array(hm_mean) + np.array(hm_std),
                color=N_MUT_COLOR[n_mut_sub],
                alpha=0.05,
            )
            frac_ax.plot(
                FTLIB_FRAC_LIST,
                hm_frac,
                color=N_MUT_COLOR[n_mut_sub],
                linestyle="dotted",
                linewidth=2,
            )

        for zs in zs_opts:
            if zs != "none":
                zs_df = lib_df[lib_df["zs"] == zs].copy()
                zs_mean = zs_df.groupby("ft_lib")["top_maxes"].mean()
                zs_std = zs_df.groupby("ft_lib")["top_maxes"].std()
                zs_frac = zs_df.groupby("ft_lib")["top_maxes"].apply(get_val_frac)
                ax.plot(
                    zs_mean.index,
                    zs_mean,
                    label=zs,
                    color=ZS_COLOR_MAP[zs],
                    marker="o",
                    linestyle="solid",
                    linewidth=2,
                )
                ax.fill_between(
                    zs_mean.index,
                    zs_mean - zs_std,
                    zs_mean + zs_std,
                    color=ZS_COLOR_MAP[zs],
                    alpha=0.05,
                )
                frac_ax.plot(
                    zs_frac.index,
                    zs_frac,
                    label=zs,
                    color=ZS_COLOR_MAP[zs],
                    marker="*",
                    linestyle="dotted",
                    linewidth=2,
                )

        ax.set_title(lib)
        ax.set_xlabel("Fraction of library")
        ax.set_ylabel("Mean of max fitness")
        frac_ax.set_ylabel("Fraction of fitness = 1")

        # cut off the last one
        ax.set_xlim(0.125, 0.5)
        ax.set_ylim(0, 1)
        frac_ax.set_xlim(0.125, 0.5)
        frac_ax.set_ylim(0, 1)

        if i % ncol != ncol - 1:
            frac_ax.set_yticklabels([])

        if i == ncol - 1:
            # create labels
            legend_list = []
            for zs in ["none"] + ZS_OPTS:
                legend_list.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="solid",
                        color=ZS_COLOR_MAP[zs],
                        label=ZS_OPTS_LEGEND[zs],
                    )
                )

            # add double
            for n_mut_sub in N_MUT_SUBS:
                legend_list.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="solid",
                        color=N_MUT_COLOR[n_mut_sub],
                        label=N_MUT_LEGEND[n_mut_sub]
                    )
                )

            # add frac
            legend_list.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    linestyle="solid",
                    label="Mean of max fitness",
                )
            )
            legend_list.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="*",
                    linestyle="dashed",
                    label="Fraction of max fitness = 1",
                )
            )

            ax.legend(handles=legend_list, loc="upper left", bbox_to_anchor=(1.12, 1))

        ax.set_title(lib)

    extra_len = len(axs.flatten()) - len(liborder)

    if extra_len > 0:
        for i in range(extra_len):
            axs.flatten()[-(i + 1)].set_visible(False)
            frac_axs[-(i + 1)].set_visible(False)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_countzs(
    plot_folder: str = "results/mlde_vs_ftmlde/onehot/collage/zs_count",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    n_mut: str = "all",
    n_sample: int = 384,
    n_top: int = 96,
    simplezs: bool = True,
    meanorfrac: str = "mean",
    liborderby: str = "single_step_DE",
):
    """bar plots for counting"""

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + N_MUT_SUBS + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + N_MUT_SUBS + ZS_OPTS + ZS_COMB_OPTS

    if meanorfrac == "mean":
        app_meanfrac = "mean of max fitness"
    else:
        app_meanfrac = "fraction of max fitness = 1"

    plot_folder = checkNgen_folder(
        os.path.join("".join([plot_folder, app_zs.replace(" ", "_")]), n_mut)
    )
    mlde_all = pd.read_csv(mlde_csv).copy()

    sup_title = f"Occurance by sampling {str(n_sample)} top{str(n_top)} {app_meanfrac}"

    mlde_df = mlde_all[
        (mlde_all["encoding"] == "one-hot")
        & (mlde_all["n_sample"] == n_sample)
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["zs"].isin(zs_opts))
    ].copy()

    # init list
    calc_df_list = []

    # prep df
    for lib in LIB_INFO_DICT.keys():
        append_df_list = []
        # get no zs double and singles
        for n_mut_sub in ["all"] + N_MUT_SUBS:
            nozs_df = mlde_df[
                (mlde_df["lib"] == lib)
                & (mlde_df["n_mut_cutoff"] == n_mut_sub)
                & (mlde_df["zs"] == "none")
            ]

            append_df_list.append(
                pd.DataFrame(
                        {
                            "lib": [lib] * len(FTLIB_FRAC_LIST),
                            "zs": [n_mut_sub] * len(FTLIB_FRAC_LIST),
                            "ft_lib": FTLIB_FRAC_LIST,
                            "mean": [nozs_df["top_maxes"].mean()] * len(FTLIB_FRAC_LIST),
                            "frac": [get_val_frac(nozs_df["top_maxes"])]
                            * len(FTLIB_FRAC_LIST),
                        }
                    )
            )

        # get mean for the rest
        lib_df = mlde_df[(mlde_df["lib"] == lib) & (mlde_df["n_mut_cutoff"] == "all")][
            ["lib", "zs", "ft_lib", "top_maxes"]
        ]

        # map ft_lib size
        lib_df["ft_lib"] = lib_df["ft_lib"].map(
            {
                numb: frac
                for numb, frac in zip(
                    sorted(lib_df["ft_lib"].unique()), FTLIB_FRAC_LIST
                )
            }
        )

        calc_df_list.append(
            pd.concat(
                append_df_list + [
                    lib_df.groupby(["lib", "zs", "ft_lib"])["top_maxes"]
                    .agg(["mean", get_val_frac])
                    .rename(columns={"get_val_frac": "frac"})
                    .reset_index(),
                ]
            ).reset_index(drop=True)
        )

    all_calc_df = pd.concat(calc_df_list).reset_index(drop=True)
    all_calc_df["zs"] = all_calc_df["zs"].map(ZS_OPTS_APPENDED)
    all_calc_df["type"] = all_calc_df["lib"].map(
        {n: v["type"] for n, v in LIB_INFO_DICT.items()}
    )

    # now plot

    # Create subplots for each type
    types = list(all_calc_df["type"].unique())

    # top enzyme bind best
    # bottom enzyme bind worst
    fig, axes = plt.subplots(
        nrows=2, ncols=len(types), figsize=(8, 6), sharex=True, sharey=True
    )

    # Plot stacked bar plots
    for i, (ax, t) in enumerate(zip(axes.flatten(), types + types)):

        if i // 2 == 0:
            # Group by 'type', 'ft_lib', and 'zs' to get the counts
            grouped = (
                all_calc_df.loc[
                    all_calc_df.groupby(["lib", "ft_lib"])[meanorfrac].idxmax()
                ]
                .groupby(["type", "ft_lib", "zs"])
                .size()
                .unstack(fill_value=0)
            )
            ax.set_title(t)
            ax.set_ylabel("Count of best")
        else:
            grouped = (
                all_calc_df.loc[
                    all_calc_df.groupby(["lib", "ft_lib"])[meanorfrac].idxmin()
                ]
                .groupby(["type", "ft_lib", "zs"])
                .size()
                .unstack(fill_value=0)
            )
            ax.set_ylabel("Count of worst")

        group = grouped.loc[t]
        group.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[
                ZS_COLOR_MAP_APPENDED[{v: k for k, v in ZS_OPTS_APPENDED.items()}[col]]
                for col in group.columns
            ],
        )

        ax.set_xlabel("Fraction of library")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        if i == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            ax.legend().set_visible(False)

    fig.suptitle(sup_title, fontsize=16, fontweight="bold")

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )

def count_zs_v_n_df(
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    simplezs: bool = True
):

    mlde_all = pd.read_csv(mlde_csv)

    if simplezs:
        zs_opts = ["none"] + N_MUT_SUBS + ZS_OPTS
    else:
        zs_opts = ["none"] + N_MUT_SUBS + ZS_OPTS + ZS_COMB_OPTS

    mlde_df = mlde_all[(mlde_all["encoding"] == "one-hot") & (mlde_all["zs"].isin(zs_opts))].copy()

    all_count_df_list = []

    for n_sample in tqdm(N_SAMPLE_LIST):
        for n_top in [96, 384]:

            calc_df_list = []

            n_mlde_df = mlde_df[
                (mlde_df["n_sample"] == n_sample) & (mlde_df["n_top"] == n_top)
            ]

            for lib in LIB_INFO_DICT.keys():

                append_df_list = []

                for n_mut_sub in ["all"] + N_MUT_SUBS:
                    # get double, single and all 
                    hm_df = n_mlde_df[
                        (n_mlde_df["lib"] == lib)
                        & (n_mlde_df["n_mut_cutoff"] == n_mut_sub)
                        & (n_mlde_df["zs"] == "none")
                    ]

                    append_df_list.append(
                            pd.DataFrame(
                            {
                                "lib": [lib] * len(FTLIB_FRAC_LIST),
                                "zs": [n_mut_sub] * len(FTLIB_FRAC_LIST),
                                "ft_lib": FTLIB_FRAC_LIST,
                                "mean": [hm_df["top_maxes"].mean()] * len(FTLIB_FRAC_LIST),
                                "frac": [get_val_frac(hm_df["top_maxes"])]
                                * len(FTLIB_FRAC_LIST),
                            }
                        )
                    )

                # get mean for the rest
                lib_df = n_mlde_df[
                    (n_mlde_df["lib"] == lib) & (n_mlde_df["n_mut_cutoff"] == "all")
                ][["lib", "zs", "ft_lib", "top_maxes"]]

                # map ft_lib size
                lib_df["ft_lib"] = lib_df["ft_lib"].map(
                    {
                        numb: frac
                        for numb, frac in zip(
                            sorted(lib_df["ft_lib"].unique()), FTLIB_FRAC_LIST
                        )
                    }
                )

                calc_df_list.append(
                    pd.concat(
                        append_df_list + [
                            lib_df.groupby(["lib", "zs", "ft_lib"])["top_maxes"]
                            .agg(["mean", get_val_frac])
                            .rename(columns={"get_val_frac": "frac"})
                            .reset_index(),
                        ]
                    ).reset_index(drop=True)
                )

            calc_df = pd.concat(calc_df_list).reset_index(drop=True)
            calc_df["zs"] = calc_df["zs"].map(ZS_OPTS_APPENDED)
            calc_df["type"] = calc_df["lib"].map(
                {n: v["type"] for n, v in LIB_INFO_DICT.items()}
            )

            # slice based on metric now

            for metric in ["mean", "frac"]:
                for minmax in ["min", "max"]:
                    if minmax == "min":
                        sliced_zs = calc_df.loc[
                            calc_df.groupby(["lib", "ft_lib"])[metric].idxmin()
                        ]
                    else:
                        sliced_zs = calc_df.loc[
                            calc_df.groupby(["lib", "ft_lib"])[metric].idxmax()
                        ]

                    # Group by 'type', 'ft_lib', and 'zs' to get the counts
                    grouped = (
                        sliced_zs.groupby(["type", "ft_lib", "zs"])
                        .size()
                        .unstack(fill_value=0)
                    )
                    grouped = grouped.reset_index()
                    grouped["minmax"] = minmax
                    grouped["metric"] = metric
                    grouped["n_sample"] = n_sample
                    grouped["n_top"] = n_top
                    
                    all_count_df_list.append(grouped.reset_index(drop=True))

    return pd.concat(all_count_df_list).fillna(0)


def plot_count_zs_v_n(
    plot_folder: str = "results/mlde_vs_ftmlde/onehot/collage/zs_count_n",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    n_mut: str = "all",
    n_top: int = 96,
    simplezs: bool = True,
):

    if simplezs:
        app_zs = ""
        zs_opts = ["none"] + N_MUT_SUBS + ZS_OPTS
    else:
        app_zs = " with ZS ensemble"
        zs_opts = ["none"] + N_MUT_SUBS + ZS_OPTS + ZS_COMB_OPTS

    plot_folder = checkNgen_folder(
        os.path.join("".join([plot_folder, app_zs.replace(" ", "_")]), n_mut)
    )

    all_df = count_zs_v_n_df(mlde_csv, simplezs=simplezs)

    sup_title = f"Occurance cross sampling size top{str(n_top)}"

    types = list(all_df["type"].unique())

    nrow = 2
    ncol = len(types)
    fig, axs = plt.subplots(nrow, ncol, figsize = (8, 6), sharex=True, sharey=True)

    frac_axs = []
    for ax in axs.flatten():
        frac_axs.append(ax.twinx())

    for i, (ax, frac_ax, t) in enumerate(zip(axs.flatten(), frac_axs, types + types)):
        
        if i // ncol == 0:
            minmax = "max"
            ax.set_title(t)
        else:
            minmax = "min"
            
        if i == 0:
            ax.set_ylabel("Count of best for mean of max")
        if i == 2:
            ax.set_ylabel("Count of worst for mean of max")

        if i % ncol == ncol - 1:
            frac_ax.set_ylabel("Count of best for fraction = 1")
            ax.set_yticklabels([])

        if i == ncol - 1:
            # create labels
            legend_list = []
            for zs in zs_opts:
                legend_list.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="solid",
                        color=ZS_COLOR_MAP_APPENDED[zs],
                        label=ZS_OPTS_APPENDED[zs],
                    )
                )

            # add frac
            legend_list.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="o",
                    linestyle="solid",
                    label="Mean of max fitness",
                )
            )
            legend_list.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="*",
                    linestyle="dashed",
                    label="Fraction of max fitness = 1",
                )
            )

            ax.legend(handles=legend_list, loc="upper left", bbox_to_anchor=(1.2, 1))
        
        mean_df = all_df[
            (all_df["n_top"] == n_top)
            & (all_df["minmax"] == minmax)
            & (all_df["metric"] == "mean")
            & (all_df["ft_lib"] == 0.125)
            & (all_df["type"] == t)
        ].sort_values(by=["n_sample"])

        frac_df = all_df[
            (all_df["n_top"] == n_top)
            & (all_df["minmax"] == minmax)
            & (all_df["metric"] == "frac")
            & (all_df["ft_lib"] == 0.125)
            & (all_df["type"] == t)
        ].sort_values(by=["n_sample"])

        for zs in zs_opts:

            print(zs)
            print(ZS_OPTS_APPENDED[zs])
            print(mean_df.columns)

            ax.plot(
                list(mean_df["n_sample"].values),
                list(mean_df[ZS_OPTS_APPENDED[zs]].values),
                marker="o",
                linestyle="solid",
                color=ZS_COLOR_MAP_APPENDED[zs]
            )

            frac_ax.plot(
                list(frac_df["n_sample"].values),
                list(frac_df[ZS_OPTS_APPENDED[zs]].values),
                marker="*",
                linestyle="dashed",
                color=ZS_COLOR_MAP_APPENDED[zs]
            )
        
        ax.set_xlabel("Number of samples")
        ax.set_xscale("log")

        ax.set_ylim(0, len(mean_df))
        frac_ax.set_ylim(0, len(frac_df))
        
    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.975)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )



def vis_sum_ftlib_v_size(
    plot_dir: str = "results/mlde_vs_ftmlde/onehot/collage/",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    simplezs: bool = True,
    liborderby: str = "single_step_DE",
):
    for n_top in [96, 384]:
        
        common_args = {
                "mlde_csv": mlde_csv,
                "n_mut": n_mut,
                "n_top": n_top,
                "simplezs": simplezs, 
            }

        plot_count_zs_v_n(
            plot_folder=os.path.join(plot_dir,"zs_count_n"),
            **common_args
        )    

        for n_sample in tqdm(N_SAMPLE_LIST):
        
            plot_ftlib_v_size(
                plot_folder = os.path.join(plot_dir, "lib_size"),
                de_folder = de_folder,
                n_sample=n_sample,
                liborderby=liborderby,
                **common_args
            )

            for meanorfrac in ["mean", "frac"]:
                plot_countzs(
                    plot_folder=os.path.join(plot_dir, "zs_count"),
                    meanorfrac=meanorfrac,
                    n_sample=n_sample,
                    liborderby=liborderby,
                    **common_args,
                )
        
def vis_sum_de_mlde(
    plot_dir: str = "results/de_vs_mlde/onehot/collage",
    mlde_csv: str = "results/mlde/all_df_comb.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    liborderby: str = "single_step_DE",
    min_ftlib: bool = True,
    simplezs: bool = True,
):

    common_args = {
        "mlde_csv": mlde_csv,
        "de_folder": de_folder,
        "min_ftlib": min_ftlib,
        "simplezs": simplezs,
    }

    for n_mut in ["all", "double"]:
        for n_top in [96, 384]:

            print(f"Plotting {n_mut} {n_top} DE vs MLDE...")
            plot_de_v_mlde(
                plot_folder=f"{plot_dir}/n_samples",
                n_mut=n_mut,
                n_top=n_top,
                liborderby=liborderby,
                **common_args,
            )

            print(f"Plotting {n_mut} {n_top} for different sample sizes...")
            plot_n_ftmlde(
                plot_folder=f"{plot_dir}/ftMLDE",
                n_mut=n_mut,
                n_top=n_top,
                liborderby=liborderby,
                **common_args,
            )

            print(f"Plotting {n_mut} {n_top} over sample sizes...")
            plot_de_mlde_ft_v_n_comb(
                plot_folder=f"{plot_dir}/n_mean_frac",
                    n_mut=n_mut,
                    n_top=n_top,
                    liborderby=liborderby,
                    **common_args
                )

            for meanorfrac in ["mean", "frac"]:
                print(f"Plotting {n_mut} {n_top} {meanorfrac} over sample sizes...")
                plot_de_mlde_ft_v_n(
                    plot_folder=f"{plot_dir}/n_mean_frac",
                    meanorfrac=meanorfrac,
                    n_mut=n_mut,
                    n_top=n_top,
                    liborderby=liborderby,
                    **common_args,
                )

                print(
                    f"Plotting {n_mut} {n_top} {meanorfrac} zs counts over sample sizes..."
                )
                plot_de_mlde_ft_count_v_n(
                    plot_folder=f"{plot_dir}/n_mean_count",
                    meanorfrac=meanorfrac,
                    n_mut=n_mut,
                    n_top=n_top,
                    **common_args,
                )

                plot_de_mlde_ft_count_v_n_comb(
                    plot_folder=f"{plot_dir}/n_mean_count",
                    meanorfrac=meanorfrac,
                    n_mut=n_mut,
                    n_top=n_top,
                    **common_args,
                )