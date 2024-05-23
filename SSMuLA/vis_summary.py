"""
A script to visualize the results cross landscapes.
"""


import os
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from ast import literal_eval


import matplotlib.pyplot as plt
import holoviews as hv


from SSMuLA.de_simulations import DE_TYPES
from SSMuLA.zs_analysis import (
    ZS_OPTS,
    ZS_COMB_OPTS,
    ZS_OPTS_LEGEND,
    SIMPLE_ZS_OPT_LEGNED,
)
from SSMuLA.landscape_global import LIB_INFO_DICT, LIB_NAMES, TrpB_names
from SSMuLA.vis import (
    LIB_COLORS,
    PRESENTATION_PALETTE_SATURATE,
    MLDE_COLORS,
    SIMPLE_ZS_COLOR_MAP,
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

ZS_METRICS = ["rho", "ndcg", "rocauc"]
ZS_N_MUTS = ["all", "double", "single"]
ZS_COMB_VIS_OPTS = ["both", "nocomb", "comb"]

N_SAMPLE_LIST = [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]

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


def get_val_percent(s: pd.Series, numb: float = 1) -> float:
    """
    Get the fraction of a column that is equal to a certain number.
    """
    return (s == numb).sum().sum() / len(s) * 100


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
            zs_opt = ZS_OPTS + ZS_COMB_OPTS
            hook = self._zs_hook
            comb_dets = "-both"
        elif comb_opt == "nocomb":
            zs_opt = ZS_OPTS
            hook = self._zs_nocombhook
            comb_dets = "-nocomb"
        else:
            zs_opt = ZS_COMB_OPTS
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
                ylabel=f"{metric} correlation",
                multi_level=False,
                title=f"ZS fitness {metric} correlation for {n_mut}",
                xlabel="Library",
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
            (lib, zs) for lib in LIB_NAMES for zs in ZS_OPTS + ZS_COMB_OPTS
        ]

    def _zs_nocombhook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in ZS_OPTS
        ]

    def _zs_combhook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in ZS_COMB_OPTS
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
    mlde_csv: str = "results/mlde/vis_3/all_df.csv",
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

    plot_folder = checkNgen_folder(os.path.join(plot_folder, n_mut, str(n_top)))
    mlde_all = pd.read_csv(mlde_csv).copy()

    if simplezs:
        zs_opts = ZS_OPTS + ["none"]
    else:
        zs_opts = ZS_OPTS + ZS_COMB_OPTS + ["none"]

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

        if simplezs:
            app_zs = ""
        else:
            app_zs = " with ZS ensemble"

        sup_title = (
            f"{ZS_OPTS_LEGEND[zs]} MLDE{app_ftlib} top{str(n_top)} vs DE{app_zs}"
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
        liborder = (
            de_sumdf[de_sumdf["de_type"] == liborderby]
            .sort_values(by=["mean_all"])["lib"]
            .values
        )

        ncol = 5
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16))
        for i, (ax, lib) in enumerate(zip(axs.flatten(), liborder)):

            if i < len(liborder):

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

                ax.plot(
                    ss_df["final_fitness"],
                    ecdf_transform(ss_df["final_fitness"]),
                    ".",
                    label="DE - single-step",
                    color=MLDE_COLORS[0],
                )
                ax.plot(
                    recomb_df["final_fitness"],
                    ecdf_transform(recomb_df["final_fitness"]),
                    ".",
                    label="DE - recombination",
                    color=MLDE_COLORS[1],
                )
                ax.plot(
                    toprecomb_df["final_fitness"],
                    ecdf_transform(toprecomb_df["final_fitness"]),
                    ".",
                    label="DE - top96 recombination",
                    color=PRESENTATION_PALETTE_SATURATE["brown"],
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

            else:
                ax.set_visible(False)

        fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

        save_plt(
            fig,
            plot_title=sup_title,
            path2folder=plot_folder,
        )


def plot_n_ftmlde(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/ftMLDE",
    mlde_csv: str = "results/mlde/vis_3/all_df.csv",
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

    plot_folder = checkNgen_folder(os.path.join(plot_folder, n_mut, str(n_top)))
    mlde_all = pd.read_csv(mlde_csv).copy()

    for n in tqdm(N_SAMPLE_LIST):

        if min_ftlib:
            app_ftlib = " (12.5% ft)"
        else:
            app_ftlib = ""

        if simplezs:
            zs_legend_map = SIMPLE_ZS_OPT_LEGNED
            zs_color_map = SIMPLE_ZS_COLOR_MAP
            app_zs = ""
        else:
            zs_legend_map = ZS_OPTS_LEGEND
            zs_color_map = ZS_COLOR_MAP
            app_zs = " with ZS ensemble"

        sup_title = f"{str(n)} MLDE{app_ftlib} top{str(n_top)} vs DE{app_zs}"

        de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
        liborder = (
            de_sumdf[de_sumdf["de_type"] == liborderby]
            .sort_values(by=["mean_all"])["lib"]
            .values
        )

        # fig, axs = plt.subplots(3, 4, figsize=(28, 16))
        # for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):

        # for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):
        # fig, axs = plt.subplots(4, 4, figsize=(28, 20))
        # for i, (ax, lib) in enumerate(zip(axs.flatten()[:len(LIB4BY4)], LIB4BY4)):

        ncol = 5
        nrow = 3
        fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16))
        for i, (ax, lib) in enumerate(zip(axs.flatten(), liborder)):

            if i < len(liborder):

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

                ax.plot(
                    ss_df["final_fitness"],
                    ecdf_transform(ss_df["final_fitness"]),
                    ".",
                    label="DE - single-step",
                    color=MLDE_COLORS[0],
                )
                ax.plot(
                    recomb_df["final_fitness"],
                    ecdf_transform(recomb_df["final_fitness"]),
                    ".",
                    label="DE - recombination",
                    color=MLDE_COLORS[1],
                )
                ax.plot(
                    toprecomb_df["final_fitness"],
                    ecdf_transform(toprecomb_df["final_fitness"]),
                    ".",
                    label="DE - top96 recombination",
                    color=PRESENTATION_PALETTE_SATURATE["brown"],
                )

                for zs, zs_color in zs_color_map.items():

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
                        label=zs_legend_map[zs],
                        color=zs_color,
                    )

                if i == ncol - 1:
                    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

                ax.set_title(lib)
                ax.set_xlabel("Max fitness achieved")
                ax.set_ylabel("ECDF")

            else:
                ax.set_visible(False)

        fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

        save_plt(
            fig,
            plot_title=sup_title,
            path2folder=plot_folder,
        )


def plot_de_mlde_ft_mean_v_n(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_frac",
    mlde_csv: str = "results/mlde/vis_4/all_df.csv",
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

    plot_folder = checkNgen_folder(os.path.join(plot_folder, n_mut, str(n_top)))
    mlde_all = pd.read_csv(mlde_csv).copy()

    ncol = 5
    nrow = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16), sharex=True, sharey=True)

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    if simplezs:
        zs_legend_map = SIMPLE_ZS_OPT_LEGNED
        zs_color_map = SIMPLE_ZS_COLOR_MAP
        app_zs = ""
    else:
        zs_legend_map = ZS_OPTS_LEGEND
        zs_color_map = ZS_COLOR_MAP
        app_zs = " with ZS ensemble"

    sup_title = f"DE vs MLDE{app_ftlib} top{str(n_top)} mean of max fitness over sample size{app_zs}"

    de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
    liborder = (
        de_sumdf[de_sumdf["de_type"] == liborderby]
        .sort_values(by=["mean_all"])["lib"]
        .values
    )

    for i, (ax, lib) in enumerate(zip(axs.flatten(), liborder)):

        if i < len(liborder):

            ss_df = pd.read_csv(f"{de_folder}/{lib}-single_step_DE.csv").copy()
            recomb_df = pd.read_csv(f"{de_folder}/{lib}-recomb_SSM.csv").copy()
            toprecomb_df = pd.read_csv(f"{de_folder}/{lib}-top96_SSM.csv").copy()

            ss_mean = ss_df["final_fitness"].mean()
            recomb_mean = recomb_df["final_fitness"].mean()
            toprecomb_mean = toprecomb_df["final_fitness"].mean()

            for zs, zs_color in zs_color_map.items():

                mlde_df = mlde_all[
                    (mlde_all["lib"] == lib)
                    & (mlde_all["n_mut_cutoff"] == n_mut)
                    & (mlde_all["n_top"] == n_top)
                    & (mlde_all["encoding"] == "one-hot")
                ].copy()

                if zs == "none":
                    mlde_df = mlde_df[(mlde_df["zs"] == zs)]

                else:
                    mlde_df = mlde_df[
                        (mlde_df["zs"] == zs)
                        & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                    ]

                mean_fit_dict = {}

                for n_samples in N_SAMPLE_LIST:
                    # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                    mlde_df_n = mlde_df[mlde_df["n_sample"] == n_samples]["top_maxes"]

                    mean_fit_dict[n_samples] = mlde_df_n.mean()

                x = list(mean_fit_dict.keys())
                y1 = list(mean_fit_dict.values())

                # Plotting the first y-axis
                ax.plot(
                    x,
                    y1,
                    color=zs_color,
                    marker="o",
                    linestyle="solid",
                    linewidth=2,
                    label=f"Mean of max fitness: {zs_legend_map[zs]}",
                )
            ax.axhline(
                y=ss_mean,
                color=PRESENTATION_PALETTE_SATURATE["orange"],
                linestyle="dotted",
                linewidth=2,
                label="Mean of final fitness: single step",
            )
            ax.axhline(
                y=recomb_mean,
                color=PRESENTATION_PALETTE_SATURATE["yellow"],
                linestyle="dashed",
                linewidth=2,
                label="Mean of final fitness: recomb",
            )
            ax.axhline(
                y=toprecomb_mean,
                color=PRESENTATION_PALETTE_SATURATE["brown"],
                linestyle="dashdot",
                linewidth=2,
                label="Mean of final fitness: top96 recomb",
            )
            # Setting labels and title
            ax.set_xlabel("Training set size")
            ax.set_ylabel("Mean of max fitness")
            ax.tick_params(axis="y")

            if i == ncol - 1:
                # create labels
                ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1))

            ax.set_title(lib)
            ax.set_ylim(0, 1)
        else:
            ax.set_visible(False)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_de_mlde_ft_percent_v_n(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_frac",
    mlde_csv: str = "results/mlde/vis_4/all_df.csv",
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

    plot_folder = checkNgen_folder(os.path.join(plot_folder, n_mut, str(n_top)))
    mlde_all = pd.read_csv(mlde_csv).copy()

    ncol = 5
    nrow = 3
    fig, axs = plt.subplots(nrow, ncol, figsize=(32, 16), sharex=True, sharey=True)

    if min_ftlib:
        app_ftlib = " (12.5% ft)"
    else:
        app_ftlib = ""

    if simplezs:
        nrow = 2
        zs_legend_map = SIMPLE_ZS_OPT_LEGNED
        zs_color_map = SIMPLE_ZS_COLOR_MAP
        app_zs = ""
    else:
        nrow = 3
        zs_legend_map = ZS_OPTS_LEGEND
        zs_color_map = ZS_COLOR_MAP
        app_zs = " with ZS ensemble"

    sup_title = (
        f"DE vs MLDE{app_ftlib} top{str(n_top)} fraction = 1 over sample size{app_zs}"
    )

    de_sumdf = pd.read_csv(f"{de_folder}/all_landscape_de_summary.csv")
    liborder = (
        de_sumdf[de_sumdf["de_type"] == liborderby]
        .sort_values(by=["mean_all"])["lib"]
        .values
    )

    for i, (ax, lib) in enumerate(zip(axs.flatten(), liborder)):

        if i < len(liborder):

            ss_df = pd.read_csv(f"{de_folder}/{lib}-single_step_DE.csv").copy()
            recomb_df = pd.read_csv(f"{de_folder}/{lib}-recomb_SSM.csv").copy()
            toprecomb_df = pd.read_csv(f"{de_folder}/{lib}-top96_SSM.csv").copy()

            ss_frac = get_val_percent(s=ss_df["final_fitness"], numb=1)
            recomb_frac = get_val_percent(s=recomb_df["final_fitness"], numb=1)
            toprecomb_frac = get_val_percent(s=toprecomb_df["final_fitness"], numb=1)

            for zs, zs_color in zs_color_map.items():

                # Common condition
                condition = (
                    (mlde_all["lib"] == lib)
                    & (mlde_all["n_mut_cutoff"] == n_mut)
                    & (mlde_all["n_top"] == n_top)
                    & (mlde_all["encoding"] == "one-hot")
                    & (mlde_all["zs"] == zs)
                )

                """
                mlde_df = mlde_all[
                    (mlde_all["lib"] == lib)
                    & (mlde_all["n_mut_cutoff"] == n_mut)
                    & (mlde_all["n_top"] == n_top)
                    & (mlde_all["encoding"] == "one-hot")
                ].copy()

                                if zs == "none":
                    mlde_df = mlde_df[(mlde_df["zs"] == zs)]

                else:
                    mlde_df = mlde_df[
                        (mlde_df["zs"] == zs)
                        & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                    ]


                """

                # Apply the additional condition if needed
                if min_ftlib and zs != "none":
                    condition &= mlde_all["ft_lib"] == mlde_all["ft_lib"].min()      

                # Apply the conditions to filter the DataFrame
                mlde_df = mlde_all[condition].copy()

                # mlde_df = mlde_all[
                #     (mlde_all["lib"] == lib)
                #     & (mlde_all["n_mut_cutoff"] == n_mut)
                #     & (mlde_all["n_top"] == n_top)
                #     & (mlde_all["encoding"] == "one-hot")
                # ].copy()

                # if min_ftlib:
                #     if zs == "none":
                #         mlde_df = mlde_df[(mlde_df["zs"] == zs)]

                #     else:
                #         mlde_df = mlde_df[
                #             (mlde_df["zs"] == zs)
                #             & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                #         ]
                # else:
                #     mlde_df = mlde_df[(mlde_df["zs"] == zs)]

                max_percent_dict = {}

                for n_samples in N_SAMPLE_LIST:
                    # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                    mlde_df_n = mlde_df[mlde_df["n_sample"] == n_samples]["top_maxes"]

                    max_percent_dict[n_samples] = get_val_percent(s=mlde_df_n, numb=1)

                x = list(max_percent_dict.keys())
                y = list(max_percent_dict.values())

                ax.plot(
                    x,
                    y,
                    color=zs_color,
                    marker="s",
                    linestyle="solid",
                    linewidth=2,
                    label=f"Fraction of max fitness = 1: {zs_legend_map[zs]}",
                )
            ax.axhline(
                y=ss_frac,
                color=PRESENTATION_PALETTE_SATURATE["orange"],
                linestyle="dotted",
                linewidth=2,
                label="Fraction of final fitness = 1: single step",
            )
            ax.axhline(
                y=recomb_frac,
                color=PRESENTATION_PALETTE_SATURATE["yellow"],
                linestyle="dashed",
                linewidth=2,
                label="Fraction of final fitness = 1: recomb",
            )
            ax.axhline(
                y=toprecomb_frac,
                color=PRESENTATION_PALETTE_SATURATE["brown"],
                linestyle="dashdot",
                linewidth=2,
                label="Fraction of final fitness = 1: top96 recomb",
            )

            ax.set_ylabel("Fraction of max fitness = 1")
            ax.tick_params(axis="y")

            if i == ncol - 1:
                # create labels
                ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1))

            ax.set_title(lib)
            ax.set_ylim(0, 100)
        else:
            ax.set_visible(False)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def plot_de_mlde_ft_meancount_v_n(
    plot_folder: str = "results/de_vs_mlde/onehot/collage/n_mean_count",
    mlde_csv: str = "results/mlde/vis_4/all_df.csv",
    de_folder: str = "results/de/DE-active/scale2max",
    n_mut: str = "all",
    n_top: int = 96,
    meanorfrac: str = "mean",
    simplezs: bool = True,
    min_ftlib: bool = False,
):
    """ """

    plot_folder = checkNgen_folder(os.path.join(plot_folder, n_mut, str(n_top)))
    mlde_all = pd.read_csv(mlde_csv).copy()

    if min_ftlib:
        app_ftlib = " (12.5% ft) "
    else:
        app_ftlib = " "

    if meanorfrac == "mean":
        app_meanfrac = "mean of max fitness"
    else:
        app_meanfrac = "fraction of max fitness = 1"

    if simplezs:
        nrow = 2
        zs_map = SIMPLE_ZS_COLOR_MAP
        app_zs = ""
    else:
        nrow = 3
        zs_map = ZS_COLOR_MAP
        app_zs = " with ZS ensemble"

    sup_title = (
        f"DE vs MLDE{app_ftlib}{app_meanfrac} top{str(n_top)} over sample size{app_zs}"
    )

    ncol = 3

    fig, axs = plt.subplots(nrow, ncol, figsize=(20, 12), sharex=True, sharey=True)

    for i, (ax, zs) in enumerate(zip(axs.flatten(), zs_map.keys())):

        mlde_df = mlde_all[
            (mlde_all["n_mut_cutoff"] == n_mut) & (mlde_all["encoding"] == "one-hot")
        ].copy()

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
                    de_val = get_val_percent(s=de_df["final_fitness"], numb=1)

                for n in N_SAMPLE_LIST:

                    condition = (
                        (mlde_df["zs"] == zs)
                        & (mlde_df["lib"] == lib)
                        & (mlde_df["n_sample"] == n)
                        & (mlde_df["n_top"] == n_top)
                    )

                    if min_ftlib and zs != "none":
                        condition &= mlde_df["ft_lib"] == mlde_df["ft_lib"].min()

                    mlde_df_n = mlde_df[condition]["top_maxes"]

                    # if min_ftlib:
                    #     if zs == "none":
                    #         mlde_df_n = mlde_df[
                    #             (mlde_df["zs"] == zs)
                    #             & (mlde_df["lib"] == lib)
                    #             & (mlde_df["n_sample"] == n)
                    #             & (mlde_df["n_top"] == n_top)
                    #         ]["top_maxes"]

                    #     else:
                    #         mlde_df_n = mlde_df[
                    #             (mlde_df["zs"] == zs)
                    #             & (mlde_df["lib"] == lib)
                    #             & (mlde_df["n_sample"] == n)
                    #             & (mlde_df["n_top"] == n_top)
                    #             & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                    #         ]["top_maxes"]
                    # else:
                    #     # no ft lib size filter
                    #     mlde_df_n = mlde_df[
                    #         (mlde_df["zs"] == zs)
                    #         & (mlde_df["lib"] == lib)
                    #         & (mlde_df["n_sample"] == n)
                    #         & (mlde_df["n_top"] == n_top)
                    #     ]["top_maxes"]
                    if meanorfrac == "mean":
                        mlde_val = mlde_df_n.mean()
                    else:
                        mlde_val = get_val_percent(s=mlde_df_n, numb=1)
                    
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

        ax.plot(
            x,
            single_step_DE,
            color=PRESENTATION_PALETTE_SATURATE["orange"],
            marker="o",
            linestyle="dotted",
            linewidth=2,
            label="Single step",
        )
        ax.plot(
            x,
            recomb_SSM,
            color=PRESENTATION_PALETTE_SATURATE["yellow"],
            marker="o",
            linestyle="dashed",
            linewidth=2,
            label="Recomb",
        )
        ax.plot(
            x,
            top96_SSM,
            color=PRESENTATION_PALETTE_SATURATE["brown"],
            marker="o",
            linestyle="solid",
            linewidth=2,
            label="Top96 recomb",
        )

        if i == ncol - 1:
            # create labels
            ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1))

        if i == 5 and simplezs:
            ax.set_visible(False)
        else:
            # Adding labels and title
            ax.set_xlabel("Number of samples")
            ax.set_ylabel("Fraction of MLDE > DE")
            ax.tick_params(axis="y")

            ax.set_title(ZS_OPTS_LEGEND[zs])
            ax.set_ylim(0, 1)

    fig.tight_layout
    fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

    save_plt(
        fig,
        plot_title=sup_title,
        path2folder=plot_folder,
    )


def vis_sum_de_mlde(
    plot_dir: str = "results/de_vs_mlde/onehot/collage",
    mlde_csv: str = "results/mlde/vis_5/all_df.csv",
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

            plot_de_v_mlde(
                plot_folder=f"{plot_dir}/n_samples",
                n_mut=n_mut,
                n_top=n_top,
                liborderby=liborderby,
                **common_args,
            )

            plot_n_ftmlde(
                plot_folder=f"{plot_dir}/ftMLDE",
                n_mut=n_mut,
                n_top=n_top,
                liborderby=liborderby,
                **common_args,
            )

            plot_de_mlde_ft_mean_v_n(
                plot_folder=f"{plot_dir}/n_mean_frac",
                n_mut=n_mut,
                n_top=n_top,
                liborderby=liborderby,
                **common_args,
            )

            plot_de_mlde_ft_percent_v_n(
                plot_folder=f"{plot_dir}/n_mean_frac",
                n_mut=n_mut,
                n_top=n_top,
                liborderby=liborderby,
                **common_args,
            )

            for meanorfrac in ["mean", "frac"]:
                plot_de_mlde_ft_meancount_v_n(
                    plot_folder=f"{plot_dir}/n_mean_count",
                    meanorfrac=meanorfrac,
                    n_mut=n_mut,
                    n_top=n_top,
                    **common_args,
                )