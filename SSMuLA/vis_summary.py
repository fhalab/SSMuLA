"""
A script to visualize the results cross landscapes.
"""


import os
from copy import deepcopy
import pandas as pd
from ast import literal_eval


import matplotlib.pyplot as plt
import holoviews as hv


from SSMuLA.de_simulations import DE_TYPES
from SSMuLA.zs_analysis import ZS_OPTS
from SSMuLA.landscape_global import LIB_INFO_DICT, LIB_NAMES, TrpB_names
from SSMuLA.vis import (
    LIB_COLORS,
    PRESENTATION_PALETTE_SATURATE,
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
ZS_OPTS_NOESM = deepcopy([zs for zs in ZS_OPTS if "esm" not in zs])
ZS_N_MUTS = ["all", "double", "single"]


TEN_COLORS = list(LIB_COLORS.values())
N_SAMPLE_LIST = [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]


def de_sum_hook(plot, element):
    plot.handles["plot"].x_range.factors = [
        (lib, de) for lib in LIB_NAMES for de in DE_TYPES
    ]


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
        input_csv: str = "results/simulations/DE-active/scale2max/all_landscape_de_summary.csv",
        output_folder: str = ""
    ) -> None:

        """
        Args:
        - input_csv: str, The input csv file path for the summary df
            ie 'results/simulations/DE-active/scale2max/all_landscape_de_summary.csv'
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
                for include_esm in [True, False]:
                    self._plot_zs_sum(
                        metric=metric,
                        n_mut=n_mut,
                        include_esm=include_esm,
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
        """

        zs_sum_df = self.input_df.copy()

        # Melt the DataFrame with the following columns:
        # lib
        # zs_type
        # corr

        zs_sum_df_melt = zs_sum_df.melt(
            id_vars=["lib", "n_mut"],
            value_vars=ZS_OPTS,
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
                zs_sum_df_melt["corr"].apply(literal_eval).apply(pd.Series),
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

    def _plot_zs_sum(self, metric: str, n_mut: str = "all", include_esm: bool = True):

        """
        Plot the bar plot summary of ZS simulations.

        Args:
        - metric: str, the metric to plot, ie: 'rho', 'ndcg', 'rocauc'
        - n_mut: str, the number of mutations to plot, ie: 'all', 'double', 'single'
        - include_esm: bool, whether to include the ESM score in the plot
        """

        if include_esm:
            plot_df = self._zs_df[(self._zs_df["metric"] == metric) & (self._zs_df["n_mut"] == n_mut)]
            hook = self._zs_hook
            esm_dets = ""
        else:
            plot_df = self._zs_df[
                (self._zs_df["metric"] == metric)
                & (self._zs_df["n_mut"] == n_mut)
                & (self._zs_df["zs_type"].isin(ZS_OPTS_NOESM))
            ]
            hook = self._zs_noesm_hook
            esm_dets = "-noesm"

        # Create the Holoviews Bars element
        save_bokeh_hv(
            hv.Bars(plot_df, kdims=["lib", "zs_type"], vdims="value").opts(
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
            plot_name=f"{self.zs_dets}-{metric}{esm_dets}-{n_mut}",
            plot_path=self.zs_dir,
        )

    def _zs_hook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in ZS_OPTS
        ]

    def _zs_noesm_hook(self, plot, element):
        """
        Plot hook to set the x_range factors for ZS plots without ESM.
        """
        plot.handles["plot"].x_range.factors = [
            (lib, zs) for lib in LIB_NAMES for zs in ZS_OPTS_NOESM
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
    de_folder: str = "results/simulations/DE-active/scale2max",
    ) -> None:

    """
    A function for plotting DE vs MLDE in a collage plot
    loop over all ZS scores for all libraries in each

    Args:
    - plot_folder: str, path to the folder where the plots will be saved
    - mlde_csv: str, path to the MLDE csv file
    """

    plot_folder = checkNgen_folder(plot_folder)
    mlde_all = pd.read_csv(mlde_csv).copy()

    for zs, mlde_title in zip(
        ["Triad_score", "esm_score", "ev_score", "none"],
        ["Triad-ftMLDE", "ESM-ftMLDE", "EVmutation-ftMLDE", "MLDE"]
    ):

        sup_title = f"{mlde_title} vs DE"

        fig, axs = plt.subplots(3, 4, figsize=(28, 16))

        for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):

            ss_df_all = pd.read_csv(
                f"{de_folder}/{lib}-single_step_DE.csv"
            )
            recomb_df_all = pd.read_csv(
                f"{de_folder}/{lib}-recomb_SSM.csv"
            )

            ss_df = ss_df_all.copy()
            recomb_df = recomb_df_all.copy()

            mlde_df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == "all")
                & (mlde_all["zs"] == zs)
                & (mlde_all["encoding"] == "one-hot")
            ].copy()

            ax.plot(
                ss_df["final_fitness"],
                ecdf_transform(ss_df["final_fitness"]),
                ".",
                label="DE - single-step",
                color=TEN_COLORS[0],
            )
            ax.plot(
                recomb_df["final_fitness"],
                ecdf_transform(recomb_df["final_fitness"]),
                ".",
                label="DE - recombination",
                color=TEN_COLORS[1],
            )

            for n, n_samples in enumerate(N_SAMPLE_LIST):
                # [24, 48, 96, 192, 288, 384, 480, 576, 960, 1920]
                mlde_df_n = mlde_df[mlde_df["n_sample"] == n_samples]["top_maxes"]
                ax.plot(
                    mlde_df_n,
                    ecdf_transform(mlde_df_n),
                    ".",
                    label=f"{mlde_title} - {str(n_samples)}",
                    color=TEN_COLORS[n + 2],
                )

            if i == 3:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

            ax.set_title(lib)
            ax.set_xlabel("Max fitness achieved")
            ax.set_ylabel("ECDF")

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
) -> None:

    """
    Plot the fitness landscape of the top maxes for ftMLDE vs DE

    Args:
    - plot_folder: str, path to the folder where the plots will be saved
    - mlde_csv: str, path to the MLDE csv file
    - de_folder: str, path to the DE folder
    """

    plot_folder = checkNgen_folder(plot_folder)
    mlde_all = pd.read_csv(mlde_csv).copy()

    for n in N_SAMPLE_LIST:

        sup_title = f"{str(n)} MLDE sample 12.5% ft library vs DE"

        fig, axs = plt.subplots(3, 4, figsize=(28, 16))

        for i, (ax, lib) in enumerate(zip(axs.flatten(), LIB_INFO_DICT.keys())):

            ss_df_all = pd.read_csv(
                f"{de_folder}/{lib}-single_step_DE.csv"
            )
            recomb_df_all = pd.read_csv(
                f"{de_folder}/{lib}-recomb_SSM.csv"
            )

            ss_df = ss_df_all.copy()
            recomb_df = recomb_df_all.copy()

            mlde_df = mlde_all[
                (mlde_all["lib"] == lib)
                & (mlde_all["n_mut_cutoff"] == "all")
                & (mlde_all["n_sample"] == n)
                & (mlde_all["encoding"] == "one-hot")
            ].copy()

            ax.plot(
                ss_df["final_fitness"],
                ecdf_transform(ss_df["final_fitness"]),
                ".",
                label="DE - single-step",
                color=TEN_COLORS[0],
            )
            ax.plot(
                recomb_df["final_fitness"],
                ecdf_transform(recomb_df["final_fitness"]),
                ".",
                label="DE - recombination",
                color=TEN_COLORS[1],
            )

            for zs_label, zs_color, zs in zip(
                ["MLDE", "Triad ftMLDE", "ESM ftMLDE", "EVmutation ftMLDE"],
                ["gray", "blue", "purple", "green"],
                ["none", "Triad_score", "esm_score", "ev_score"],
            ):  
                if zs == "none":
                    mlde_df_n = mlde_df[
                        (mlde_df["zs"] == zs)
                    ]["top_maxes"]

                else:
                    mlde_df_n = mlde_df[
                        (mlde_df["zs"] == zs) & (mlde_df["ft_lib"] == mlde_df["ft_lib"].min())
                    ]["top_maxes"]

                ax.plot(
                    mlde_df_n,
                    ecdf_transform(mlde_df_n),
                    ".",
                    label=zs_label,
                    color=PRESENTATION_PALETTE_SATURATE[zs_color],
                )

            if i == 3:
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

            ax.set_title(lib)
            ax.set_xlabel("Max fitness achieved")
            ax.set_ylabel("ECDF")

        fig.suptitle(sup_title, fontsize=16, fontweight="bold", y=0.9125)

        save_plt(
            fig,
            plot_title=sup_title,
            path2folder=plot_folder,
        )