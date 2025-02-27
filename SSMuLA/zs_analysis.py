"""A script for analyzing zs scores"""

from __future__ import annotations

import os
from ast import literal_eval

from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, ttest_ind
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
import seaborn as sns

# Basic plotting
import holoviews as hv

# Large data plotting
from holoviews.operation.datashader import rasterize

from SSMuLA.landscape_global import LibData, n_mut_cutoff_dict, LIB_INFO_DICT
from SSMuLA.vis import (
    JSON_THEME,
    LIGHT_COLORS,
    one_decimal_x,
    one_decimal_y,
    fixmargins,
    save_bokeh_hv,
    FZL_PALETTE,
    glasbey_category10,
    save_svg
)
from SSMuLA.zs_data import MSA_DF
from SSMuLA.util import ndcg_scale, checkNgen_folder

hv.extension("bokeh")
hv.renderer("bokeh").theme = JSON_THEME

ZS_OPTS = [
    "ed_score",
    "ev_score",
    "esm_score",
    "esmif_score",
    "coves_score",
    "Triad_score",
]

ZS_COMB_OPTS = [
    "Triad-ev_score",
    "Triad-esm_score",
    "Triad-esmif_score",
    "coves-ev_score",
    "coves-esm_score",
    "ev-esm_score",
    "ev-esm-esmif_score",
    "Triad-ev-esm-esmif_score",
    "two-best_score",
]

SIMPLE_ZS_OPT_LEGNED = {
    "none": "Random",
    "ed_score": "Hamming distance",
    "Triad_score": "Triad",
    "ev_score": "EVmutation",
    "esm_score": "ESM-2",
    "esmif_score": "ESM-IF",
    "coves_score": "CoVES",
}

ZS_OPTS_LEGEND = {
    "none": "Random",
    "ed_score": "Hamming distance",
    "Triad_score": "Triad",
    "ev_score": "EVmutation",
    "esm_score": "ESM-2",
    "esmif_score": "ESM-IF",
    "coves_score": "CoVES",
    "Triad-esmif_score": "Triad + ESM-IF",  # prev struc-comb
    "Triad-ev_score": "Triad + EVmutation",
    "Triad-esm_score": "Triad + ESM-2",
    "coves-ev_score": "CoVES + EVmutation",
    "coves-esm_score": "CoVES + ESM-2",
    "ev-esm_score": "EVmutation + ESM-2",  # prev msanoif-comb
    "two-best_score": "EVmutation + ESM-IF",
    "ev-esm-esmif_score": "EVmutation + ESM-2 + ESM-IF",  # prev msa-comb
    "Triad-ev-esm-esmif_score": "Triad + EVmutation + ESM-2 + ESM-IF",  # prev structnev-esm-esmif
}

ZS_METRICS = ["rho", "ndcg", "rocauc"]

ZS_METRIC_MAP_TITLE = {
    "rho": "Fitness ranking\n(Spearman's ρ)",
    "rocauc": "Active / inactive classification\n(ROC-AUC)",
}

ZS_METRIC_MAP_LABEL = {
    "rho": "Spearman's ρ",
    "rocauc": "ROC-AUC",
}

ZS_METRIC_BASELINE = {
    "rho": 0,
    "rocauc": 0.5,
}

ZS_N_MUTS = ["all", "double", "single"]


SIX_ZS_COLORS = {
    "ed_score": FZL_PALETTE["blue"],
    "ev_score": FZL_PALETTE["green"],
    "esm_score": FZL_PALETTE["purple"],
    "esmif_score": FZL_PALETTE["yellow"],
    "coves_score": FZL_PALETTE["brown"],
    "Triad_score": FZL_PALETTE["orange"],
}


class ZS_Analysis(LibData):

    """
    A class to process and visualize the ZS data
    """

    def __init__(
        self,
        input_csv: str,
        scale_fit: str,
        n_mut_cutoff: int = 0,
        ev_esm_folder: str = "ev_esm",
        triad_folder: str = "triad",
        esmif_folder: str = "esmif",
        coves_folder: str = "coves/100_processed",
        filter_min_by: str = "none",
        zs_comb_dir: str = "results/zs_comb",
        zs_vis_dir: str = "results/zs_vis",
    ) -> None:

        """
        Args:
        - input_csv, str: path to the input csv file
        - scale_fit, str: ways to scale the fitness
            'parent' means the parent fitness = 1
            'max' means max fitness = 1
        - n_mut_cutoff, int: the number of mutations cutoff
        - ev_esm_folder, str: the folder for the ev and esm scores
        - triad_folder, str: the folder for the triad scores
        - esmif_folder, str: the folder for the esm inverse folding scores
        - coves_folder, str: the folder for the coves scores
        - filter_min_by, str: the filter for the minimum fitness
        - zs_comb_dir, str: the folder for the ZS combed with fitness outputs
        - zs_vis_dir, str: the folder for the ZS vis outputs
        """

        super().__init__(input_csv, scale_fit)

        self._n_mut_cutoff = n_mut_cutoff
        self._ev_esm_folder = os.path.normpath(ev_esm_folder)
        self._triad_folder = os.path.normpath(triad_folder)
        self._esmif_folder = os.path.normpath(esmif_folder)
        self._coves_folder = os.path.normpath(coves_folder)
        self._filter_min_by = filter_min_by
        self._zs_comb_dir = checkNgen_folder(zs_comb_dir)
        self._zs_vis_dir = checkNgen_folder(zs_vis_dir)

        if self._n_mut_cutoff > 0:
            self._n_mut_subdir = n_mut_cutoff_dict[self._n_mut_cutoff]
        else:
            self._n_mut_subdir = "all"

        print(f"Get fitness data without stop codon from {self._input_csv}...")
        print(f"Get ev esm data from {self.ev_esm_path}...")
        print(f"Get triad data from {self.triad_path}...")
        print(f"Get inverse folding data from {self.esmif_path}...")
        print(f"Get coves data from {self.coves_path}...")

        print(f"Save combed zs data to {self.zf_comb_path}...")
        self._zs_df = self._get_zs_df()
        self.zs_df.to_csv(self.zf_comb_path, index=False)

        self._roc, self._zs_coord_dict = self._plot_roc()
        self._zs_fit_plot_dict = self._plot_zs_vs_fitness()

    def _get_zs_df(self) -> pd.DataFrame:

        """
        Get the ZS dataframe
        """
        df = pd.merge(
            pd.merge(
                pd.merge(
                    pd.merge(self.df_no_stop, self.ev_esm_df, on="muts"),
                    self.esmif_df,
                    on="muts",
                ),
                self.triad_df,
                on="AAs",
            ),
            self.coves_df,
            on="muts",
        )

        # some easy zs comb
        df["Triad-ev_score"] = -1 * (df["Triad_rank"] + df["ev_rank"])
        df["Triad-esm_score"] = -1 * (df["Triad_rank"] + df["esm_rank"])
        df["Triad-esmif_score"] = -1 * (df["Triad_rank"] + df["esmif_rank"])

        df["coves-ev_score"] = -1 * (df["coves_rank"] + df["ev_rank"])
        df["coves-esm_score"] = -1 * (df["coves_rank"] + df["esm_rank"])

        df["ev-esm-esmif_score"] = -1 * (
            df["ev_rank"] + df["esm_rank"] + df["esmif_rank"]
        )
        df["ev-esm_score"] = -1 * (df["ev_rank"] + df["esm_rank"])
        df["Triad-ev-esm-esmif_score"] = -1 * (
            df["Triad_rank"] + df["ev_rank"] + df["esm_rank"] + df["esmif_rank"]
        )

        df["two-best_score"] = -1 * (df["ev_rank"] + df["esmif_rank"])

        for comb_opt in [
            "Triad-ev",
            "Triad-esm",
            "Triad-esmif",
            "coves-ev",
            "coves-esm",
            "ev-esm-esmif",
            "ev-esm",
            "Triad-ev-esm-esmif",
            "two-best",
        ]:
            df[f"{comb_opt}_rank"] = df[f"{comb_opt}_score"].rank(ascending=False)

        if self._n_mut_cutoff > 0:
            return df[df["n_mut"] <= self._n_mut_cutoff].copy()
        else:
            return df.copy()

    def _plot_roc(self) -> hv.Overlay:

        """
        Plot the ROC curve
        """

        print(f"Plotting {self.lib_name} zs true active roc...")

        df = self.zs_df.copy()

        print(f"df has columns: {df.columns}")

        zs_coord_dict = {zs: {} for zs in ZS_OPTS + ZS_COMB_OPTS}

        roc_plots = []

        for zs in ZS_OPTS + ZS_COMB_OPTS:

            if zs not in df.columns:
                print(f"{zs} not in {df.columns} for {self.lib_name}")
                zs_coord_dict[zs]["rho"] = np.nan
                zs_coord_dict[zs]["ndcg"] = np.nan
                zs_coord_dict[zs]["rocauc"] = np.nan
                continue

            print(
                f"number of values in {self.lib_name} {zs}: {np.sum(~np.isnan(df[zs]))}"
            )
            print(f"number of nan in {self.lib_name} {zs}: {np.sum(np.isnan(df[zs]))}")

            if zs in ZS_OPTS:
                line_style = "solid"
            else:
                line_style = "dashed"

            slice_df = df.dropna(subset=[zs]).copy()
            y_true_active = slice_df["active"].values
            y_true_fitness = slice_df["fitness"].values
            y_score = slice_df[zs].values

            if len(y_score) == 0:
                print(f"{zs} has no data -> skip")
                zs_coord_dict[zs]["rho"] = np.nan
                zs_coord_dict[zs]["ndcg"] = np.nan
                zs_coord_dict[zs]["rocauc"] = np.nan
            else:
                # calc rho and ndcg
                zs_coord_dict[zs]["rho"] = spearmanr(y_true_fitness, y_score)[0]
                zs_coord_dict[zs]["ndcg"] = ndcg_scale(y_true_fitness, y_score)

                # roc curves
                roc_name = (
                    f"{self.lib_name} active variant zero-shot predictor ROC curves"
                )

                fpr, tpr, _ = roc_curve(y_true_active, y_score, pos_label=True)
                temp = pd.DataFrame(
                    {"False Positive Rate": fpr, "True Positive Rate": tpr}
                )

                roc_plots.append(
                    hv.Curve(
                        temp,
                        kdims=["False Positive Rate"],
                        vdims=["True Positive Rate"],
                        label=ZS_OPTS_LEGEND[zs],
                    ).opts(
                        height=400,
                        width=700,
                        xlim=(0, 1),
                        ylim=(0, 1),
                        hooks=[one_decimal_x, one_decimal_y, fixmargins],
                        color=hv.Cycle("Category10"),
                        line_dash=line_style,
                    )
                )

                roc_auc = auc(fpr, tpr)
                zs_coord_dict[zs]["rocauc"] = roc_auc

        roc_plots.append(
            hv.Curve(
                ((0, 1), (0, 1)),
                "False Positive Rate",
                "True Positive Rate",
            ).opts(
                color="grey",
                line_dash="dashed",
            )
        )

        roc = hv.Overlay(roc_plots).opts(
            height=400,
            width=700,
            legend_position="right",
            legend_offset=(5, 0),
            xlim=(0, 1),
            ylim=(0, 1),
            hooks=[one_decimal_x, one_decimal_y, fixmargins],
            title=roc_name,
        )

        save_bokeh_hv(roc, plot_name=roc_name, plot_path=self.roc_folder)

        for k, v in zs_coord_dict.items():
            print(f"{k}: {v}")

        return roc, zs_coord_dict

    def _plot_zs_vs_fitness(self):

        """
        Plot the zero-shot predictors against the fitness with fraction of active
        """

        print(f"Plotting {self.lib_name} zs vs fitness...")

        zs_fit_plot_dict = {}

        for zs in ZS_OPTS + ZS_COMB_OPTS:

            zs_title = f"{self.lib_name} {ZS_OPTS_LEGEND[zs]} vs fitness"

            xs = np.linspace(min(self.zs_df[zs]), max(self.zs_df[zs]), 100)

            samples = np.array([self._get_frac_active(zs, thresh) for thresh in xs]).T

            p = rasterize(hv.Scatter(self.zs_df, kdims=zs, vdims="fitness")).opts(
                cmap=["black", LIGHT_COLORS["yellow"]],
                hooks=[fixmargins, one_decimal_y],
                ylabel="fitness",
            )

            p = (
                p
                * hv.Curve((xs, samples[0]), label="fraction active").opts(
                    color=hv.Cycle("Category10")
                )
                * hv.Curve((xs, samples[2]), label="mean fitness").opts(
                    line_dash="dashed", color=hv.Cycle("Category10")
                )
            )

            p = p.opts(
                height=200,
                width=400,
                show_legend=True,
                legend_position="right",
                legend_offset=(5, 0),
                ylabel="fraction or fitness",
                xlabel=ZS_OPTS_LEGEND[zs],
                title=zs_title,
            )

            zs_fit_plot_dict[zs] = p

            save_bokeh_hv(
                p,
                plot_name=zs_title,
                plot_path=self.zs_fit_plot_folder,
            )

        return zs_fit_plot_dict

    def _get_frac_active(self, zs: str, score: float):

        """
        Get the fraction of active sequences

        Args:
        - zs, str: the zero-shot predictor
        - score, float: the score threshold
        """

        df = self.zs_df[self.zs_df[zs] > score]

        try:
            frac_active = sum(df["active"]) / len(df)
        except ZeroDivisionError:
            frac_active = 0

        frac_library = len(df) / len(self.zs_df)
        mean_fitness = df["fitness"].mean()
        max_fitness = df["fitness"].max()

        return frac_active, frac_library, mean_fitness, max_fitness

    @property
    def df_no_stop(self) -> pd.DataFrame:
        """
        Returns the dataframe without stop codons
        """

        df = self.input_df[~self.input_df["AAs"].str.contains("\*")].copy()

        # check wt n_mut = 0 and update to 0 if not
        if df[df["muts"] == "WT"]["n_mut"].values[0] != 0:
            print("WT n_mut is not 0 -> update to 0")
            df.loc[df["muts"] == "WT", "n_mut"] = 0

        # add ed_score and rank based on n_mut
        df["ed_score"] = -1 * df["n_mut"]
        # rank the smaller the better
        df["ed_rank"] = df["ed_score"].rank(ascending=False)

        if self._filter_min_by in ["none", "", None]:
            return df.copy()
        elif self._filter_min_by == "active":
            return df[df["active"]].copy()
        elif self._filter_min_by == "0":
            return df[df["fitness"] >= 0].copy()
        elif self._filter_min_by == "min0":
            df["fitness"] = df["fitness"].apply(lambda x: max(0, x))
            return df.copy()
        else:
            print(f"{self._filter_min_by} not valid -> no filter beyond no stop codon")
            return df.copy()

    @property
    def ev_esm_path(self) -> str:
        """
        Returns the path to the ev and esm scores
        """
        return f"{self._ev_esm_folder}/{self.lib_name}/{self.lib_name}.csv"

    @property
    def ev_esm_df(self) -> pd.DataFrame:
        """
        Returns the dataframe with the ESM and EV scores
        """

        df = pd.read_csv(self.ev_esm_path)

        # find parent row
        parent_row = df[df["muts"] == "WT"]

        for zs in ["esm", "ev"]:

            if f"{zs}_score" not in df.columns:
                df[f"{zs}_score"] = np.nan
            else:
                parent_zs = df.loc[parent_row.index, f"{zs}_score"].values[0]

                if np.isnan(parent_zs):
                    df.loc[parent_row.index, f"{zs}_score"] = 0

            # Add rank column for each score
            df[f"{zs}_rank"] = df[f"{zs}_score"].rank(ascending=False)

        # find if its fit or fitness
        if "fit" in df.columns:
            fit_col = "fit"
        else:
            fit_col = "fitness"

        # prevent duplicates
        drop_cols = [fit_col, "combo"]

        if "active" in df.columns:
            drop_cols.append("active")

        if "AAs" in df.columns:
            drop_cols.append("AAs")

        if "n_mut" in df.columns:
            drop_cols.append("n_mut")

        return df.drop(columns=drop_cols).copy()

    @property
    def triad_path(self) -> str:

        """
        Returns the path to triad scores
        """

        return f"{self._triad_folder}/{self.lib_name}/{self.lib_name}.csv"

    @property
    def triad_df(self) -> pd.DataFrame:

        """
        Returns the dataframe with the triad scores
        """

        df = pd.read_csv(self.triad_path)
        # flip triad score
        df["Triad_score"] = df["Triad_score"] * -1

        return df.copy()

    @property
    def esmif_path(self) -> str:

        """
        Returns the path to the esm inverse folding scores

        ie esmif/DHFR_esmif_scores.csv
        """

        return f"{self._esmif_folder}/{self.lib_name}_esmif_scores.csv"

    @property
    def esmif_df(self) -> pd.DataFrame:
        """
        Returns the dataframe with the esm inverse folding scores

        Start with columns 'seqid,log_likelihood'
        Rename to 'muts,esmif_score' and add `esmif_rank`

        Note WT is WT

        Returns:
        - df, pd.DataFrame: the dataframe with the esm inverse folding scores
        """

        df = pd.read_csv(self.esmif_path)

        # Rename columns
        df.columns = ["muts", "esmif_score"]

        # Add rank column for each score
        df["esmif_rank"] = df["esmif_score"].rank(ascending=False)

        return df.copy()

    @property
    def coves_path(self) -> str:
        """
        Returns the path to the coves scores
        """

        return f"{self._coves_folder}/{self.lib_name}.csv"

    @property
    def coves_df(self) -> pd.DataFrame:

        """
        Returns the dataframe with the coves scores
        """

        df = pd.read_csv(self.coves_path)

        # Add rank column for each score
        df["coves_rank"] = df["coves_score"].rank(ascending=False)

        return df.copy()

    @property
    def zs_df(self) -> pd.DataFrame:
        """
        Returns the dataframe with the ZS scores

        Merge the fitness data with the ev, esm, triad, and esmif scores
        """

        return self._zs_df

    @property
    def zf_comb_folder(self) -> str:
        """
        Returns the path to the ZF combined with fitness
        """
        return checkNgen_folder(
            os.path.join(
                self._zs_comb_dir,
                self._filter_min_by,
                self.scale_type,
                self._n_mut_subdir,
            )
        )

    @property
    def zf_comb_path(self) -> str:
        """
        Returns the path to the ZF combined with fitness
        """
        return os.path.join(self.zf_comb_folder, f"{self.lib_name}.csv")

    @property
    def roc_folder(self) -> str:
        """
        Returns the folder path to the ROC curve
        """
        return checkNgen_folder(
            os.path.join(
                self._zs_vis_dir,
                "roc",
                self._filter_min_by,
                self.scale_type,
                self._n_mut_subdir,
            )
        )

    @property
    def roc(self) -> hv.Overlay:
        """
        Returns the ROC curve
        """
        return self._roc

    @property
    def zs_coord_dict(self) -> dict:
        """
        Returns the dictionary of zs coordinates
        """
        return self._zs_coord_dict

    @property
    def zs_fit_plot_folder(self) -> dict:
        """
        Returns the folder path of zs vs fitness plots
        """
        return checkNgen_folder(
            os.path.join(
                self._zs_vis_dir,
                "vs_fitness",
                self._filter_min_by,
                self.scale_type,
                self._n_mut_subdir,
            )
        )

    @property
    def zs_fit_plot_dict(self) -> dict:
        """
        Returns the dictionary of zs vs fitness plots
        """
        return self._zs_fit_plot_dict


def run_zs_analysis(
    scale_types: list = ["max"],
    data_folder: str = "data",
    ev_esm_folder: str = "ev_esm",
    triad_folder: str = "triad",
    esmif_folder: str = "esmif",
    filter_min_by: str = "none",
    n_mut_cutoff_list: list[int] = [0, 1, 2],
    zs_comb_dir: str = "results/zs_comb",
    zs_vis_dir: str = "results/zs_vis",
    zs_sum_dir: str = "results/zs_sum",
) -> None:

    """
    Run the ZS analysis

    Args:
    - scale_types, list: the types of scaling
    - data_folder, str: the folder for the data
    - ev_esm_folder, str: the folder for the ev and esm scores
    - triad_folder, str: the folder for the triad scores
    - esmif_folder, str: the folder for the esm inverse folding scores
    - filter_min_by, str: the filter for the minimum fitness
    - n_mut_cutoff_list, list[int]: the list of number of mutations cutoff
    - zs_comb_dir, str: the folder for the ZS combed with fitness outputs
    - zs_vis_dir, str: the folder for the ZS vis outputs
    - zs_sum_dir, str: the folder for the ZS summary outputs
    """

    for scale_type in scale_types:

        zs_stat_df = pd.DataFrame()

        for n_mut_cutoff in n_mut_cutoff_list:

            for lib_path in tqdm(
                sorted(
                    glob(f"{os.path.normpath(data_folder)}/*/scale2{scale_type}/*.csv")
                )
            ):

                zs = ZS_Analysis(
                    input_csv=lib_path,
                    scale_fit=scale_type,
                    n_mut_cutoff=n_mut_cutoff,
                    ev_esm_folder=ev_esm_folder,
                    triad_folder=triad_folder,
                    esmif_folder=esmif_folder,
                    filter_min_by=filter_min_by,
                    zs_comb_dir=zs_comb_dir,
                    zs_vis_dir=zs_vis_dir,
                )

                zs_stat_df = zs_stat_df._append(
                    {
                        "lib": zs.lib_name,
                        "n_mut": n_mut_cutoff_dict[n_mut_cutoff],
                        "scale_type": scale_type,
                        **zs.zs_coord_dict,
                    },
                    ignore_index=True,
                )

        zs_stat_df.to_csv(
            f"{checkNgen_folder(os.path.join(zs_sum_dir, filter_min_by))}/zs_stat_scale2{scale_type}.csv",
            index=False,
        )


def plot_zs_corr(
    n_mut_cutoff="all",
    active_cutoff=1,
    lib_stat_csv="results/landscape/lib_stats.csv",
    zs_df_dir="results/zs_comb/none/scale2max/",
    save_dir="figs",
    fig_id="3d",
    ifsave=True,
    addtitle=False
):
    """
    Plot the ZS correlations

    Args:
    - n_mut_cutoff, str: the number of mutations cutoff
    - actvie_cutoff, int: the active cutoff, default 1%
    - lib_list, list: the list of libraries
    - zs_df_dir, str: the folder for the ZS dataframes
    - save_dir, str: the folder to save the figure
    - fig_id, str: the figure id
    - ifsave, bool: if save the figure
    - addtitle, bool: if add title to the figure
    """

    lib_stat = pd.read_csv(lib_stat_csv)

    lib_list = lib_stat[(lib_stat["percent_active"] >= active_cutoff)]["lib"].tolist()

    corr_dict = {}
    for lib in lib_list:

        lib_csv = os.path.join(zs_df_dir, n_mut_cutoff, f"{lib}.csv")

        df = pd.read_csv(lib_csv)
        corr_dict[lib] = df[ZS_OPTS].corr(method="spearman", min_periods=1)

    zs_names = [ZS_OPTS_LEGEND[z] for z in ZS_OPTS]
    avg_corr_df = pd.DataFrame(
        np.nanmean(np.stack(list(corr_dict.values())), axis=0),
        index=zs_names,
        columns=zs_names,
    )
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(avg_corr_df, dtype=bool))
    # Modifying the mask to exclude the diagonal
    np.fill_diagonal(mask, False)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(
        avg_corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"shrink": 0.8},
        linewidths=0.8,
        square=True,
        vmin=0,
        vmax=1,
    )  # Setting vmin and vmax
    cbar = hm.collections[0].colorbar
    # Add a title to the color bar
    cbar.set_label(r"Spearman's $\rho$", rotation=270, labelpad=20)

    # Setting vmin and vmax
    ax.set_xlabel("ZS predictor")
    ax.set_ylabel("ZS predictor")
    # Add titles and labels as necessary
    if addtitle:
        plt.title(f"ZS correlations for {n_mut_cutoff}", fontweight="bold")
    plt.show()
    if ifsave:
        save_dir = checkNgen_folder(save_dir)
        fig.savefig(
            f"{save_dir}/{fig_id}.svg", dpi=300, bbox_inches="tight", format="svg"
        )

def parse_zs(
    zs_sum_csv: str = "results/zs_sum/none/zs_stat_scale2max.csv",
    zs_parse_csv: str = "results/zs_sum/none/zs_stat_parsed.csv",
) -> pd.DataFrame:

    """
    A function to parse the zs_sum csv file into a pandas dataframe
    """

    zs_df = pd.read_csv(zs_sum_csv)

    # init zs_df
    zs_df_list = [zs_df[["lib", "n_mut"]]]
    # Create new columns for each score type
    for c in ZS_OPTS + ZS_COMB_OPTS:

        zs_name = c.replace("_score", "")
        zs_df_list.append(
            zs_df[f"{zs_name}_score"]
            .str.replace(": nan", ": None")
            .apply(literal_eval)
            .apply(pd.Series)
            .rename(columns={m: f"{zs_name}_{m}" for m in ZS_METRICS})
        )

    zs_df_expend = pd.concat(zs_df_list, axis=1)

    # note this n_mut == all is necearry to get the expanded df with all zs options
    zs_mut_df_list = [zs_df_expend[zs_df_expend["n_mut"] == "all"]["lib"]]
    for n_mut in ZS_N_MUTS:
        slice_df = (
            zs_df_expend[zs_df_expend["n_mut"] == n_mut]
            .drop(columns=["lib", "n_mut"])
            .reset_index(drop=True)
        )
        zs_mut_df_list.append(
            slice_df.rename(columns={c: f"{n_mut}_{c}" for c in slice_df.columns})
        )

    zs_parsed_df = pd.concat(zs_mut_df_list, axis=1)
    zs_parsed_df["type"] = zs_parsed_df["lib"].map(
        {n: v["type"] for n, v in LIB_INFO_DICT.items()}
    )

    zs_append_msa = pd.merge(zs_parsed_df, MSA_DF, on="lib")
    
    zs_append_msa.to_csv(zs_parse_csv, index=False)

    return zs_append_msa


def plot_app_type_zs(
    metric: str,
    n_mut: str,
    slice_zs: pd.DataFrame,
    fig_name = "4b",
    y_min = None,
    y_max = None,
    y_annotation = None,
    save_dir: str = "figs",
    if_save: bool = True,
):

    """
    Plot the ZS scores for the different types across different landscapes

    Args:
    - metric, str: the metric to plot
    - n_mut, str: the number of mutations, all, single, or double
    - slice_zs, pd.DataFrame: the dataframe with the ZS scores
    - save_dir, str: the folder to save the figure
    - fig_name, str: the figure name
    """
    fig, axes = plt.subplots(1, 6, figsize=(6, 3.6), sharey=True)

    for z, zs in enumerate(ZS_OPTS):
        ax = axes.flatten()[z]
        x = f"{n_mut}_" + zs.split("_")[0] + f"_{metric}"
        bar_type_df = slice_zs[["lib", "type", x]].sort_values(["lib", "type"]).copy()

        if len(bar_type_df["type"].unique()) == 1:
            bar_order = ["Enzyme activity"]
            do_ttest = False
            bar_width = 0.3
        else:
            bar_order = ["Binding", "Enzyme activity"]
            do_ttest = True
            bar_width = 0.6

        sns.boxplot(
            x="type",
            y=x,
            data=bar_type_df,
            order=bar_order,
            width=bar_width,
            ax=ax,
            boxprops={
                "facecolor": "None",
                "edgecolor": FZL_PALETTE["gray"],
            },
        )
        sns.stripplot(
            x="type",
            y=x,
            data=bar_type_df,
            order=bar_order,
            hue="lib",
            hue_order=bar_type_df["lib"].unique(),
            jitter=True,
            size=7.5,
            palette=glasbey_category10[:12],
            marker="o",
            alpha=0.8,
            ax=ax,
        )

        labels = [
            label.get_text().replace("Enzyme activity", "Enzyme\nactivity")
            for label in ax.get_xticklabels()
        ]

        if metric == "rho":
            if y_annotation is None:
                y_annotation = 0.6

        elif metric == "rocauc" :
            if y_annotation is None:
                y_annotation = 1

        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)

        ax.set_xlabel("")
        ax.set_xticklabels(labels, rotation=90, ha="center")
        ax.set_title(
            ZS_OPTS_LEGEND[x.split("_")[1] + "_score"].replace(" ", "\n"), fontsize=10
        )
        ax.axhline(ZS_METRIC_BASELINE[metric], color="gray", lw=1.2, ls="--")

        # Hide the top and right spine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if z != len(ZS_OPTS) - 1:
            ax.legend().remove()
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1.025))
        if z == 0:
            ax.set_ylabel(ZS_METRIC_MAP_LABEL[metric])
        else:
            ax.set_ylabel("")

        if do_ttest:
            t_val, p_value = ttest_ind(
                list(bar_type_df[bar_type_df["type"] == "Binding"][x]),
                list(bar_type_df[bar_type_df["type"] == "Enzyme activity"][x]),
                equal_var=False,
                nan_policy="omit",
            )
            print(f"{zs} t={t_val:.3f} and p={p_value:.3f}")

            # Draw a line between points
            p = 0.1
            q = 1
            annot_y = y_annotation
            if p_value < 0.05:
                ax.text((p + q) * 0.5, annot_y, "*", ha="center", va="bottom", color="gray")

    plt.tight_layout(pad=0, h_pad=-0.0, w_pad=0.5)

    if if_save:
        save_svg(fig, f"{fig_name}_{metric}", save_dir)