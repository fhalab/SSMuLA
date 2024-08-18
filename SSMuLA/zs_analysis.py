"""A script for analyzing zs scores"""

from __future__ import annotations

import os
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, auc

# Basic plotting
import holoviews as hv

# Large data plotting
from holoviews.operation.datashader import rasterize

from SSMuLA.landscape_global import LibData, n_mut_cutoff_dict
from SSMuLA.vis import (
    JSON_THEME,
    LIGHT_COLORS,
    one_decimal_x,
    one_decimal_y,
    fixmargins,
    save_bokeh_hv,
)
from SSMuLA.util import ndcg_scale, checkNgen_folder

hv.extension("bokeh")
hv.renderer("bokeh").theme = JSON_THEME

ZS_OPTS = ["ed_score", "Triad_score", "ev_score", "esm_score", "esmif_score", "coves_score"]
ZS_COMB_OPTS = [
    "Triad-ev_score",
    "Triad-esm_score",
    "struc-comb_score",
    "msanoif-comb_score",
    "msa-comb_score",
    "structnmsa-comb_score",
    "two-best_score",
]

SIMPLE_ZS_OPT_LEGNED = {
    "none": "Random",
    "ed_score": "Edit distance",
    "Triad_score": "Triad",
    "ev_score": "EVmutation",
    "esm_score": "ESM",
    "esmif_score": "ESM-IF",
    "coves_score": "Coves",
}

ZS_OPTS_LEGEND = {
    "none": "Random",
    "ed_score": "Edit distance",
    "Triad_score": "Triad",
    "ev_score": "EVmutation",
    "esm_score": "ESM",
    "esmif_score": "ESM-IF",
    "coves_score": "Coves",
    "struc-comb_score": "Triad + ESM-IF",
    "Triad-ev_score": "Triad + EVmutation",
    "Triad-esm_score": "Triad + ESM",
    "msanoif-comb_score": "EVmutation + ESM",
    "two-best_score": "EVmutation + ESM-IF",
    "msa-comb_score": "EVmutation + ESM + ESM-IF",
    "structnmsa-comb_score": "Triad + EVmutation + ESM + ESM-IF",
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

        df["struc-comb_score"] = -1 * (df["Triad_rank"] + df["esmif_rank"])
        df["msa-comb_score"] = -1 * (df["ev_rank"] + df["esm_rank"] + df["esmif_rank"])
        df["msanoif-comb_score"] = -1 * (df["ev_rank"] + df["esm_rank"])
        df["structnmsa-comb_score"] = -1 * (
            df["Triad_rank"] + df["ev_rank"] + df["esm_rank"] + df["esmif_rank"]
        )

        df["two-best_score"] = -1 * (df["ev_rank"] + df["esmif_rank"])

        for comb_opt in [
            "Triad-ev",
            "Triad-esm",
            "struc-comb",
            "msa-comb",
            "msanoif-comb",
            "structnmsa-comb",
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

        zs_coord_dict = {zs: {} for zs in ZS_OPTS + ZS_COMB_OPTS}

        roc_plots = []

        for zs in ZS_OPTS + ZS_COMB_OPTS:

            print(f"number of nan in {self.lib_name} {zs}: {np.sum(np.isnan(df[zs]))}")

            if zs in ZS_OPTS:
                line_style = "solid"
            else:
                line_style = "dashed"

            df = df.dropna(subset=[zs])
            y_true_active = df["active"].values
            y_true_fitness = df["fitness"].values
            y_score = df[zs].values

            # skip if y_score only has nan
            if np.isnan(y_score).all():
                continue

            # calc rho and ndcg
            zs_coord_dict[zs]["rho"] = spearmanr(y_true_fitness, y_score)[0]
            zs_coord_dict[zs]["ndcg"] = ndcg_scale(y_true_fitness, y_score)

            # roc curves
            roc_name = f"{self.lib_name} active variant zero-shot predictor ROC curves"

            fpr, tpr, _ = roc_curve(y_true_active, y_score, pos_label=True)
            temp = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})

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