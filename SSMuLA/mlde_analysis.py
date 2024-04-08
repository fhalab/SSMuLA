"""A function for parsing the mlde results"""

from __future__ import annotations

import os
import re
import itertools
from glob import glob
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Basic plotting
import holoviews as hv

from SSMuLA.aa_global import DEFAULT_LEARNED_EMB_COMBO
from SSMuLA.landscape_global import n_mut_cutoff_dict, LIB_NAMES
from SSMuLA.zs_analysis import ZS_OPTS_LEGEND
from SSMuLA.vis import save_bokeh_hv, one_decimal_x, one_decimal_y, fixmargins
from SSMuLA.util import checkNgen_folder, get_file_name

hv.extension("bokeh")


DEFAULT_MLDE_METRICS = [
    "maxes",
    "means",
    "ndcgs",
    "rhos_pearson",
    "rhos_spearman",
    "iftruemax",
]


class MLDEParser:
    """A class for parsing EACH mlde result npy file"""

    def __init__(
        self, mlde_npy_path: str, mlde_results_dir: str = "results/mlde/saved"
    ):

        """
        Args:
        - mlde_npy: str, the path to the mlde npy file
            ie. 'results/mlde/saved/none/none-double/scale2max/GB1/one-hot_boosting|ridge_sample384_top96.npy'
        - mlde_results_dir: str, the directory where the mlde results are saved

        Note:

        {'data_config': {'input_csv': 'results/zs_comb/none/scale2max/all/GB1.csv',
            'zs_predictor': 'none',
            'encoding': ['one-hot'],
            'ft_libs': [149361],
            'scale_fit': 'max',
            'filter_min_by': 'none',
            'n_mut_cutoff': 2},
            'model_config': {'model_classes': ['boosting', 'ridge']},
            'train_config': {'n_sample': [384],
            'n_splits': 5,
            'n_replicate': 100,
            'n_worker': 1,
            'global_seed': 42,
            'verbose': False,
            'save_model': False},
            'eval_config': {'n_top': 96}}
        """

        self._mlde_npy_path = mlde_npy_path
        self._mlde_results_dir = mlde_results_dir

        # get all npy keys as properties
        # should be
        # ['config',
        #  'top_seqs',
        #  'maxes',
        #  'means',
        #  'ndcgs',
        #  'rhos_pearson',
        #  'unique',
        #  'labelled',
        #  'y_preds']
        for attr, val in self.npy_item.items():
            # need to specify what the rhos are from the oringal run 
            if attr != "rhos":
                setattr(self, attr, val)
            else:
                setattr(self, "rhos_pearson", val)

        if not hasattr(self, "config"):
            print(f"no config found for {self._mlde_npy_path}")
            pass

        # set all config_dict keys as properties
        # should be ['data_config', 'model_config', 'train_config', 'eval_config']
        for attr, val in self.config.items():
            setattr(self, attr, val)
            for k, v in val.items():
                setattr(self, k, v)
                if isinstance(v, list):
                    setattr(self, f"{k}_len", len(v))

        # TODO:
        # in the process of transfering all ZS to all, double, single folder

        if "all" not in self.input_csv:
            self.input_csv = os.path.join(
                os.path.dirname(self.input_csv), "all", os.path.basename(self.input_csv)
            )

        self._metric_df = self._get_metric_df()

    def _get_metric_df(self) -> pd.DataFrame:
        """Return the metric df"""

        # set up df for all metrics
        metric_df = pd.DataFrame(
            {
                "encoding": self.encoding_index,
                "model": self.models_index,
                "n_sample": self.n_sample_index,
                "ft_lib": self.lib_index,
                "repeats": self.repeats_index,
            }
        )

        metric_df["encoding"] = metric_df["encoding"].map(
            {i: v for i, v in enumerate(self.encoding)}
        )
        metric_df["model"] = metric_df["model"].map(
            {i: v for i, v in enumerate(self.model_classes)}
        )
        metric_df["n_sample"] = metric_df["n_sample"].map(
            {i: v for i, v in enumerate(self.n_sample)}
        )
        metric_df["ft_lib"] = metric_df["ft_lib"].map(
            {i: v for i, v in enumerate(self.ft_libs)}
        )
        metric_df["n_mut_cutoff"] = n_mut_cutoff_dict[self.n_mut_cutoff]
        metric_df["lib"] = get_file_name(self.input_csv)
        metric_df["zs"] = self.zs_predictor
        metric_df["n_top"] = self.n_top

        # calc if true max in topn
        self.iftruemax = np.any(self.top_seqs == self.max_fit_seq, axis=-1).astype(int)

        # calc spearman rho that is the same shape as the ndcg and pearson rho
        # in the shape of
        # (
        #     self.encoding_len,
        #     self.model_classes_len,
        #     self.n_sample_len,
        #     self.ft_libs_len,
        #     self.n_replicate,
        # )

        rho_spearman = np.full(self.output_shape, np.nan)

        # Iterate over the tensor to calculate Spearman correlation for each slice
        for i in range(self.encoding_len):
            for j in range(self.model_classes_len):
                for k in range(self.n_sample_len):
                    for n in range(self.ft_libs_len):
                        for m in range(self.n_replicate):
                            # Extract the current slice
                            y_pred = self.y_preds[i, j, k, n, m, :]

                            # Calculate the Spearman correlation coefficient
                            # Note: spearmanr returns a tuple (correlation, p-value), so we select [0] to get the correlation
                            rho_spearman[i, j, k, n, m] = spearmanr(
                                y_pred, self.true_ys
                            )[0]

        # set the rho_spearman as a property
        self.rhos_spearman = deepcopy(rho_spearman)

        # get all metrics as properties
        for m in DEFAULT_MLDE_METRICS:
            m_array = getattr(self, m)
            # get rid of nan col
            try:
                metric_df[m] = m_array[:, ~np.isnan(m_array).any(axis=0)].flatten()
            except Exception as e:
                print(
                    self._mlde_npy_path,
                    m_array.shape,
                    m_array[:, ~np.isnan(m_array).any(axis=0)].shape,
                )
                print(str(e))

        return metric_df

    @property
    def npy_item(self) -> dict:
        """Return the npy item"""
        return np.load(self._mlde_npy_path, allow_pickle=True).item()

    @property
    def npy_item_keys(self) -> list[str]:
        """Return the keys of the npy item"""
        return deepcopy(list(self.npy_item.keys()))

    @property
    def output_shape(self) -> tuple:

        """
        Return the shape of the output for
        maxes, means, ndcgs, rhos_pearson, rhos_spearman, true_max, unique, and labelled

            len(encodings),
            len(model_classes),
            len(n_samples),
            len(ft_libs),
            n_replicate,
        """

        return (
            self.encoding_len,
            self.model_classes_len,
            self.n_sample_len,
            self.ft_libs_len,
            self.n_replicate,
        )

    @property
    def top_seq_output_shape(self) -> tuple:

        """
        Return the shape of the output for top_seqs

            len(encodings),
            len(model_classes),
            len(n_samples),
            len(ft_libs),
            n_replicate,
            n_top,
        """

        return (
            self.encoding_len,
            self.model_classes_len,
            self.n_sample_len,
            self.ft_libs_len,
            self.n_replicate,
            self.n_top,
        )

    @property
    def encoding_index(self) -> np.ndarray:
        """Return the encoding index"""
        return np.array(
            [
                [i]
                * self.output_shape[1]
                * self.output_shape[2]
                * self.output_shape[3]
                * self.output_shape[4]
                for i in range(self.output_shape[0])
            ]
        ).flatten()

    @property
    def models_index(self) -> np.ndarray:
        """Return the models index"""
        return np.array(
            [
                [i] * self.output_shape[2] * self.output_shape[3] * self.output_shape[4]
                for i in range(self.output_shape[1])
                for _ in range(self.output_shape[0])
            ]
        ).flatten()

    @property
    def n_sample_index(self) -> np.ndarray:
        """Return the n_sample index"""
        return np.array(
            [
                [i] * self.output_shape[3] * self.output_shape[4]
                for i in range(self.output_shape[2])
                for _ in range(self.output_shape[1])
                for _ in range(self.output_shape[0])
            ]
        ).flatten()

    @property
    def lib_index(self) -> np.ndarray:
        """Return the lib index"""
        return np.array(
            [
                [i] * self.output_shape[4]
                for i in range(self.output_shape[3])
                for _ in range(self.output_shape[2])
                for _ in range(self.output_shape[1])
                for _ in range(self.output_shape[0])
            ]
        ).flatten()

    @property
    def repeats_index(self) -> np.ndarray:
        """Return the repeats index"""
        return np.array(
            [
                i
                for i in range(self.output_shape[4])
                for _ in range(self.output_shape[3])
                for _ in range(self.output_shape[2])
                for _ in range(self.output_shape[1])
                for _ in range(self.output_shape[0])
            ]
        )

    @property
    def metric_df(self) -> pd.DataFrame:
        """Return the metric df"""
        return self._metric_df

    @property
    def input_df(self) -> str:
        """Return the input csv"""
        return pd.read_csv(self.input_csv)

    @property
    def filtered_df(self) -> pd.DataFrame:
        """Return the filtered df"""
        # make sure no stop codon
        df = self.input_df[~self.input_df["AAs"].str.contains("\*")].copy()

        if self.filter_min_by in ["none", "", None]:
            return df.copy()
        elif self.filter_min_by == "active":
            return df[df["active"]].copy()
        elif self.filter_min_by == "0":
            return df[df["fitness"] >= 0].copy()
        elif self.filter_min_by == "min0":
            df["fitness"] = df["fitness"].apply(lambda x: max(0, x))
            return df.copy()
        else:
            print(f"{self.filter_min_by} not valid -> no filter beyond no stop codon")
            return df.copy()

    @property
    def max_fit_seq(self) -> np.ndarray:
        """Return the max fit seq"""
        return self.input_df.loc[self.input_df["fitness"].idxmax()]["AAs"]

    @property
    def true_ys(self) -> np.ndarray:
        """Return the true ys"""
        return self.filtered_df["fitness"].copy()


def get_all_metric_df(mlde_results_dir: str = "results/mlde/saved") -> pd.DataFrame:
    """Return the metric df for all mlde results"""
    mlde_npy_paths = sorted(glob(f"{mlde_results_dir}/**/*.npy", recursive=True))
    # one-hot needs redo
    mlde_parsers = [MLDEParser(mlde_npy_path) for mlde_npy_path in tqdm(mlde_npy_paths)]
    return pd.concat([mlde_parser.metric_df for mlde_parser in mlde_parsers])


class MLDEVis:

    """A class for visualizing MLDE results"""

    def __init__(
        self,
        mlde_results_dir: str = "results/mlde/saved",
        mlde_vis_dir: str = "results/mlde/vis",
    ) -> None:

        """
        Args:
        - mlde_results_dir: str, the directory where the mlde results are saved
        - mlde_vis_dir: str, the directory where the mlde visualizations are saved
        """

        self._mlde_results_dir = mlde_results_dir
        self._mlde_vis_dir = checkNgen_folder(mlde_vis_dir)

        self._all_df = get_all_metric_df(self._mlde_results_dir)
        self._all_df.to_csv(os.path.join(self._mlde_vis_dir, "all_df.csv"), index=False)

        encoding_lists = deepcopy(
            [[encoding] for encoding in self._all_df["encoding"].unique()]
            + deepcopy([DEFAULT_LEARNED_EMB_COMBO])
        )
        models = self._all_df["model"].unique()
        n_tops = self._all_df["n_top"].unique()

        with tqdm() as pbar:
            pbar.reset(
                len(ZS_OPTS_LEGEND)
                * len(encoding_lists)
                * len(models)
                * len(n_tops)
                * len(DEFAULT_MLDE_METRICS)
            )

            for metric in DEFAULT_MLDE_METRICS:
                metric_subfolder = checkNgen_folder(
                    os.path.join(self._mlde_vis_dir, metric)
                )

                for zs in ZS_OPTS_LEGEND.keys():

                    zs_subfolder = checkNgen_folder(os.path.join(metric_subfolder, zs))

                    for encoding_list in encoding_lists:
                        print(encoding_list)
                        for model in models:
                            for n_top in n_tops:

                                self.zs_encode_model_ntop_metirc(
                                    zs,
                                    encoding_list,
                                    model,
                                    n_top,
                                    metric,
                                    zs_subfolder,
                                )
                                pbar.update()

            pbar.close()

    def zs_encode_model_ntop_metirc(
        self,
        zs: str,
        encoding_list: list[str],
        model: str,
        n_top: int,
        metric: str,
        plot_path: str,
    ):

        """
        Plot
        """

        if len(encoding_list) > 1:
            encoding = "ESM2"
        else:
            encoding = encoding_list[0]

        plot_name = f"{ZS_OPTS_LEGEND[zs]} {encoding} {model} {n_top} {metric}"

        save_bokeh_hv(
            hv.Violin(
                self._all_df[
                    (self._all_df["zs"] == zs)
                    & (self._all_df["encoding"].isin(encoding_list))
                    & (self._all_df["model"] == model)
                    & (self._all_df["n_top"] == n_top)
                ]
                .sort_values(["lib", "n_mut_cutoff"], ascending=[True, False])
                .copy(),
                kdims=["lib", "n_mut_cutoff"],
                vdims=[metric],
            ).opts(
                width=1200,
                height=400,
                violin_color="n_mut_cutoff",
                show_legend=True,
                legend_position="top",
                legend_offset=(0, 5),
                title=plot_name,
                ylim=(0, 1),
                hooks=[one_decimal_x, one_decimal_y, fixmargins, lib_ncut_hook],
            ),
            plot_name=plot_name,
            plot_path=plot_path,
        )


def lib_ncut_hook(plot, element):

    plot.handles["plot"].x_range.factors = [
        (lib, n_mut) for lib in LIB_NAMES for n_mut in ["single", "double", "all"]
    ]
    plot.handles["xaxis"].major_label_text_font_size = "0pt"
    # plot.handles['xaxis'].group_text_font_size = '0pt'
    # plot.handles['yaxis'].axis_label_text_font_size = '10pt'
    # plot.handles['yaxis'].axis_label_text_font_style = 'normal'
    # plot.handles['xaxis'].axis_label_text_font_style = 'normal'