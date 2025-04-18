"""A function for parsing the mlde results"""

from __future__ import annotations

import os
import math
from glob import glob
from tqdm import tqdm
from copy import deepcopy

import warnings

import numpy as np
import pandas as pd

from scipy.stats import stats, spearmanr, ttest_ind, t
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FormatStrFormatter

import seaborn as sns

from SSMuLA.aa_global import DEFAULT_LEARNED_EMB_COMBO
from SSMuLA.landscape_global import (
    n_mut_cutoff_dict,
    LIB_INFO_DICT,
    N_SAMPLE_LIST,
)
from SSMuLA.de_simulations import (
    DE_TYPES,
    DE_N_TEST,
    DE_LEGEND_MAP,
    get_de_lib,
    get_de_avg,
)
from SSMuLA.zs_analysis import (
    ZS_OPTS,
    ZS_OPTS_LEGEND,
    SIMPLE_ZS_OPT_LEGNED,
    SIX_ZS_COLORS,
    map_zs_labels,
)
from SSMuLA.alde_analysis import avg_alde_df, get_ftalde_libavg
from SSMuLA.finetune_analysis import parse_finetune_df, avg_finetune_df
from SSMuLA.vis import (
    save_bokeh_hv,
    plot_zs_violin,
    FZL_PALETTE,
    GRAY_COLORS,
    save_svg,
    glasbey_category10,
    generate_related_color,
)
from SSMuLA.util import checkNgen_folder, get_file_name

# Suppress all warnings
warnings.filterwarnings("ignore")


TOTAL_N_LIST = [n + 96 for n in N_SAMPLE_LIST]
N_TICK_LIST = [120, 192, 288, 480, 1056, 2016]

FTLIB_FRAC_LIST = [0.0625, 0.125, 0.25, 0.5, 1]


DEFAULT_MLDE_METRICS = [
    "all_maxes",  # max of predicted values
    "all_means",  # mean of predicted values
    "top_maxes",  # topn
    "top_means",  # topn
    "ndcgs",
    "rhos",
    "if_truemaxs",
    "truemax_inds",
]


ENCODING_STYLE = {
    "one-hot": "solid",
    "esm2_t33_650M_UR50D-flatten_site": "dashed",
    "esm2_t33_650M_UR50D-mean_site": "dashdot",
    "esm2_t33_650M_UR50D-mean_all": "dotted",
}


ENCODING_DETS = {
    "one-hot": "One-hot",
    "esm2_t33_650M_UR50D-flatten_site": "ESM-2 flatten over targeted sites",
    "esm2_t33_650M_UR50D-mean_site": "ESM-2 mean pooling over sites",
    "esm2_t33_650M_UR50D-mean_all": "ESM-2 mean pooling over full sequence",
}

ENCODING_COLOR = {
    "one-hot": "gray",
    "esm2_t33_650M_UR50D-flatten_site": "yellow",
    "esm2_t33_650M_UR50D-mean_all": "blue",
    "esm2_t33_650M_UR50D-mean_site": "green",
}

MLDE_ALDE_COLORS = [
    sns.color_palette("colorblind")[9],
    sns.color_palette("colorblind")[0],
    sns.color_palette("colorblind")[8],
    sns.color_palette("colorblind")[1],
    sns.color_palette("colorblind")[6],
    sns.color_palette("colorblind")[4],
    sns.color_palette("colorblind")[5],
    sns.color_palette("colorblind")[3],
]

FINETUNE_COLORS = [
    generate_related_color(3, 1, 2),
    sns.color_palette("colorblind")[2],
]

FTMLDE_COLOR_LIST = ["gray", "blue", "green", "purple", "yellow", "brown", "orange"]

PLOT_MLDE_METRICS = ["top_maxes", "if_truemaxs"]

PLOT_DE_METRICS = ["mean_all", "fraction_max"]

PLOT_LINE_PERFORMANCE_YAXIS = [
    "Average max fitness achieved",
    "Fraction reaching the global optimum",
]

PERFORMANCE_YAXIS_DICT = {
    "top_maxes": "Average max fitness achieved",
    "if_truemaxs": "Fraction reaching the global optimum",
}

ATTRIBUTE_MAPPING = {
    "percent_active": "Percent active",
    "frac_loc_opt_total": "Fraction of local optima",
    "fraction_non-magnitude": "Fraction of non-magnitude epistasis",
    "loc": "Cauchy peak location",
    "kurt": "Kurtosis (tailedness)",
    "numb_kde_peak": "Number of KDE peaks",
}

ATTRIBUTE_LIST = deepcopy(list(ATTRIBUTE_MAPPING.values()))

MLAL_ORDER = [
    "MLDE",
    "ftMLDE",
    "ALDE",
    "ftALDE",
    "ALDE x 3",
    "ftALDE x 3",
    "ALDE x 4",
    "ftALDE x 4",
]

DE_MLAL_ORDER = MLAL_ORDER + [
    "DE: Recomb",
    "DE: Single step",
    "DE: Top96 recomb",
]

ALL_FT_ORDER = [
    "Random",
    "Hamming distance",
    "EVmutation",
    "ESM-2",
    "ESM-IF",
    "CoVES",
    "Triad",
    "Hamming distance + EVmutation",
    "Hamming distance + ESM-2",
    "Hamming distance + ESM-IF",
    "Hamming distance + CoVES",
    "Hamming distance + Triad",
    "Triad + EVmutation",
    "Triad + ESM-2",
    "Triad + ESM-IF",
    "CoVES + EVmutation",
    "CoVES + ESM-2",
    "EVmutation + ESM-IF",
]

ALDE_MARKER_STYLES = {2: "o", 3: "s", 4: "X"}

ALDE_METRICS = ["Mean", "Frac"]

ALDE_METRIC_MAP = {
    "Mean": "Average max fitness improved",
    "Frac": "Fraction reaching the global optimum improved",
}

MLDE_METRIC_MAP = {
    "top_maxes": "Average max fitness achieved",
    "if_truemaxs": "Fraction reaching the global optimum",
}

ALDE_MODEL_OPTS = ["Boosting Ensemble", "DNN Ensemble"]

ALDE_ACQ_OPTS = ["GREEDY", "UCB", "TS"]


def get_mlde_avg_sdf(
    mlde_all: pd.DataFrame,
    n_top: int,
    n_mut_cutoff: str,
    zs: str,
    lib_list: list,
    model_list: list = ["boosting"],
    encoding_list: list = ["one-hot"],
    ft_frac: float = 0.125,
):
    """
    Get the average MLDE for a given set of libraries, zs values, and ft_frac

    Args:
    - mlde_all, pd.DataFrame: DataFrame containing all MLDE data
    - n_top, int: Number of top mutations to consider
    - n_mut_cutoff, str: Mutation cutoff to consider
    - zs, str: ZS value to consider
    - lib_list, list: List of libraries to consider
    - model_list, list: List of models to consider
    - ft_frac, float: fraction of the focused training set
    """

    slice_mlde = mlde_all[
        (mlde_all["lib"].isin(lib_list))
        & (mlde_all["zs"] == zs)
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["n_mut_cutoff"] == n_mut_cutoff)
        & (mlde_all["rep"].isin(np.arange(50)))
        & (
            mlde_all["model"].isin(model_list)
            & (mlde_all["encoding"].isin(encoding_list))
        )
    ]

    if zs != "none":
        lib_dfs = []
        for lib in lib_list:
            lib_df = slice_mlde[slice_mlde["lib"] == lib].copy()

            n_site = len(LIB_INFO_DICT[lib]["positions"])

            if n_mut_cutoff == "all":
                sample_space = 20 ** n_site
            elif n_mut_cutoff == "double":
                sample_space = math.comb(n_site, 2) * 20 ** 2
            elif n_mut_cutoff == "single":
                sample_space = n_site * 20

            ft_lib_unique = np.array(sorted(lib_df["ft_lib"].unique()))
            ft_lib_frac = ft_lib_unique / sample_space

            lib_df["ft_lib_size"] = lib_df["ft_lib"].map(
                {numb: frac for numb, frac in zip(ft_lib_unique, ft_lib_frac)}
            )

            lib_dfs.append(lib_df[lib_df["ft_lib_size"] == ft_frac])
        slice_mlde = pd.concat(lib_dfs)

    if len(lib_list) == 1:
        avg_mlde = (
            slice_mlde[["n_sample", "top_maxes", "if_truemaxs"]]
            .groupby("n_sample")
            .agg(["mean", "std"])
            .reset_index()
        )
    else:
        # STEP 1: average over replicates within each library
        per_lib_avg = (
            slice_mlde.groupby(["lib", "n_sample"])[["top_maxes", "if_truemaxs"]]
            .mean()
            .reset_index()
        )

        # STEP 2: now aggregate over libraries
        avg_mlde = (
            per_lib_avg.groupby("n_sample")[["top_maxes", "if_truemaxs"]]
            .agg(["mean", "std"])
            .reset_index()
        )

    avg_mlde.columns = ["{}_{}".format(i, j) for i, j in avg_mlde.columns]
    avg_mlde = avg_mlde.rename(columns={"n_sample_": "n_sample"}).set_index("n_sample")
    return avg_mlde


def get_ftmlde_libavg(
    mlde_csv: str,
    lib_list: list,
    n_sample: int,
    models: list = ["boosting"],
    n_top: int = 96,
    add_ensemble: bool = True,
) -> pd.DataFrame:

    """
    Get the ftMLDE data for each of the library, number of rounds, models, and acquisition method.

    Args:
        mlde_csv (str): Path to the FT-MLDE CSV file.
        lib_list (list): List of libraries to filter.
        n_sample (int): Number of samples.
        models (list): List of models to filter.
        n_top (int): Number of top samples.
        add_ensemble (bool): Whether to include ensemble models.

    Returns:
        pd.DataFrame: Filtered DataFrame containing FT-MLDE data.
    """

    mlde_df = pd.read_csv(mlde_csv)

    if lib_list is None:
        lib_list = deepcopy(mlde_df["lib"].unique().tolist())

    common_slice = mlde_df[
        (mlde_df["encoding"] == "one-hot")
        & (mlde_df["model"].isin(models))
        & (mlde_df["n_sample"] == n_sample)
        & (mlde_df["n_top"] == n_top)
        & (mlde_df["lib"].isin(lib_list))
    ]

    noft_df = common_slice[
        (common_slice["zs"] == "none") & (common_slice["n_mut_cutoff"] == "all")
    ]

    if not add_ensemble:
        ftzs = ZS_OPTS[1:]
    else:
        ftzs = ZS_OPTS[1:] + [
            "Triad-ev_score",
            "Triad-esm_score",
            "Triad-esmif_score",
            "coves-ev_score",
            "coves-esm_score",
            "two-best_score",
        ]

    no_combo_df = common_slice[
        (common_slice["n_mut_cutoff"] == "all")
        & (common_slice["zs"].isin(ftzs))
        & (common_slice["ft_lib"].isin([0.125 * 20 ** 3, 0.125 * 20 ** 4]))
    ].copy()

    ds_only_df = common_slice[
        (common_slice["n_mut_cutoff"] == "double") & (common_slice["zs"] == "none")
    ].copy()
    ds_only_df["zs"] = ds_only_df["zs"].replace({"none": "ed_score"})

    ds_comb_df = common_slice[
        (common_slice["n_mut_cutoff"] == "double")
        & (common_slice["zs"].isin(ZS_OPTS[1:]))
        & (common_slice["ft_lib"].isin([0.125 * 3 * 20 ** 2, 0.125 * 6 * 20 ** 2]))
    ].copy()

    ds_comb_df["zs"] = ds_comb_df["zs"].replace(
        {
            "ev_score": "ds-ev",
            "esmif_score": "ds-esmif",
            "esm_score": "ds-esm",
            "coves_score": "ds-coves",
            "Triad_score": "ds-Triad",
        }
    )

    slice_df = pd.concat(
        [noft_df, ds_only_df, no_combo_df, ds_comb_df], ignore_index=True
    )

    # Convert 'Category' column to categorical with defined order
    slice_df["zs"] = pd.Categorical(
        slice_df["zs"],
        categories=["none"]
        + [ZS_OPTS[0]]
        + ftzs
        + [
            "ds-ev",
            "ds-esmif",
            "ds-esm",
            "ds-coves",
            "ds-Triad",
        ],
        ordered=True,
    )

    slice_df = (
        slice_df[["lib", "zs", "top_maxes", "if_truemaxs"]]
        .groupby(["lib", "zs"])
        .mean()
        .reset_index()
        .sort_values(by=["zs", "lib"])
    )

    # map zs labels
    slice_df["zs"] = slice_df["zs"].apply(map_zs_labels)

    return slice_df.reset_index(drop=True).copy()


class MLDEParser:
    """
    A class for parsing EACH mlde result npy file with indexing
    """

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
        # [
        #  'config',
        #  'maxes',
        #  'means',
        #  'ndcgs',
        #  'rhos',
        #  'if_truemaxa',
        #  'truemax_inds',
        #  'top_seqs',
        #  'unique',
        #  'labelled',
        #  'y_preds',
        #  'y_trues',
        # ]

        for attr, val in self.npy_item.items():
            setattr(self, attr, val)

        if not hasattr(self, "y_preds"):
            print(f"no y_preds found for {self._mlde_npy_path}")
            pass

        # TODO CLEAN UP WITH NEW MLDE
        if not hasattr(self, "maxes_all"):
            setattr(self, "maxes_all", np.max(self.y_preds, axis=-1))

        if not hasattr(self, "means_all"):
            setattr(self, "means_all", np.mean(self.y_preds, axis=-1))

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

        if not hasattr(self, "max_fit_seq"):
            setattr(
                self,
                "max_fit_seq",
                self.filtered_df.loc[self.filtered_df["fitness"].idxmax()]["AAs"],
            )

        if not hasattr(self, "if_truemaxs"):
            setattr(
                self,
                "if_truemaxs",
                np.any(self.top_seqs == self.max_fit_seq, axis=-1).astype(int),
            )

        if not hasattr(self, "truemax_inds"):

            # init with nan
            truemax_inds = np.full(self.top_seqs.shape[:-1], np.nan)

            # Iterate over all possible indices of the first 5 dimensions
            for i in range(self.top_seqs.shape[0]):
                for j in range(self.top_seqs.shape[1]):
                    for k in range(self.top_seqs.shape[2]):
                        for n in range(self.top_seqs.shape[3]):
                            for m in range(self.top_seqs.shape[4]):
                                # Find the index in the last dimension where the element is max_fit_seq
                                match_indices = np.where(
                                    self.top_seqs[i, j, k, n, m] == self.max_fit_seq
                                )[0]
                                if match_indices.size > 0:
                                    # If there is at least one match, take the first one
                                    truemax_inds[i, j, k, n, m] = match_indices[0]
            setattr(self, "truemax_inds", truemax_inds)

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
        metric_df = pd.DataFrame()

        # Fill the DataFrame using nested loops
        for i, encoding in enumerate(self.encoding):
            for j, model_class in enumerate(self.model_classes):
                for k, n_sample in enumerate(self.n_sample):
                    for n, ft_lib in enumerate(self.ft_libs):
                        for m in range(self.n_replicate):  # Replicate index
                            row = {
                                "encoding": encoding,
                                "model": model_class,
                                "n_sample": n_sample,
                                "ft_lib": ft_lib,
                                "rep": m,
                            }
                            # Adding metric values to the row
                            for metric in DEFAULT_MLDE_METRICS:
                                row[metric] = getattr(self, metric)[i, j, k, n, m]
                            # Append the row to the DataFrame
                            metric_df = metric_df._append(row, ignore_index=True)

        # add other details as additional columns
        metric_df["n_mut_cutoff"] = n_mut_cutoff_dict[self.n_mut_cutoff]
        metric_df["lib"] = get_file_name(self.input_csv)
        metric_df["zs"] = self.zs_predictor.replace("_score_score", "_score")
        metric_df["n_top"] = self.n_top
        metric_df["scale_fit"] = self.scale_fit

        return metric_df

    @property
    def npy_item(self) -> dict:
        """Return the npy item"""
        try:
            # Use context manager to open and load the file
            with open(self._mlde_npy_path, "rb") as f:
                return np.load(f, allow_pickle=True).item()
        except Exception as e:
            print(f"Error loading {self._mlde_npy_path}: {e}")
            return {}

    @property
    def npy_item_keys(self) -> list[str]:
        """Return the keys of the npy item"""
        return deepcopy(list(self.npy_item.keys()))

    @property
    def output_shape(self) -> tuple:

        """
        Return the shape of the output for
        maxes, means, ndcgs, rhos, true_max, unique, and labelled

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


def get_all_metric_df(mlde_results_dir: str = "results/mlde/saved") -> pd.DataFrame:
    """Return the metric df for all mlde results"""
    mlde_npy_paths = sorted(glob(f"{mlde_results_dir}/**/*.npy", recursive=True))
    # one-hot needs redo
    mlde_parsers = []
    for mlde_npy_path in tqdm(mlde_npy_paths):
        try:
            mlde_parsers.append(MLDEParser(mlde_npy_path))
        except Exception as e:
            print(f"Error parsing {mlde_npy_path}: {e}")

    # mlde_parsers = [MLDEParser(mlde_npy_path) for mlde_npy_path in tqdm(mlde_npy_paths)]

    df_list = []

    for mlde_parser in mlde_parsers:
        df_list.append(mlde_parser.metric_df)
        del mlde_parser

    return pd.concat(df_list)


def get_heuristic_lib_order(
    lib_stat_csv: str,
    epistasis_csv: str,
    loc_csv: str,
    loc_opt: str = "frac_loc_opt_total",
) -> list[str]:

    """
    Get the heuristic library order based on the given DE type.
    """

    lib_stat = pd.read_csv(lib_stat_csv)
    epistasis_df = pd.read_csv(epistasis_csv)
    loc_df = pd.read_csv(loc_csv)

    epistasis_df_non_mag = epistasis_df[
        (epistasis_df["n_mut"] == "all")
        & (epistasis_df["summary_type"] == "fraction")
        & (epistasis_df["epistasis_type"] == "magnitude")
    ][["lib", "value"]]
    epistasis_df_non_mag["value"] = 1 - epistasis_df_non_mag["value"]
    epistasis_df_non_mag = epistasis_df_non_mag.rename(
        columns={"value": "non-magnitude"}
    )

    merge_df = pd.merge(
        pd.merge(lib_stat, epistasis_df_non_mag.reset_index(drop=True), on="lib"),
        loc_df[["lib", loc_opt]],
        on="lib",
    )

    merge_df["heuristics"] = (
        merge_df["percent_active"]
        / 100
        / merge_df["non-magnitude"]
        / merge_df["frac_loc_opt_total"]
        / (merge_df["n_site"] * 20)
    )

    return merge_df.sort_values(by="heuristics")["lib"].tolist()


class MLDESum:

    """A class for visualizing MLDE results"""

    def __init__(
        self,
        mlde_results_dir: str = "results/mlde/saved",
        mlde_vis_dir: str = "results/mlde/vis",
        all_encoding: bool = True,
        encoding_lists: list[str] = [],
        ifvis: bool = False,
    ) -> None:

        """
        Args:
        - mlde_results_dir: str, the directory where the mlde results are saved
        - mlde_vis_dir: str, the directory where the mlde visualizations are saved
        - all_encoding: bool, whether to visualize all encodings
        - encoding_lists: list[str], the list of encodings to visualize
        """

        self._mlde_results_dir = mlde_results_dir
        self._mlde_vis_dir = checkNgen_folder(mlde_vis_dir)

        self._all_df = get_all_metric_df(self._mlde_results_dir)
        self._all_df.to_csv(os.path.join(self._mlde_vis_dir, "all_df.csv"), index=False)

        if ifvis:
            if all_encoding or len(encoding_lists) == 0:
                encoding_lists = deepcopy(
                    [[encoding] for encoding in self._all_df["encoding"].unique()]
                    + deepcopy([DEFAULT_LEARNED_EMB_COMBO])
                )
            else:
                encoding_lists = [[encoding_list] for encoding_list in encoding_lists]

            models = self._all_df["model"].unique()
            n_tops = self._all_df["n_top"].unique()
            n_samples = self._all_df["n_sample"].unique()

            with tqdm() as pbar:
                pbar.reset(
                    len(ZS_OPTS_LEGEND)
                    * len(encoding_lists)
                    * len(models)
                    * len(n_samples)
                    * len(n_tops)
                    * len(DEFAULT_MLDE_METRICS)
                )

                for metric in DEFAULT_MLDE_METRICS:
                    metric_subfolder = checkNgen_folder(
                        os.path.join(self._mlde_vis_dir, metric)
                    )

                    for zs in ZS_OPTS_LEGEND.keys():

                        zs_subfolder = checkNgen_folder(
                            os.path.join(metric_subfolder, zs)
                        )

                        for encoding_list in encoding_lists:
                            for model in models:
                                for n_sample in n_samples:
                                    for n_top in n_tops:

                                        self.zs_encode_model_ntop_metirc(
                                            zs,
                                            encoding_list,
                                            model,
                                            n_sample,
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
        n_sample: int,
        n_top: int,
        metric: str,
        plot_path: str,
    ):

        """
        Plot
        """

        if len(encoding_list) > 1:
            encoding = "ESM-2"
        else:
            encoding = encoding_list[0]

        plot_name = "{} {} {} {} sample top {} {}".format(
            ZS_OPTS_LEGEND[zs], encoding, model, n_sample, n_top, metric
        )

        save_bokeh_hv(
            plot_zs_violin(
                all_df=self._all_df,
                zs=zs,
                encoding_list=encoding_list,
                model=model,
                n_sample=n_sample,
                n_top=n_top,
                metric=metric,
                plot_name=plot_name,
            ),
            plot_name=plot_name,
            plot_path=plot_path,
        )


def comb_mlde_dfs(
    mlde_csv: str = "results/mlde/vis_5/all_df.csv",
    mlde_csv2: str = "results/mlde_hm2zs/vis/all_df.csv",
    save_path: str = "results/mlde/mlde_df_comb.csv",
    onehotonly: bool = True,
):
    """A function for combining the seperate mlde runs"""

    df = pd.read_csv(mlde_csv)
    df2 = pd.read_csv(mlde_csv2)

    if onehotonly:
        df = df[df["encoding"] == "one-hot"].copy()
        df2 = df2[df2["encoding"] == "one-hot"].copy()

        save_path = save_path.replace(".csv", "_onehot.csv")
        # check the folder
        checkNgen_folder(save_path)

    # take all the single, all together with no zs double from the main
    # add the new zs + ds from the second csv
    df0 = df[df["n_mut_cutoff"].isin(["all", "single"])]
    df1 = df[(df["n_mut_cutoff"] == "double") & (df["zs"] == "none")]

    df_comb = pd.concat([df0, df1, df2])

    df_comb.to_csv(save_path, index=False)


def plot_mlde_emb(
    sliced_mlde,
    lib_list,
    zs,
    fig_name,
    n_top=96,
    fig_dir="figs",
    ifsave=True,
):

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, mlde_metric, y_label in zip(
        axes,
        PLOT_MLDE_METRICS,
        PLOT_LINE_PERFORMANCE_YAXIS,
    ):

        for i, (emb_opt, ls) in enumerate(ENCODING_STYLE.items()):
            avg_mlde_df = get_mlde_avg_sdf(
                sliced_mlde,
                n_top,
                n_mut_cutoff="all",
                zs=zs,
                encoding_list=[emb_opt],
                lib_list=lib_list,
            )

            ax.plot(
                TOTAL_N_LIST,
                avg_mlde_df[f"{mlde_metric}_mean"],
                marker="o",
                linestyle=ls,
                linewidth=2,
                color=FZL_PALETTE[ENCODING_COLOR[emb_opt]],
                label=ENCODING_DETS[emb_opt],
            )
            ax.fill_between(
                TOTAL_N_LIST,
                avg_mlde_df[f"{mlde_metric}_mean"] - avg_mlde_df[f"{mlde_metric}_std"],
                avg_mlde_df[f"{mlde_metric}_mean"] + avg_mlde_df[f"{mlde_metric}_std"],
                color=FZL_PALETTE[ENCODING_COLOR[emb_opt]],
                alpha=0.05,
            )

        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])

        ax.set_ylim(0, 1.0)

        ax.set_xscale("log")
        # label the orignial xticks labels
        ax.set_xticks(N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())

        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

        # Get the handles and labels from the legend
        handles, labels = ax.get_legend_handles_labels()

        [
            handles[index].set_dashes([6, 2])
            for index, value in enumerate(ENCODING_STYLE.values())
            if value == "dashed"
        ]

        if ax == axes[1]:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1.025))

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def get_model_comp_dict(
    mlde_csv, lib_list, n_top=96, alde_dir="/disk2/fli/alde4ssmula"
):

    mlde_all = pd.read_csv(mlde_csv)

    avg_mlde_df_dict = {}

    # just mlde
    avg_mlde_df_dict["MLDE boosting"] = get_mlde_avg_sdf(
        mlde_all,
        n_top,
        n_mut_cutoff="all",
        zs="none",
        lib_list=lib_list,
        model_list=["boosting"],
        ft_frac=0.125,
    )

    avg_mlde_df_dict["MLDE ridge"] = get_mlde_avg_sdf(
        mlde_all,
        n_top,
        n_mut_cutoff="all",
        zs="none",
        lib_list=lib_list,
        model_list=["ridge"],
        ft_frac=0.125,
    )

    for eq_n in [2, 3, 4]:

        avg_mlde_df_dict[f"ALDE boosting x {eq_n}"] = avg_alde_df(
            eq_n, lib_list=lib_list, alde_model="Boosting Ensemble", alde_dir=alde_dir
        )

        avg_mlde_df_dict[f"ALDE DNN x {eq_n}"] = avg_alde_df(
            eq_n, lib_list=lib_list, alde_model="DNN Ensemble", alde_dir=alde_dir
        )

    return avg_mlde_df_dict


def plot_mlde_dict(
    avg_mlde_df_dict,
    fig_name,
    n_top=96,
    n_corr=384,
    ifvline=False,
    fig_dir="figs",
    ifsave=True,
):

    # for avg cross number of samples
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    line_styles = ["dashed", "solid"] + ["dotted", "solid"] * 3

    for ax, mlde_metric, y_label in zip(
        axes,
        PLOT_MLDE_METRICS,
        PLOT_LINE_PERFORMANCE_YAXIS,
    ):

        for i, (mlde_opts, mlde_df) in enumerate(avg_mlde_df_dict.items()):

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                label=mlde_opts.replace("Average ", "").replace(" x 2", ""),
                marker="o",
                linestyle=line_styles[i],
                linewidth=2,
                color=MLDE_ALDE_COLORS[i],
            )
            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=MLDE_ALDE_COLORS[i],
                alpha=0.05,
            )

        if ifvline:
            ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")

        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(0, 1.0)

        ax.set_xscale("log")
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

        # label the orignial xticks labels
        ax.set_xticks(N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())

        # Get the handles and labels from the legend
        # fix the dash handle
        handles, labels = ax.get_legend_handles_labels()

        # Find the index of the word "dashed" and fix spacing
        [
            handles[index].set_dashes([6, 2])
            for index, value in enumerate(line_styles)
            if value == "dashed"
        ]

        if ax == axes[1]:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1.025))

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def get_mlde_avg_dict(
    mlde_csv: str,
    alde_dir: str,
    lib_list: list,
    n_top: int = 96,
    alde_rounds: list = [2, 3, 4],
    add_ensemble: bool = False,
):

    """
    Get the average MLDE for a given set of libraries with defaul 0.125 ft_frac
    for all MLDE, ALDE, and focused training options

    Args:
    - mlde_csv: str, path to the MLDE csv, empty string means skip
    - alde_dir: str, path to the ALDE directory, empty string means skip
    - lib_list: list, list of libraries
    - n_top: int, n_top
    - add_ensemble: bool, whether to add ensemble options,
        ie Triad + EVmutation", "Triad + ESM", "Triad + ESM-IF", "EVmutation + ESM-IF
    """

    avg_mlde_df_dict = {}

    if mlde_csv != "":

        mlde_all = pd.read_csv(mlde_csv)

        mlde_kwargs = dict(
            mlde_all=mlde_all,
            n_top=n_top,
            lib_list=lib_list,
            ft_frac=0.125,
        )

        # just mlde
        avg_mlde_df_dict["MLDE"] = get_mlde_avg_sdf(
            n_mut_cutoff="all", zs="none", **mlde_kwargs
        )

        avg_mlde_df_dict["ftMLDE: Hamming distance"] = get_mlde_avg_sdf(
            n_mut_cutoff="double", zs="none", **mlde_kwargs
        )

        for zs in ZS_OPTS[1:]:
            avg_mlde_df_dict[f"ftMLDE: {ZS_OPTS_LEGEND[zs]}"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs=zs, **mlde_kwargs
            )

        # get average of all ftMLDE
        avg_mlde_df_dict["Average ftMLDE"] = pd.DataFrame(
            np.mean(
                np.stack([v for k, v in avg_mlde_df_dict.items() if "ftMLDE: " in k]),
                axis=0,
            ),
            columns=avg_mlde_df_dict["MLDE"].columns,
            index=N_SAMPLE_LIST,
        )

        # add double site
        for zs in ZS_OPTS[1:]:
            avg_mlde_df_dict[f"ds-ftMLDE: {ZS_OPTS_LEGEND[zs]}"] = get_mlde_avg_sdf(
                n_mut_cutoff="double", zs=zs, **mlde_kwargs
            )

        # if add ensemble
        if add_ensemble:
            avg_mlde_df_dict["Triad + EVmutation"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs="Triad-ev_score", **mlde_kwargs
            )

            avg_mlde_df_dict["Triad + ESM-2"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs="Triad-esm_score", **mlde_kwargs
            )

            avg_mlde_df_dict["Triad + ESM-IF"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs="Triad-esm_score", **mlde_kwargs
            )

            avg_mlde_df_dict["CoVES + EVmutation"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs="coves-ev_score", **mlde_kwargs
            )

            avg_mlde_df_dict["CoVES + ESM-2"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs="coves-esm_score", **mlde_kwargs
            )

            avg_mlde_df_dict["EVmutation + ESM-IF"] = get_mlde_avg_sdf(
                n_mut_cutoff="all", zs="two-best_score", **mlde_kwargs
            )

        col_names = list(avg_mlde_df_dict["MLDE"].columns)
    else:

        col_names = [
            "top_maxes_mean",
            "top_maxes_std",
            "if_truemaxs_mean",
            "if_truemaxs_std",
        ]

    if alde_dir != "":
        # now add alde
        alde_kwags = dict(
            lib_list=lib_list,
            alde_model="Boosting Ensemble",
            alde_encoding="onehot",
            alde_acq="GREEDY",
            alde_dir=alde_dir,
        )

        for eq_n in alde_rounds:

            avg_mlde_df_dict[f"ALDE x {eq_n}"] = avg_alde_df(eq_n, **alde_kwags)

            # add zs
            for zs in ZS_OPTS:
                avg_mlde_df_dict[
                    f"ftALDE x {eq_n}: {ZS_OPTS_LEGEND[zs]}"
                ] = avg_alde_df(eq_n, zs=zs.replace("_score", ""), **alde_kwags)

            # take ftALDE average
            avg_ftalde_df = pd.DataFrame(
                np.mean(
                    np.stack(
                        [
                            v.to_numpy()
                            for k, v in avg_mlde_df_dict.items()
                            if f"ftALDE x {eq_n}" in k
                        ]
                    ),
                    axis=0,
                ),
                index=np.array(N_SAMPLE_LIST).flatten().ravel(),
                columns=col_names,  # avg_mlde_df_dict["MLDE"].columns,  # make all column nameas are the same
            )

            try:
                # add name to the index
                avg_ftalde_df.index.name = "n_sample"

            except:
                pass

            avg_mlde_df_dict[f"Average ftALDE x {eq_n}"] = avg_ftalde_df

            # also now add in ds ftalde
            for zs in ZS_OPTS[1:]:
                avg_mlde_df_dict[
                    f"ds-ftALDE x {eq_n}: {ZS_OPTS_LEGEND[zs]}"
                ] = avg_alde_df(eq_n, zs="ds-" + zs.replace("_score", ""), **alde_kwags)

    return avg_mlde_df_dict


def get_ftmlal_libavg(
    mlde_csv: str,
    alde_csv: str,
    lib_list: list,
    n_sample: int,
    mlde_models: list = ["boosting"],
    alde_models: list = ["Boosting Ensemble"],
    alde_acquisition: list = ["GREEDY"],
    n_top: int = 96,
    add_ensemble: bool = True,
) -> pd.DataFrame:

    """
    Get the ftMLDE and ftALDE data for each of the library, number of rounds, models, and acquisition method.

    Args:
        mlde_csv (str): Path to the FT-MLDE CSV file.
        alde_csv (str): Path to the FT-ALDE CSV file.
        lib_list (list): List of libraries to filter.
        n_sample (int): Number of samples.
        mlde_models (list): List of MLDE models to filter.
        alde_models (list): List of ALDE models to filter.
        alde_acquisition (list): List of acquisition methods to filter.
        n_top (int): Number of top samples.
        add_ensemble (bool): Whether to include ensemble models.

    Returns:
        pd.DataFrame: Filtered DataFrame containing FT-MLDE and FT-ALDE data.
    """

    ftmlde_libdf = get_ftmlde_libavg(
        mlde_csv=mlde_csv,
        lib_list=lib_list,
        n_sample=n_sample,
        models=mlde_models,
        n_top=n_top,
        add_ensemble=add_ensemble,
    )

    if lib_list is None:
        lib_list = deepcopy(ftmlde_libdf["lib"].unique().tolist())

    ftalde_libs = []

    for n in [2, 3, 4]:
        ftalde_libs.append(
            get_ftalde_libavg(
                alde_csv=alde_csv,
                lib_list=lib_list,
                n_total=n_sample + n_top,
                n_round=n,
                models=alde_models,
                acquisition=alde_acquisition,
            )
        )

    ftalde_libdf = (
        pd.concat(ftalde_libs, ignore_index=True).reset_index(drop=True).copy()
    )

    return (
        pd.concat([ftmlde_libdf, ftalde_libdf], ignore_index=True)
        .reset_index(drop=True)
        .copy()
    )


def plot_single_mlde_vs_de(
    de_csv: str,
    mlde_csv: str,
    alde_dir: str,
    lib_list: list,
    metric_idx: int,  # for mlde_metric, de_metric, and y_label
    fig_name: str | None = None,
    iflegend: bool = False,
    n_top: int = 96,
    n_corr: int = 384,
    ax=None,
):

    """
    Plot a single MLDE vs DE comparison on a given subplot axis.

    Args:
        n_top (int): Number of top variants.
        n_corr (int): Number of corrected variants.
        ax (matplotlib.axes): Axis to plot on.
    """

    line_styles = [
        "dashed",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
    ]
    mlde_metric = PLOT_MLDE_METRICS[metric_idx]
    de_metric = PLOT_DE_METRICS[metric_idx]
    y_label = PLOT_LINE_PERFORMANCE_YAXIS[metric_idx]

    if isinstance(lib_list, list):
        de_avg = get_de_avg(de_csv, lib_list)

    else:
        de_avg = get_de_lib(de_csv, lib_list)

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv, alde_dir=alde_dir, lib_list=lib_list, n_top=n_top
    )

    # MLDE Plot
    for i, mlde_opts in enumerate(
        [
            "MLDE",
            "Average ftMLDE",
            "ALDE x 2",
            "Average ftALDE x 2",
            "ALDE x 3",
            "Average ftALDE x 3",
            "ALDE x 4",
            "Average ftALDE x 4",
        ]
    ):
        mlde_df = avg_mlde_df_dict[mlde_opts]

        ax.plot(
            TOTAL_N_LIST,
            mlde_df[f"{mlde_metric}_mean"],
            label=mlde_opts.replace("Average ", "").replace(" x 2", ""),
            marker="o",
            linestyle=line_styles[i],
            linewidth=2,
            color=MLDE_ALDE_COLORS[i],
        )

        # only std for avg
        if metric_idx == 0:

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"] - mlde_df[f"{mlde_metric}_std"],
                mlde_df[f"{mlde_metric}_mean"] + mlde_df[f"{mlde_metric}_std"],
                color=MLDE_ALDE_COLORS[i],
                alpha=0.05,
            )

    # DE Plot
    de_colors = sns.color_palette("Greys", 4)
    de_ls = ["dotted", "dashed", "dashdot"]

    for d, de in enumerate(DE_TYPES):
        ax.axhline(
            de_avg.loc[de, f"{de_metric}_mean"],
            label=f"DE: {DE_LEGEND_MAP[de]}",
            color=de_colors[d + 1],
            linestyle=de_ls[d],
            linewidth=2,
        )
        # add shaded region for std
        ax.fill_between(
            [57] + TOTAL_N_LIST,
            de_avg.loc[de, f"{de_metric}_mean"] - de_avg.loc[de, f"{de_metric}_std"],
            de_avg.loc[de, f"{de_metric}_mean"] + de_avg.loc[de, f"{de_metric}_std"],
            color=de_colors[d + 1],
            alpha=0.05,
        )

        # if four-site lib
        if "GB1" in lib_list or "TEV" in lib_list or "TrpB4" in lib_list:
            factor = 4
            marker = "d"
        else:
            factor = 3
            marker = "^"

        # Annotate points
        ax.scatter(
            19 * factor + DE_N_TEST[de],
            de_avg.loc[de, f"{de_metric}_mean"],
            marker=marker,
            facecolors="none",
            edgecolors=de_colors[d + 1],
            linewidth=1.2,
            s=36,
        )

        ax.scatter(
            factor * -22 * math.log(1 - 0.95) + DE_N_TEST[de],
            de_avg.loc[de, f"{de_metric}_mean"],
            marker=marker,
            color=de_colors[d + 1],
            s=40,
        )

    # Formatting
    ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
    ax.set_xlim(57, TOTAL_N_LIST[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xscale("log")
    ax.set_xticks([57, 76] + N_TICK_LIST)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Total number of variants")
    ax.set_ylabel(y_label)

    # add title
    if fig_name is not None:
        ax.set_title(fig_name)

    # Add legend if specified
    if iflegend:
        # Get the handles and labels from the legend
        handles, labels = ax.get_legend_handles_labels()
        # fix the dash handle
        [
            handles[index].set_dashes([6, 2])
            for index, value in enumerate(line_styles)
            if value == "dashed"
        ]
        # Manually create legend handles with assigned colors
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="none",
                    markeredgecolor="black",
                    markerfacecolor="none",
                    markersize=6,
                    label="3-site unique",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="d",
                    color="none",
                    markeredgecolor="black",
                    markerfacecolor="none",
                    markersize=6,
                    label="4-site unique",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="none",
                    markerfacecolor="black",
                    markersize=6,
                    label=r"3-site 95% coverage",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="d",
                    color="none",
                    markerfacecolor="black",
                    markersize=6,
                    label=r"4-site 95% coverage",
                ),
            ]
        )

        # Manually create legend labels
        labels.extend(
            [
                "3-site unique",
                "4-site unique",
                r"3-site 95% coverage",
                r"4-site 95% coverage",
            ]
        )

        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1.025))

    return ax


def plot_agg_de_vs_mlde(
    de_csv: str,
    mlde_csv: str,
    alde_dir: str,
    lib_stat_csv: str,
    epistasis_csv: str,
    loc_csv: str,
    fig_names: list[str],
    lib_list: list | None = None,
    n_top: int = 96,
    n_corr: int = 384,
    fig_dir: str = "figs",
    ifsave: bool = True,
):
    """
    Plot aggregated single DE vs MLDE
    """

    if lib_list is None:
        lib_list = get_heuristic_lib_order(
            lib_stat_csv=lib_stat_csv,
            epistasis_csv=epistasis_csv,
            loc_csv=loc_csv,
        )

    for metric_idx, fig_name in enumerate(fig_names):

        # create 4 by 4 subplots
        fig, axs = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)

        # for each plot_de_vs_mlde
        for i, ax in enumerate(axs.flatten()):
            lib_name = lib_list[i]

            if i != 7:
                iflegend = False
            else:
                iflegend = True

            # Plot the DE vs MLDE
            plot_single_mlde_vs_de(
                de_csv=de_csv,
                mlde_csv=mlde_csv,
                alde_dir=alde_dir,
                lib_list=[lib_name],
                metric_idx=metric_idx,
                fig_name=lib_name,
                iflegend=iflegend,
                ax=ax,
            )

            # only have y labels if in the first column
            if i % 4 != 0:
                ax.set_ylabel("")
            # only have x labels if in the last row
            if i < 12:
                ax.set_xlabel("")

        if ifsave:
            save_svg(fig, fig_name, fig_dir)


def plot_de_vs_mlde(
    de_csv: str,
    mlde_csv: str,
    alde_dir: str,
    lib_list: list,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    """
    Plot DE vs MLDE

    Args:
    - de_csv: str, path to the DE csv
    - mlde_csv: str, path to the MLDE csv
    - alde_dir: str, path to the ALDE directory
    - lib_list: list, list of libraries
    - fig_name: str, name of the figure
    - n_top: int, n_top
    - n_corr: int, n_corr, where the vertical line is drawn
    - ifsave: bool, whether to save the figure
    - fig_dir: str, directory to save the figure
    """

    de_avg = get_de_avg(de_csv, lib_list)

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv, alde_dir=alde_dir, lib_list=lib_list, n_top=n_top
    )

    # for avg cross number of samples
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    line_styles = [
        "dashed",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
        "dotted",
        "solid",
    ]

    for ax, mlde_metric, de_metric, y_label in zip(
        axes,
        PLOT_MLDE_METRICS,
        PLOT_DE_METRICS,
        PLOT_LINE_PERFORMANCE_YAXIS,
    ):

        for i, (mlde_opts) in enumerate(
            [
                "MLDE",
                "Average ftMLDE",
                "ALDE x 2",
                "Average ftALDE x 2",
                "ALDE x 3",
                "Average ftALDE x 3",
                "ALDE x 4",
                "Average ftALDE x 4",
            ]
        ):
            mlde_df = avg_mlde_df_dict[mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                label=mlde_opts.replace("Average ", "").replace(" x 2", ""),
                marker="o",
                linestyle=line_styles[i],
                linewidth=2,
                color=MLDE_ALDE_COLORS[i],
            )

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=MLDE_ALDE_COLORS[i],
                alpha=0.05,
            )

        de_colors = sns.color_palette("Greys", 4)
        de_ls = ["dotted", "dashed", "dashdot"]

        for d, de in enumerate(DE_TYPES):
            ax.axhline(
                de_avg.loc[de, f"{de_metric}_mean"],
                label=f"DE: {DE_LEGEND_MAP[de]}",
                color=de_colors[d + 1],
                linestyle=de_ls[d],
                linewidth=2,
            )

            ax.fill_between(
                [57] + TOTAL_N_LIST,  # N_SAMPLE_LIST,
                de_avg.loc[de, f"{de_metric}_mean"]
                - de_avg.loc[de, f"{de_metric}_std"],
                de_avg.loc[de, f"{de_metric}_mean"]
                + de_avg.loc[de, f"{de_metric}_std"],
                color=de_colors[d + 1],
                alpha=0.05,
            )

            # annotate unique screen
            ax.scatter(
                3 * 19 + DE_N_TEST[de],
                de_avg.loc[de, f"{de_metric}_mean"],
                marker="^",
                facecolors="none",
                edgecolors=de_colors[d + 1],
                linewidth=1.2,
                s=36,
            )
            ax.scatter(
                4 * 19 + DE_N_TEST[de],
                de_avg.loc[de, f"{de_metric}_mean"],
                marker="d",
                facecolors="none",
                edgecolors=de_colors[d + 1],
                linewidth=1.2,
                s=36,
            )

        # add vline
        ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")

        # ax.set_xlim(N_SAMPLE_LIST[0], N_SAMPLE_LIST[-1])
        ax.set_xlim(57, TOTAL_N_LIST[-1])
        ax.set_ylim(0, 1.0)

        ax.set_xscale("log")
        # label the orignial xticks labels
        ax.set_xticks([57, 76] + N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())

        # ax.set_xlabel("Number of training samples")
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

        if ax == axes[1]:
            # Get the handles and labels from the legend
            handles, labels = ax.get_legend_handles_labels()
            # fix the dash handle
            [
                handles[index].set_dashes([6, 2])
                for index, value in enumerate(line_styles)
                if value == "dashed"
            ]
            # Manually create legend handles with assigned colors
            handles.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="black",
                        linestyle="none",
                        markeredgecolor="black",
                        markerfacecolor="none",
                        markersize=6,
                        label="3-site",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="d",
                        color="black",
                        linestyle="none",
                        markeredgecolor="black",
                        markerfacecolor="none",
                        markersize=6,
                        label="4-site",
                    ),
                ]
            )

            # Manually create legend labels
            labels.extend(
                [
                    "3-site DE",
                    "4-site DE",
                ]
            )

            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1.025))

    if ifsave:
        save_svg(fig, fig_name, fig_dir)

    return de_avg, avg_mlde_df_dict


def plot_ftmlde_simple(
    mlde_csv: str,
    lib_list: list,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifzoomy: bool = False,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv, alde_dir="", lib_list=lib_list, n_top=n_top
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    mlde_color_list = FTMLDE_COLOR_LIST + ["dark_brown"]

    # Define the data and labels for MLDE and ALDE cases
    options = [
        "MLDE",
        "ftMLDE: Hamming distance",
        "ftMLDE: EVmutation",
        "ftMLDE: ESM-2",
        "ftMLDE: ESM-IF",
        "ftMLDE: CoVES",
        "ftMLDE: Triad",
        "ds-ftMLDE: EVmutation",
    ]

    line_styles = ["solid"] * 7 + ["dashed"]

    if ifzoomy:
        y_mins = [0.5, 0]
    else:
        y_mins = [0, 0]

    # Loop over

    for ax, mlde_metric, y_label, y_min in zip(
        axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS, y_mins
    ):

        for i, mlde_opts in enumerate(options):
            mlde_df = avg_mlde_df_dict[mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                label=mlde_opts.replace("Average ", ""),
                marker="o",
                linewidth=2,
                linestyle=line_styles[i],
                color=FZL_PALETTE[mlde_color_list[i % len(mlde_color_list)]],
            )

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=FZL_PALETTE[mlde_color_list[i % len(mlde_color_list)]],
                alpha=0.05,
            )

        ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(y_min, 1.0)
        ax.set_xscale("log")
        # label the orignial xticks labels
        ax.set_xticks(N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

    # add first legend to be different zs
    mlde_color_dict = {
        l.replace("ftMLDE: ", "")
        .replace("MLDE", "Random sampling")
        .replace("ds-", "Hamming distance\n"): FZL_PALETTE[c]
        for (l, c) in zip(options, mlde_color_list)
    }
    # Create legend for line colors using the color dictionary
    color_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2,
            label=label,
            marker="o",
            linestyle=line_styles[i],
        )
        for i, (label, color) in enumerate(mlde_color_dict.items())
    ]

    [
        color_handles[index].set_dashes([6, 2])
        for index, value in enumerate(line_styles)
        if value == "dashed"
    ]
    # Add the legends to the figure
    axes[1].legend(handles=color_handles, loc="upper left", bbox_to_anchor=(1, 1.025))

    # Adjust the layout to prevent overlapping
    plt.tight_layout(h_pad=1.5)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def plot_single_ftmlde_doubles(
    mlde_csv: str,
    lib_list: list,
    metric_idx: int,  # Index for metric selection
    fig_name: str | None = None,
    iflegend: bool = True,
    ifzoomy: bool = True,
    n_top: int = 96,
    n_corr: int = 384,
    ax=None,
):
    """
    Plot a single FTMLDE doubles comparison on a given subplot axis.

    Args:
        metric_idx (int): Index for metric selection.
        ax (matplotlib.axes): Axis to plot on.
    """

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv, alde_dir="", lib_list=lib_list, n_top=n_top
    )

    mlde_metric = PLOT_MLDE_METRICS[metric_idx]
    y_label = PLOT_LINE_PERFORMANCE_YAXIS[metric_idx]

    # Define the different MLDE options to be plotted
    options = [
        "MLDE",
        "ftMLDE: Hamming distance",
        "ftMLDE: EVmutation",
        "ftMLDE: ESM-2",
        "ftMLDE: ESM-IF",
        "ftMLDE: CoVES",
        "ftMLDE: Triad",
    ]

    if not ifzoomy or "TrpB3A" in lib_list:
        y_min = 0
    else:
        y_min = 0.5

    # Plot MLDE variations
    for i, mlde_opts in enumerate(options):
        mlde_df = avg_mlde_df_dict[mlde_opts]

        ax.plot(
            TOTAL_N_LIST,
            mlde_df[f"{mlde_metric}_mean"],
            label=mlde_opts.replace("Average ", ""),
            marker="o",
            linewidth=2,
            color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
        )

        # only std for avg
        if metric_idx == 0:
            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
                alpha=0.05,
            )

        # Plot double-site variant in gray with dashed linestyle
        if i > 1:
            mlde_df = avg_mlde_df_dict["ds-" + mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                marker="o",
                linestyle="dashed",
                linewidth=2,
                color=GRAY_COLORS[
                    "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                ],
            )

            # only std for avg
            if metric_idx == 0:

                ax.fill_between(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"].values.flatten()
                    - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                    mlde_df[f"{mlde_metric}_mean"].values.flatten()
                    + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                    color=GRAY_COLORS[
                        "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                    ],
                    alpha=0.05,
                )

    # Add vertical reference lines
    ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
    ax.axvline(96 + n_top, color="gray", linewidth=0.5, linestyle="dotted")

    # Set axis limits and scales
    ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
    ax.set_ylim(y_min, 1.0)
    ax.set_xscale("log")
    ax.set_xticks(N_TICK_LIST)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Total number of variants")
    ax.set_ylabel(y_label)

    if iflegend:

        # add first legend to be different zs
        mlde_color_dict = {
            l.replace("ftMLDE: ", "").replace("MLDE", "Random sampling"): FZL_PALETTE[c]
            for (l, c) in zip(options, FTMLDE_COLOR_LIST)
        }
        ds_color_dict = {
            l.replace("ftMLDE: ", "Hamming distance + "): GRAY_COLORS["gray-" + c]
            for (l, c) in zip(options[2:], FTMLDE_COLOR_LIST[2:])
        }

        # Create legend for line colors using the color dictionary
        color_handles = [
            Line2D([0], [0], color=color, lw=2, label=label, marker="o")
            for label, color in mlde_color_dict.items()
        ]

        gray_handles = [
            Line2D(
                [0], [0], color=color, lw=2, label=label, marker="o", linestyle="dashed"
            )
            for label, color in ds_color_dict.items()
        ]
        for h in gray_handles:
            h.set_dashes([6, 2])

        # # Add the legends to the figure
        ax.legend(
            handles=color_handles + gray_handles,
            loc="upper left",
            bbox_to_anchor=(1, 1.025),
        )

    # Add title if specified
    if fig_name is not None:
        ax.set_title(fig_name)

    return ax


def plot_agg_ftmlde_doubles(
    mlde_csv: str,
    lib_stat_csv: str,
    epistasis_csv: str,
    loc_csv: str,
    fig_names: list[str],
    lib_list: list | None = None,
    n_top: int = 96,
    n_corr: int = 384,
    fig_dir: str = "figs",
    ifsave: bool = True,
):
    """
    Plot aggregated FTMLDE doubles across multiple libraries.
    """

    # If no library list is provided, get it heuristically
    if lib_list is None:
        lib_list = get_heuristic_lib_order(
            lib_stat_csv=lib_stat_csv,
            epistasis_csv=epistasis_csv,
            loc_csv=loc_csv,
        )

    # Loop through each metric to create a figure
    for metric_idx, fig_name in enumerate(fig_names):

        # Create a 4x4 subplot grid
        fig, axs = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)

        # Iterate through libraries and plot them
        for i, ax in enumerate(axs.flatten()):
            lib_name = lib_list[i]

            # Only show the legend on one subplot
            if i != 7:
                iflegend = False
            else:
                iflegend = True

            # Plot FTMLDE doubles
            plot_single_ftmlde_doubles(
                mlde_csv=mlde_csv,
                lib_list=[lib_name],
                metric_idx=metric_idx,
                fig_name=lib_name,
                iflegend=iflegend,
                ifzoomy=False,
                n_top=n_top,
                n_corr=n_corr,
                ax=ax,
            )

            # Hide unnecessary axis labels
            if i % 4 != 0:
                ax.set_ylabel("")
            if i < 12:
                ax.set_xlabel("")

        # Save the figure if required
        if ifsave:
            save_svg(fig, fig_name, fig_dir)


def plot_ftmlde_doubles(
    mlde_csv: str,
    lib_list: list,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifzoomy: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv, alde_dir="", lib_list=lib_list, n_top=n_top
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4), sharex=True
    )  # Create a 2x2 grid of subplots

    # Define the data and labels for MLDE and ALDE cases
    options = [
        "MLDE",
        "ftMLDE: Hamming distance",
        "ftMLDE: EVmutation",
        "ftMLDE: ESM-2",
        "ftMLDE: ESM-IF",
        "ftMLDE: CoVES",
        "ftMLDE: Triad",
    ]

    if not ifzoomy or "TrpB3A" in lib_list:
        y_mins = [0, 0]
    else:
        y_mins = [0.5, 0]

    # Loop over  MLDE
    for ax, mlde_metric, y_label, y_min in zip(
        axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS, y_mins
    ):

        for i, mlde_opts in enumerate(options):
            mlde_df = avg_mlde_df_dict[mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                label=mlde_opts.replace("Average ", ""),
                marker="o",
                # linestyle="dotted",
                linewidth=2,
                color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
            )

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
                alpha=0.05,
            )
            if i > 1:
                # plot the double site same color but different linestyle and marker
                mlde_df = avg_mlde_df_dict["ds-" + mlde_opts]

                ax.plot(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"],
                    label=mlde_opts.replace("Average ", ""),
                    marker="o",
                    # markersize=7.5,
                    linestyle="dashed",
                    linewidth=2,
                    color=GRAY_COLORS[
                        "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                    ],
                )

                ax.fill_between(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"].values.flatten()
                    - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                    mlde_df[f"{mlde_metric}_mean"].values.flatten()
                    + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                    color=GRAY_COLORS[
                        "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                    ],
                    alpha=0.05,
                )

        ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
        ax.axvline(96 + n_top, color="gray", linewidth=0.5, linestyle="dotted")
        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(y_min, 1.0)
        ax.set_xscale("log")
        # label the orignial xticks labels
        ax.set_xticks(N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

    # add first legend to be different zs
    mlde_color_dict = {
        l.replace("ftMLDE: ", "").replace("MLDE", "Random sampling"): FZL_PALETTE[c]
        for (l, c) in zip(options, FTMLDE_COLOR_LIST)
    }
    ds_color_dict = {
        l.replace("ftMLDE: ", "Hamming distance + "): GRAY_COLORS["gray-" + c]
        for (l, c) in zip(options[2:], FTMLDE_COLOR_LIST[2:])
    }

    # Create legend for line colors using the color dictionary
    color_handles = [
        Line2D([0], [0], color=color, lw=2, label=label, marker="o")
        for label, color in mlde_color_dict.items()
    ]

    gray_handles = [
        Line2D([0], [0], color=color, lw=2, label=label, marker="o", linestyle="dashed")
        for label, color in ds_color_dict.items()
    ]
    for h in gray_handles:
        h.set_dashes([6, 2])

    # # Add the legends to the figure
    axes[1].legend(
        handles=color_handles + gray_handles,
        loc="upper left",
        bbox_to_anchor=(1, 1.025),
    )

    # Adjust the layout to prevent overlapping
    plt.tight_layout(h_pad=1.5)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def plot_single_ftmlde_ensemble(
    mlde_csv: str,
    lib_list: list,
    metric_idx: int,  # Index for metric selection
    fig_name: str | None = None,
    iflegend: bool = True,
    ifzoomy: bool = True,
    n_top: int = 96,
    n_corr: int = 384,
    ax=None,
):
    """
    Plot a single FTMLDE ensemble comparison on a given subplot axis.

    Args:
        metric_idx (int): Index for metric selection.
        ax (matplotlib.axes): Axis to plot on.
    """

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv,
        alde_dir="",
        lib_list=lib_list,
        n_top=n_top,
        add_ensemble=True,
    )

    mlde_metric = PLOT_MLDE_METRICS[metric_idx]
    y_label = PLOT_LINE_PERFORMANCE_YAXIS[metric_idx]

    options = [
        "MLDE",
        "ftMLDE: EVmutation",
        "ftMLDE: ESM-2",
        "ftMLDE: ESM-IF",
        "ftMLDE: CoVES",
        "ftMLDE: Triad",
    ]

    ensemble_opts = [
        "Triad + EVmutation",
        "Triad + ESM-2",
        "Triad + ESM-IF",
        "CoVES + EVmutation",
        "CoVES + ESM-2",
        "EVmutation + ESM-IF",
    ]

    mlde_color_list = ["gray", "green", "purple", "yellow", "brown", "orange", "blue"]

    y_min = 0 if (not ifzoomy or "TrpB3A" in lib_list) else 0.5

    # Plot MLDE and Ensembles
    for i, mlde_opts in enumerate(options):
        mlde_df = avg_mlde_df_dict[mlde_opts]

        ax.plot(
            TOTAL_N_LIST,
            mlde_df[f"{mlde_metric}_mean"],
            label=mlde_opts.replace("Average ", ""),
            marker="o",
            linewidth=2,
            color=FZL_PALETTE[mlde_color_list[i % len(mlde_color_list)]],
        )

        # only std for avg
        if metric_idx == 0:
            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=FZL_PALETTE[mlde_color_list[i % len(mlde_color_list)]],
                alpha=0.05,
            )

        j = i + 1  # no gray gray

        # plot the ensemble same color but different linestyle and marker
        mlde_df = avg_mlde_df_dict[ensemble_opts[j - 1]]

        ax.plot(
            TOTAL_N_LIST,
            mlde_df[f"{mlde_metric}_mean"],
            marker="o",
            linestyle="dashed",
            linewidth=2,
            color=GRAY_COLORS["gray-" + mlde_color_list[j % len(mlde_color_list)]],
        )

        # only std for avg
        if metric_idx == 0:
            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=GRAY_COLORS["gray-" + mlde_color_list[j % len(mlde_color_list)]],
                alpha=0.05,
            )

    # Add reference lines
    ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
    ax.axvline(96 + n_top, color="gray", linewidth=0.5, linestyle="dotted")

    ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
    ax.set_ylim(y_min, 1.0)
    ax.set_xscale("log")
    ax.set_xticks(N_TICK_LIST)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())

    ax.set_xlabel("Total number of variants")
    ax.set_ylabel(y_label)

    if iflegend:
        # add first legend to be different zs
        mlde_color_dict = {
            l.replace("ftMLDE: ", "").replace("MLDE", "Random sampling"): FZL_PALETTE[c]
            for (l, c) in zip(options, mlde_color_list)
        }
        ds_color_dict = {
            l.replace("ftMLDE: ", "Triad + "): GRAY_COLORS["gray-" + c]
            for (l, c) in zip(options[1:4], mlde_color_list[1:4])
        }

        ds_color_dict["CoVES + EVmutation"] = GRAY_COLORS["gray-" + mlde_color_list[-3]]
        ds_color_dict["CoVES + ESM-2"] = GRAY_COLORS["gray-" + mlde_color_list[-2]]

        ds_color_dict["EVmutation + ESM-IF"] = GRAY_COLORS[
            "gray-" + mlde_color_list[-1]
        ]
        # Create legend for line colors using the color dictionary
        color_handles = [
            Line2D([0], [0], color=color, lw=2, label=label, marker="o")
            for label, color in mlde_color_dict.items()
        ]

        gray_handles = [
            Line2D(
                [0], [0], color=color, lw=2, label=label, marker="o", linestyle="dashed"
            )
            for label, color in ds_color_dict.items()
        ]

        for h in gray_handles:
            h.set_dashes([6, 2])

        # Add the legends to the figure
        ax.legend(
            handles=color_handles + gray_handles,
            loc="upper left",
            bbox_to_anchor=(1, 1.025),
        )

    if fig_name:
        ax.set_title(fig_name)

    return ax


def plot_agg_ftmlde_ensemble(
    mlde_csv: str,
    lib_stat_csv: str,
    epistasis_csv: str,
    loc_csv: str,
    fig_names: list[str],
    lib_list: list | None = None,
    n_top: int = 96,
    n_corr: int = 384,
    fig_dir: str = "figs",
    ifsave: bool = True,
):
    """
    Plot aggregated FTMLDE ensemble across multiple libraries.
    """

    if lib_list is None:
        lib_list = get_heuristic_lib_order(
            lib_stat_csv=lib_stat_csv,
            epistasis_csv=epistasis_csv,
            loc_csv=loc_csv,
        )

    for metric_idx, fig_name in enumerate(fig_names):

        fig, axs = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)

        for i, ax in enumerate(axs.flatten()):
            lib_name = lib_list[i]

            iflegend = True if i == 7 else False

            plot_single_ftmlde_ensemble(
                mlde_csv=mlde_csv,
                lib_list=[lib_name],
                metric_idx=metric_idx,
                fig_name=lib_name,
                iflegend=iflegend,
                ifzoomy=False,
                n_top=n_top,
                n_corr=n_corr,
                ax=ax,
            )

            if i % 4 != 0:
                ax.set_ylabel("")
            if i < 12:
                ax.set_xlabel("")

        if ifsave:
            save_svg(fig, fig_name, fig_dir)


def plot_ftmlde_ensemble(
    mlde_csv: str,
    lib_list: list,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifzoomy: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv,
        alde_dir="",
        lib_list=lib_list,
        n_top=n_top,
        add_ensemble=True,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 4), sharex=True
    )  # Create a 2x2 grid of subplots

    # Define the data and labels for MLDE and ALDE cases
    options = [
        "MLDE",
        "ftMLDE: EVmutation",
        "ftMLDE: ESM-2",
        "ftMLDE: ESM-IF",
        "ftMLDE: CoVES",
        "ftMLDE: Triad",
    ]

    ensemble_opts = [
        "Triad + EVmutation",
        "Triad + ESM-2",
        "Triad + ESM-IF",
        "CoVES + EVmutation",
        "CoVES + ESM-2",
        "EVmutation + ESM-IF",
    ]

    mlde_color_list = ["gray", "green", "purple", "yellow", "brown", "orange", "blue"]

    if (not ifzoomy) or ("TrpB3A" in lib_list):
        y_mins = [0, 0]
    else:
        y_mins = [0.5, 0]

    # Loop over
    for ax, mlde_metric, y_label, y_min in zip(
        axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS, y_mins
    ):

        for i, mlde_opts in enumerate(options):
            mlde_df = avg_mlde_df_dict[mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                label=mlde_opts.replace("Average ", ""),
                marker="o",
                # linestyle="dotted",
                linewidth=2,
                color=FZL_PALETTE[mlde_color_list[i % len(mlde_color_list)]],
            )

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=FZL_PALETTE[mlde_color_list[i % len(mlde_color_list)]],
                alpha=0.05,
            )

            j = i + 1  # no gray gray

            # plot the ensemble same color but different linestyle and marker
            mlde_df = avg_mlde_df_dict[ensemble_opts[j - 1]]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                # label=mlde_opts.replace("Average ", ""),
                marker="o",
                # markersize=7.5,
                linestyle="dashed",
                linewidth=2,
                color=GRAY_COLORS["gray-" + mlde_color_list[j % len(mlde_color_list)]],
            )

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=GRAY_COLORS["gray-" + mlde_color_list[j % len(mlde_color_list)]],
                alpha=0.05,
            )

        ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
        ax.axvline(96 + n_top, color="gray", linewidth=0.5, linestyle="dotted")
        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(y_min, 1.0)
        ax.set_xscale("log")
        # label the orignial xticks labels
        ax.set_xticks(N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())

        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

    # add first legend to be different zs
    mlde_color_dict = {
        l.replace("ftMLDE: ", "").replace("MLDE", "Random sampling"): FZL_PALETTE[c]
        for (l, c) in zip(options, mlde_color_list)
    }
    ds_color_dict = {
        l.replace("ftMLDE: ", "Triad + "): GRAY_COLORS["gray-" + c]
        for (l, c) in zip(options[1:4], mlde_color_list[1:4])
    }

    ds_color_dict["CoVES + EVmutation"] = GRAY_COLORS["gray-" + mlde_color_list[-3]]
    ds_color_dict["CoVES + ESM-2"] = GRAY_COLORS["gray-" + mlde_color_list[-2]]

    ds_color_dict["EVmutation + ESM-IF"] = GRAY_COLORS["gray-" + mlde_color_list[-1]]
    # Create legend for line colors using the color dictionary
    color_handles = [
        Line2D([0], [0], color=color, lw=2, label=label, marker="o")
        for label, color in mlde_color_dict.items()
    ]

    gray_handles = [
        Line2D([0], [0], color=color, lw=2, label=label, marker="o", linestyle="dashed")
        for label, color in ds_color_dict.items()
    ]

    for h in gray_handles:
        h.set_dashes([6, 2])
    # # Add the legends to the figure
    axes[1].legend(
        handles=color_handles + gray_handles,
        loc="upper left",
        bbox_to_anchor=(1, 1.025),
    )

    # Adjust the layout to prevent overlapping
    plt.tight_layout(h_pad=1.5)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def plot_single_ftalde(
    alde_dir: str,
    lib_list: list,
    metric_idx: int,
    rd: int = 4,
    fig_name: str | None = None,
    iflegend: bool = True,
    ifzoomy: bool = False,
    n_top: int = 96,
    n_corr: int = 384,
    ax=None,
):
    """
    Plot a single FTALDE comparison on a given subplot axis.

    Args:
        metric_idx (int): Index for metric selection.
        ax (matplotlib.axes): Axis to plot on.
    """

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv="", alde_dir=alde_dir, lib_list=lib_list, n_top=n_top
    )

    mlde_metric = PLOT_MLDE_METRICS[metric_idx]
    y_label = PLOT_LINE_PERFORMANCE_YAXIS[metric_idx]

    opt_list = [""] + [ZS_OPTS_LEGEND[z] for z in ZS_OPTS]

    if not ifzoomy or "TrpB3A" in lib_list:
        y_min = 0
    else:
        y_min = 0.5

    options = [f"ALDE x {rd}"] + [f"ftALDE x {rd}: {o}" for o in opt_list[1:]]

    for i, mlde_opts in enumerate(options):
        mlde_df = avg_mlde_df_dict[mlde_opts]

        ax.plot(
            TOTAL_N_LIST,
            mlde_df[f"{mlde_metric}_mean"],
            label=mlde_opts.replace("Average ", ""),
            marker=ALDE_MARKER_STYLES[rd],
            linewidth=2,
            color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
        )

        if metric_idx == 0:
            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"] - mlde_df[f"{mlde_metric}_std"],
                mlde_df[f"{mlde_metric}_mean"] + mlde_df[f"{mlde_metric}_std"],
                color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
                alpha=0.05,
            )

        if i > 1:
            mlde_df = avg_mlde_df_dict["ds-" + mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                marker=ALDE_MARKER_STYLES[rd],
                linestyle="dashed",
                linewidth=2,
                color=GRAY_COLORS[
                    "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                ],
            )

            if metric_idx == 0:
                ax.fill_between(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"] - mlde_df[f"{mlde_metric}_std"],
                    mlde_df[f"{mlde_metric}_mean"] + mlde_df[f"{mlde_metric}_std"],
                    color=GRAY_COLORS[
                        "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                    ],
                    alpha=0.05,
                )

    ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
    ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
    ax.set_ylim(y_min, 1.0)
    ax.set_xscale("log")
    ax.set_xticks(N_TICK_LIST)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("Total number of variants")
    ax.set_ylabel(y_label)

    if iflegend:
        mlde_color_dict = {
            (l if l else "Random sampling"): FZL_PALETTE[c]
            for (l, c) in zip(opt_list, FTMLDE_COLOR_LIST)
        }
        ds_color_dict = {
            "Hamming distance + " + l.replace(" ALDE x 4", ""): GRAY_COLORS["gray-" + c]
            for (l, c) in zip(opt_list[2:], FTMLDE_COLOR_LIST[2:])
        }

        color_handles = [
            Line2D([0], [0], color=color, lw=2, label=label)
            for label, color in mlde_color_dict.items()
        ]
        gray_handles = [
            Line2D([0], [0], color=color, lw=2, label=label, linestyle="dashed")
            for label, color in ds_color_dict.items()
        ]
        for h in gray_handles:
            h.set_dashes([6, 2])

        ax.legend(
            handles=color_handles + gray_handles,
            loc="upper left",
            bbox_to_anchor=(1, 1.025),
        )

    if fig_name is not None:
        ax.set_title(fig_name)

    return ax


def plot_agg_ftalde(
    alde_dir: str,
    lib_stat_csv: str,
    epistasis_csv: str,
    loc_csv: str,
    fig_names: list[str],
    lib_list: list | None = None,
    rd: int = 4,
    n_top: int = 96,
    n_corr: int = 384,
    fig_dir: str = "figs",
    ifsave: bool = True,
):
    """
    Plot aggregated FTALDE across multiple libraries.
    """

    if lib_list is None:
        lib_list = get_heuristic_lib_order(
            lib_stat_csv=lib_stat_csv,
            epistasis_csv=epistasis_csv,
            loc_csv=loc_csv,
        )

    for metric_idx, fig_name in enumerate(fig_names):

        fig, axs = plt.subplots(4, 4, figsize=(18, 16), sharex=True, sharey=True)

        for i, ax in enumerate(axs.flatten()):
            lib_name = lib_list[i]

            iflegend = i == 7  # Show legend only in one subplot

            plot_single_ftalde(
                alde_dir=alde_dir,
                lib_list=[lib_name],
                metric_idx=metric_idx,
                rd=rd,
                fig_name=lib_name,
                iflegend=iflegend,
                ifzoomy=False,
                n_top=n_top,
                n_corr=n_corr,
                ax=ax,
            )

            if i % 4 != 0:
                ax.set_ylabel("")
            if i < 12:
                ax.set_xlabel("")

        if ifsave:
            save_svg(fig, fig_name, fig_dir)


def plot_ftalde(
    alde_dir: str,
    lib_list: list,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    ifzoomy: bool = False,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv="", alde_dir=alde_dir, lib_list=lib_list, n_top=n_top
    )

    fig, axes = plt.subplots(
        3, 2, figsize=(10, 9.6), sharex=True
    )  # Create a 2x2 grid of subplots

    n = [2, 3, 4]
    # Define the data and labels for MLDE and ALDE cases
    opt_list = [""] + [ZS_OPTS_LEGEND[z] for z in ZS_OPTS]
    # ftALDE x 2: Hamming distance

    if not ifzoomy or "TrpB3A" in lib_list:
        y_mins = [0, 0]
    else:
        y_mins = [0.5, 0]

    # Loop over both MLDE and ALDE data in the first and second rows respectively
    for r, rd in enumerate(n):
        options = [f"ALDE x {rd}"] + [f"ftALDE x {rd}: {o}" for o in opt_list[1:]]

        for m, (mlde_metric, y_label, y_min) in enumerate(
            zip(PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS, y_mins)
        ):
            for i, mlde_opts in enumerate(options):
                mlde_df = avg_mlde_df_dict[mlde_opts]

                axes[r, m].plot(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"],
                    label=mlde_opts.replace("Average ", ""),
                    marker=ALDE_MARKER_STYLES[rd],
                    # linestyle="dotted",
                    linewidth=2,
                    color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
                )

                axes[r, m].fill_between(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"] - mlde_df[f"{mlde_metric}_std"],
                    mlde_df[f"{mlde_metric}_mean"] + mlde_df[f"{mlde_metric}_std"],
                    color=FZL_PALETTE[FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]],
                    alpha=0.05,
                )
                if i > 1:
                    # plot the double site same color but different linestyle and marker
                    mlde_df = avg_mlde_df_dict["ds-" + mlde_opts]

                    axes[r, m].plot(
                        TOTAL_N_LIST,
                        mlde_df[f"{mlde_metric}_mean"],
                        label=mlde_opts.replace("Average ", ""),
                        marker=ALDE_MARKER_STYLES[rd],
                        # markersize=7.5,
                        linestyle="dashed",
                        linewidth=2,
                        color=GRAY_COLORS[
                            "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                        ],
                    )

                    axes[r, m].fill_between(
                        TOTAL_N_LIST,
                        mlde_df[f"{mlde_metric}_mean"].values.flatten()
                        - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                        mlde_df[f"{mlde_metric}_mean"].values.flatten()
                        + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                        color=GRAY_COLORS[
                            "gray-" + FTMLDE_COLOR_LIST[i % len(FTMLDE_COLOR_LIST)]
                        ],
                        alpha=0.05,
                    )

            axes[r, m].axvline(
                n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted"
            )
            axes[r, m].set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
            axes[r, m].set_ylim(y_min, 1.0)
            axes[r, m].set_xscale("log")

            # Use FuncFormatter to display the original values on the log-scaled axis
            axes[r, m].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{int(x)}")
            )
            axes[r, m].xaxis.set_minor_locator(plt.NullLocator())

            if r == 2:
                # label the orignial xticks labels
                axes[r, m].set_xticks(N_TICK_LIST)
                axes[r, m].set_xlabel("Total number of variants")

            axes[r, m].set_ylabel(y_label)

    # add first legend to be different zs
    mlde_color_dict = {
        (l if l else "Random sampling"): FZL_PALETTE[c]
        for (l, c) in zip(opt_list, FTMLDE_COLOR_LIST)
    }
    # Create legend for line colors using the color dictionary
    color_handles = [
        Line2D([0], [0], color=color, lw=2, label=label)
        for label, color in mlde_color_dict.items()
    ]
    ds_color_dict = {
        "Hamming distance + " + l.replace(" ALDE x 4", ""): GRAY_COLORS["gray-" + c]
        for (l, c) in zip(opt_list[2:], FTMLDE_COLOR_LIST[2:])
    }

    gray_handles = [
        Line2D([0], [0], color=color, lw=2, label=label, linestyle="dashed")
        for label, color in ds_color_dict.items()
    ]

    marker_style_dict = {f"ALDE x {r}": s for r, s in ALDE_MARKER_STYLES.items()}

    # Create legend for marker styles using the marker dictionary
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            markersize=7,
            label=label,
        )
        for label, marker in marker_style_dict.items()
    ]

    # Add the legends to the figure
    axes[0, 1].legend(
        handles=color_handles + gray_handles,
        loc="upper left",
        bbox_to_anchor=(1, 1.025),
    )
    axes[1, 1].legend(
        handles=marker_handles, loc="upper left", bbox_to_anchor=(1, 1.025)
    )

    # Adjust the layout to prevent overlapping
    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def plot_alde_opt(
    alde_dir: str,
    lib_list: list,
    fig_name: str,
    n_top: int = 96,
    n_corr: int = 384,
    n_rounds: list[int] = [2, 3, 4],
    ifzoomy: bool = False,
    ifsave: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot ALDE with different optimization strategies
    """

    line_styles = ["dashed", "solid"] * len(n_rounds)

    selected_alde_colors = MLDE_ALDE_COLORS[:4] + MLDE_ALDE_COLORS[-2:]

    avg_mlde_df_dict = {}

    for n in n_rounds:
        for alde_model in ALDE_MODEL_OPTS:
            for alde_acq in ALDE_ACQ_OPTS:
                avg_mlde_df_dict[f"{alde_model} {alde_acq} x {n}"] = avg_alde_df(
                    eq_n=n,
                    lib_list=lib_list,
                    zs="",
                    alde_model=alde_model,
                    alde_encoding="onehot",
                    alde_acq=alde_acq,
                    alde_dir=alde_dir,
                )

    fig, axes = plt.subplots(
        3, 2, figsize=(10, 9.6), sharex=True
    )  # Create a 2x2 grid of subplots

    # Define the data and labels for ALDE
    opt_list = [f"{m} {a}" for a in ALDE_ACQ_OPTS for m in ALDE_MODEL_OPTS]

    if not ifzoomy or "TrpB3A" in lib_list:
        y_mins = [0, 0]
    else:
        y_mins = [0.5, 0]

    # Loop over ALDE data
    for r, rd in enumerate(n_rounds):
        options = [f"{o} x {rd}" for o in opt_list]

        for m, (mlde_metric, y_label, y_min) in enumerate(
            zip(PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS, y_mins)
        ):
            for i, mlde_opts in enumerate(options):
                mlde_df = avg_mlde_df_dict[mlde_opts]

                axes[r, m].plot(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"],
                    label=mlde_opts.replace("Average ", ""),
                    marker=ALDE_MARKER_STYLES[rd],
                    linestyle=line_styles[i],
                    linewidth=2,
                    color=selected_alde_colors[i],
                )

                axes[r, m].fill_between(
                    TOTAL_N_LIST,
                    mlde_df[f"{mlde_metric}_mean"] - mlde_df[f"{mlde_metric}_std"],
                    mlde_df[f"{mlde_metric}_mean"] + mlde_df[f"{mlde_metric}_std"],
                    color=selected_alde_colors[i],
                    alpha=0.05,
                )

            axes[r, m].axvline(
                n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted"
            )
            axes[r, m].set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
            axes[r, m].set_ylim(y_min, 1.0)
            axes[r, m].set_xscale("log")

            # Use FuncFormatter to display the original values on the log-scaled axis
            axes[r, m].xaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{int(x)}")
            )
            axes[r, m].xaxis.set_minor_locator(plt.NullLocator())

            if r == 2:
                # label the orignial xticks labels
                axes[r, m].set_xticks(N_TICK_LIST)
                axes[r, m].set_xlabel("Total number of variants")

            axes[r, m].set_ylabel(y_label)

    # add first legend to be different alde opts
    # also reorg so that all the odd ones first then all the even ones
    mlde_color_dict = {l: c for (l, c) in zip(opt_list, selected_alde_colors)}

    # Create legend for line colors using the color dictionary
    color_handles = [
        Line2D([0], [0], color=color, lw=2, label=label, linestyle=line_styles[i])
        for i, (label, color) in enumerate(mlde_color_dict.items())
    ]

    # sort color handles so that the odd ones are first
    color_handles = sorted(
        color_handles, key=lambda x: (color_handles.index(x) % 2 == 1)
    )

    # Create alde legend rounds
    marker_style_dict = {f"ALDE x {r}": s for r, s in ALDE_MARKER_STYLES.items()}

    # Create legend for marker styles using the marker dictionary
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="black",
            linestyle="None",
            markersize=7,
            label=label,
        )
        for label, marker in marker_style_dict.items()
    ]

    # Add the legends to the figure
    axes[0, 1].legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(1, 1.025),
    )
    axes[1, 1].legend(
        handles=marker_handles, loc="upper left", bbox_to_anchor=(1, 1.025)
    )

    # Adjust the layout to prevent overlapping
    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def plot_finetune(
    mlde_csv: str,
    alde_dir: str,
    finetune_dir: str,
    lib_list: list,
    fig_name: str,
    zs: str = "ev_score",
    n_top: int = 96,
    n_corr: int = 384,
    ifzoomy: bool = False,
    ifsave: bool = True,
    fig_dir: str = "figs",
):
    """
    Plot finetune results together with
    MLDE, ftMLDE: EVmutation, (boosting, one-hot)
    ALDE x 2, ftALDE x 2: EVmutation, (boosting, one-hot, greedy)
    ALDE x 3, ftALDE x 3: EVmutation,
    ALDE x 4, ftALDE x 4: EVmutation,
    Fine-tuning, ftFine-tuning: EVmutation

    Args:
        mlde_csv (str): Path to the MLDE CSV file.
        alde_dir (str): Path to the ALDE directory.
        finetune_dir (str): Path to the fine-tuning directory.
            Assume containing `ev` and `none` subfolder
        lib_list (list): List of libraries.
        n_sample (int): Number of samples.
        fig_name (str): Name of the figure to save.
        models (list, optional): List of models. Defaults to ["boosting"].
        n_top (int, optional): Number of top mutations. Defaults to 96.
        zs (str, optional): ZS option. Defaults to "ev_score".
        ifsave (bool, optional): Whether to save the figure. Defaults to True.
    """

    avg_mlde_df_dict = {}

    if mlde_csv != "":

        mlde_all = pd.read_csv(mlde_csv)

        mlde_kwargs = dict(
            mlde_all=mlde_all,
            n_top=n_top,
            lib_list=lib_list,
            ft_frac=0.125,
        )

        # just mlde
        avg_mlde_df_dict["MLDE"] = get_mlde_avg_sdf(
            n_mut_cutoff="all", zs="none", **mlde_kwargs
        )

        avg_mlde_df_dict[f"ftMLDE: {ZS_OPTS_LEGEND[zs]}"] = get_mlde_avg_sdf(
            n_mut_cutoff="all", zs=zs, **mlde_kwargs
        )

    if alde_dir != "":
        # now add alde
        alde_kwags = dict(
            lib_list=lib_list,
            alde_model="Boosting Ensemble",
            alde_encoding="onehot",
            alde_acq="GREEDY",
            alde_dir=alde_dir,
        )

        for eq_n in [2, 3, 4]:

            avg_mlde_df_dict[f"ALDE x {eq_n}"] = avg_alde_df(eq_n, **alde_kwags)

            # add ev
            avg_mlde_df_dict[f"ftALDE x {eq_n}: {ZS_OPTS_LEGEND[zs]}"] = avg_alde_df(
                eq_n, zs=zs.replace("_score", ""), **alde_kwags
            )

    # get finetune df
    if finetune_dir != "":
        finetune_kwargs = dict(
            lib_list=lib_list,
            n_top=n_top,
        )

        avg_mlde_df_dict["Fine-tuning"] = avg_finetune_df(
            parse_finetune_df(
                finetune_dir=os.path.join(
                    finetune_dir, "none"
                ),  # ie results/finetuning/ev or none
                **finetune_kwargs,
            ),
            N_SAMPLE_LIST,
        )

        avg_mlde_df_dict[f"ftFine-tuning: {ZS_OPTS_LEGEND[zs]}"] = avg_finetune_df(
            parse_finetune_df(
                finetune_dir=os.path.join(
                    finetune_dir, zs.replace("_score", "")
                ),  # ie results/finetuning/ev or none
                **finetune_kwargs,
            ),
            N_SAMPLE_LIST,
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True)

    mlde_color_list = MLDE_ALDE_COLORS + FINETUNE_COLORS

    # Define the data and labels for MLDE and ALDE cases
    options = [
        "MLDE",
        "ftMLDE: EVmutation",
        "ALDE x 2",
        "ftALDE x 2: EVmutation",
        "ALDE x 3",
        "ftALDE x 3: EVmutation",
        "ALDE x 4",
        "ftALDE x 4: EVmutation",
        "Fine-tuning",
        f"ftFine-tuning: {ZS_OPTS_LEGEND[zs]}",
    ]

    line_styles = (
        [
            "dashed",
            "solid",
        ]
        + [
            "dashed",
            "solid",
        ]
        * 3
        + ["dashed", "solid"]
    )

    if ifzoomy:
        y_mins = [0.5, 0]
    else:
        y_mins = [0, 0]

    # Loop over

    for ax, mlde_metric, y_label, y_min in zip(
        axes, PLOT_MLDE_METRICS, PLOT_LINE_PERFORMANCE_YAXIS, y_mins
    ):

        for i, mlde_opts in enumerate(options):
            mlde_df = avg_mlde_df_dict[mlde_opts]

            ax.plot(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"],
                label=mlde_opts.replace("Average ", ""),
                marker="o",
                linewidth=2,
                linestyle=line_styles[i],
                color=mlde_color_list[i],
            )

            ax.fill_between(
                TOTAL_N_LIST,
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                - mlde_df[f"{mlde_metric}_std"].values.flatten(),
                mlde_df[f"{mlde_metric}_mean"].values.flatten()
                + mlde_df[f"{mlde_metric}_std"].values.flatten(),
                color=mlde_color_list[i],
                alpha=0.05,
            )

        ax.axvline(n_corr + n_top, color="gray", linewidth=0.5, linestyle="dotted")
        ax.set_xlim(TOTAL_N_LIST[0], TOTAL_N_LIST[-1])
        ax.set_ylim(y_min, 1.0)
        ax.set_xscale("log")
        # label the orignial xticks labels
        ax.set_xticks(N_TICK_LIST)

        # Use FuncFormatter to display the original values on the log-scaled axis
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel("Total number of variants")
        ax.set_ylabel(y_label)

        if ax == axes[1]:
            # Get the handles and labels from the legend
            handles, labels = ax.get_legend_handles_labels()
            # fix the dash handle
            [
                handles[index].set_dashes([6, 2])
                for index, value in enumerate(line_styles)
                if value == "dashed"
            ]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1.025))

    # Adjust the layout to prevent overlapping
    plt.tight_layout(h_pad=1.5)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def get_ftmlde_stat(
    mlde_csv: str,
    lib_list: list,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top: int = 96,
) -> pd.DataFrame:

    """
    Get the MLDE statistics for each landscape given number of samples
    """

    mlde_df = pd.read_csv(mlde_csv)

    # get mlde average
    mlde_avg = (
        mlde_df[
            (mlde_df["zs"] == "none")
            & (mlde_df["encoding"] == "one-hot")
            & (mlde_df["model"].isin(models))
            & (mlde_df["n_mut_cutoff"] == "all")
            & (mlde_df["n_sample"] == n_sample)
            & (mlde_df["n_top"] == n_top)
            & (mlde_df["lib"].isin(lib_list))
        ][["lib", "top_maxes", "if_truemaxs"]]
        .groupby("lib")
        .mean()
    )

    for z, zs in enumerate(ZS_OPTS):
        rename_cols = {
            "top_maxes": "top_maxes_" + zs.replace("_score", ""),
            "if_truemaxs": "if_truemaxs_" + zs.replace("_score", ""),
        }

        # for hamming distance
        if z == 0:
            mlde_avg = pd.merge(
                mlde_avg,
                (
                    mlde_df[
                        (mlde_df["zs"] == "none")
                        & (mlde_df["encoding"] == "one-hot")
                        & (mlde_df["model"].isin(models))
                        & (mlde_df["n_mut_cutoff"] == "double")
                        & (mlde_df["n_sample"] == n_sample)
                        & (mlde_df["n_top"] == n_top)
                        & (mlde_df["lib"].isin(lib_list))
                    ][
                        [
                            "lib",
                            "top_maxes",
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

            slice_ftmlde = mlde_df[
                (mlde_df["zs"] == zs)
                & (mlde_df["encoding"] == "one-hot")
                & (mlde_df["model"].isin(models))
                & (mlde_df["n_mut_cutoff"] == "all")
                & (mlde_df["n_sample"] == n_sample)
                & (mlde_df["n_top"] == n_top)
                & (mlde_df["lib"].isin(lib_list))
                & (mlde_df["ft_lib"].isin([0.125 * 20 ** 3, 0.125 * 20 ** 4]))
            ].copy()

            mlde_avg = pd.merge(
                mlde_avg,
                (
                    slice_ftmlde[
                        [
                            "lib",
                            "top_maxes",
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

    return mlde_avg.reset_index().copy()


# Helper function to create scatter plots with common settings
def scatter_plot(
    ax, x_data, y_data, y_data2, x_label, y_label, title_label, clist, xlabel_scale=None
):
    ax.scatter(
        x_data,
        y_data,
        edgecolors=clist,
        facecolors="none",
        s=100,
        alpha=0.8,
        linewidth=1.2,
    )
    ax.scatter(
        x_data,
        y_data2,
        c=clist,
        marker="X",
        linewidth=1.2,
        s=100,
    )
    ax.set_xlabel(x_label)
    if xlabel_scale:
        ax.set_xscale(xlabel_scale)
    ax.set_title(title_label)
    if y_label != "":
        ax.set_ylabel(y_label)


# Helper function to create scatter plots with common settings
def scatter_plot2(
    ax,
    x_data,
    y_data,
    y_data2,
    y_data3,
    x_label,
    y_label,
    title_label,
    clist,
    xlabel_scale=None,
):
    ax.scatter(
        x_data,
        y_data,
        c=clist,
        marker="x",
        s=100,
        alpha=0.8,
        linewidth=1.2,
    )

    ax.scatter(
        x_data,
        y_data2,
        marker="o",
        edgecolors=clist,
        facecolors="none",
        s=100,
        alpha=0.8,
        linewidth=1.2,
    )
    ax.scatter(
        x_data,
        y_data3,
        c=clist,
        marker="X",
        linewidth=1.2,
        s=100,
    )
    ax.set_xlabel(x_label)
    if xlabel_scale:
        ax.set_xscale(xlabel_scale)
    ax.set_title(title_label)
    if y_label != "":
        ax.set_ylabel(y_label)


# Helper function to create the title with spearman correlation
def create_spearman_title(data1, data2, y_data, y_data2, label):
    return r"$\rho_{{DE}}$: {:.2f}, $\rho_{{MLDE}}$: {:.2f}, $\rho_{{ftMLDE}}$: {:.2f}".format(
        spearmanr(data1, data2)[0],
        spearmanr(y_data, data2)[0],
        spearmanr(y_data2, data2)[0],
    )


def plot_mlde_attribute_corr(
    mlde_csv: str,
    attribute_csv: str,
    lib_list: list,
    fig_name: str,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top=96,
    add_landscape_legend: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    merge_mldedf = get_ftmlde_stat(
        mlde_csv=mlde_csv,
        lib_list=lib_list,
        models=models,
        n_sample=n_sample,
        n_top=n_top,
    )

    pooled_ft = merge_mldedf[merge_mldedf["lib"].isin(lib_list)][
        ["top_maxes_" + zs.replace("_score", "") for zs in ZS_OPTS]
    ].mean(axis=1, skipna=True)

    # Load and filter data
    all_landscape_attribute = pd.read_csv(attribute_csv)
    landscape_attribute_df = (
        all_landscape_attribute[all_landscape_attribute["lib"].isin(lib_list)]
        .reset_index(drop=True)
        .sort_values("lib")
    )

    single_step_de = landscape_attribute_df["single_step_DE_mean_all"]

    # Create figure and axes
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.2), sharey=True)

    # Define y and y2 values
    y = merge_mldedf["top_maxes"].values - single_step_de.values
    y2 = pooled_ft.values - single_step_de.values
    clist = glasbey_category10[: len(lib_list)]

    # Scatter plot data and corresponding labels
    plot_data = [
        ("percent_active", "Percent active", "log"),
        ("frac_loc_opt_total", "Fraction of local optima", "log"),
        ("fraction_non-magnitude", "Fraction of non-magnitude epistasis", None),
        ("loc", "Cauchy peak location", "symlog"),
        ("kurt", "Kurtosis (tailedness)", "symlog"),
        ("numb_kde_peak", "Number of KDE peaks", None),
    ]

    # Iterate through the axes and plot each
    for i, (x_col, x_label, scale) in enumerate(plot_data):
        if i % 3 == 0:
            y_label = "Average max fitness improvement"
        else:
            y_label = ""

        row, col = divmod(i, 3)
        scatter_plot(
            ax[row, col],
            landscape_attribute_df[x_col],
            # single_step_de,
            y,
            y2,
            x_label,
            y_label,
            create_spearman_title(
                single_step_de,
                landscape_attribute_df[x_col],
                y,
                y2,
                x_col,
            ),
            clist,
            xlabel_scale=scale,
        )

    # Formatting the Cauchy peak location axis (symlog)
    ax[1, 0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Create legends for landscapes and markers
    legend_list = [
        Line2D([0], [0], marker="o", linestyle="none", color=lc, label=l, alpha=0.8)
        for l, lc in zip(landscape_attribute_df["lib"], clist)
    ]

    legend_list2 = [
        # Line2D(
        #     [0],
        #     [0],
        #     marker="x",
        #     linestyle="none",
        #     alpha=0.8,
        #     label="Single-step DE",
        #     markeredgecolor="black",
        #     markerfacecolor="black",
        # ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            alpha=0.8,
            label="MLDE over DE",
            markeredgecolor="black",
            markerfacecolor="none",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            alpha=0.8,
            label="ftMLDE over DE",
            markeredgecolor="black",
            markerfacecolor="black",
        ),
    ]

    # Add legends
    if add_landscape_legend:
        ax[1, 2].legend(
            handles=legend_list,
            loc="upper left",
            title="Landscapes",
            bbox_to_anchor=(1, 1.0425),
        )

    ax[0, 2].legend(
        handles=legend_list2,
        loc="upper left",
        title="MLDE or ftMLDE",
        bbox_to_anchor=(1, 1.0315),
    )

    # Show plot
    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def slice_alde_ftalde(n: int, alde_all: pd.DataFrame) -> pd.DataFrame:
    sliced_alde = alde_all[
        (alde_all["rounds"] == n)
        & (alde_all["Encoding"] == "onehot")
        & (alde_all["Model"] == "Boosting Ensemble")
        & (alde_all["Acquisition"] == "GREEDY")
        & (alde_all["n_samples"] == 384)
        & (alde_all["n_mut_cutoff"] == "all")
    ]

    ftalde = (
        sliced_alde[sliced_alde["zs"] != ""][["Protein", "Mean", "Frac"]]
        .groupby("Protein")
        .mean()
        .reset_index()
        .rename(
            columns={
                "Protein": "lib",
                "Mean": f"zs_top_max_{n}",
                "Frac": f"zs_frac_{n}",
            }
        )
    )

    alde = (
        sliced_alde[sliced_alde["zs"].isna()][["Protein", "Mean", "Frac"]]
        .groupby("Protein")
        .mean()
        .reset_index()
        .rename(columns={"Protein": "lib", "Mean": f"top_max_{n}", "Frac": f"frac_{n}"})
    )
    return pd.merge(alde, ftalde, on="lib")


def plot_alde_attribute_corr(
    alde_csv: str,
    attribute_csv: str,
    lib_list: list,
    fig_name: str,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top=96,
    add_landscape_legend: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    landscape_attribute_df = pd.read_csv(attribute_csv)
    alde_all = pd.read_csv(alde_csv)

    # prep merge alde
    alde_dfs = [slice_alde_ftalde(n=i, alde_all=alde_all) for i in [4, 3, 2]]

    # Perform the merge step-by-step
    alde_n = landscape_attribute_df
    for df in alde_dfs:
        alde_n = pd.merge(alde_n, df, on="lib")

    merge_df = alde_n[alde_n["lib"].isin(lib_list)].sort_values("lib")

    single_step_de = merge_df["single_step_DE_mean_all"]

    # Prepare data
    alde_2 = merge_df["top_max_2"] - single_step_de
    alde_3 = merge_df["top_max_3"] - single_step_de
    alde_4 = merge_df["top_max_4"] - single_step_de
    ftalde_2 = merge_df["zs_top_max_2"] - single_step_de
    ftalde_3 = merge_df["zs_top_max_3"] - single_step_de
    ftalde_4 = merge_df["zs_top_max_4"] - single_step_de

    clist = glasbey_category10[:12]

    # Updated style dictionary
    style_dict = {
        "ALDE x 2": {"facecolors": "none", "marker": "o"},
        "ALDE x 3": {"facecolors": "none", "marker": "s"},
        "ALDE x 4": {"facecolors": "none", "marker": "X"},
        "ftALDE x 2": {"facecolors": clist, "marker": "o", "alpha": 0.5},
        "ftALDE x 3": {"facecolors": clist, "marker": "s", "alpha": 0.5},
        "ftALDE x 4": {"facecolors": clist, "marker": "X", "alpha": 0.5},
    }

    # Define a reusable function for scatter plots and titles
    def create_alde_scatter(
        ax, x_data, y_data, xlabel, title_fmt, x_scale="linear", y_label=None
    ):
        for y, label in zip(
            [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
            [
                "ALDE x 2",
                "ALDE x 3",
                "ALDE x 4",
                "ftALDE x 2",
                "ftALDE x 3",
                "ftALDE x 4",
            ],
        ):
            ax.scatter(
                x_data, y, edgecolors=clist, s=100, linewidth=1.2, **style_dict[label]
            )

        ax.set_xscale(x_scale)
        ax.set_xlabel(xlabel)
        if y_label:
            ax.set_ylabel(y_label)
        # Title with two lines including both MLDE/ftMLDE and ALDE/ftALDE correlations
        ax.set_title(
            (
                title_fmt.format(
                    spearmanr(merge_df["single_step_DE_mean_all"], x_data)[0],
                    spearmanr(alde_2, x_data)[0],
                    spearmanr(ftalde_2, x_data)[0],
                )
                + "\n"
                + r"$\rho_{{ALDE x 3}}$: {:.2f}, $\rho_{{ftALDE x 3}}$: {:.2f}".format(
                    spearmanr(alde_3, x_data)[0], spearmanr(ftalde_3, x_data)[0]
                )
                + "\n"
                + r"$\rho_{{ALDE x 4}}$: {:.2f}, $\rho_{{ftALDE x 4}}$: {:.2f}".format(
                    spearmanr(alde_4, x_data)[0], spearmanr(ftalde_4, x_data)[0]
                )
            )
        )

    # Create figure and axes
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), sharey=True)

    # Scatter plots
    create_alde_scatter(
        ax[0, 0],
        merge_df["percent_active"],
        [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
        "Percent active",
        r"$\rho_{{DE}}$: {:.2f}, $\rho_{{ALDE x 2}}$: {:.2f}, $\rho_{{ftALDE x 2}}$: {:.2f}",
        x_scale="log",
        y_label="Average max fitness improvement",
    )

    create_alde_scatter(
        ax[0, 1],
        merge_df["frac_loc_opt_total"],
        [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
        "Fraction of local optima",
        r"$\rho_{{DE}}$: {:.2f}, $\rho_{{ALDE x 2}}$: {:.2f}, $\rho_{{ftALDE x 2}}$: {:.2f}",
        x_scale="log",
    )

    create_alde_scatter(
        ax[0, 2],
        merge_df["fraction_non-magnitude"],
        [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
        "Fraction of non-magnitude epistasis",
        r"$\rho_{{DE}}$: {:.2f}, $\rho_{{ALDE x 2}}$: {:.2f}, $\rho_{{ftALDE x 2}}$: {:.2f}",
    )

    create_alde_scatter(
        ax[1, 0],
        merge_df["loc"],
        [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
        "Cauchy peak location",
        r"$\rho_{{DE}}$: {:.2f}, $\rho_{{ALDE x 2}}$: {:.2f}, $\rho_{{ftALDE x 2}}$: {:.2f}",
        x_scale="symlog",
        y_label="Average max fitness improvement",
    )
    ax[1, 0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    create_alde_scatter(
        ax[1, 1],
        merge_df["kurt"],
        [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
        "Kurtosis (tailedness)",
        r"$\rho_{{DE}}$: {:.2f}, $\rho_{{ALDE x 2}}$: {:.2f}, $\rho_{{ftALDE x 2}}$: {:.2f}",
        x_scale="symlog",
    )

    create_alde_scatter(
        ax[1, 2],
        merge_df["numb_kde_peak"],
        [alde_2, alde_3, alde_4, ftalde_2, ftalde_3, ftalde_4],
        "Number of KDE peaks",
        r"$\rho_{{DE}}$: {:.2f}, $\rho_{{ALDE x 2}}$: {:.2f}, $\rho_{{ftALDE x 2}}$: {:.2f}",
    )

    # Legends
    legend_list = [
        Line2D([0], [0], marker="o", linestyle="none", color=lc, label=l, alpha=0.8)
        for l, lc in zip(merge_df["lib"], clist)
    ]
    ax[0, 2].legend(
        handles=legend_list,
        loc="upper left",
        title="Landscapes",
        bbox_to_anchor=(1, 1.0325),
    )

    # Updated legend_list2 using new style_dict
    legend_list2 = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=style_dict[label]["marker"],
            markerfacecolor=(
                "black" if style_dict[label]["facecolors"] != "none" else "none"
            ),
            markeredgecolor="black",
            alpha=0.5,
            label=label,
        )
        for label in [
            "ALDE x 2",
            "ALDE x 3",
            "ALDE x 4",
            "ftALDE x 2",
            "ftALDE x 3",
            "ftALDE x 4",
        ]
    ]
    ax[1, 2].legend(
        handles=legend_list2,
        loc="upper left",
        title="Improvement over DE",
        bbox_to_anchor=(1, 1.0315),
    )
    plt.tight_layout()

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def get_demlalft_attribute_stats(
    mlde_csv: str,
    alde_csv: str,
    attribute_csv: str,
    lib_list: list,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top: int = 96,
) -> pd.DataFrame:

    """
    Prep for getting correlation and fold stats
    """

    mlde_df = get_ftmlde_stat(
        mlde_csv=mlde_csv,
        lib_list=lib_list,
        models=models,
        n_sample=n_sample,
        n_top=n_top,
    ).sort_values("lib")

    pooled_ft = mlde_df[mlde_df["lib"].isin(lib_list)][
        ["top_maxes_" + zs.replace("_score", "") for zs in ZS_OPTS]
    ].mean(axis=1, skipna=True)

    # Load and filter data
    all_landscape_attribute = pd.read_csv(attribute_csv)
    landscape_attribute_df = (
        all_landscape_attribute[all_landscape_attribute["lib"].isin(lib_list)]
        .reset_index(drop=True)
        .sort_values("lib")
    )

    # merge landscape_attribute_df with mlde_df and pooled_ft
    mlft_df = mlde_df[["lib", "top_maxes"]].rename(columns={"top_maxes": "MLDE"})
    # add ftmlde
    mlft_df["ftMLDE"] = pooled_ft.values

    # merge with landscape_attribute_df
    merged_df = pd.merge(
        landscape_attribute_df,
        mlft_df,
        on="lib",
        how="outer",
    )

    # now add alde
    alde_all = pd.read_csv(alde_csv)

    # prep merge alde
    alde_dfs = [slice_alde_ftalde(n=i, alde_all=alde_all) for i in [4, 3, 2]]

    # Perform the merge step-by-step
    alde_n = landscape_attribute_df
    for df in alde_dfs:
        alde_n = pd.merge(alde_n, df, on="lib")

    # Ensure columns are properly filtered and renamed
    merge_aldedf = alde_n[alde_n["lib"].isin(lib_list)].sort_values("lib")

    # Select only "top_max" columns and "lib"
    selected_columns = [c for c in merge_aldedf.columns if "top_max" in c or c == "lib"]

    # Apply selection to the DataFrame
    merge_aldedf = merge_aldedf[selected_columns].rename(
        columns={
            "top_max_2": "ALDE",
            "zs_top_max_2": "ftALDE",
            "top_max_3": "ALDE x 3",
            "zs_top_max_3": "ftALDE x 3",
            "top_max_4": "ALDE x 4",
            "zs_top_max_4": "ftALDE x 4",
        }
    )[["lib", "ALDE", "ftALDE", "ALDE x 3", "ftALDE x 3", "ALDE x 4", "ftALDE x 4"]]

    # merge all dataframes
    return (
        pd.merge(
            merged_df,
            merge_aldedf,
            on="lib",
            how="outer",
        )
        .sort_values("lib")
        .reset_index(drop=True)
        .rename(
            columns={
                "lib": "Landscape",
                "n_site": "Number of sites",
                "single_step_DE_mean_all": "Single-step DE",
                **ATTRIBUTE_MAPPING,
            }
        )
    )


def get_attribute_corr(
    mlde_csv: str | None = None,
    alde_csv: str | None = None,
    attribute_csv: str | None = None,
    lib_list: list | None = None,
    df: pd.DataFrame | None = None,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top: int = 96,
) -> pd.DataFrame:

    """
    Get the correlation stats for each landscape given number of samples
    """

    if df is None:
        assert (
            mlde_csv is not None
            and alde_csv is not None
            and attribute_csv is not None
            and lib_list is not None
        ), "Either df or all other arguments must be provided"
        df = (
            get_demlalft_attribute_stats(
                mlde_csv=mlde_csv,
                alde_csv=alde_csv,
                attribute_csv=attribute_csv,
                lib_list=lib_list,
                models=models,
                n_sample=n_sample,
                n_top=n_top,
            )
            .set_index("Landscape")
            .copy()
        )

    # Get the correlation stats
    # each row for
    # Percent active, Fraction of local optima
    # Fraction of non-magnitude epistasis, Cauchy peak location,
    # Kurtosis (tailedness), Number of KDE peaks
    # each column for Single-step DE	MLDE	ftMLDE	ALDE	ftALDE	ALDE x 3	ftALDE x 3	ALDE x 4	ftALDE x 4
    # Get the correlation stats

    # Initialize an empty dictionary to store correlations
    corr_dict = {col: [] for col in ATTRIBUTE_LIST}
    method_list = [
        "Single-step DE",
        "MLDE",
        "ftMLDE",
        "ALDE",
        "ftALDE",
        "ALDE x 3",
        "ftALDE x 3",
        "ALDE x 4",
        "ftALDE x 4",
    ]

    method_rename = []

    # Compute Spearman correlation for each attribute against each method
    for col2 in method_list:

        if col2 != "Single-step DE":
            val2 = df[col2].values - df["Single-step DE"].values
            method_rename.append(col2 + " over DE")
        else:
            val2 = df[col2].values
            method_rename.append(col2)

        for col in ATTRIBUTE_LIST:
            corr_dict[col].append(spearmanr(df[col].values, val2)[0])

    # Convert dictionary to DataFrame
    return pd.DataFrame(corr_dict, index=method_rename).T.round(2)


def get_fold_stats_simple(
    mlde_csv: str | None = None,
    alde_csv: str | None = None,
    attribute_csv: str | None = None,
    lib_list: list | None = None,
    df: pd.DataFrame | None = None,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top: int = 96,
) -> pd.DataFrame:
    """
    Compute fold changes for MLDE, ALDE, and focused training strategies.
    """

    if df is None:

        assert (
            mlde_csv is not None
            and alde_csv is not None
            and attribute_csv is not None
            and lib_list is not None
        ), "Either stats_df or all other arguments must be provided"

        # Get preprocessed statistics
        df = get_demlalft_attribute_stats(
            mlde_csv=mlde_csv,
            alde_csv=alde_csv,
            attribute_csv=attribute_csv,
            lib_list=lib_list,
            models=models,
            n_sample=n_sample,
            n_top=n_top,
        )

    df = df.set_index("Landscape").copy()

    # Extract necessary columns
    single_step_de = df["Single-step DE"]

    return df[MLAL_ORDER].div(single_step_de, axis=0).round(2)


def get_fold_stats(
    mlde_csv: str,
    alde_csv: str,
    attribute_csv: str,
    lib_list: list,
    models: list = ["boosting"],
    n_sample: int = 384,
    n_top: int = 96,
):
    merge_mldedf = get_ftmlde_stat(
        mlde_csv=mlde_csv,
        lib_list=lib_list,
        models=models,
        n_sample=n_sample,
        n_top=n_top,
    ).sort_values("lib")

    pooled_ft = merge_mldedf[merge_mldedf["lib"].isin(lib_list)][
        ["top_maxes_" + zs.replace("_score", "") for zs in ZS_OPTS]
    ].mean(axis=1, skipna=True)

    # Load and filter data
    all_landscape_attribute = pd.read_csv(attribute_csv)
    landscape_attribute_df = (
        all_landscape_attribute[all_landscape_attribute["lib"].isin(lib_list)]
        .reset_index(drop=True)
        .sort_values("lib")
    )

    single_step_de = landscape_attribute_df["single_step_DE_mean_all"]

    # Combine arrays into a DataFrame
    fold_df = pd.DataFrame(
        {
            "MLDE": merge_mldedf["top_maxes"].values / single_step_de.values,
            "ftMLDE": pooled_ft.values / single_step_de.values,
        }
    )

    # Add row index (if needed)
    fold_df.index = merge_mldedf["lib"]

    # now add alde
    alde_all = pd.read_csv(alde_csv)

    # prep merge alde
    alde_dfs = [slice_alde_ftalde(n=i, alde_all=alde_all) for i in [4, 3, 2]]

    # Perform the merge step-by-step
    alde_n = landscape_attribute_df
    for df in alde_dfs:
        alde_n = pd.merge(alde_n, df, on="lib")

    merge_df = alde_n[alde_n["lib"].isin(lib_list)].sort_values("lib")

    alde_fold_df = pd.DataFrame(
        {
            "ALDE": merge_df["top_max_2"] / single_step_de.values,
            "ftALDE": merge_df["zs_top_max_2"] / single_step_de.values,
            "ALDE x 3": merge_df["top_max_3"] / single_step_de.values,
            "ftALDE x 3": merge_df["zs_top_max_3"] / single_step_de.values,
            "ALDE x 4": merge_df["top_max_4"] / single_step_de.values,
            "ftALDE x 4": merge_df["zs_top_max_4"] / single_step_de.values,
        }
    )

    alde_fold_df.index = merge_df["lib"]

    return pd.concat([fold_df, alde_fold_df], axis=1, ignore_index=False).round(2)


def slice_lib_mlde(
    mlde_csv: str,
    lib_list: list,
    n_sample: int,
    models: list = ["boosting"],
    n_top: int = 96,
):

    mlde_df = pd.read_csv(mlde_csv)

    common_slice = mlde_df[
        (mlde_df["encoding"] == "one-hot")
        & (mlde_df["model"].isin(models))
        & (mlde_df["n_sample"] == n_sample)
        & (mlde_df["n_top"] == n_top)
        & (mlde_df["lib"].isin(lib_list))
    ]

    noft_df = common_slice[
        (common_slice["zs"] == "none") & (common_slice["n_mut_cutoff"] == "all")
    ]

    no_combo_df = common_slice[
        (common_slice["n_mut_cutoff"] == "all")
        & (common_slice["zs"].isin(ZS_OPTS[1:]))
        & (common_slice["ft_lib"].isin([0.125 * 20 ** 3, 0.125 * 20 ** 4]))
    ].copy()

    ds_only_df = common_slice[
        (common_slice["n_mut_cutoff"] == "double") & (common_slice["zs"] == "none")
    ].copy()
    ds_only_df["zs"] = ds_only_df["zs"].replace({"none": "ed_score"})

    ds_comb_df = common_slice[
        (common_slice["n_mut_cutoff"] == "double")
        & (common_slice["zs"].isin(["ev_score", "esmif_score", "esm_score"]))
        & (common_slice["ft_lib"].isin([0.125 * 3 * 20 ** 2, 0.125 * 6 * 20 ** 2]))
    ].copy()

    ds_comb_df["zs"] = ds_comb_df["zs"].replace(
        {
            "ev_score": "ds-ev",
            "esmif_score": "ds-esmif",
            "esm_score": "ds-esm",
        }
    )

    slice_df = pd.concat(
        [noft_df, ds_only_df, no_combo_df, ds_comb_df], ignore_index=True
    )

    # Convert 'Category' column to categorical with defined order
    slice_df["zs"] = pd.Categorical(
        slice_df["zs"],
        categories=["none"]
        + ZS_OPTS
        + [
            "ds-ev",
            "ds-esmif",
            "ds-esm",
        ],
        ordered=True,
    )

    slice_df = (
        slice_df[["lib", "zs", "top_maxes", "if_truemaxs"]]
        .groupby(["lib", "zs"])
        .mean()
        .reset_index()
        .sort_values(by=["zs", "lib"])
    )

    slice_df["type"] = slice_df["lib"].map(lambda x: LIB_INFO_DICT[x]["type"])

    return slice_df


def plot_mlde_type(
    mlde_csv: str,
    lib_list: list,
    n_sample: int,
    fig_name: str,
    ymin1: float = -0.15,
    ymax1: float = 0.5,
    ymin2: float = -0.8,
    ymax2: float = 1.0,
    h_pad: float = 0,
    models: list = ["boosting"],
    n_top: int = 96,
    ifpad_y1: bool = False,
    set_ymin2_scale: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    slice_df = slice_lib_mlde(
        mlde_csv=mlde_csv,
        lib_list=lib_list,
        n_sample=n_sample,
        models=models,
        n_top=n_top,
    )

    fig, axes = plt.subplots(2, 9, figsize=(8.4, 6.4))

    for i, x in enumerate(PLOT_MLDE_METRICS):

        rand_df = (
            slice_df[slice_df["zs"] == "none"][["lib", "type", x]]
            .reset_index(drop=True)
            .copy()
        )

        for z, zs in enumerate(
            ZS_OPTS
            + [
                "ds-ev",
                "ds-esmif",
                "ds-esm",
            ]
        ):

            ax = axes[i, z]
            bar_type_df = (
                slice_df[slice_df["zs"] == zs][["lib", "type", x]]
                .reset_index(drop=True)
                .copy()
            )

            # subtract random
            merg_df = pd.merge(bar_type_df, rand_df, on=["lib", "type"], how="outer")
            merg_df["delta"] = merg_df[x + "_x"] - merg_df[x + "_y"]

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
                y="delta",
                data=merg_df,
                width=bar_width,
                ax=ax,
                order=bar_order,
                boxprops={
                    "facecolor": "None",
                    "edgecolor": FZL_PALETTE["gray"],
                },
            )

            sns.stripplot(
                x="type",
                y="delta",
                data=merg_df,
                order=bar_order,
                hue="lib",
                hue_order=merg_df["lib"].unique(),
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

            if i == 0:
                ymax = ymax1
                ax.set_xlabel("")
                ax.set_xticks([])  # Removes x-axis ticks
                ax.set_ylim(ymin1, ymax1)
            else:
                ymax = ymax2
                ax.set_xlabel("")
                ax.set_ylim(ymin2, ymax2)
                ax.set_xticklabels(labels, rotation=90, ha="center")
                if set_ymin2_scale:
                    ax.set_yticks([-0.5, 0, 0.5, 1])

            if z == 0:
                ax.set_ylabel(MLDE_METRIC_MAP[x])
                if ifpad_y1 and i == 0:
                    ax.set_ylabel(MLDE_METRIC_MAP[x], labelpad=12)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])  # Removes y-axis ticks

            # # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            if "ds-" in zs:
                ax_title = (
                    "Hamming\ndistance\n"
                    + ZS_OPTS_LEGEND[zs.replace("ds-", "") + "_score"]
                )
            else:
                ax_title = ZS_OPTS_LEGEND[zs]
            ax.legend().remove()
            ax.set_title(
                ax_title.replace(" ", "\n"),
                fontdict={
                    "fontsize": 10,
                },
            )

            # # Hide the top and right spine
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.axhline(0, color="gray", lw=1, ls="dotted")

            if do_ttest:
                t_val, p_value = ttest_ind(
                    list(bar_type_df[bar_type_df["type"] == "Binding"][x]),
                    list(bar_type_df[bar_type_df["type"] == "Enzyme activity"][x]),
                    equal_var=False,
                )
                print(f"{zs} : t={t_val:.3f} and p={p_value:.3f}")

                # Draw a line between points
                p = 0.1
                q = 1
                annot_y = ymax
                if p_value < 0.05:
                    # ax.plot([p+0.5*(q-p), q-0.5*(q-p)], [annot_y, annot_y], color='gray', lw=1.5)
                    ax.text(
                        (p + q) * 0.5,
                        annot_y,
                        "*",
                        ha="center",
                        va="bottom",
                        color="gray",
                    )

    axes[0, 8].legend(loc="upper left", bbox_to_anchor=(1, 1.12))
    plt.tight_layout(pad=0, h_pad=h_pad, w_pad=0.5)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def plot_mlde_type_split(
    mlde_csv: str,
    lib_list: list,
    n_sample: int,
    fig_name: str,
    ymin1: float = -0.15,
    ymax1: float = 0.5,
    ymin2: float = -0.8,
    ymax2: float = 1.0,
    h_pad: float = 0,
    models: list = ["boosting"],
    n_top: int = 96,
    ifpad_y1: bool = False,
    set_ymin2_scale: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    slice_df = slice_lib_mlde(
        mlde_csv=mlde_csv,
        lib_list=lib_list,
        n_sample=n_sample,
        models=models,
        n_top=n_top,
    )

    for i, x in enumerate(PLOT_MLDE_METRICS):
        fig, axes = plt.subplots(1, 7, figsize=(7, 3.6))

        rand_df = (
            slice_df[slice_df["zs"] == "none"][["lib", "type", x]]
            .reset_index(drop=True)
            .copy()
        )

        for z, zs in enumerate(ZS_OPTS + ["ds-ev"]):

            ax = axes[z]
            bar_type_df = (
                slice_df[slice_df["zs"] == zs][["lib", "type", x]]
                .reset_index(drop=True)
                .copy()
            )

            # subtract random
            merg_df = pd.merge(bar_type_df, rand_df, on=["lib", "type"], how="outer")
            merg_df["delta"] = merg_df[x + "_x"] - merg_df[x + "_y"]

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
                y="delta",
                data=merg_df,
                width=bar_width,
                ax=ax,
                order=bar_order,
                boxprops={"facecolor": "None", "edgecolor": FZL_PALETTE["gray"]},
            )
            sns.stripplot(
                x="type",
                y="delta",
                data=merg_df,
                order=bar_order,
                hue="lib",
                hue_order=merg_df["lib"].unique(),
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

            if i == 0:
                ymax = ymax1
                ax.set_ylim(ymin1, ymax1)

            else:
                ymax = ymax2
                ax.set_ylim(ymin2, ymax2)
                ax.set_yticks([-0.5, 0, 0.5, 1])

            if z == 0:
                ax.set_ylabel(PLOT_LINE_PERFORMANCE_YAXIS[i])
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.legend().remove()
            ax.set_xlabel("")
            if "ds-" in zs:
                ax_title = (
                    "Hamming\ndistance\n"
                    + ZS_OPTS_LEGEND[zs.replace("ds-", "") + "_score"]
                )
            else:
                ax_title = ZS_OPTS_LEGEND[zs]
            ax.set_title(
                ax_title.replace(" ", "\n"),
                fontdict={
                    "fontsize": 10,
                },
            )
            ax.set_xticklabels(labels, rotation=90, ha="center")
            ax.yaxis.set_label_coords(-0.64, 0.55)

            # # Hide the top and right spine
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.axhline(0, color="gray", lw=1, ls="dotted")

            if do_ttest:

                t_val, p_value = ttest_ind(
                    list(bar_type_df[bar_type_df["type"] == "Binding"][x]),
                    list(bar_type_df[bar_type_df["type"] == "Enzyme activity"][x]),
                    equal_var=False,
                )

                print(f"{zs} : t={t_val:.3f} and p={p_value:.3f}")

                # # Draw a line between points
                p = 0.1
                q = 1
                annot_y = ymax
                if p_value < 0.05:
                    # ax.plot([p+0.5*(q-p), q-0.5*(q-p)], [annot_y, annot_y], color='gray', lw=1.5)
                    ax.text(
                        (p + q) * 0.5,
                        annot_y,
                        "*",
                        ha="center",
                        va="bottom",
                        color="gray",
                    )
        # axes[0,0].yaxis.set_label_coords(-0.8, 0.55)
        # axes[1,0].yaxis.set_label_coords(-0.8, 0.42)
        axes[6].legend(loc="upper left", bbox_to_anchor=(1.01, 1.075))
        plt.tight_layout(pad=0, h_pad=1, w_pad=0.5)

        if ifsave:
            save_svg(fig, f"{fig_name}_{x}", fig_dir)


def plot_alde_type(
    alde_csv: str,
    lib_list: list,
    n_total: int,
    n_round: int,
    fig_name: str,
    ymin1: float = -0.15,
    ymax1: float = 0.5,
    ymin2: float = -0.8,
    ymax2: float = 1.0,
    h_pad: float = 0,
    models: list = ["Boosting Ensemble"],
    acquisition: list = ["GREEDY"],
    ifpad_y1: bool = False,
    set_ymin2_scale: bool = True,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    alde_all = pd.read_csv(alde_csv)

    # Replace NaN values in column 'zs' with the string "none"
    alde_all["zs"] = alde_all["zs"].fillna("none")

    slice_df = alde_all[
        (alde_all["rounds"] == n_round)
        & (alde_all["Encoding"] == "onehot")
        & (alde_all["Model"].isin(models))
        & (alde_all["Acquisition"].isin(acquisition))
        & (alde_all["n_samples"] == n_total)
        & (alde_all["Protein"].isin(lib_list))
        # & (alde_all["n_mut_cutoff"] == "all")
    ].copy()

    fig, axes = plt.subplots(2, 9, figsize=(8.4, 6.4))

    # Convert 'Category' column to categorical with defined order
    slice_df["zs"] = pd.Categorical(
        slice_df["zs"],
        categories=["none"]
        + [o.replace("_score", "") for o in ZS_OPTS]
        + [
            "ds-esmif",
            "ds-ev",
            "ds-coves",
        ],
        ordered=True,
    )

    slice_df = slice_df.sort_values(by=["zs", "Protein"])

    slice_df["zs"] = slice_df["zs"].replace(ZS_OPTS_LEGEND)
    slice_df["type"] = slice_df["Protein"].map(lambda x: LIB_INFO_DICT[x]["type"])

    for i, x in enumerate(ALDE_METRICS):

        rand_df = (
            slice_df[slice_df["zs"] == "Random"][["Protein", "type", x]]
            .reset_index(drop=True)
            .copy()
        )

        for z, zs in enumerate(list(slice_df["zs"].unique()[1:10])):
            ax = axes[i, z]
            bar_type_df = (
                slice_df[slice_df["zs"] == zs][["Protein", "type", x]]
                .reset_index(drop=True)
                .copy()
            )

            # subtract random
            merg_df = pd.merge(
                bar_type_df, rand_df, on=["Protein", "type"], how="outer"
            )
            merg_df["delta"] = merg_df[x + "_x"] - merg_df[x + "_y"]

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
                y="delta",
                data=merg_df,
                width=bar_width,
                ax=ax,
                order=bar_order,
                boxprops={
                    "facecolor": "None",
                    "edgecolor": FZL_PALETTE["gray"],
                },
            )

            sns.stripplot(
                x="type",
                y="delta",
                data=merg_df,
                order=bar_order,
                hue="Protein",
                hue_order=merg_df["Protein"].unique(),
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

            if i == 0:
                ymax = ymax1
                ax.set_xlabel("")
                ax.set_xticks([])  # Removes x-axis ticks
                ax.set_ylim(ymin1, ymax1)
            else:
                ymax = ymax2
                ax.set_xlabel("")
                ax.set_ylim(ymin2, ymax2)
                ax.set_xticklabels(labels, rotation=90, ha="center")
                if set_ymin2_scale:
                    ax.set_yticks([-0.5, 0, 0.5, 1])

            if z == 0:
                ax.set_ylabel(ALDE_METRIC_MAP[x])
                if ifpad_y1 and i == 0:
                    ax.set_ylabel(ALDE_METRIC_MAP[x], labelpad=12)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])  # Removes y-axis ticks

            # # ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            if "ds-" in zs:
                ax_title = (
                    "Hamming\ndistance\n"
                    + ZS_OPTS_LEGEND[zs.replace("ds-", "") + "_score"]
                )
            else:
                ax_title = ZS_OPTS_LEGEND[zs + "_score"]
            ax.legend().remove()
            ax.set_title(
                ax_title.replace(" ", "\n"),
                fontdict={
                    "fontsize": 10,
                },
            )

            # # Hide the top and right spine
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.axhline(0, color="gray", lw=1, ls="dotted")

            if do_ttest:
                t_val, p_value = ttest_ind(
                    list(bar_type_df[bar_type_df["type"] == "Binding"][x]),
                    list(bar_type_df[bar_type_df["type"] == "Enzyme activity"][x]),
                    equal_var=False,
                )
                print(f"{zs} : t={t_val:.3f} and p={p_value:.3f}")

                # Draw a line between points
                p = 0.1
                q = 1
                annot_y = ymax
                if p_value < 0.05:
                    # ax.plot([p+0.5*(q-p), q-0.5*(q-p)], [annot_y, annot_y], color='gray', lw=1.5)
                    ax.text(
                        (p + q) * 0.5,
                        annot_y,
                        "*",
                        ha="center",
                        va="bottom",
                        color="gray",
                    )

    axes[0, 8].legend(loc="upper left", bbox_to_anchor=(1, 1.12))
    plt.tight_layout(pad=0, h_pad=h_pad, w_pad=0.5)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def get_demlal_libavg(
    mlde_csv: str,
    alde_csv: str,
    de_csv: str,
    n_sample: int,
    lib_list: list | None = None,
    models: list = ["boosting"],
    n_top: int = 96,
    ifmapde: bool = False,
):

    mlde_df = pd.read_csv(mlde_csv)

    if lib_list is None:
        lib_list = deepcopy(mlde_df["lib"].unique().tolist())

    common_slice = mlde_df[
        (mlde_df["encoding"] == "one-hot")
        & (mlde_df["model"].isin(models))
        & (mlde_df["n_sample"] == n_sample)
        & (mlde_df["n_top"] == n_top)
        & (mlde_df["lib"].isin(lib_list))
    ]

    noft_df = common_slice[
        (common_slice["zs"] == "none") & (common_slice["n_mut_cutoff"] == "all")
    ]

    no_combo_df = common_slice[
        (common_slice["n_mut_cutoff"] == "all")
        & (common_slice["zs"].isin(ZS_OPTS[1:]))
        & (common_slice["ft_lib"].isin([0.125 * 20 ** 3, 0.125 * 20 ** 4]))
    ].copy()

    ds_only_df = common_slice[
        (common_slice["n_mut_cutoff"] == "double") & (common_slice["zs"] == "none")
    ].copy()
    ds_only_df["zs"] = ds_only_df["zs"].replace({"none": "ed_score"})

    merge_mlde = pd.concat([noft_df, no_combo_df, ds_only_df], ignore_index=True)
    merge_mlde["method"] = merge_mlde["zs"].apply(
        lambda x: "MLDE" if x == "none" else "ftMLDE"
    )
    merge_mlde = merge_mlde[["method", "lib", "top_maxes", "if_truemaxs"]].copy()

    slice_alde_list = []

    alde_all = pd.read_csv(alde_csv)
    alde_all["zs"] = alde_all["zs"].fillna("none")

    for r in [2, 3, 4]:

        slice_alde = alde_all[
            (alde_all["rounds"] == r)
            & (alde_all["Encoding"] == "onehot")
            & (alde_all["Model"] == "Boosting Ensemble")
            & (alde_all["Acquisition"] == "GREEDY")
            & (alde_all["n_samples"] == n_sample + n_top)
            & (alde_all["Protein"].isin(lib_list))
            & (
                alde_all["zs"].isin(
                    ["none"] + [z.replace("_score", "") for z in ZS_OPTS]
                )
            )
        ].copy()

        slice_alde["method"] = slice_alde["zs"].apply(
            lambda x: f"ALDE x {str(r)}" if x == "none" else f"ftALDE x {str(r)}"
        )
        slice_alde_list.append(slice_alde)

    slice_aldes = pd.concat(slice_alde_list)

    merge_mlal = pd.concat(
        [
            merge_mlde,
            slice_aldes[["zs", "Protein", "Mean", "Frac", "method"]].rename(
                columns={
                    "Protein": "lib",
                    "Mean": "top_maxes",
                    "Frac": "if_truemaxs",
                }
            ),
        ]
    )

    de_all = pd.read_csv(de_csv)

    if ifmapde:
        # rename column de_type with DE_LEGEND_MAP if they are in the map and keep the rest what it is
        de_all["de_type"] = de_all["de_type"].map(DE_LEGEND_MAP)

    merge_demlal = pd.concat(
        [
            de_all[de_all["lib"].isin(lib_list)][
                ["lib", "mean_all", "fraction_max", "de_type"]
            ].rename(
                {
                    "mean_all": "top_maxes",
                    "fraction_max": "if_truemaxs",
                    "de_type": "method",
                },
                axis=1,
            ),
            merge_mlal[["lib", "top_maxes", "if_truemaxs", "method"]],
        ]
    )

    return (
        merge_demlal[["lib", "top_maxes", "if_truemaxs", "method"]]
        .groupby(["lib", "method"])
        .mean()
        .reset_index()
    ).copy()


def comp_de_mlde_alde_lib(
    mlde_csv: str,
    alde_csv: str,
    de_csv: str,
    lib_list: list,
    n_sample: int,
    fig_name: str,
    models: list = ["boosting"],
    n_top: int = 96,
    ifannotate: bool = False,
    ifsave: bool = True,
    fig_dir: str = "figs",
):

    merge_demlal_avg = get_demlal_libavg(
        mlde_csv=mlde_csv,
        alde_csv=alde_csv,
        de_csv=de_csv,
        lib_list=lib_list,
        n_sample=n_sample,
        models=models,
        n_top=n_top,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    order = [
        "DE",
        "MLDE",
        "ftMLDE",
        "ALDE x 2",
        "ftALDE x 2",
        "ALDE x 3",
        "ftALDE x 3",
        "ALDE x 4",
        "ftALDE x 4",
    ]

    merge_demlal_avg_sliced = merge_demlal_avg[
        ~merge_demlal_avg["method"].isin(["recomb_SSM", "top96_SSM"])
    ].replace("single_step_DE", "DE")

    for i, m in enumerate(PLOT_MLDE_METRICS):

        ax = axes[i]

        sns.boxplot(
            # x="Type", y="Rho",
            x="method",
            y=m,
            data=merge_demlal_avg_sliced.replace(" x 2", ""),
            width=0.6,
            ax=ax,
            order=order,
            # color=FZL_PALETTE["light_gray"],# palette="pastel", showmeans=True
            boxprops={
                "facecolor": "None",
                "edgecolor": FZL_PALETTE["gray"],
            },
        )
        # Perform pairwise t-tests between each pair of groups in the `model` column
        pairs = list(combinations(order, 2))
        p_values = []

        # Perform t-tests and store p-values
        for pair in pairs:
            group1 = merge_demlal_avg_sliced[
                merge_demlal_avg_sliced["method"] == pair[0]
            ][m]
            group2 = merge_demlal_avg_sliced[
                merge_demlal_avg_sliced["method"] == pair[1]
            ][m]
            t_stat, p_value = stats.ttest_ind(group1, group2)
            p_values.append((pair, p_value))

        # Filter significant results (p < 0.05)
        significant_pairs = [pair for pair, p_value in p_values if p_value < 0.05]

        # add mean as x for each model
        # Calculate means for each category (model)
        means = merge_demlal_avg_sliced.groupby("method")[m].mean().reindex(order)

        # Add mean as a scatter plot (dot) for each model category
        ax.scatter(
            x=range(len(means)),  # x-values corresponding to the boxplot positions
            y=means,
            color=FZL_PALETTE["black"],  # Color of the mean dot
            marker="x",  # Shape of the mean marker (dot)
            alpha=0.8,
            zorder=20,  # Ensure dots appear on top of the boxplot
            label="Approach Mean",
        )

        sns.stripplot(
            x="method",
            y=m,
            data=merge_demlal_avg_sliced.replace(" x 2", ""),
            hue="lib",
            hue_order=merge_demlal_avg_sliced["lib"].unique(),
            jitter=True,
            size=7.5,
            alpha=0.8,
            ax=ax,
            palette=glasbey_category10[:12],  # Updated palette
            order=order,
        )

        ax.set_xlabel("")
        # Get current xtick labels
        xticks = ax.get_xticklabels()

        # Update the labels, replacing ' x 2' with ''
        for label in xticks:
            new_text = label.get_text().replace(" x 2", "")
            label.set_text(new_text)

        # rotate x-axis labels
        ax.set_xticklabels(xticks, rotation=90, ha="center")
        ax.set_ylabel(PLOT_LINE_PERFORMANCE_YAXIS[i])
        ax.set_ylim(-0.05, 1.05)
        # # Hide the top and right spine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend().remove()

    def add_stat_annotation(merge_mlal_avg, ax, pair, y_offset=0.06, level=1):
        """
        Adds a line and a star between the two groups indicating significance,
        with staggered y positions to avoid overlap.

        Parameters:
        - ax: The matplotlib axes object.
        - pair: The tuple of two categories between which the significance is drawn.
        - y_offset: The base offset for the lines.
        - level: The staggered level to place the current annotation.
        """
        x1, x2 = order.index(pair[0]), order.index(pair[1])  # Get positions of the bars
        y_max = merge_mlal_avg[m].max()  # Maximum y-value to determine the line height
        h = 0.02  # The height difference between the lines
        y = y_max + y_offset * level  # Increase the y-position for each level

        # Plot the line
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1, c=FZL_PALETTE["gray"])

        # Add the star to indicate significance
        ax.text(
            (x1 + x2) * 0.5,
            y + h / 4,
            "*",
            ha="center",
            va="bottom",
            color=FZL_PALETTE["gray"],
        )

    if ifannotate:
        # Add significance lines for each significant pair with increasing levels
        for i, pair in enumerate(significant_pairs):
            add_stat_annotation(ax, pair, level=i + 1)

    # Get handles and labels from the existing legend
    handles, labels = ax.get_legend_handles_labels()

    # Move the "Model Mean" label to the last position
    handles.append(handles.pop(0))  # Move the first handle (Model Mean) to the end
    labels.append(labels.pop(0))  # Move the first label (Model Mean) to the end

    # Add a reordered legend
    axes[1].legend(
        handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.02, 1.02)
    )
    # plt.legend(handles=[handles[0]], labels=[labels[0]], loc="upper left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(pad=0, h_pad=0, w_pad=1)

    if ifsave:
        save_svg(fig, fig_name, fig_dir)


def get_prospective_libavg(
    mlde_csv: str, alde_csv: str, n_top: int, n_sample: int, n_cutoff: int = 480
):
    """
    Filters and processes datasets based on given n_top and n_sample values.

    Args:
        n_top (int): Number of top variants.
        n_sample (int): Number of sampled variants.

    Returns:
        pd.DataFrame: Processed and averaged results grouped by 'lib'.
    """

    # Load datasets
    mlde_all = pd.read_csv(mlde_csv)
    alde_all = pd.read_csv(alde_csv)
    alde_all["zs"] = alde_all["zs"].fillna("none")

    n_total = n_top + n_sample

    # Define filters for different libraries
    if n_total <= n_cutoff:
        n_mut_cutff = "double"
        alde_ds_zs = "ed"
        alde_ds_pre = "ds-"
    else:
        n_mut_cutff = "all"
        alde_ds_zs = "none"
        alde_ds_pre = ""

    hm_ev = mlde_all[
        (
            mlde_all["lib"].isin(
                ["DHFR"] + [f"TrpB3{chr(i)}" for i in range(ord("A"), ord("I") + 1)]
            )
        )
        & (mlde_all["n_mut_cutoff"] == n_mut_cutff)
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["zs"] == "ev_score")
        & (mlde_all["encoding"] == "one-hot")
        & (mlde_all["model"] == "boosting")
        & (mlde_all["n_sample"] == n_sample)
    ][["lib", "top_maxes", "if_truemaxs"]].reset_index(drop=True)

    gb1 = (
        alde_all[
            (alde_all["Protein"] == "GB1")
            & (alde_all["n_mut_cutoff"] == n_mut_cutff)
            & (alde_all["zs"] == alde_ds_pre + "esmif")
            & (alde_all["rounds"] == 4)
            & (alde_all["n_samples"] == n_total)
            & (alde_all["Encoding"] == "onehot")
            & (alde_all["Model"] == "Boosting Ensemble")
            & (alde_all["Acquisition"] == "GREEDY")
        ][["Protein", "Mean", "Frac"]]
        .rename(columns={"Protein": "lib", "Mean": "top_maxes", "Frac": "if_truemaxs"})
        .reset_index(drop=True)
    )

    pards = mlde_all[
        (mlde_all["lib"].isin(["ParD2", "ParD3"]))
        & (mlde_all["n_mut_cutoff"] == "all")
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["zs"] == "none")
        & (mlde_all["encoding"] == "one-hot")
        & (mlde_all["model"] == "boosting")
        & (mlde_all["n_sample"] == n_sample)
    ][["lib", "top_maxes", "if_truemaxs"]].reset_index(drop=True)

    t7 = mlde_all[
        (mlde_all["lib"] == "T7")
        & (mlde_all["n_mut_cutoff"] == n_mut_cutff)
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["zs"] == "none")
        & (mlde_all["encoding"] == "one-hot")
        & (mlde_all["model"] == "boosting")
        & (mlde_all["n_sample"] == n_sample)
    ][["lib", "top_maxes", "if_truemaxs"]].reset_index(drop=True)

    tev = (
        alde_all[
            (alde_all["Protein"] == "TEV")
            & (alde_all["n_mut_cutoff"] == "all")
            & (alde_all["zs"] == alde_ds_zs)
            & (alde_all["rounds"] == 4)
            & (alde_all["n_samples"] == n_total)
            & (alde_all["Encoding"] == "onehot")
            & (alde_all["Model"] == "Boosting Ensemble")
            & (alde_all["Acquisition"] == "GREEDY")
        ][["Protein", "Mean", "Frac"]]
        .rename(columns={"Protein": "lib", "Mean": "top_maxes", "Frac": "if_truemaxs"})
        .reset_index(drop=True)
    )

    trpb4 = (
        alde_all[
            (alde_all["Protein"] == "TrpB4")
            & (alde_all["n_mut_cutoff"] == n_mut_cutff)
            & (alde_all["zs"] == alde_ds_pre + "ev")
            & (alde_all["rounds"] == 4)
            & (alde_all["n_samples"] == n_total)
            & (alde_all["Encoding"] == "onehot")
            & (alde_all["Model"] == "Boosting Ensemble")
            & (alde_all["Acquisition"] == "GREEDY")
        ][["Protein", "Mean", "Frac"]]
        .rename(columns={"Protein": "lib", "Mean": "top_maxes", "Frac": "if_truemaxs"})
        .reset_index(drop=True)
    )

    # Combine all filtered data
    all_lib_chosen = pd.concat([hm_ev, gb1, pards, t7, tev, trpb4]).reset_index(
        drop=True
    )

    # Compute mean for each library and round to 2 decimal places
    all_lib_avg = all_lib_chosen.groupby("lib").mean().round(2)

    return all_lib_avg


def get_demlft_improvement_tables(
    mlde_csv: str | None = None,
    de_csv: str | None = None,
    alde_dir: str | None = None,
    lib_list: list | None = None,
    de_avg: dict | None = None,
    avg_mlde_df_dict: dict | None = None,
    n_top: int = 96,
    n_sample_list: list = N_SAMPLE_LIST,
    de_types: list = DE_TYPES,
    plot_de_metrics: list = PLOT_DE_METRICS,
    plot_mlde_metrics: list = PLOT_MLDE_METRICS,
    plot_line_performance_yaxis: list = PLOT_LINE_PERFORMANCE_YAXIS,
    de_legend_map: dict = DE_LEGEND_MAP,
) -> pd.DataFrame:
    """
    Computes percent improvement table for improvement of MLDE over DE.

    Parameters:
        mlde_csv (str): Path to the MLDE CSV file.
        de_csv (str): Path to the DE CSV file.
        alde_dir (str): Path to the ALDE directory.
        lib_list (list): List of libraries.
        de_avg (pd.DataFrame): DataFrame containing the average DE metrics.
        avg_mlde_df_dict (dict): Dictionary containing MLDE metric dataframes.
        n_sample_list (list): List of training sample sizes.
        de_types (list): List of DE types.
        plot_de_metrics (list): List of DE metric names.
        plot_mlde_metrics (list): List of corresponding MLDE metric names.
        plot_line_performance_yaxis (list): Labels for performance metrics.
        de_legend_map (dict): Mapping of DE types to their legends.

    Returns:
        pd.DataFrame: Percent improvement table.
    """

    if de_avg is None:
        assert (
            de_csv is not None and lib_list is not None
        ), "de_csv and lib_list must be provided."
        de_avg = get_de_avg(de_csv, lib_list)

    if avg_mlde_df_dict is None:
        assert (
            mlde_csv is not None and alde_dir is not None and lib_list is not None
        ), "mlde_csv, alde_dir, and lib_list must be provided."
        avg_mlde_df_dict = get_mlde_avg_dict(
            mlde_csv=mlde_csv, alde_dir=alde_dir, lib_list=lib_list, n_top=n_top
        )

    # First Table: MLDE improvement over DE
    sum_de_mlde = pd.DataFrame(index=n_sample_list)
    sum_de_mlde.index.name = "Number of training sample"

    for de in de_types:
        for i, (de_metric, mlde_metric) in enumerate(
            zip(plot_de_metrics, plot_mlde_metrics)
        ):
            improve_stat = (
                (
                    avg_mlde_df_dict["MLDE"][f"{mlde_metric}_mean"]
                    - de_avg.loc[de, f"{de_metric}_mean"]
                )
                / de_avg.loc[de, f"{de_metric}_mean"]
                * 100
            )

            col_info = {n: v for n, v in zip(n_sample_list, improve_stat)}

            sum_de_mlde = sum_de_mlde.merge(
                pd.Series(
                    col_info,
                    index=n_sample_list,
                    name=f"{plot_line_performance_yaxis[i]} percent improvement from DE: {de_legend_map[de]}",
                ).to_frame(),
                left_index=True,
                right_index=True,
                how="left",
            )

    return sum_de_mlde.applymap(lambda x: round(x, 2))


def get_ft_improvement_tables(
    mlde_csv: str | None = None,
    alde_dir: str | None = None,
    lib_list: list | None = None,
    avg_mlde_df_dict: dict | None = None,
    n_top: int = 96,
    list_of_tables: list = ["MLDE", "ALDE x 4"],
) -> pd.DataFrame:

    """
    Computes percent improvement tables for improvement of ftA/MLDE over A/MLDE.

    Parameters:
        mlde_csv (str): Path to the MLDE CSV file.
        alde_dir (str): Path to the ALDE directory.
        lib_list (list): List of libraries.
        de_avg (pd.DataFrame): DataFrame containing the average DE metrics.
        avg_mlde_df_dict (dict): Dictionary containing MLDE metric dataframes.
        n_top (int): Number of top variants.

    Returns:
        tuple: Two pandas DataFrames for percent improvements.
    """

    avg_mlde_df_dict = get_mlde_avg_dict(
        mlde_csv=mlde_csv, alde_dir=alde_dir, lib_list=lib_list, n_top=n_top
    )

    output_df_dict = {}

    ft_keys = [k for k in avg_mlde_df_dict.keys() if "Average" in k]
    baseline_keys = [k.replace("Average ft", "") for k in ft_keys]
    labels = [f"{k} from {k.replace('Average ft', '')}" for k in ft_keys]

    output_df_dict[
        "Average focused training improvement"
    ] = compute_relative_improvements_table(
        avg_mlde_df_dict, ft_keys, baseline_keys, labels
    )

    for t in list_of_tables:

        zs_keys = [
            f"ft{t}: {SIMPLE_ZS_OPT_LEGNED[v]}"
            for v in ZS_OPTS
            if SIMPLE_ZS_OPT_LEGNED[v] != "Random"
        ]
        baseline_zs_keys = [t] * len(zs_keys)
        zs_labels = [
            SIMPLE_ZS_OPT_LEGNED[v]
            for v in ZS_OPTS
            if SIMPLE_ZS_OPT_LEGNED[v] != "Random"
        ]

        output_df_dict[f"ft{t} improvement"] = compute_relative_improvements_table(
            avg_mlde_df_dict, zs_keys, baseline_zs_keys, zs_labels
        )

    return output_df_dict


def welch_ttest(mean1, std1, n1, mean2, std2, n2):
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)
    t_stat = (mean1 - mean2) / np.sqrt(se1 ** 2 + se2 ** 2)
    df = (se1 ** 2 + se2 ** 2) ** 2 / ((se1 ** 4 / (n1 - 1)) + (se2 ** 4 / (n2 - 1)))
    p_value = 2 * t.sf(np.abs(t_stat), df)
    return p_value


def format_with_significance(val, p):
    # if p < 0.001:
    #     return f"{val:.2f}⁽***⁾"
    # elif p < 0.01:
    #     return f"{val:.2f}⁽**⁾"
    if p < 0.05:
        return f"{val:.2f}*"
    else:
        return f"{val:.2f}"


# def compute_relative_improvements_table(
#     df_dict,
#     ft_keys,
#     baseline_keys,
#     labels,
#     columns_map={
#         "top_maxes_mean": "Average max fitness achieved",
#         "if_truemaxs_mean": "Fraction reaching the global optimum",
#     },
#     std_map={
#         "top_maxes_mean": "top_maxes_std",
#         "if_truemaxs_mean": "if_truemaxs_std",
#     },
#     n_landscape=12
# ):
#     assert len(ft_keys) == len(baseline_keys) == len(labels), "Keys and labels must be same length"

#     merged_data = {}

#     for ft_key, base_key, label in zip(ft_keys, baseline_keys, labels):
#         data = {}
#         for metric_key, pretty_name in columns_map.items():
#             ft_mean = df_dict[ft_key][metric_key]
#             base_mean = df_dict[base_key][metric_key]

#             ft_std = df_dict[ft_key][std_map[metric_key]]
#             base_std = df_dict[base_key][std_map[metric_key]]

#             vals = []
#             for idx in ft_mean.index:
#                 imp = ((ft_mean[idx] - base_mean[idx]) / base_mean[idx]) * 100
#                 p = welch_ttest(
#                     ft_mean[idx], ft_std[idx], n_landscape,
#                     base_mean[idx], base_std[idx], n_landscape
#                 )
#                 vals.append(format_with_significance(imp, p))

#             data[pretty_name] = vals

#         improvement_df = pd.DataFrame(data, index=ft_mean.index)
#         merged_data[label] = improvement_df

#     final_df = pd.concat(merged_data.values(), axis=1, keys=merged_data.keys())
#     final_df.index.name = "Number of training samples"
#     return final_df


def compute_relative_improvements_table(
    df_dict,
    ft_keys,
    baseline_keys,
    labels,
    columns_map={
        "top_maxes_mean": "Average max fitness achieved",
        "if_truemaxs_mean": "Fraction reaching the global optimum",
    },
):
    """
    Computes relative improvements between ft_keys and baseline_keys and returns a formatted table.

    Parameters:
        df_dict (dict): Dictionary of DataFrames keyed by method names.
        ft_keys (list): List of keys (strings) for ft-models in df_dict.
        baseline_keys (list): List of keys (strings) for corresponding baselines.
        labels (list): List of column group labels for output table.
        columns_map (dict): Mapping of internal column names to display names.

    Returns:
        pd.DataFrame: Combined improvement table with labeled MultiIndex columns.
    """
    assert (
        len(ft_keys) == len(baseline_keys) == len(labels)
    ), "Keys and labels must be same length"

    merged_data = {}

    for ft_key, base_key, label in zip(ft_keys, baseline_keys, labels):
        improvement_df = (
            ((df_dict[ft_key] - df_dict[base_key]) / df_dict[base_key] * 100)[
                list(columns_map.keys())
            ]
            .rename(columns=columns_map)
            .applymap(lambda x: round(x, 2))
        )
        merged_data[label] = improvement_df

    final_df = pd.concat(merged_data.values(), axis=1, keys=merged_data.keys())
    final_df.index.name = "Number of training samples"
    return final_df


def get_per_landscape_metric_table(
    mlde_csv: str,
    n_mut_cutoff: str,
    zs: str,
    lib_list: list,
    n_top: int = 96,
    model_list: list = ["boosting"],
    encoding_list: list = ["one-hot"],
    ft_frac: float = 0.125,
    metrics: list = ["top_maxes", "if_truemaxs"],
):
    """
    Extracts per-landscape average metrics for t-testing across landscapes.

    Returns one row per (library, n_sample), averaged over replicates.

    Args:
        mlde_all (pd.DataFrame): Full MLDE results dataframe.
        n_top (int): Number of top variants.
        n_mut_cutoff (str): Mutation cutoff level.
        zs (str): ZS strategy.
        lib_list (list): Libraries to include.
        model_list (list): ML models to filter.
        encoding_list (list): Encodings to filter.
        ft_frac (float): Focused training fraction.
        metrics (list): Metrics to extract.

    Returns:
        pd.DataFrame: Rows = (lib, n_sample), Columns = metrics.
    """

    mlde_all = pd.read_csv(mlde_csv)

    slice_mlde = mlde_all[
        (mlde_all["lib"].isin(lib_list))
        & (mlde_all["zs"] == zs)
        & (mlde_all["n_top"] == n_top)
        & (mlde_all["n_mut_cutoff"] == n_mut_cutoff)
        & (mlde_all["rep"].isin(np.arange(50)))
        & (mlde_all["model"].isin(model_list))
        & (mlde_all["encoding"].isin(encoding_list))
    ].copy()

    # Filter to specific ft_frac (if ZS is used)
    if zs != "none":
        lib_dfs = []
        for lib in lib_list:
            lib_df = slice_mlde[slice_mlde["lib"] == lib].copy()
            n_site = len(LIB_INFO_DICT[lib]["positions"])

            if n_mut_cutoff == "all":
                sample_space = 20 ** n_site
            elif n_mut_cutoff == "double":
                sample_space = math.comb(n_site, 2) * 20 ** 2
            elif n_mut_cutoff == "single":
                sample_space = n_site * 20

            ft_lib_unique = np.array(sorted(lib_df["ft_lib"].unique()))
            ft_lib_frac = ft_lib_unique / sample_space

            lib_df["ft_lib_size"] = lib_df["ft_lib"].map(
                {numb: frac for numb, frac in zip(ft_lib_unique, ft_lib_frac)}
            )

            lib_df = lib_df[lib_df["ft_lib_size"] == ft_frac]
            lib_dfs.append(lib_df)

        slice_mlde = pd.concat(lib_dfs)

    # Group by lib and n_sample, average over replicates
    per_landscape_df = (
        slice_mlde.groupby(["lib", "n_sample"])[metrics].mean().reset_index()
    )

    return per_landscape_df


def plot_ft_fact(
    lib_list: list,
    fig_name: str,
    mlde_csv: str = "results/mlde/all_results.csv",
    n_sample: int = 384,
    n_top: int = 96,
    if_save: bool = True,
    fig_dir: str = "figs",
):

    mlde_df = pd.read_csv(mlde_csv)
    slice_mlde = (
        mlde_df[
            (mlde_df["lib"].isin(lib_list))
            & (mlde_df["zs"].isin(ZS_OPTS[1:]))
            & (mlde_df["n_mut_cutoff"] == "all")
            & (mlde_df["encoding"] == "one-hot")
            & (mlde_df["n_sample"] == n_sample)
            & (mlde_df["model"] == "boosting")
        ]
        .sort_values(["lib", "zs"])
        .copy()
    )
    # map lib to number of sites use len(LIB_INFO_DICT[lib]["positions"])
    slice_mlde["n_sites"] = slice_mlde["lib"].map(
        lambda x: len(LIB_INFO_DICT[x]["positions"])
    )

    # for avg cross number of samples
    frac_ft_dfs = []
    for frac in FTLIB_FRAC_LIST[:-1]:
        for zs in ZS_OPTS[1:]:
            for n_site in [3, 4]:
                lib_list = slice_mlde[slice_mlde["n_sites"] == n_site]["lib"].unique()
                slice_ft_df = get_mlde_avg_sdf(
                    slice_mlde, n_top, "all", zs, lib_list, ft_frac=frac
                ).reset_index(drop=True)
                slice_ft_df["ft_lib_size"] = frac
                slice_ft_df["zs"] = zs
                slice_ft_df["n_site"] = n_site
                frac_ft_dfs.append(slice_ft_df)

    frac_ft_df = pd.concat(frac_ft_dfs).reset_index(drop=True)

    ms = ["top_maxes_mean", "if_truemaxs_mean"]
    titles = ["Average max fitness achieved", "Fraction reaching the global optimum"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex=True)

    for j, site in enumerate([3, 4]):

        sliced_df = frac_ft_df[frac_ft_df["n_site"] == site].copy()

        sliced_df["ft_lib_size"] = sliced_df["ft_lib_size"] * 100

        # Convert 'Category' column to categorical with defined order
        sliced_df["zs"] = pd.Categorical(
            sliced_df["zs"], categories=list(SIX_ZS_COLORS.keys()), ordered=True
        )

        # Sort DataFrame by 'Category'
        sliced_df = sliced_df.sort_values(by="zs")

        for i in range(2):
            ax = axes[j, i]
            # set x log

            sns.stripplot(
                data=sliced_df,
                x="ft_lib_size",
                y=ms[i],
                hue="zs",
                size=7.5,
                alpha=0.8,
                jitter=True,
                marker="o",
                ax=ax,
                palette=list(SIX_ZS_COLORS.values()),
            )

            sns.boxplot(
                x="ft_lib_size",
                y=ms[i],
                data=sliced_df,
                width=0.45,
                boxprops={"facecolor": "None", "edgecolor": FZL_PALETTE["gray"]},
                ax=ax,
            )

            ax.set_ylabel("")
            ax.set_yticklabels([f"{y:.2f}" for y in ax.get_yticks()])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend().set_visible(False)
            ax.set_title(f"{str(site)}-site landscapes", fontsize=10)

            if j == 1:
                ax.set_ylabel(titles[i])
                if i == 0:
                    ax.set_xlabel(
                        "Percentage of full landscape covered by focused training library"
                    )
                    # shift to the right
                    ax.yaxis.set_label_coords(-0.2, 1)
                    ax.xaxis.set_label_coords(1.2, -0.2)
                else:
                    ax.set_xlabel("")
                    ax.yaxis.set_label_coords(-0.2, 1)

    legend_list = []
    for zs, c in SIX_ZS_COLORS.items():
        legend_list.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c,
                label=ZS_OPTS_LEGEND[zs],
            )
        )
    axes[0, 1].legend(handles=legend_list, loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout(w_pad=-2)
    if if_save:
        save_svg(fig=fig, plot_title=fig_name, path2folder=fig_dir)