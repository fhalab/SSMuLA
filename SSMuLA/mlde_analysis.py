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
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
import pickle

# Basic plotting
import holoviews as hv
import bokeh
from bokeh.io import export_svg
from bokeh.models import NumeralTickFormatter


import panel as pn

pn.config.comms = "vscode"

# Making graphs
import matplotlib.pyplot as plt

hv.extension("bokeh")


from SSMuLA.landscape_global import n_mut_cutoff_dict
from SSMuLA.util import checkNgen_folder


default_metrics = ["maxes", "means", "ndcgs", "rhos"]


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

        """

        self._mlde_npy_path = mlde_npy_path
        self._mlde_results_dir = mlde_results_dir

        # get all npy keys as properties
        # should be ['config', 'top_seqs', 'maxes', 'means', 'ndcgs', 'rhos', 'unique', 'labelled', 'y_preds']
        for attr, val in self.npy_item.items():
            setattr(self, attr, val)

        # set all config_dict keys as properties
        # should be ['data_config', 'model_config', 'train_config', 'eval_config']
        for attr, val in self.config.items():
            setattr(self, attr, val)
            for k, v in val.items():
                setattr(self, k, v)
                if isinstance(v, list):
                    setattr(self, f"{k}_len", len(v))

        # get all metrics as properties
        for m in default_metrics:
            print(m)
            setattr(
                self,
                f"{m}_df",
                pd.DataFrame(
                    {
                        "Encoding": self.encoding_index,
                        "Models": self.models_index,
                        "N_Sample": self.n_sample_index,
                        "Ft_Lib": self.lib_index,
                        "Repeats": self.repeats_index,
                        m: getattr(self, m).flatten(),
                    }
                ),
            )

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
        Return the shape of the output for maxes, means, ndcgs, rhos, unique, and labelled

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

    """
    {'data_config': {'input_csv': 'results/zs_comb/none/scale2max/GB1.csv',
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