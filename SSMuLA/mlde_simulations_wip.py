"""A script for MLDE simulations"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from copy import deepcopy

from typing import Callable, Dict

import xgboost as xgb
from sklearn import linear_model
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

from SSMuLA.aa_global import ALL_AAS
from SSMuLA.fitness_process_vis import LibData
from SSMuLA.util import checkNgen_folder


# Copied from ProFET (Ofer & Linial, DOI: 10.1093/bioinformatics/btv345)
# Original comment by the ProFET authors: 'Acquired from georgiev's paper of
# AAscales using helper script "GetTextData.py". + RegEx cleaning DOI: 10.1089/cmb.2008.0173'
gg_1 = {
    "Q": -2.54,
    "L": 2.72,
    "T": -0.65,
    "C": 2.66,
    "I": 3.1,
    "G": 0.15,
    "V": 2.64,
    "K": -3.89,
    "M": 1.89,
    "F": 3.12,
    "N": -2.02,
    "R": -2.8,
    "H": -0.39,
    "E": -3.08,
    "W": 1.89,
    "A": 0.57,
    "D": -2.46,
    "Y": 0.79,
    "S": -1.1,
    "P": -0.58,
}
gg_2 = {
    "Q": 1.82,
    "L": 1.88,
    "T": -1.6,
    "C": -1.52,
    "I": 0.37,
    "G": -3.49,
    "V": 0.03,
    "K": 1.47,
    "M": 3.88,
    "F": 0.68,
    "N": -1.92,
    "R": 0.31,
    "H": 1,
    "E": 3.45,
    "W": -0.09,
    "A": 3.37,
    "D": -0.66,
    "Y": -2.62,
    "S": -2.05,
    "P": -4.33,
}
gg_3 = {
    "Q": -0.82,
    "L": 1.92,
    "T": -1.39,
    "C": -3.29,
    "I": 0.26,
    "G": -2.97,
    "V": -0.67,
    "K": 1.95,
    "M": -1.57,
    "F": 2.4,
    "N": 0.04,
    "R": 2.84,
    "H": -0.63,
    "E": 0.05,
    "W": 4.21,
    "A": -3.66,
    "D": -0.57,
    "Y": 4.11,
    "S": -2.19,
    "P": -0.02,
}
gg_4 = {
    "Q": -1.85,
    "L": 5.33,
    "T": 0.63,
    "C": -3.77,
    "I": 1.04,
    "G": 2.06,
    "V": 2.34,
    "K": 1.17,
    "M": -3.58,
    "F": -0.35,
    "N": -0.65,
    "R": 0.25,
    "H": -3.49,
    "E": 0.62,
    "W": -2.77,
    "A": 2.34,
    "D": 0.14,
    "Y": -0.63,
    "S": 1.36,
    "P": -0.21,
}
gg_5 = {
    "Q": 0.09,
    "L": 0.08,
    "T": 1.35,
    "C": 2.96,
    "I": -0.05,
    "G": 0.7,
    "V": 0.64,
    "K": 0.53,
    "M": -2.55,
    "F": -0.88,
    "N": 1.61,
    "R": 0.2,
    "H": 0.05,
    "E": -0.49,
    "W": 0.72,
    "A": -1.07,
    "D": 0.75,
    "Y": 1.89,
    "S": 1.78,
    "P": -8.31,
}
gg_6 = {
    "Q": 0.6,
    "L": 0.09,
    "T": -2.45,
    "C": -2.23,
    "I": -1.18,
    "G": 7.47,
    "V": -2.01,
    "K": 0.1,
    "M": 2.07,
    "F": 1.62,
    "N": 2.08,
    "R": -0.37,
    "H": 0.41,
    "E": 0,
    "W": 0.86,
    "A": -0.4,
    "D": 0.24,
    "Y": -0.53,
    "S": -3.36,
    "P": -1.82,
}
gg_7 = {
    "Q": 0.25,
    "L": 0.27,
    "T": -0.65,
    "C": 0.44,
    "I": -0.21,
    "G": 0.41,
    "V": -0.33,
    "K": 4.01,
    "M": 0.84,
    "F": -0.15,
    "N": 0.4,
    "R": 3.81,
    "H": 1.61,
    "E": -5.66,
    "W": -1.07,
    "A": 1.23,
    "D": -5.15,
    "Y": -1.3,
    "S": 1.39,
    "P": -0.12,
}
gg_8 = {
    "Q": 2.11,
    "L": -4.06,
    "T": 3.43,
    "C": -3.49,
    "I": 3.45,
    "G": 1.62,
    "V": 3.93,
    "K": -0.01,
    "M": 1.85,
    "F": -0.41,
    "N": -2.47,
    "R": 0.98,
    "H": -0.6,
    "E": -0.11,
    "W": -1.66,
    "A": -2.32,
    "D": -1.17,
    "Y": 1.31,
    "S": -1.21,
    "P": -1.18,
}
gg_9 = {
    "Q": -1.92,
    "L": 0.43,
    "T": 0.34,
    "C": 2.22,
    "I": 0.86,
    "G": -0.47,
    "V": -0.21,
    "K": -0.26,
    "M": -2.05,
    "F": 4.2,
    "N": -0.07,
    "R": 2.43,
    "H": 3.55,
    "E": 1.49,
    "W": -5.87,
    "A": -2.01,
    "D": 0.73,
    "Y": -0.56,
    "S": -2.83,
    "P": 0,
}
gg_10 = {
    "Q": -1.67,
    "L": -1.2,
    "T": 0.24,
    "C": -3.78,
    "I": 1.98,
    "G": -2.9,
    "V": 1.27,
    "K": -1.66,
    "M": 0.78,
    "F": 0.73,
    "N": 7.02,
    "R": -0.99,
    "H": 1.52,
    "E": -2.26,
    "W": -0.66,
    "A": 1.31,
    "D": 1.5,
    "Y": -0.95,
    "S": 0.39,
    "P": -0.66,
}
gg_11 = {
    "Q": 0.7,
    "L": 0.67,
    "T": -0.53,
    "C": 1.98,
    "I": 0.89,
    "G": -0.98,
    "V": 0.43,
    "K": 5.86,
    "M": 1.53,
    "F": -0.56,
    "N": 1.32,
    "R": -4.9,
    "H": -2.28,
    "E": -1.62,
    "W": -2.49,
    "A": -1.14,
    "D": 1.51,
    "Y": 1.91,
    "S": -2.92,
    "P": 0.64,
}
gg_12 = {
    "Q": -0.27,
    "L": -0.29,
    "T": 1.91,
    "C": -0.43,
    "I": -1.67,
    "G": -0.62,
    "V": -1.71,
    "K": -0.06,
    "M": 2.44,
    "F": 3.54,
    "N": -2.44,
    "R": 2.09,
    "H": -3.12,
    "E": -3.97,
    "W": -0.3,
    "A": 0.19,
    "D": 5.61,
    "Y": -1.26,
    "S": 1.27,
    "P": -0.92,
}
gg_13 = {
    "Q": -0.99,
    "L": -2.47,
    "T": 2.66,
    "C": -1.03,
    "I": -1.02,
    "G": -0.11,
    "V": -2.93,
    "K": 1.38,
    "M": -0.26,
    "F": 5.25,
    "N": 0.37,
    "R": -3.08,
    "H": -1.45,
    "E": 2.3,
    "W": -0.5,
    "A": 1.66,
    "D": -3.85,
    "Y": 1.57,
    "S": 2.86,
    "P": -0.37,
}
gg_14 = {
    "Q": -1.56,
    "L": -4.79,
    "T": -3.07,
    "C": 0.93,
    "I": -1.21,
    "G": 0.15,
    "V": 4.22,
    "K": 1.78,
    "M": -3.09,
    "F": 1.73,
    "N": -0.89,
    "R": 0.82,
    "H": -0.77,
    "E": -0.06,
    "W": 1.64,
    "A": 4.39,
    "D": 1.28,
    "Y": 0.2,
    "S": -1.88,
    "P": 0.17,
}
gg_15 = {
    "Q": 6.22,
    "L": 0.8,
    "T": 0.2,
    "C": 1.43,
    "I": -1.78,
    "G": -0.53,
    "V": 1.06,
    "K": -2.71,
    "M": -1.39,
    "F": 2.14,
    "N": 3.13,
    "R": 1.32,
    "H": -4.18,
    "E": -0.35,
    "W": -0.72,
    "A": 0.18,
    "D": -1.98,
    "Y": -0.76,
    "S": -2.42,
    "P": 0.36,
}
gg_16 = {
    "Q": -0.18,
    "L": -1.43,
    "T": -2.2,
    "C": 1.45,
    "I": 5.71,
    "G": 0.35,
    "V": -1.31,
    "K": 1.62,
    "M": -1.02,
    "F": 1.1,
    "N": 0.79,
    "R": 0.69,
    "H": -2.91,
    "E": 1.51,
    "W": 1.75,
    "A": -2.6,
    "D": 0.05,
    "Y": -5.19,
    "S": 1.75,
    "P": 0.08,
}
gg_17 = {
    "Q": 2.72,
    "L": 0.63,
    "T": 3.73,
    "C": -1.15,
    "I": 1.54,
    "G": 0.3,
    "V": -1.97,
    "K": 0.96,
    "M": -4.32,
    "F": 0.68,
    "N": -1.54,
    "R": -2.62,
    "H": 3.37,
    "E": -2.29,
    "W": 2.73,
    "A": 1.49,
    "D": 0.9,
    "Y": -2.56,
    "S": -2.77,
    "P": 0.16,
}
gg_18 = {
    "Q": 4.35,
    "L": -0.24,
    "T": -5.46,
    "C": -1.64,
    "I": 2.11,
    "G": 0.32,
    "V": -1.21,
    "K": -1.09,
    "M": -1.34,
    "F": 1.46,
    "N": -1.71,
    "R": -1.49,
    "H": 1.87,
    "E": -1.47,
    "W": -2.2,
    "A": 0.46,
    "D": 1.38,
    "Y": 2.87,
    "S": 3.36,
    "P": -0.34,
}
gg_19 = {
    "Q": 0.92,
    "L": 1.01,
    "T": -0.73,
    "C": -1.05,
    "I": -4.18,
    "G": 0.05,
    "V": 4.77,
    "K": 1.36,
    "M": 0.09,
    "F": 2.33,
    "N": -0.25,
    "R": -2.57,
    "H": 2.17,
    "E": 0.15,
    "W": 0.9,
    "A": -4.22,
    "D": -0.03,
    "Y": -3.43,
    "S": 2.67,
    "P": 0.04,
}

# Package all georgiev parameters
georgiev_parameters = [
    gg_1,
    gg_2,
    gg_3,
    gg_4,
    gg_5,
    gg_6,
    gg_7,
    gg_8,
    gg_9,
    gg_10,
    gg_11,
    gg_12,
    gg_13,
    gg_14,
    gg_15,
    gg_16,
    gg_17,
    gg_18,
    gg_19,
]


def get_georgiev_params_for_aa(aa: str):
    return [gg[aa] for gg in georgiev_parameters]

def get_georgiev_params_for_seq(s):
    return np.concatenate([get_georgiev_params_for_aa(aa) for aa in s])

def seqs_to_georgiev(seqs):
    return np.stack([get_georgiev_params_for_seq(s) for s in seqs])


def generate_onehot(seqs: list) -> np.ndarray:
    """
    Builds a onehot encoding for a given combinatorial space.
    """
    # Make a dictionary that links amino acid to index
    one_hot_dict = {aa: i for i, aa in enumerate(ALL_AAS)}

    # Build an array of zeros
    onehot_array = np.zeros([len(seqs), len(seqs[0]), len(ALL_AAS)])

    # Loop over all combos. This should all be vectorized at some point.
    for i, combo in enumerate(seqs):

        # Loop over the combo and add ones as appropriate
        for j, character in enumerate(combo):

            # Add a 1 to the appropriate position
            onehot_ind = one_hot_dict[character]
            onehot_array[i, j, onehot_ind] = 1

    # Return the flattened array
    return onehot_array


def generate_georgiev(seqs: list) -> np.ndarray:
    """
    Builds a georgiev encoding for a given combinatorial space.
    """
    X = seqs_to_georgiev(seqs)
    return X


encoding_dict = {
    'one-hot' : generate_onehot,
    'georgiev' : generate_georgiev
}


######## MODELS ########
# code modified from https://github.com/google-research/slip/blob/main/models.py
class KerasModelWrapper:
    """Wraps a Keras model to have the sklearn model interface."""

    def __init__(
        self,
        model_build_fn: Callable,
        sequence_length: int,
        vocab_size: int,
        fit_kwargs: Dict = dict(),
    ):
        """Initialize a KerasModelWrapper.
        Args:
          model_build_fn: A function that when called with arguments
            `model_build_fn(sequence_length, vocab_size)` returns a Keras model.
          sequence_length: The length of input sequences.
          vocab_size: The one-hot dimension size for input sequences.
          fit_kwargs: An optional dictionary of keyword arguments passed to the
            Keras model.fit(**fit_kwargs). See
              https://keras.io/api/models/model_training_apis/ for more details.
        """
        self._model_build_fn = model_build_fn
        self._fit_kwargs = fit_kwargs
        self._sequence_length = sequence_length
        self._vocab_size = vocab_size

    def fit(self, X, y):
        self._model = self._model_build_fn(self._sequence_length, self._vocab_size)
        self._model.fit(X, y, **self._fit_kwargs)

    def predict(self, X):
        return self._model.predict(x=X, verbose=0).squeeze(axis=1)


def build_linear_model(model_kwargs):
    default_kwargs = {
        "ridge_alpha": 1.0,
        "ridge_fit_intercept": True,
    }
    kwargs = default_kwargs.copy()
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]

    model = linear_model.Ridge(**model_kwargs)
    return model


def build_boosting_model(model_kwargs):
    default_kwargs = {
        "objective": "reg:tweedie",
        "early_stopping_rounds": 10,
        "nthread": -1,
    }
    kwargs = default_kwargs.copy()
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]

    model = xgb.XGBRegressor(**model_kwargs)
    return model


def get_model(model_name, model_kwargs: Dict):
    """Returns model, flatten_inputs."""
    if model_name == "ridge":
        return build_linear_model(model_kwargs)
    elif model_name == "boosting":
        return build_boosting_model(model_kwargs)
    else:
        raise NotImplementedError


def ndcg(y_true, y_pred):
    y_true_normalized = y_true - min(y_true)
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))


class MLDEDataset(LibData):
    """Base class for labeled datasets linking sequence to fitness."""

    def __init__(self, input_csv: str, zs_predictor: str, scale_fit: str = "max"):
        """
        Args:
        - input_csv, str: path to the input csv file WITH ZS,
            ie. results/zs_comb/none/scale2max/DHFR.csv
        """

        super().__init__(input_csv, scale_fit)

        assert "zs" in self._input_csv, "Make sure the input csv has ZS scores"

        self._zs_predictor = zs_predictor

    def sample_top(self, cutoff: int, n_sample: int, seed: int) -> np.ndarray:
        """
        Samples n_samples from the top ZS scores 
        based on a given cutoff in the ranking and a seed.
        
        Args:
        - cutoff : number cutoff in the ranking
        - n_sample : number of samples to take
        - seed: seed for reproducibility
        
        Returns:
            1D np.ndarray of sampled sequences
        """
        if self.df_length <= cutoff:
            sorted = self.input_df
        else:
            sorted = self.sorted_df[:cutoff]
        
        options = sorted["AAs"].values
        np.random.seed(seed)
        return np.random.choice(options, n_sample, replace=False)

    def encode_X(self, encoding: str):
        """
        Encodes the input features based on the encoding type.
        """
        if encoding == 'one-hot':
            self.X = np.array(encoding_dict[encoding](self.all_combos)) 
            self.X = self.X.reshape(self.X.shape[0],-1) 

        self.input_dim = self.X.shape[1]
        self.n_residues = self.input_dim/len(ALL_AAS)
    
    def get_mask(self, seqs: list) -> list:
        """
        Returns an index mask for given sequences.
        """
        return list(self.input_df[self.input_df['AAs'].isin(seqs)].index)
    
    @property
    def sorted_df(self):
        if self._zs_predictor not in self.input_df.columns:
            print(f"ZS predictor {self._zs_predictor} not in input dataframe - dataframe NOT sorted.")
            return self.input_df.copy()
        else:
            return self.input_df.sort_values(by=self._zs_predictor, ascending=False).copy()
    
    @property
    def all_combos(self):
        return self.input_df['AAs'].values
    
    @property
    def y(self):
        return self.input_df['fitness']

class MLDESim(MLDEDataset):
    """
    Class for training and evaluating MLDE models
    for a given dataset, encoding, and model class.
    """

    def __init__(self, 
                 input_csv: str,
                 zs_predictor: str,
                 encoding: str,
                 ft_libs: list[int] | None,
                 model_class: str,
                 n_sample: int,
                 n_split: int = 5,
                 n_replicate: int = 100,
                 n_topseq: int = 384,
                 n_topns: list[int] = [96, 384],
                 n_worker: int = 1,
                 global_seed: int = 42,
                 verbose: bool = False,
                 save_model: bool = False,
                 scale_fit: str = "max",
                 save_path: str = "results/mlde",
                 ) -> None:
        
        """
        Args:
        - input_csv: str, path to the input csv file WITH ZS,
            ie. 'results/zs_comb/none/scale2max/DHFR.csv'
        - zs_predictor: str, name of the ZS predictor
        - encoding: str, encoding type
        - ft_libs: list[int] | None = None, list of sizes of focused training libraries
            ie. [149361, 32000, 16000, 8000, 4000]
        - model_class: str, model class
            ie. 'boosting'
        - n_sample: int, number of samples to train on
        - n_split: int = 5, number of splits for cross-validation
        - n_replicate: int = 100, number of replicates
        - n_topseq: int = 384, number of top sequences to save
        - n_topns: list[int] = [96, 384], number of top sequences to calculate max and mean fitness
        - n_worker: int = 1, number of workers for parallel processing
        - global_seed: int = 42, global seed for reproducibility
        - verbose: bool = False, verbose output
        - save_model: bool = False, save models
        - scale_fit: str, scaling type
        - save_path: str, path to save results
        """

        super().__init__(input_csv, zs_predictor, scale_fit)

        assert self.input_df[self.input_df["AA"].str.contains("\*")] == 0, "Make sure there are no stop codons in the input data"

        self._encoding = encoding

        if ft_libs is not None:
            self._ft_libs = ft_libs
        else:
            self._ft_libs = [self.df_length]

        self._n_solution = len(self._ft_libs)

        self._model_class = model_class
        self._n_sample = n_sample
        self._n_split = n_split
        self._n_replicate = n_replicate
        self._n_topseq = n_topseq
        self._n_topns = n_topns
        self._n_worker = n_worker
        self._verbose = verbose
        self._save_model = save_model
        self._save_path = checkNgen_folder(os.path.normpath(save_path))

        # init
        self.top_seqs = np.full((self._n_solution, self._n_replicate, self._n_topseq), "")
        self.ndcgs = np.zeros((self._n_solution, self._n_replicate))
        self.maxes = np.zeros((self._n_solution, self._n_replicate))
        self.means = np.zeros((self._n_solution, self._n_replicate))
        self.unique = np.zeros((self._n_solution, self._n_replicate))
        self.labelled = np.zeros((self._n_solution, self._n_replicate))

        # set up all random seeds
        np.random.seed(global_seed)
        random.seed(global_seed)
        self._subset_seeds = deepcopy([random.randint(0, 1000000) for _ in range(self._n_replicate)])

        self.encode_X(encoding=self._encoding)

        self.X_train_all = np.array(self.X)

        self.y_train_all = np.array(self.y)
        self.y_preds_all = np.zeros((self.df_length, self._n_replicate, self._n_split))

    def train_all(self):
        """
        Loops through all libraries to be sampled from (n_solutions) and 
        for each solution trains n_replicates of models. 
        Each model is an ensemble of n_splits models, 
        each trained on 90% of the subset selected randomly.

        Output: results for each of the models
        """
        with tqdm() as pbar:
            pbar.reset(self._n_solution * self._n_replicate * self._n_split)
            pbar.set_description("Training and evaluating")

            for k in range(self._n_solution):

                cutoff = self._ft_libs[k]
   
                for j in range(self._n_replicate):
                    
                    seqs = self.sample_top(
                        cutoff, self._n_sample, self._subset_seeds[j]
                    )

                    uniques = np.unique(seqs)
                    self.unique[k, j] = len(uniques)
                    mask = self.get_mask(uniques)
                    self.labelled[k, j] = len(mask)
                    combos_train = []

                    if self._save_model:
                        save_dir = checkNgen_folder(os.path.join(self._save_path, str(k), str(j)))
                   
                    for i in range(self._n_split):
                        if self._n_split > 1:
                            # boostrap ensembling with 90% of the data
                            train_mask, validation_mask = train_test_split(
                                mask, test_size=0.1, random_state=i
                            )
                        else:
                            train_mask = mask
                            validation_mask = mask  # used for validation if desired

                        X_train = self.X_train_all[train_mask]
                        y_train = self.y_train_all[train_mask]
                        combos_train += list(self.all_combos[train_mask])
                        X_validation = self.X_train_all[validation_mask]
                        y_validation = self.y_train_all[validation_mask]
                        y_preds, clf = self.train_single(
                            X_train, y_train, X_validation, y_validation
                        )

                        # remove this if you don't want to save the model
                        if self._save_model:
                            filename = "split" + str(i) + ".model"
                            clf.save_model(os.path.join(save_dir, filename))

                        self.y_preds_all[:, j, i] = y_preds
                        pbar.update()

                    # need to redo some of these, loop was in wrong place? 
                    # but it should be fine cause it gets replaced to the correct mean at the end
                    means = np.mean(self.y_preds_all, axis=2)
                    y_preds = means[:, j]

                    (
                        self.maxes[k, j],
                        self.means[k, j],
                        self.top_seqs[k, j, :],
                    ) = self.get_mlde_results(y_preds)
                    ndcg_value = ndcg(self.y_train_all, y_preds)
                    self.ndcgs[k, j] = ndcg_value

        pbar.close

        return (
            self.top_seqs,
            self.maxes,
            self.means,
            self.ndcgs,
            self.unique,
            self.labelled,
        )

    def train_single(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
    ):
        
        """
        Trains a single supervised ML model. 
        Returns the predictions on the training set and the trained model.
        """

        if self._model_class == "boosting":
            clf = get_model(
                self._model_class, model_kwargs={"nthread": self._num_worker}
            )
            eval_set = [(X_validation, y_validation)]
            clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            clf = get_model(self._model_class, model_kwargs={})
            clf.fit(X_train, y_train)

        y_preds = clf.predict(self.X_train_all)

        return y_preds, clf

    def get_mlde_results(
        self, y_preds: np.ndarray, topn: int
    ) -> tuple:
        
        """
        Calculates the MLDE results for a given set of predictions. 
        Returns the max and mean of the top 96 sequences and the top 500 sequences.

        Args:
            y_preds: the predictions on the training data
        """

        df = self.input_df.copy()
        df["y_preds"] = y_preds

        ##optionally filter out the sequences in the training set
        # data2 = data2[~data2['Combo'].isin(unique_seqs)]

        sorted = df.sort_values(by=["y_preds"], ascending=False)

        top_fit = sorted.iloc[:topn, :]["fitness"]
        max_fit = np.max(top_fit)
        mean_fit = np.mean(top_fit)

        # save the top n sequeneces
        top_seqs = sorted.iloc[:self._n_topseq, :]["AAs"].values

        ##for checking how many predictions are in the training set
        # top_seqs_96 = sorted.iloc[:96,:]['Combo'].values
        # print(len(np.intersect1d(np.array(unique_seqs), top_seqs_96)))

        return max_fit, mean_fit, top_seqs


def run_all_mlde(encodings: list[str],
                 model_classes: list[str],
                 n_sample: int,
                 n_subset: int,):
    
    """A function to run all MLDE simulations."""

    # init all outputs
    all_ndcgs = np.zeros((len(encodings), len(model_classes), len(n_sample), config['data_config']['n_solutions'], config['train_config']['n_subsets']))


"""
all_maxes = np.copy(all_ndcgs)
all_means = np.copy(all_ndcgs)
all_unique = np.copy(all_ndcgs)
all_labelled = np.copy(all_ndcgs)

all_top_seqs = np.full((len(encodings), len(model_classes), len(n_sampless), config['data_config']['n_solutions'], config['train_config']['n_subsets'], 500), 'VDGV')


for i, encoding in enumerate(encodings):
    for j, model_class in enumerate(model_classes):
        for k, n_samples in enumerate(n_sampless):
            
            #keep track of how long the computation took
            start = time.time()

            exp_name2 = encoding + '_' + model_class + '_' + str(n_samples)
            save_dir2 = os.path.join(os.getcwd(), 'MLDE_lite', 'saved', exp_name, exp_name2)

            print('\n###' + exp_name2 + '###')
            
            # Create save folder
            # if not os.path.exists(save_dir2):
            #     os.makedirs(save_dir2)

            mlde_sim = MLDESim(save_path=save_dir2,
                encoding = encoding, 
                model_class = model_class, 
                n_samples = n_samples, 
                model_config=config['model_config'],
                data_config=config['data_config'],
                train_config=config['train_config'])

            input_csv: str,
            zs_predictor: str,
            encoding: str,
            ft_libs: list[int] | None,
            model_class: str,
            n_sample: int,
            n_split: int = 5,
            n_replicate: int = 100,
            n_topseq: int = 384,
            n_topns: list[int] = [96, 384],
            n_worker: int = 1,
            global_seed: int = 42,
            verbose: bool = False,
            save_model: bool = False,
            scale_fit: str = "max",
            save_path: str = "results",
            
            top_seqs, maxes, means, ndcgs, unique, labelled =  mlde_sim.train_all()
            
            all_top_seqs[i, j, k, :, :, :] = top_seqs
            all_ndcgs[i, j, k, :, :] = ndcgs
            all_maxes[i, j, k, :, :] = maxes
            all_means[i, j, k, :, :] = means
            all_unique[i, j, k, :, :] = unique
            all_labelled[i, j, k, :, :] = labelled

            end = time.time()
            print('Time: ' + str(end-start))

mlde_results = {}
mlde_results['top_seqs'], mlde_results['maxes'], mlde_results['means'], mlde_results['ndcgs'], mlde_results['unique'], mlde_results['labelled'] = all_top_seqs, all_maxes, all_means, all_ndcgs, all_unique, all_labelled

np.save(os.path.join(save_dir, 'mlde_results.npy'), mlde_results)
"""