"""A scirpt for the MLDE_lite"""

from __future__ import annotations

from typing import Callable, Dict

import os
import time
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy
from tqdm.auto import tqdm

import xgboost as xgb
from sklearn import linear_model
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

from SSMuLA.aa_global import ALL_AAS, georgiev_parameters
from SSMuLA.landscape_global import LibData, LIB_INFO_DICT
from SSMuLA.util import checkNgen_folder, get_file_name


def get_georgiev_params_for_aa(aa):
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


encoding_dict = {"one-hot": generate_onehot, "georgiev": generate_georgiev}


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
            sorted = self.input_df.copy()
        else:
            sorted = self.sorted_df[:cutoff].copy()

        options = sorted["AAs"].values
        np.random.seed(seed)
        return np.random.choice(options, n_sample, replace=False).copy()

    def encode_X(self, encoding: str):
        """
        Encodes the input features based on the encoding type.
        """
        if encoding == "one-hot":
            self.X = np.array(encoding_dict[encoding](self.all_combos))
            self.X = self.X.reshape(self.X.shape[0], -1)

        self._n_sites = self.X.shape[1] / len(ALL_AAS)

        assert self._n_sites == self.n_site, "Number of sites do not match"

    def get_mask(self, seqs: list) -> list:
        """
        Returns an index mask for given sequences.
        """
        return list(self.input_df[self.input_df["AAs"].isin(seqs)].index)

    @property
    def sorted_df(self):
        if self._zs_predictor not in self.input_df.columns:
            print(
                f"ZS predictor {self._zs_predictor} not in input dataframe - dataframe NOT sorted."
            )
            return self.input_df.copy()
        else:
            return self.input_df.sort_values(
                by=self._zs_predictor, ascending=False
            ).copy()

    @property
    def all_combos(self):
        return self.input_df["AAs"].values.copy()

    @property
    def y(self):
        return self.input_df["fitness"].copy()


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


class MLDESim(MLDEDataset):
    """Class for training and evaluating MLDE models."""

    def __init__(
        self,
        input_csv: str,
        zs_predictor: str,
        encoding: str,
        model_class: str,
        n_sample: int,
        n_split: int = 5,
        n_replicate: int = 100,
        n_top: int = 384,
        n_worker: int = 1,
        global_seed: int = 42,
        verbose: bool = False,
        save_model: bool = False,
        ft_libs: list[float] = [1],
        scale_fit: str = "max",
        save_path: str = "results/mlde",
    ) -> None:

        """
        Args:
        - input_csv: str, path to the input csv file WITH ZS,
            ie. 'results/zs_comb/none/scale2max/DHFR.csv'
        - zs_predictor: str, name of the ZS predictor
        - encoding: str, encoding type
        - ft_libs: list[float] = [1], list of percent of focused training libraries
            ie. [1, 0.5, 0.25, 0.125]
        - model_class: str, model class
            ie. 'boosting'
        - n_sample: int, number of samples to train on
        - n_split: int = 5, number of splits for cross-validation
        - n_replicate: int = 100, number of replicates
        - n_top: int = 384, number of top sequences to calculate max and mean fitness
        - n_worker: int = 1, number of workers for parallel processing
        - global_seed: int = 42, global seed for reproducibility
        - verbose: bool = False, verbose output
        - save_model: bool = False, save models
        - scale_fit: str, scaling type
        - save_path: str, path to save results
        """

        super().__init__(input_csv, zs_predictor, scale_fit)

        assert (
            len(self.input_df[self.input_df["AAs"].str.contains("\*")]) == 0
        ), "Make sure there are no stop codons in the input data"

        self._encoding = encoding

        if ft_libs != [1]:
            self._ft_libs = [int(f * len(ALL_AAS) ** self.n_site) for f in ft_libs if f < 1]
            print(self._ft_libs)
        else:
            self._ft_libs = [self.df_length]

        self._n_solution = len(self._ft_libs)

        self._model_class = model_class
        self._n_sample = n_sample
        self._n_split = n_split
        self._n_replicate = n_replicate
        self._n_top = n_top
        self._n_worker = n_worker
        self._verbose = verbose
        self._save_model = save_model
        self._save_path = checkNgen_folder(os.path.normpath(save_path))

        # init
        self.top_seqs = np.full((self._n_solution, self._n_replicate, self._n_top), "")
        self.ndcgs = np.zeros((self._n_solution, self._n_replicate))
        self.maxes = np.copy(self.ndcgs)
        self.means = np.copy(self.ndcgs)
        self.unique = np.copy(self.ndcgs)
        self.labelled = np.copy(self.ndcgs)
        self.y_preds = np.zeros((self._n_solution, self._n_replicate, self.df_length))

        # set up all random seeds
        np.random.seed(global_seed)
        random.seed(global_seed)
        self._subset_seeds = deepcopy(
            [random.randint(0, 1000000) for _ in range(self._n_replicate)]
        )

        self.encode_X(encoding=self._encoding)

        self.X_train_all = np.array(self.X)
        self.y_train_all = np.array(self.y)

        # init for all the predictions for taking avg over n_splits
        self.y_preds_all = np.zeros((self.df_length, self._n_replicate, self._n_split))

    def train_all(self):
        """
        Loops through all libraries to be sampled from (n_solutions) and
        for each solution trains n_subsets of models.
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
                    # need to check if the seeding process works the same way

                    seqs = self.sample_top(
                        cutoff, self._n_sample, self._subset_seeds[j]
                    )

                    uniques = np.unique(seqs)
                    self.unique[k, j] = len(uniques)
                    mask = self.get_mask(uniques)
                    self.labelled[k, j] = len(mask)
                    combos_train = []

                    if self._save_model:
                        save_dir = checkNgen_folder(
                            os.path.join(self._save_path, str(k), str(j))
                        )

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

                    # TODO check if loop in the right place?
                    # it gets replaced to the correct mean at the end
                    means = np.mean(self.y_preds_all, axis=2)
                    y_preds = means[:, j]

                    (
                        self.maxes[k, j],
                        self.means[k, j],
                        self.top_seqs[k, j, :],
                    ) = self.get_mlde_results(y_preds)

                    ndcg_value = ndcg(self.y_train_all, y_preds)
                    self.ndcgs[k, j] = ndcg_value

                    self.y_preds[k, j, :] = y_preds

        pbar.close

        return (
            self.top_seqs,
            self.maxes,
            self.means,
            self.ndcgs,
            self.unique,
            self.labelled,
            self.y_preds,
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
                self._model_class, model_kwargs={"nthread": self._n_worker}
            )
            eval_set = [(X_validation, y_validation)]
            clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            clf = get_model(self._model_class, model_kwargs={})
            clf.fit(X_train, y_train)

        y_preds = clf.predict(self.X_train_all)

        return y_preds, clf

    def get_mlde_results(
        self,
        y_preds: np.ndarray,
    ) -> tuple:
        """
        Calculates the MLDE results for a given set of predictions.
        Returns the max and mean of the top 96 sequences and the top 500 sequences.

        Args:
            data2: pandas dataframe with all sequences and fitness labels in the combinatorial space
            y_preds: the predictions on the training data
            unique_seqs: the unique sequences in the training data
        """

        df = self.input_df.copy()
        df["y_preds"] = y_preds

        ##optionally filter out the sequences in the training set
        # data2 = data2[~data2['Combo'].isin(unique_seqs)]

        sorted = df.sort_values(by=["y_preds"], ascending=False)

        top_fit = sorted.iloc[: self._n_top, :]["fitness"]
        max_fit = np.max(top_fit)
        mean_fit = np.mean(top_fit)

        # save the top
        top_seqs = sorted.iloc[: self._n_top, :]["AAs"].astype(str).values

        ##for checking how many predictions are in the training set
        # top_seqs_96 = sorted.iloc[:96,:]['Combo'].values
        # print(len(np.intersect1d(np.array(unique_seqs), top_seqs_96)))

        return max_fit, mean_fit, top_seqs


def run_mlde_lite(
    input_csv: str,
    zs_predictor: str,
    scale_fit: str = "max",
    filter_min_by: str = "none",
    encodings: list[str] = ["one-hot"],
    ft_libs: list[float] = [1],
    model_classes: list[str] = ["boosting", "ridge"],
    n_samples: list[int] = [384],
    n_split: int = 5,
    n_replicate: int = 100,
    n_top: int = 384,
    n_worker: int = 1,
    global_seed: int = 42,
    verbose: bool = False,
    save_model: bool = False,
    mlde_folder: str = "results/mlde",
    exp_name="",
):

    """
    Run MLDE

    Args:
    - input_csv: str, path to the input csv file WITH ZS,
            ie. 'results/zs_comb/none/scale2max/DHFR.csv'
    - zs_predictor: str, name of the ZS predictor
    - ft_libs: list[float] = [1], list of percent of focused training libraries
            ie. [1, 0.5, 0.25, 0.125]

    """

    if len(exp_name) == 0:
        exp_name = get_file_name(input_csv)

    save_dir = checkNgen_folder(
        os.path.join(
            os.path.normpath(mlde_folder),
            "saved",
            zs_predictor,
            filter_min_by,
            scale_fit,
            exp_name,
        )
    )

    config_folder = checkNgen_folder(os.path.dirname(save_dir.replace("saved", "configs")))
    config_path = os.path.join(config_folder, f"{exp_name}_{n_top}.json")

    # Load JSON config file
    with open(config_path, "w") as f:
        config_dict = {
            "data_config": {
                "input_csv": input_csv,
                "zs_predictor": zs_predictor,
                "encoding": encodings,
                "ft_libs": ft_libs,
            },
            "model_config": {
                "model_classes": model_classes,
            },
            "train_config": {
                "n_sample": n_samples,
                "n_splits": n_split,
                "n_replicate": n_replicate,
                "n_worker": n_worker,
                "global_seed": global_seed,
                "verbose": verbose,
                "save_model": save_model,
            },
            "eval_config": {"n_top": n_top},
        }
        json.dump(config_dict, f, indent=4)

    print("Save directory:\t {}".format(save_dir))
    print("Config file:\t {}".format(config_path))
    for key, value in config_dict.items():
        print(f"{key}:\t {value}")

    # Start training
    all_ndcgs = np.zeros(
        (
            len(encodings),
            len(model_classes),
            len(n_samples),
            len(ft_libs),
            n_replicate,
        )
    )
    all_maxes = np.copy(all_ndcgs)
    all_means = np.copy(all_ndcgs)
    all_unique = np.copy(all_ndcgs)
    all_labelled = np.copy(all_ndcgs)

    all_top_seqs = np.full(
        (
            len(encodings),
            len(model_classes),
            len(n_samples),
            len(ft_libs),
            n_replicate,
            n_top,
        ),
        "",
    )
    all_y_preds = np.zeros(
        (
            len(encodings),
            len(model_classes),
            len(n_samples),
            len(ft_libs),
            n_replicate,
            len(pd.read_csv(input_csv)),
        )
    )

    for i, encoding in enumerate(encodings):
        for j, model_class in enumerate(model_classes):
            for k, n_sample in enumerate(n_samples):

                # keep track of how long the computation took
                start = time.time()

                exp_name_dets = f"{encoding}_{model_class}_sample{str(n_sample)}_top{str(n_top)}"

                print(f"Running {exp_name_dets}...")

                mlde_sim = MLDESim(
                    input_csv=input_csv,
                    zs_predictor=zs_predictor,
                    encoding=encoding,
                    ft_libs=ft_libs,
                    model_class=model_class,
                    n_sample=n_sample,
                    n_split=n_split,
                    n_replicate=n_replicate,
                    n_top=n_top,
                    n_worker=n_worker,
                    global_seed=global_seed,
                    verbose=verbose,
                    save_model=save_model,
                    scale_fit=scale_fit,
                    save_path=save_dir,
                )

                (
                    top_seqs,
                    maxes,
                    means,
                    ndcgs,
                    unique,
                    labelled,
                    y_preds,
                ) = mlde_sim.train_all()

                all_top_seqs[i, j, k, :, :, :] = top_seqs
                all_ndcgs[i, j, k, :, :] = ndcgs
                all_maxes[i, j, k, :, :] = maxes
                all_means[i, j, k, :, :] = means
                all_unique[i, j, k, :, :] = unique
                all_labelled[i, j, k, :, :] = labelled
                all_y_preds[i, j, k, :, :, :] = y_preds

                end = time.time()
                print("Time: " + str(end - start))

    mlde_results = {}
    (
        mlde_results["top_seqs"],
        mlde_results["maxes"],
        mlde_results["means"],
        mlde_results["ndcgs"],
        mlde_results["unique"],
        mlde_results["labelled"],
        mlde_results["y_preds"],
    ) = (
        all_top_seqs,
        all_maxes,
        all_means,
        all_ndcgs,
        all_unique,
        all_labelled,
        all_y_preds,
    )

    comb_exp_dets = "|".join(encodings) + "_" + "|".join(model_classes)
    np.save(os.path.join(save_dir, f"{comb_exp_dets}_sample{str(n_sample)}_top{str(n_top)}.npy"), mlde_results)


def run_all_mlde(
    zs_folder: str = "results/zs_comb",
    filter_min_by: str = "none",
    scale_type: str = "scale2max",
    zs_predictors: list[str] = ["none", "Triad", "ev", "esm"],
    ft_lib_fracs: list[float] = [0.5, 0.25, 0.125],
    encodings: list[str] = ["one-hot"],
    model_classes: list[str] = ["boosting", "ridge"],
    n_samples: list[int] = [384],
    n_split: int = 5,
    n_replicate: int = 100,
    n_tops: list[int] = [96, 384],
    n_worker: int = 1,
    global_seed: int = 42,
    verbose: bool = False,
    save_model: bool = False,
    mlde_folder: str = "results/mlde",
):
    """
    Run all MLDE give zs combined csvs
    """

    for input_csv in sorted(
        glob(f"{os.path.normpath(zs_folder)}/{filter_min_by}/{scale_type}/*.csv")
    ):
        for zs in zs_predictors:
            if zs == "none":
                ft_libs = [1]
            else:
                zs = f"{zs}_score"
                ft_libs = ft_lib_fracs

            for n_top in n_tops:

                run_mlde_lite(
                    input_csv=input_csv,
                    zs_predictor=zs,
                    scale_fit=scale_type.split("scale2")[1],
                    filter_min_by=filter_min_by,
                    encodings=encodings,
                    ft_libs=ft_libs,
                    model_classes=model_classes,
                    n_samples=n_samples,
                    n_split=n_split,
                    n_replicate=n_replicate,
                    n_top=n_top,
                    n_worker=n_worker,
                    global_seed=global_seed,
                    verbose=verbose,
                    save_model=save_model,
                    mlde_folder=mlde_folder,
                    exp_name="",
                )