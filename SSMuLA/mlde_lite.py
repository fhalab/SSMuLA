"""A scirpt for the MLDE_lite"""

from __future__ import annotations

import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

import os
import gc
import time
import json
import random
import tables
import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy
from tqdm.auto import tqdm

import xgboost as xgb
from scipy.stats import spearmanr
from sklearn import linear_model
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split

from SSMuLA.aa_global import (
    ALL_AAS,
    georgiev_parameters,
    DEFAULT_LEARNED_EMB_COMBO,
    DEFAULT_LEARNED_EMB_DIR,
)
from SSMuLA.landscape_global import LibData, LIB_INFO_DICT, n_mut_cutoff_dict
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


def generate_precomp_emb(emb_path, seqs: list) -> list:
    """
    Generate the emb from precomputed leanred embeddings
    """

    X = []

    # Close all the open files
    tables.file._open_files.close_all()

    # Open the embeddings table
    with tables.open_file(emb_path, mode="r") as emb_table:

        emb_table.flush()

        # Convert the list of AAs to strings
        bseqs = [aa.encode("utf-8") for aa in seqs]

        # Get the AAs dataset
        AAs_dataset = emb_table.root.AAs[:]

        # Iterate over each AA and find its index
        for seq in bseqs:
            try:
                idx = np.where(AAs_dataset == seq)[0][0]  # Get the first occurrence
                # Use the index to extract the corresponding embeddings
                embedding = emb_table.root.emb[idx]
                X.append(embedding)
            except IndexError:
                print(f"AA '{seq}' not found in the dataset.")
                X.append(None)  # Append None if AA is not found

        emb_table.close()

    return X


encoding_dict = {"one-hot": generate_onehot, "georgiev": generate_georgiev}


class MLDEDataset(LibData):
    """Base class for labeled datasets linking sequence to fitness."""

    def __init__(
        self,
        input_csv: str,
        zs_predictor: str,
        scale_fit: str = "max",
        filter_min_by: str = "none",
        n_mut_cutoff: int = 0,
    ):
        """
        Args:
        - input_csv, str: path to the input csv file WITH ZS,
            ie. results/zs_comb/none/scale2max/all/DHFR.csv
        - n_mut_cutoff: number of mutations cutoff, default 0 meaning no cutoff
        """

        super().__init__(input_csv, scale_fit)

        assert "zs" in self._input_csv, "Make sure the input csv has ZS scores"

        self._zs_predictor = zs_predictor
        self._filter_min_by = filter_min_by
        self._n_mut_cutoff = n_mut_cutoff

        self._n_mut_cuttoff_df = self._get_n_mut_cuttoff_df()

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

        df = self._n_mut_cuttoff_df.copy()

        if len(df) <= cutoff or self._zs_predictor not in df.columns:
            df_sorted = df.copy()
        else:
            df_sorted = (
                df.sort_values(by=self._zs_predictor, ascending=False)
                .copy()[:cutoff]
                .copy()
            )

        options = df_sorted["AAs"].values

        n_choice = (
            min(n_sample, len(options)) if len(options) > n_sample else len(options)
        )

        np.random.seed(seed)
        return np.random.choice(options, n_choice, replace=False).copy()

    def encode_X(self, encoding: str):
        """
        Encodes the input features based on the encoding type.
        """
        if encoding == "one-hot":
            self.X = np.array(encoding_dict[encoding](self.all_combos))
            # need to flatten the array
            self.X = self.X.reshape(self.X.shape[0], -1)

            self._n_sites = self.X.shape[1] / len(ALL_AAS)

            assert self._n_sites == self.n_site, "Number of sites do not match"

        elif "esm" in encoding:
            esm, combo = encoding.split("-")
            emb_paths = glob(
                f"{os.path.join(DEFAULT_LEARNED_EMB_DIR, combo)}/{self.lib_name}-{esm}*.h5"
            )
            assert len(emb_paths) == 1, "More than one or no learned embeddings found"
            emb_path = emb_paths[0]
            print(f"Given {esm} and {combo} loading {emb_path}")
            self.X = np.array(generate_precomp_emb(emb_path, self.all_combos))
            # no reshape needed
        else:
            raise NotImplementedError

    def _get_mask(self, seqs: list) -> list:
        """
        Returns an index mask for given sequences.
        """
        # Find indices where values in 'A' match each value in 'values_to_find'
        indices = []
        for seq in seqs:
            matched_indices = self._n_mut_cuttoff_df.index[self._n_mut_cuttoff_df["AAs"] == seq].tolist()
            assert len(matched_indices) == 1, f"{len(matched_indices)} {seq} found in the dataframe"
            indices.append(matched_indices[0])
        return indices
        # return list(self._n_mut_cuttoff_df[self._n_mut_cuttoff_df["AAs"].isin(seqs)].index)

    def _get_n_mut_cuttoff_df(self):
        """
        Get the dataframe with n_mut cutoff
        """
        if self._n_mut_cutoff > 0:
            df = self.filtered_df[self.filtered_df["n_mut"] <= self._n_mut_cutoff]
            print(f"All {n_mut_cutoff_dict[self._n_mut_cutoff]} total {len(df)}")
        else:
            df = self.filtered_df
        return df.copy()

    @property
    def filtered_df(self):

        df = self.input_df.copy()
        # make sure no stop codon
        df = df[~df["AAs"].str.contains("\*")]

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
    def len_filtered_df(self):
        return len(self.filtered_df)
    
    @property
    def all_combos(self):
        return self.filtered_df["AAs"].values.copy()

    @property
    def y(self):
        """
        Return all the fitness values
        """
        return self.filtered_df["fitness"].copy()

    @property
    def max_fit_seq(self) -> np.ndarray:
        """Return the max fit seq"""
        return self.filtered_df.loc[self.filtered_df["fitness"].idxmax()]["AAs"]

    @property
    def n_mut_cuttoff_df(self):
        """
        Return the dataframe with n_mut cutoff
        """
        return self._n_mut_cuttoff_df

    @property
    def len_n_mut_cuttoff_df(self):
        """
        Return the length of the dataframe with n_mut cutoff
        """
        return len(self._n_mut_cuttoff_df)


def build_linear_model(model_kwargs):
    default_kwargs = {
        "ridge_alpha": 1.0,
        "ridge_fit_intercept": True,
    }
    kwargs = deepcopy(default_kwargs)
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
    kwargs = deepcopy(default_kwargs)
    for key in default_kwargs.keys():
        if key in model_kwargs:
            kwargs[key] = model_kwargs[key]

    model = xgb.XGBRegressor(**model_kwargs)
    return model


def get_model(model_name: str, model_kwargs: dict):
    """Returns model, flatten_inputs."""
    if model_name == "ridge":
        return build_linear_model(model_kwargs)
    elif model_name == "boosting":
        return build_boosting_model(model_kwargs)
    else:
        raise NotImplementedError


def ndcg(y_true, y_pred) -> float:
    """
    Calculate the NDCG score for a given set of true and predicted values.
    """
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
        boosting_n_worker: int = 1,
        global_seed: int = 42,
        verbose: bool = False,
        save_model: bool = False,
        ft_libs: list[float] = [1],
        scale_fit: str = "max",
        filter_min_by: str = "none",
        n_mut_cutoff: int = 0,
        save_path: str = "results/mlde",
    ) -> None:

        """
        Args:
        - input_csv: str, path to the input csv file WITH ZS,
            ie. 'results/zs_comb/none/scale2max/all/DHFR.csv'
        - zs_predictor: str, name of the ZS predictor
        - encoding: str, encoding type,
            ie `one-hot`, or `esm2_t33_650M_UR50D-flatten_site`
        - ft_libs: list[float] = [1], list of percent of focused training libraries
            ie. [1, 0.5, 0.25, 0.125]
        - model_class: str, model class
            ie. 'boosting'
        - n_sample: int, number of samples to train on
        - n_split: int = 5, number of splits for cross-validation
        - n_replicate: int = 100, number of replicates
        - n_top: int = 384, number of top sequences to calculate max and mean fitness
        - boosting_n_worker: int = 1, number of workers for parallel processing
        - global_seed: int = 42, global seed for reproducibility
        - verbose: bool = False, verbose output
        - save_model: bool = False, save models
        - scale_fit: str, scaling type
        - filter_min_by: str, filter minimum fitness by
        - n_mut_cutoff: int = 0, number of mutations cutoff, default 0 meaning no cutoff
        - save_path: str, path to save results
        """

        super().__init__(
            input_csv, zs_predictor, scale_fit, filter_min_by, n_mut_cutoff
        )

        assert (
            len(self.filtered_df[self.filtered_df["AAs"].str.contains("\*")]) == 0
        ), "Make sure there are no stop codons in the input data"

        self._encoding = encoding

        if ft_libs != [1]:
            ft_lib_mut_numbs = [
                int(f * len(ALL_AAS) ** self.n_site) for f in ft_libs if f < 1
            ]
            print(
                "{} focused training library sizes valid only if bigger than {} df size {}".format(
                    ft_lib_mut_numbs,
                    n_mut_cutoff_dict[self._n_mut_cutoff],
                    self.len_n_mut_cuttoff_df,
                )
            )
            ft_lib_w_nums = [
                ft_lib
                if ft_lib <= self.len_n_mut_cuttoff_df
                else self.len_n_mut_cuttoff_df
                for ft_lib in ft_lib_mut_numbs
            ]
            # use set to filter out duplicates
            self._ft_libs = deepcopy(sorted(list(set(ft_lib_w_nums)), reverse=True))
            #  [x if x <= criteria else 0 for x in original_list]
            print(f"Valid focused training library sizes: {self._ft_libs}")

        else:
            self._ft_libs = [self.len_n_mut_cuttoff_df]

        self._n_solution = len(self._ft_libs)

        if self._n_solution == 0:
            print("No valid focused training library sizes")

        self._model_class = model_class
        self._n_sample = n_sample
        self._n_split = n_split
        self._n_replicate = n_replicate
        self._n_top = n_top
        self._boosting_n_worker = boosting_n_worker
        self._verbose = verbose
        self._save_model = save_model
        self._save_path = checkNgen_folder(os.path.normpath(save_path))

        # init
        self.top_seqs = np.full(
            (self._n_solution, self._n_replicate, self._n_top),
            "".join(["n"] * self.n_site),
        )

        self.ndcgs = np.zeros((self._n_solution, self._n_replicate))
        self.rhos = np.zeros(self.ndcgs.shape)

        # for all
        self.all_maxes = np.zeros(self.ndcgs.shape)
        self.all_means = np.zeros(self.ndcgs.shape)

        # for the top n
        self.top_maxes = np.zeros(self.ndcgs.shape)
        self.top_means = np.zeros(self.ndcgs.shape)

        self.if_truemaxs = np.zeros(self.ndcgs.shape)
        self.truemax_inds = np.zeros(self.ndcgs.shape)

        self.unique = np.zeros(self.ndcgs.shape)
        self.labelled = np.zeros(self.ndcgs.shape)

        self.y_preds = np.zeros((self._n_solution, self._n_replicate, self.df_length))
        self.y_trues = np.zeros(self.y_preds.shape)

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
        self.y_preds_nsplits = np.zeros((self.df_length, self._n_replicate, self._n_split))

    def train_all(self) -> dict:
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

            if self._n_solution > 0:

                for k in range(self._n_solution):

                    cutoff = self._ft_libs[k]

                    for j in range(self._n_replicate):
                        # need to check if the seeding process works the same way

                        seqs = self.sample_top(
                            cutoff=cutoff,
                            n_sample=self._n_sample,
                            seed=self._subset_seeds[j],
                        )

                        uniques = np.unique(seqs)
                        self.unique[k, j] = len(uniques)
                        mask = self._get_mask(seqs)
                        self.labelled[k, j] = len(mask)
                        combos_train = []

                        if self._save_model:
                            save_model_dir = checkNgen_folder(
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

                            if self._save_model:
                                filename = "split" + str(i) + ".model"
                                clf.save_model(os.path.join(save_model_dir, filename))

                            self.y_preds_nsplits[:, j, i] = y_preds
                            
                            pbar.update()
                        
                        # TODO check if loop in the right place?
                        # it gets replaced to the correct mean at the end
                        means = np.mean(self.y_preds_nsplits, axis=2)
                        y_preds = means[:, j]

                        self.all_maxes[k, j] = np.max(y_preds)
                        self.all_means[k, j] = np.mean(y_preds)

                        (
                            self.top_maxes[k, j],
                            self.top_means[k, j],
                            self.if_truemaxs[k, j],
                            self.truemax_inds[k, j],
                            self.top_seqs[k, j, :],
                        ) = self.get_mlde_results(y_preds)

                        self.ndcgs[k, j] = ndcg(self.y_train_all, y_preds)
                        self.rhos[k, j] = spearmanr(self.y_train_all, y_preds)[0]

                        self.y_preds[k, j, :] = y_preds
                        self.y_trues[k, j, :] = self.y_train_all

            else:
                print("No valid focused training library sizes. No output updates.")

        pbar.close

        return {
            "all_maxes": self.all_maxes,
            "all_means": self.all_means,
            "ndcgs": self.ndcgs,
            "rhos": self.rhos,
            "if_truemaxs": self.if_truemaxs,
            "truemax_inds": self.truemax_inds,
            "top_maxes": self.top_maxes,
            "top_means": self.top_means,
            "top_seqs": self.top_seqs,
            "unique": self.unique,
            "labelled": self.labelled,
            "y_preds": self.y_preds,
            "y_trues": self.y_trues,
        }
            
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
                self._model_class, model_kwargs={"nthread": self._boosting_n_worker}
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
        Returns the max and mean of the top n sequences and the top n sequences.

        Args:
        - y_preds:np.array, the predictions on the training data

        Returns:
        - max_fit: float, the max fitness in the top n sequences
        - mean_fit: float, the mean fitness in the top n sequences
        - if_truemax: int, if the top n sequences contain the max fitness sequence
        - truemax_ind: int, the index of the max fitness sequence
        - top_seqs: list, the top n sequences
        """

        df = self.filtered_df.copy()
        df["y_preds"] = y_preds

        ##optionally filter out the sequences in the training set
        # data2 = data2[~data2['AAs'].isin(unique_seqs)]

        df_sorted = df.sort_values(by=["y_preds"], ascending=False).reset_index(drop=True)

        top_fit = df_sorted.iloc[: self._n_top, :]["fitness"]
        max_fit = np.max(top_fit)
        mean_fit = np.mean(top_fit)

        # get the ind of the true max from df_sorted
        truemax_ind = df_sorted[df_sorted["AAs"] == self.max_fit_seq].index[0]

        # save the top seq
        top_seqs = df_sorted.iloc[: self._n_top, :]["AAs"].astype(str).values

        #  if true max in topn
        if_truemax = int(self.max_fit_seq in top_seqs)

        ##for checking how many predictions are in the training set
        # top_seqs_96 = df_sorted.iloc[:96,:]['Combo'].values
        # print(len(np.intersect1d(np.array(unique_seqs), top_seqs_96)))

        return max_fit, mean_fit, if_truemax, truemax_ind, top_seqs

    @property
    def used_ft_libs(self):
        """Return the actual ft_libs number for each"""
        return self._ft_libs


def pad_to_shape(source_array: np.ndarray, target_shape, pad_value: float = np.nan):
    """
    Pads a source array with a pad_value to match the target_shape and
    return the same if the source and target shapes match.

    Args:
    - source_array: numpy array to be padded
    - target_shape: tuple of the desired shape
    - pad_value: value to pad with, default is np.nan
    """

    padding = [(0, max(0, t - s)) for s, t in zip(source_array.shape, target_shape)]

    return np.pad(source_array, padding, mode="constant", constant_values=pad_value)


def run_mlde_lite(
    input_csv: str,
    zs_predictor: str,
    scale_fit: str = "max",
    filter_min_by: str = "none",
    n_mut_cutoff: int = 0,
    encodings: list[str] = ["one-hot"],
    ft_libs: list[float] = [1],
    model_classes: list[str] = ["boosting", "ridge"],
    n_samples: list[int] = [384],
    n_split: int = 5,
    n_replicate: int = 50,
    n_top: int = 384,
    boosting_n_worker: int = 1,
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
            ie. 'results/zs_comb/none/scale2max/all/DHFR.csv'
    - zs_predictor: str, name of the ZS predictor
    - ft_libs: list[float] = [1], list of percent of focused training libraries
            ie. [1, 0.5, 0.25, 0.125]
    """

    if len(exp_name) == 0:
        exp_name = get_file_name(input_csv)

    save_dir = checkNgen_folder(
        os.path.join(
            checkNgen_folder(os.path.normpath(mlde_folder)),
            "saved",
            zs_predictor,
            f"{filter_min_by}-{n_mut_cutoff_dict[n_mut_cutoff]}",
            f"scale2{scale_fit}",
            exp_name,
        )
    )

    config_folder = checkNgen_folder(
        os.path.dirname(save_dir.replace("saved", "configs"))
    )

    print("Save directory:\t {}".format(save_dir))

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
    all_rhos = np.zeros(all_ndcgs.shape)
    all_all_maxes = np.zeros(all_ndcgs.shape)
    all_all_means = np.zeros(all_ndcgs.shape)
    all_top_maxes = np.zeros(all_ndcgs.shape)
    all_top_means = np.zeros(all_ndcgs.shape)
    all_unique = np.zeros(all_ndcgs.shape)
    all_labelled = np.zeros(all_ndcgs.shape)
    all_if_truemaxs = np.zeros(all_ndcgs.shape)
    all_truemax_inds = np.zeros(all_ndcgs.shape)

    all_top_seqs = np.full(
        (
            len(encodings),
            len(model_classes),
            len(n_samples),
            len(ft_libs),
            n_replicate,
            n_top,
        ),
        "".join(["n"] * len(LIB_INFO_DICT[get_file_name(input_csv)]["positions"])),
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

    all_y_trues = np.zeros(all_y_preds.shape)

    for i, encoding in enumerate(encodings):
        for j, model_class in enumerate(model_classes):
            for k, n_sample in enumerate(n_samples):

                # keep track of how long the computation took
                start = time.time()

                exp_name_dets = (
                    f"{encoding}_{model_class}_sample{str(n_sample)}_top{str(n_top)}"
                )

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
                    boosting_n_worker=boosting_n_worker,
                    global_seed=global_seed,
                    verbose=verbose,
                    save_model=save_model,
                    scale_fit=scale_fit,
                    filter_min_by=filter_min_by,
                    n_mut_cutoff=n_mut_cutoff,
                    save_path=save_dir,
                )

                mlde_sim_dict = mlde_sim.train_all()

                all_all_maxes[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["all_maxes"], (len(ft_libs), n_replicate)
                )
                all_all_means[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["all_means"], (len(ft_libs), n_replicate)
                )
                all_ndcgs[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["ndcgs"], (len(ft_libs), n_replicate)
                )
                all_rhos[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["rhos"], (len(ft_libs), n_replicate)
                )
                
                all_if_truemaxs[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["if_truemaxs"], (len(ft_libs), n_replicate)
                )
                all_truemax_inds[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["truemax_inds"], (len(ft_libs), n_replicate)
                )

                all_top_maxes[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["top_maxes"], (len(ft_libs), n_replicate)
                )
                all_top_means[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["top_means"], (len(ft_libs), n_replicate)
                )
                all_top_seqs[i, j, k, :, :, :] = pad_to_shape(
                    mlde_sim_dict["top_seqs"], (len(ft_libs), n_replicate, n_top)
                )
                all_unique[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["unique"], (len(ft_libs), n_replicate)
                )
                all_labelled[i, j, k, :, :] = pad_to_shape(
                    mlde_sim_dict["labelled"], (len(ft_libs), n_replicate)
                )
                all_y_preds[i, j, k, :, :, :] = pad_to_shape(
                    mlde_sim_dict["y_preds"], (len(ft_libs), n_replicate, mlde_sim.len_filtered_df)
                )
                all_y_trues[i, j, k, :, :, :] = pad_to_shape(
                    mlde_sim_dict["y_trues"], (len(ft_libs), n_replicate, mlde_sim.len_filtered_df)
                )

                end = time.time()
                print("Time: " + str(end - start))

    config_dict = {
        "data_config": {
            "input_csv": input_csv,
            "zs_predictor": zs_predictor,
            "encoding": encodings,
            "n_site": mlde_sim.n_site,
            "len_filtered_df": mlde_sim.len_filtered_df,
            "ft_libs": mlde_sim.used_ft_libs,
            "scale_fit": scale_fit,
            "filter_min_by": filter_min_by,
            "n_mut_cutoff": n_mut_cutoff,
            "max_fit_seq": mlde_sim.max_fit_seq,
        },
        "model_config": {
            "model_classes": model_classes,
        },
        "train_config": {
            "n_sample": n_samples,
            "n_splits": n_split,
            "n_replicate": n_replicate,
            "boosting_n_worker": boosting_n_worker,
            "global_seed": global_seed,
            "verbose": verbose,
            "save_model": save_model,
        },
        "eval_config": {"n_top": n_top},
    }

    # put all in npy
    mlde_results = {}

    (
        mlde_results["config"],
        mlde_results["all_maxes"],
        mlde_results["all_means"],
        mlde_results["ndcgs"],
        mlde_results["rhos"],
        mlde_results["if_truemaxs"],
        mlde_results["truemax_inds"],
        mlde_results["top_maxes"],
        mlde_results["top_means"],
        mlde_results["top_seqs"],
        mlde_results["unique"],
        mlde_results["labelled"],
        mlde_results["y_preds"],
        mlde_results["y_trues"],
    ) = (
        config_dict,
        all_all_maxes,
        all_all_means,
        all_ndcgs,
        all_rhos,
        all_if_truemaxs,
        all_truemax_inds,
        all_top_maxes,
        all_top_means,
        all_top_seqs,
        all_unique,
        all_labelled,
        all_y_preds,
        all_y_trues,
    )

    comb_exp_dets = "|".join(encodings) + "_" + "|".join(model_classes)
    sample_dets = "|".join([str(n) for n in n_samples])
    comb_full_dets = f"{comb_exp_dets}_sample{sample_dets}_top{str(n_top)}"
    np_path = os.path.join(
        save_dir, f"{comb_full_dets}.npy"
    )
    print(f"Saving {np_path}...")
    np.save(np_path, mlde_results, allow_pickle=True)

    # Delete the variable
    del mlde_results

    config_path = os.path.join(
        config_folder, f"{comb_full_dets}.json"
    )

    # Record JSON config file
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)

    print("Config file:\t {}".format(config_path))
    for key, value in config_dict.items():
        print(f"{key}:\t {value}")

    # Manually run the garbage collector to free up the memory
    gc.collect()


def run_all_mlde(
    zs_folder: str = "results/zs_comb",
    filter_min_by: str = "none",
    n_mut_cutoffs: list[int] = [0, 1, 2],
    scale_type: str = "scale2max",
    zs_predictors: list[str] = ["none", "Triad", "ev", "esm", "esmif"],
    ft_lib_fracs: list[float] = [0.5, 0.25, 0.125],
    encodings: list[str] = ["one-hot"],
    model_classes: list[str] = ["boosting", "ridge"],
    n_samples: list[int] = [384],
    n_split: int = 5,
    n_replicate: int = 50,
    n_tops: list[int] = [96, 384],
    boosting_n_worker: int = 1,
    global_seed: int = 42,
    verbose: bool = False,
    save_model: bool = False,
    mlde_folder: str = "results/mlde",
):
    """
    Run all MLDE give zs combined csv
    """

    mlde_folder = checkNgen_folder(os.path.normpath(mlde_folder))

    for input_csv in sorted(
        glob(f"{os.path.normpath(zs_folder)}/{filter_min_by}/{scale_type}/all/*.csv")
    ):
        for n_mut_cutoff in n_mut_cutoffs:
            for zs_predictor in zs_predictors:
                if zs_predictor == "none":
                    ft_libs = [1]
                else:
                    zs_predictor = f"{zs_predictor}_score"
                    ft_libs = ft_lib_fracs

                for n_top in n_tops:

                    print(
                        "Running MLDE for {} with {} zero-shot predictor, \
                            {} mut number, {} top output...".format(
                            input_csv,
                            zs_predictor,
                            n_mut_cutoff_dict[n_mut_cutoff],
                            n_top,
                        )
                    )

                    run_mlde_lite(
                        input_csv=input_csv,
                        zs_predictor=zs_predictor,
                        scale_fit=scale_type.split("scale2")[1],
                        filter_min_by=filter_min_by,
                        n_mut_cutoff=n_mut_cutoff,
                        encodings=encodings,
                        ft_libs=ft_libs,
                        model_classes=model_classes,
                        n_samples=n_samples,
                        n_split=n_split,
                        n_replicate=n_replicate,
                        n_top=n_top,
                        boosting_n_worker=boosting_n_worker,
                        global_seed=global_seed,
                        verbose=verbose,
                        save_model=save_model,
                        mlde_folder=mlde_folder,
                        exp_name="",
                    )


def run_all_mlde_parallelized(
    zs_folder: str = "results/zs_comb",
    filter_min_by: str = "none",
    n_mut_cutoffs: list[int] = [0, 1, 2],
    scale_type: str = "scale2max",
    zs_predictors: list[str] = ["none", "Triad", "ev", "esm", "esmif"],
    ft_lib_fracs: list[float] = [0.5, 0.25, 0.125],
    encodings: list[str] = ["one-hot"] + DEFAULT_LEARNED_EMB_COMBO,
    model_classes: list[str] = ["boosting", "ridge"],
    n_samples: list[int] = [384],
    n_split: int = 5,
    n_replicate: int = 100,
    n_tops: list[int] = [96, 384],
    boosting_n_worker: int = 1,
    n_job: int = 128,
    global_seed: int = 42,
    verbose: bool = False,
    save_model: bool = False,
    mlde_folder: str = "results/mlde",
    all_libs: bool = True,
    lib_list: list[str] = [],
):
    
    """
    Run all MLDE give zs combined csv
    
    Args:
    - zs_folder: str, path to the folder with zero-shot combined CSV files
    - filter_min_by: str, filter minimum fitness by
    - n_mut_cutoffs: list[int], list of number of mutations cutoff
    - scale_type: str, scaling type
    - zs_predictors: list[str], list of zero-shot predictors
    - ft_lib_fracs: list[float], list of percent of focused training libraries
    - encodings: list[str], list of encoding types
    - model_classes: list[str], list of model classes
    - n_samples: list[int], list of number of samples to train on
    - n_split: int, number of splits for cross-validation
    - n_replicate: int, number of replicates
    - n_tops: list[int], list of number of top sequences to calculate max and mean fitness
    - boosting_n_worker: int, number of workers for parallel processing
    - n_job: int, number of jobs to run in parallel
    - global_seed: int, global seed for reproducibility
    - verbose: bool, verbose output
    - save_model: bool, save models
    - mlde_folder: str, path to save results
    - all_libs: bool, run all libraries
    - lib_list: list[str], list of libraries to run
    """

    mlde_folder = checkNgen_folder(os.path.normpath(mlde_folder))

    # Create a list to hold tasks for parallel execution
    tasks = []

    if all_libs or len(lib_list) == 0:
        lib_csv_list = sorted(
            glob(f"{os.path.normpath(zs_folder)}/{filter_min_by}/{scale_type}/all/*.csv")
        )
    else:
        lib_csv_list = sorted(
            [f"{os.path.normpath(zs_folder)}/{filter_min_by}/{scale_type}/all/{lib}.csv" for lib in lib_list]
        )

    # Iterate over each combination of parameters to create tasks
    for input_csv in tqdm(lib_csv_list):
        for n_mut_cutoff in n_mut_cutoffs:
            for zs_predictor in zs_predictors:
                # Determine feature libraries based on the predictor
                if zs_predictor == "none":
                    ft_libs = [1]
                else:
                    zs_predictor = f"{zs_predictor}_score"
                    ft_libs = ft_lib_fracs

                for n_top in n_tops:
                    # Print a message if verbose is True
                    if verbose:
                        print(
                            f"Queuing MLDE for {input_csv} with {zs_predictor} zero-shot predictor, "
                            + f"{n_mut_cutoff} mut number, {n_top} top output..."
                        )

                    exp_name = get_file_name(input_csv)
                    scale_fit = scale_type.split("scale2")[1]

                    # need to make and check all subfolders before running...
                    mlde_save_sub_dir = checkNgen_folder(
                        os.path.join(
                            checkNgen_folder(os.path.normpath(mlde_folder)),
                            "saved",
                            zs_predictor,
                            f"{filter_min_by}-{n_mut_cutoff_dict[n_mut_cutoff]}",
                            f"scale2{scale_fit}",
                            exp_name,
                        )
                    )

                    # do the same for config
                    checkNgen_folder(
                        os.path.dirname(mlde_save_sub_dir.replace("saved", "configs"))
                    )

                    # Append the task arguments as a tuple to the tasks list
                    tasks.append(
                        {
                            "input_csv": input_csv,
                            "zs_predictor": zs_predictor,
                            "scale_fit": scale_fit,
                            "filter_min_by": filter_min_by,
                            "n_mut_cutoff": n_mut_cutoff,
                            "encodings": encodings,
                            "ft_libs": ft_libs,
                            "model_classes": model_classes,
                            "n_samples": n_samples,
                            "n_split": n_split,
                            "n_replicate": n_replicate,
                            "n_top": n_top,
                            "boosting_n_worker": boosting_n_worker,
                            "global_seed": global_seed,
                            "verbose": verbose,
                            "save_model": save_model,
                            "mlde_folder": mlde_folder,
                            "exp_name": "",
                        }
                    )

    # Run tasks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_job) as executor:
        # Submit tasks
        future_to_task = {
            executor.submit(run_mlde_lite, **task): task for task in tasks
        }

        # Retrieve results as tasks are completed
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if verbose:
                    print(f"Task completed: {task}")
                    print(f"Result: {result}")
            except Exception as exc:
                # Print the task details and exception info
                print(f"Task generated an exception: {task}")
                print(f"Exception type: {type(exc).__name__}")
                print(f"Exception message: {exc}")
                # Print the full traceback to help identify where the exception was raised
                print("Traceback:")
                traceback.print_tb(exc.__traceback__)