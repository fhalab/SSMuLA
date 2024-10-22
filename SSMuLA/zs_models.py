"""
A script for generating zs scores gratefully adapted from EmreGuersoy's work
"""

# Import packages
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# EvCouplings
from evcouplings.couplings import CouplingsModel
from evcouplings.align.alignment import Alignment

# ESM
import esm

# from esm.model.msa_transformer import MSATransformer

# Pytorch
import torch

# Other
from tqdm import tqdm
from typing import List, Tuple, Optional


import warnings

# from transformers import BertTokenizer, BertModel, BertForMaskedLM, pipeline


class ZeroShotPrediction:
    """
    Zero-Shot Analysis on a given sequence and model.

    Input:
            - Model Path obtained from EvCouplings
            - Sequence or mutation list stored as csv, 
                where variant column must be [wtAA]_[pos][newAA]
    Output: - Scores for each variant
    """

    def __init__(self, df, wt_seq):
        # make sure no stopbonds are present
         
        self.df = df
        self.wt_sequence = wt_seq

    def _get_n_df(self, n: int = 1):
        """Get n data frame with n mutants"""
        return self.df[self.df["combo"].apply(lambda x: len(x) == n)].copy()


class ddG(ZeroShotPrediction):
    def __init__(self, df=None):
        super().__init__(df)
        pass


class EvMutation(ZeroShotPrediction):
    """
    Perform the EvMutation analysis on a given sequence and model.

    Input:
            - Model Path obtained from EvCouplings
            - Sequence or mutation list stored as csv
    Output: - EvMutation (delta) score for each variant
    """

    def __init__(self, df, wt_sequence, model_path):
        super().__init__(df, wt_sequence)
        self.model_path = model_path
        print("Loading model...")
        self.model = self.load_model()
        print("Model loaded")
        self.idx_map = self.model.index_map

    def check_idx(self, pos: list = [1]):
        """Check if the position is in the index map"""
        all_pos = all([p in self.idx_map for p in pos])

        if all_pos:
            return True
        else:
            return False

    def load_model(self):
        self.model = CouplingsModel(self.model_path)
        return self.model

    def _get_hamiltonian(self, mt, wt, pos):
        delta_E, _, _ = self.model.delta_hamiltonian([(pos, wt, mt)])
        return delta_E

    def _get_n_hamiltonian(self, combo: list):
        """Get the hamiltonian for n mutants"""
        delta_E, _, _ = self.model.delta_hamiltonian(combo)
        return delta_E

    def upload_dms_coupling(self, dms_coupling):
        """Input: - dms_coupling: Path to the dms_coupling file (csv)"""
        self.dms_coupling = dms_coupling

        return self.dms_coupling

    def get_single_mutant_scores(self):
        """Get the single mutant scores for the dms_coupling file"""

        # Left join the dms_coupling file with the data set
        self.single_mutant_scores = pd.merge(
            self.dms_coupling,
            self.df,
            how="left",
            left_on=["pos", "wt"],
            right_on=["pos", "wt"],
        )

        return self.single_mutant_scores

    def run_evmutation(self, df, _multi=True, _epistatic=False):
        """
        Run EvMutation for all variants in the data set

        Input:  - df: Data set containing the variants, loops trough column = 'combo'
                - _mean: If True, the mean of the probabilities is calculated. 
                    If False, the sum of the probabilities is calculated
        Output: - Score for each variant"""
        score = np.zeros(len(df))
        wt_sequence = list(self.wt_sequence)

        for i, combo in enumerate(df["combo"]):
            # Prepare mut list for EvMutation
            mut_list = []
            # Check if positions are in index map
            if self.check_idx(df["pos"].iloc[i]):

                if _multi:

                    single_mutant_scores = np.zeros((1, len(combo)))

                    for j, mt in enumerate(combo):
                        if mt == "WT" or mt == "NA":
                            score[i] = np.nan
                            continue

                        else:
                            pos_wt = int(
                                df["pos"].iloc[i][j] - 1
                            )  # Position of the mutation with python indexing
                            pos_ev = int(
                                df["pos"].iloc[i][j]
                            )  # Position of the mutation with python indexing
                            # Get single scores
                            single_mutant_scores[0, j] = self._get_hamiltonian(
                                mt, wt_sequence[pos_wt], pos_ev
                            )  # TODO: Improve at one point
                            mut_list.append((pos_ev, wt_sequence[pos_wt], mt))

                    # Run EvMutation
                    score[i] = self._get_n_hamiltonian(mut_list)

                    if _epistatic:
                        score[i] = score[i] - np.sum(single_mutant_scores)

                    # TODO: Get epistatic scores dE = dE_combo - sum(dE_single_mutant)
                else:
                    if combo == "WT" or combo == "NA":
                        score[i] = np.nan
                        continue

                    else:
                        mt = combo[0]
                        pos_wt = int(df["pos"].iloc[i][0] - 1)
                        pos_ev = int(df["pos"].iloc[i][0])
                        mut_list.append((pos_ev, wt_sequence[pos_wt], mt))
                        # score[i] = self._get_n_hamiltonian(mut_list)
                        score[i] = self._get_hamiltonian(
                            mt, wt_sequence[pos_wt], pos_ev
                        )
            else:
                score[i] = np.nan
                continue

        return score

    def _get_n_score(self, n: list = [1]):
        """Get any score for each variant in the data set"""
        df_n_list = []

        # Get the n mutant scores
        for i in n:
            # Filter out n mutants
            df_n = self._get_n_df(i)
            if df_n.empty:  # Check if the DataFrame is empty after filtering
                assert "Data set is empty"
                continue

            if i == 1:
                score_n = self.run_evmutation(df_n, _multi=False)
            else:
                score_n = self.run_evmutation(df_n, _multi=True)

            # Add column with number of mutations

            df_n.loc[:, "ev_score"] = score_n
            df_n.loc[:, "n_mut"] = i
            # score_n = pd.DataFrame(score_n, columns=['ev_score'])

            # if fit or fitness
            if "fit" in df_n.columns:
                fit_col = "fit"
            elif "fitness" in df_n.columns:
                fit_col = "fitness"

            # Choose only the columns we want
            df_n = df_n[["muts", fit_col, "n_mut", "ev_score"]]
            df_n_list.append(df_n)

        return pd.concat(df_n_list, axis=0)


class ESM(ZeroShotPrediction):
    def __init__(self, df, wt_seq, logits_path="", regen_esm=False):
        super().__init__(df, wt_seq)
        self.df = df
        self.wt_sequence = wt_seq
        (
            self.model,
            self.alphabet,
            self.batch_converter,
            self.device,
        ) = self._infer_model()
        self.mask_string, self.cls_string, self.eos_string = (
            self.alphabet.mask_idx,
            self.alphabet.cls_idx,
            self.alphabet.eos_idx,
        )
        self.alphabet_size = len(self.alphabet)

        if logits_path != "" and os.path.exists(logits_path) and not(regen_esm):
            print(f"{logits_path} exists and regen_esm = {regen_esm}. Loading...")
            self.logits = np.load(logits_path)
        else:
            print(f"Generating {logits_path}...")
            self.logits = self._get_logits()

    def _infer_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        # model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        model = model.to(device)
        print("Using device:", device)
        return model, alphabet, batch_converter, device

    def _get_logits(self):

        data_wt = [("WT", self.wt_sequence)]
        # Get Batch tokens for WT data
        batch_labels_wt, batch_strs_wt, batch_tokens_wt = self.batch_converter(data_wt)

        logits = np.zeros((len(self.wt_sequence), self.alphabet_size))

        for (i, seq) in enumerate(data_wt[0][1]):
            batch_tokens_masked = batch_tokens_wt.clone()
            batch_tokens_masked[0, i] = self.alphabet.mask_idx
            batch_tokens_masked = batch_tokens_masked.to(self.device)

            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked)["logits"], dim=-1
                ).cpu().numpy()

            logits[i] = token_probs[0, i+1]

        return logits

    def _get_mutant_prob(self, mt, wt, pos):
        """Get the probability of the mutant given the wild type sequence at certain position."""
        
        wt_idx = self.alphabet.get_idx(wt)
        mt_idx = self.alphabet.get_idx(mt)

        return self.logits[pos, mt_idx] - self.logits[pos, wt_idx]

    def run_esm(self, df, _sum=True):
        """
        Run ESM model for all variants in the data set

        Input:  - logits: Logits of the wild type sequence
                - df: Data set containing the variants, loops trough column = 'Combo' and 'Pos'
                - _sum: If True, the sum of the probabilities is calculated. 
                    If False, the mean of the probabilities is calculated

        Output: - Score for each variant
        """
        score = np.zeros(len(df))
        wt_sequence = list(self.wt_sequence)

        if _sum:
            for i, combo in enumerate(df["combo"]):
                s = np.zeros(len(combo))
                for j, mt in enumerate(combo):
                    if mt == "WT":
                        score[i] = 0
                        continue

                    elif mt == "NA":
                        score[i] = np.nan
                        continue

                    else:
                        pos = (
                            int(df["pos"].iloc[i][j]) - 1
                        )  # Position of the mutation with python indexing
                        wt = wt_sequence[pos]
                        s[j] = self._get_mutant_prob(mt=mt, wt=wt, pos=pos)
                    
                    score[i] += s.sum()

        else:
            for i, combo in enumerate(df["combo"]):

                mt = combo[0]

                if mt == "WT":
                    score[i] = 0
                    continue

                elif mt == "NA":
                    score[i] = np.nan
                    continue

                else:
                    pos = int(df["pos"].iloc[i][0] - 1)
                    wt = wt_sequence[pos]
                    score[i] = self._get_mutant_prob(mt=mt, wt=wt, pos=pos)

        return score

    def _get_n_df(self, n: int = 1):
        """Get n data frame with n mutants"""

        return self.df[self.df["combo"].apply(lambda x: len(x) == n)].copy()

    def _get_n_score(self, n: list = [1]):
        """Get any score for each variant in the data set"""
        df_n_list = []

        # Get the n mutant scores
        for i in n:
            # Filter out n mutants
            df_n = self._get_n_df(i)
            if df_n.empty:  # Check if the DataFrame is empty after filtering
                assert "Data set is empty"
                continue

            if i == 1:
                score_n = self.run_esm(df_n, _sum=False)
            else:
                score_n = self.run_esm(df_n, _sum=True)

            # Add column with number of mutations

            df_n.loc[:, "esm_score"] = score_n
            df_n.loc[:, "n_mut"] = i
            # score_n = pd.DataFrame(score_n, columns=['ev_score'])
            # if fit or fitness
            if "fit" in df_n.columns:
                fit_col = "fit"
            elif "fitness" in df_n.columns:
                fit_col = "fitness"

            # Choose only the columns we want
            df_n = df_n[["muts", fit_col, "n_mut", "esm_score"]]
            df_n_list.append(df_n)

        return pd.concat(df_n_list, axis=0)
