"""Test MLDE."""

import sys
import os

from glob import glob

from datetime import datetime

from SSMuLA.plm_finetune import train_predict_per_protein
from SSMuLA.util import checkNgen_folder

if __name__ == "__main__":

    # log outputs
    f = open(
        os.path.join(
            checkNgen_folder("logs/finetune"),
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.out",
        ),
        "w",
    )
    sys.stdout = f

    
    for landscape in sorted(glob("results/zs_comb/all/*.csv")):
        for i in range(5):
            train_predict_per_protein(
                df_csv=landscape,
                rep=i,
            )
    
    f.close()

    """
    train_predict_per_protein(
        df_csv: str,  # csv file with landscape data
        rep: int,  # replicate number
        checkpoint: str = "facebook/esm2_t33_650M_UR50D",  # model checkpoint
        n_sample: int = 384,  # number of train+val
        zs_predictor: str = "none",  # zero-shot predictor
        ft_frac: float = 0.125,  # fraction of data for focused sampling
        plot_dir: str = "results/finetuning/plot",  # directory to save the plot
        model_dir: str = "results/finetuning/model",  # directory to save the model
        pred_dir: str = "results/finetuning/predictions",  # directory to save the predictions
        train_kwargs: dict = {},  # additional training arguments
    )
    """