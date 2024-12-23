# modify from repo:
# https://github.com/RSchmirler/data-repo_plm-finetune-eval/blob/main/notebooks/finetune/Finetuning_per_protein.ipynb

# import dependencies
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

import re
import numpy as np
import pandas as pd
from copy import deepcopy

import transformers, datasets
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import EsmModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, set_seed

import peft
from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    inject_adapter_in_model,
    LoraConfig,
)

from evaluate import load
from datasets import Dataset

from tqdm import tqdm
import random

from scipy import stats
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from SSMuLA.util import checkNgen_folder, get_file_name


# # Set environment variables to run Deepspeed from a notebook
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "9994"  # modify if RuntimeError: Address already in use
# os.environ["RANK"] = "0"
# os.environ["LOCAL_RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"


# Deepspeed config for optimizer CPU offload

ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto",
        },
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
        },
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

RAND_SEED_LIST = deepcopy([random.randint(0, 1000000) for _ in range(50)])

# load ESM2 models
def load_esm_model(checkpoint, num_labels, half_precision, full=False, deepspeed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if half_precision and deepspeed:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels, torch_dtype=torch.float16
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_labels
        )

    if full == True:
        return model, tokenizer

    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
    )

    model = inject_adapter_in_model(peft_config, model)

    # Unfreeze the prediction head
    for (param_name, param) in model.classifier.named_parameters():
        param.requires_grad = True

    return model, tokenizer


# Set random seeds for reproducibility of your trainings run
def set_seeds(s):
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    set_seed(s)


# Dataset creation
def create_dataset(tokenizer, seqs, labels):
    tokenized = tokenizer(seqs, max_length=1024, padding=True, truncation=True)
    dataset = Dataset.from_dict(tokenized)
    dataset = dataset.add_column("labels", labels)

    return dataset


def plot_train_val(
    history: list,  # training history
    landscape: str,  # landscape name
):

    # Get loss, val_loss, and the computed metric from history
    loss = [x["loss"] for x in history if "loss" in x]
    val_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

    # Get spearman (for regression) or accuracy value (for classification)
    if [x["eval_spearmanr"] for x in history if "eval_spearmanr" in x] != []:
        metric = [x["eval_spearmanr"] for x in history if "eval_spearmanr" in x]
    else:
        metric = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]

    epochs = [x["epoch"] for x in history if "loss" in x]

    # Create a figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Plot loss and val_loss on the first y-axis
    line1 = ax1.plot(epochs, loss, label="train_loss")
    line2 = ax1.plot(epochs, val_loss, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Plot the computed metric on the second y-axis
    line3 = ax2.plot(epochs, metric, color="red", label="val_metric")
    ax2.set_ylabel("Metric")
    ax2.set_ylim([0, 1])

    # Combine the lines from both y-axes and create a single legend
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower left")

    # add title to the figure
    plt.title(f"Training curves for {landscape}")

    # return the plot but do not save it
    return fig


def save_model(model, filepath):
    # Saves all parameters that were changed during finetuning

    # Create a dictionary to hold the non-frozen parameters
    non_frozen_params = {}

    # Iterate through all the model parameters
    for param_name, param in model.named_parameters():
        # If the parameter has requires_grad=True, add it to the dictionary
        if param.requires_grad:
            non_frozen_params[param_name] = param

    # Save only the finetuned parameters
    torch.save(non_frozen_params, filepath)


# Main training fuction
def train_per_protein(
    checkpoint,  # model checkpoint
    train_df,  # training data
    valid_df,  # validation data
    seed,  # random seed
    num_labels=1,  # 1 for regression, >1 for classification
    # effective training batch size is batch * accum
    # we recommend an effective batch size of 8
    batch=8,  # for training
    accum=2,  # gradient accumulation
    val_batch=16,  # batch size for evaluation
    epochs=10,  # training epochs
    lr=3e-4,  # recommended learning rate
    deepspeed=False,  # if gpu is large enough disable deepspeed for training speedup
    mixed=True,  # enable mixed precision training
    full=False,  # enable training of the full model (instead of LoRA)
    # gpu = 1           #gpu selection (1 for first gpu)
):

    print("Model used:", checkpoint, "\n")

    # Set all random seeds
    set_seeds(seed)

    # load model
    model, tokenizer = load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)

    # Preprocess inputs
    # Replace uncommon AAs with "X"
    train_df["seq"] = train_df["seq"].str.replace(
        "|".join(["O", "B", "U", "Z", "J"]), "X", regex=True
    )
    valid_df["seq"] = valid_df["seq"].str.replace(
        "|".join(["O", "B", "U", "Z", "J"]), "X", regex=True
    )

    # Create Datasets
    train_set = create_dataset(
        tokenizer, list(train_df["seq"]), list(train_df["fitness"])
    )
    valid_set = create_dataset(
        tokenizer, list(valid_df["seq"]), list(valid_df["fitness"])
    )

    # Huggingface Trainer arguments
    args = TrainingArguments(
        "./",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        deepspeed=ds_config if deepspeed else None,
        fp16=mixed,
    )

    # Metric definition for validation data
    def compute_metrics(eval_pred):
        if num_labels > 1:  # for classification
            metric = load("accuracy")
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
        else:  # for regression
            metric = load("spearmanr")
            predictions, labels = eval_pred

        return metric.compute(predictions=predictions, references=labels)

    # Trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    return tokenizer, model, trainer.state.log_history


def train_predict_per_protein(
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
):
    """ """

    landscape = get_file_name(df_csv)

    seed = RAND_SEED_LIST[rep]

    df = pd.read_csv(df_csv)

    if zs_predictor == "none":
        df_sorted = df.copy()
    elif zs_predictor not in df.columns:
        print(f"{zs_predictor} not in the dataframe")
        df_sorted = df.copy()
    else:
        df_sorted = (
            df.sort_values(by=zs_predictor, ascending=False)
            .copy()[: int(len(df) * ft_frac)]
            .copy()
        )

    # randomly sample rows from the dataframe
    train_val_df = (
        df_sorted.sample(n=n_sample, random_state=seed).reset_index(drop=True).copy()
    )

    # split the train_val_df into 90%training and 10% validation sets
    train_df = (
        train_val_df.sample(frac=0.9, random_state=seed).reset_index(drop=True).copy()
    )
    valid_df = train_val_df.drop(train_df.index).reset_index(drop=True).copy()

    tokenizer, model, history = train_per_protein(
        checkpoint, train_df, valid_df, seed=seed, **train_kwargs
    )

    # save the model
    save_model(
        model,
        os.path.join(
            checkNgen_folder(os.path.join(model_dir, landscape)),
            f"{landscape}_{str(rep)}.pth",
        ),
    )

    # plot the training history
    fig = plot_train_val(history, landscape)

    # save the plot
    fig.savefig(
        os.path.join(
            checkNgen_folder(os.path.join(plot_dir, landscape)),
            f"{landscape}_{str(rep)}.png",
        )
    )

    # # Evaluate the model on the test set
    # #Use reloaded model
    # model = model_reload
    # del model_reload

    # # Set the device to use
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model.to(device)

    # create Dataset
    test_set = create_dataset(tokenizer, list(df["seq"]), list(df["fitness"]))
    # make compatible with torch DataLoader
    test_set = test_set.with_format("torch", device=device)

    # Create a dataloader for the test dataset
    test_dataloader = DataLoader(test_set, batch_size=16, shuffle=False)

    # Put the model in evaluation mode
    model.eval()

    # Make predictions on the test dataset
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # add batch results(logits) to predictions
            predictions += model.float()(
                input_ids, attention_mask=attention_mask
            ).logits.tolist()

    # flatten the prediction to one single list
    # save predictions as a new column in the test dataframe
    df["predictions"] = np.array(predictions).flatten()

    # save the dataframe
    df.to_csv(
        os.path.join(
            checkNgen_folder(os.path.join(pred_dir, landscape)),
            f"{landscape}_{str(rep)}.csv",
        ),
        index=False,
    )

    return df