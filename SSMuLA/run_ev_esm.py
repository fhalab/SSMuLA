"""
A script for generating zs scores gratefully adapted from EmreGuersoy's work
"""

# Import packages
import glob
import json

import numpy as np
import pandas as pd
import random
from collections import Counter


import argparse
from pathlib import Path

from tqdm import tqdm
from typing import List, Tuple, Optional


import warnings

from SSMuLA.zs_calc import calc_all_zs
from SSMuLA.util import get_file_name, checkNgen_folder


# TODO clean up path to be dataset name independent

def create_parser():
    parser = argparse.ArgumentParser(description="Run zero-shot predictions")

    parser.add_argument(
        "--landscape_folder", 
        type=str, 
        default="data/processed",
        metavar="LSF",
        help="A folder path for all landscape data"
    )

    parser.add_argument(
        "--dataset_list",
        type=json.loads,
        metavar="dsl",
        default=[],
        help="default dataset list empty to use glob for all",
    )

    parser.add_argument(
        "--output_folder", 
        type=str, 
        default="results/zs",
        metavar="OPF",
        help="A output folder path with landscape subfolders"
    )

    parser.add_argument(
        "--zs_model_names",
        type=str,
        metavar="ZSMN",
        help="A str of name(s) of zero-shot models to use seperated by comma, \
            available: 'esm', 'ev', developing: 'ddg', 'Bert', all: 'all' runs all currently available models \
        ie, 'esm, ev'",
    )

    parser.add_argument(
        "--ev_model_folder", 
        type=str, 
        default="data/",
        metavar="EVF",
        help="folder for evmodels"
    )

    parser.add_argument(
        "--regen_esm",
        type=bool,
        default=False,
        metavar="RG",
        help="if regenerate esm logits or load directly"
    )

    parser.add_argument(
        "--rerun_zs",
        type=bool,
        default=False,
        metavar="RR",
        help="if append new zs to current csv or create new output"
    )

    return parser


def main(args):

    # Input processing

    calc_all_zs(landscape_folder = args.landscape_folder, 
                dataset_list=args.dataset_list,
                output_folder = args.output_folder,
                zs_model_names = args.zs_model_names,
                ev_model_folder = args.ev_model_folder,
                regen_esm = args.regen_esm,
                rerun_zs = args.rerun_zs)

# Run EvMutation
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)