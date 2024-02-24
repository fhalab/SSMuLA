"""A script for calculating pairwise epistasis"""

from __future__ import annotations

# General imports
import glob
import os
import re
import pickle
import datetime

# Data manipulation
import pandas as pd
import numpy as np

# Basic plotting
import bokeh
from bokeh.io import export_svg
from bokeh.models import NumeralTickFormatter
from bokeh.io import output_notebook
output_notebook()


import panel as pn
pn.config.comms = "vscode"

# Large data plotting
import datashader as ds
from holoviews.operation.datashader import datashade, rasterize

# Making graphs
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from tqdm.auto import tqdm
from multiprocessing import Pool
from operator import itemgetter

import holoviews as hv
hv.extension('bokeh')

from SSMuLA.vis import JSON_THEME, one_decimal_x, one_decimal_y, fixmargins
from SSMuLA.aa_global import ALL_AAS

from bokeh.themes.theme import Theme

hv.renderer('bokeh').theme = JSON_THEME


def make_new_sequence(input_seq: str, new_AA: str, position: int) -> str:
    """
    Make a new sequence by replacing the amino acid at a specific position.

    Args:
        - input_seq (str): The input sequence.
        - new_AA (str): The new amino acid to replace.
        - position (int): The position in the sequence to replace.

    Returns:
        - str: The new sequence with the replaced amino acid.
    """
    seq_list = list(input_seq)
    seq_list[position] = new_AA
    return ''.join(seq_list)

def hamming(str1: str, str2: str) -> int:
    """
    Calculate the Hamming distance between two strings.

    Args:
        - str1 (str): The first string.
        - str2 (str): The second string.

    Returns:
        - int: The Hamming distance between the two strings.
    """
    assert len(str1) == len(str2)

    distance = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            distance += 1
    return distance