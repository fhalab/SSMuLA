"""
Contains classes for dataprocessing in this work
"""
from __future__ import annotations

# Import modules
import os
from glob import glob

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from Bio import SeqIO

# Import custom modules
from SSMuLA.util import checkNgen_folder

