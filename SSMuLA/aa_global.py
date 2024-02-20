"""Parameters for training and testing"""

from __future__ import annotations

from collections import Counter

import re

RAND_SEED = 42

TRANSLATE_DICT = {
    "AAA": "K",
    "AAT": "N",
    "AAC": "N",
    "AAG": "K",
    "ATA": "I",
    "ATT": "I",
    "ATC": "I",
    "ATG": "M",
    "ACA": "T",
    "ACT": "T",
    "ACC": "T",
    "ACG": "T",
    "AGA": "R",
    "AGT": "S",
    "AGC": "S",
    "AGG": "R",
    "TAA": "*",
    "TAT": "Y",
    "TAC": "Y",
    "TAG": "*",
    "TTA": "L",
    "TTT": "F",
    "TTC": "F",
    "TTG": "L",
    "TCA": "S",
    "TCT": "S",
    "TCC": "S",
    "TCG": "S",
    "TGA": "*",
    "TGT": "C",
    "TGC": "C",
    "TGG": "W",
    "CAA": "Q",
    "CAT": "H",
    "CAC": "H",
    "CAG": "Q",
    "CTA": "L",
    "CTT": "L",
    "CTC": "L",
    "CTG": "L",
    "CCA": "P",
    "CCT": "P",
    "CCC": "P",
    "CCG": "P",
    "CGA": "R",
    "CGT": "R",
    "CGC": "R",
    "CGG": "R",
    "GAA": "E",
    "GAT": "D",
    "GAC": "D",
    "GAG": "E",
    "GTA": "V",
    "GTT": "V",
    "GTC": "V",
    "GTG": "V",
    "GCA": "A",
    "GCT": "A",
    "GCC": "A",
    "GCG": "A",
    "GGA": "G",
    "GGT": "G",
    "GGC": "G",
    "GGG": "G",
}

# Amino acid code conversion
AA_DICT = {
    "Ala": "A",
    "Cys": "C",
    "Asp": "D",
    "Glu": "E",
    "Phe": "F",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Lys": "K",
    "Leu": "L",
    "Met": "M",
    "Asn": "N",
    "Pro": "P",
    "Gln": "Q",
    "Arg": "R",
    "Ser": "S",
    "Thr": "T",
    "Val": "V",
    "Trp": "W",
    "Tyr": "Y",
    "Ter": "*",
}

# the upper case three letter code for the amino acids
ALL_AAS_TLC_DICT = {k.upper(): v for k, v in AA_DICT.items() if v != "*"}

# the upper case three letter code for the amino acids
ALL_AAS_TLC = list(ALL_AAS_TLC_DICT.keys())

# All canonical amino acids
ALL_AAS = list(ALL_AAS_TLC_DICT.values())
ALL_AA_STR = "".join(ALL_AAS)
AA_NUMB = len(ALL_AAS)
ALLOWED_AAS = set(ALL_AAS)

# Create a new dictionary with values as keys and counts as values
CODON_COUNT_PER_AA = {aa: Counter(TRANSLATE_DICT.values())[aa] for aa in ALL_AAS + ["*"]}

# Create a dictionary that links each amino acid to an index
AA_TO_IND = {aa: i for i, aa in enumerate(ALL_AAS)}

# Define a expressions for parsing mutations in the format Wt##Mut
MUT_REGEX = re.compile("^([A-Z])([0-9]+)([A-Z])$")

START_AA_IND = 1
