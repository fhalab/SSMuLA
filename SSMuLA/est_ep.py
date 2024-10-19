"""
A script to estimate pairwise epistasis in a given dataset.
"""

import os
from glob import glob
import pandas as pd
import numpy as np

from SSMuLA.landscape_global import LIB_INFO_DICT, lib2prot
from SSMuLA.fitness_process_vis import parse_lib_stat

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import itertools

# Suppress PDB construction warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

def get_ca_distance(structure, residue1, residue2, chain_id='A'):
    """
    Calculate the C-alpha distance between two residues within a structure object.

    Parameters:
    - structure: PDB structure object.
    - residue1, residue2: Residue numbers (integers) of the residues to measure.
    - chain_id: ID of the chain where the residues are located (default is chain 'A').

    Returns:
    - distance: Distance between the C-alpha atoms of the specified residues.
    """
    # Select chain and residues
    chain = structure[0][chain_id]  # Assume using the first model
    res1 = chain[residue1]
    res2 = chain[residue2]

    # Fetch the 'CA' atoms if they exist
    if 'CA' in res1 and 'CA' in res2:
        ca1 = res1['CA']
        ca2 = res2['CA']
        # Calculate distance
        distance = ca1 - ca2
        return distance
    else:
        return None

def calculate_pairwise_distances(pdb_file, residues_dict, chain_id='A'):
    """
    Calculate pairwise C-alpha distances for a set of residues specified in a dictionary.

    Parameters:
    - pdb_file: Path to the PDB file.
    - residues_dict: Dictionary mapping indices to residue numbers.
    - chain_id: Chain ID to look for residues.

    Returns:
    - distances: Dictionary of tuple (residue pair) to distance.
    """
    # Parse the PDB file
    parser = PDBParser()
    structure = parser.get_structure('PDB', pdb_file)

    # Calculate distances for all pairs
    distances = {}
    for (idx1, res1), (idx2, res2) in itertools.combinations(residues_dict.items(), 2):
        distance = get_ca_distance(structure, res1, res2, chain_id)
        distances[(res1, res2)] = distance

    return distances


def all_lib_pairwise_dist(
    data_dir: str = "data",
):
        
    """
    Calculate pairwise distances for all libraries in the specified data directory.
    
    Args:
    - data_dir: Directory containing PDB files for each library.

    Returns:
    - pwd: DataFrame containing mean and standard deviation of distances for each library.
    """
    df = pd.DataFrame(columns=["lib", "res1", "res2", "dist"])

    chain_id = "A"
    for lib, l_d in LIB_INFO_DICT.items():
        pdb_path =  os.path.join(data_dir, f"{lib2prot(lib)}/{lib2prot(lib)}.pdb")
        
        parser = PDBParser()
        structure = parser.get_structure("PDB", pdb_path)

        for (idx1, res_id1), (idx2, res_id2) in itertools.combinations(
            l_d["positions"].items(), 2
        ):
            df = df._append(
                {
                    "lib": lib,
                    "res1": res_id1,
                    "res2": res_id2,
                    "dist": get_ca_distance(structure, res_id1, res_id2, chain_id),
                },
                ignore_index=True,
            )

    pwd = df[["lib", "dist"]].groupby(["lib"]).agg(["mean", "std"])
    pwd.columns = ["mean", "std"]
    
    return pwd