"""
For generating CoVES based zs scores

NOTE: 
- have to use atom3d evn for atom3d gen
- have to be run in the coves env for coves
"""

from __future__ import annotations

import os
from glob import glob

import atom3d.datasets as da

from SSMuLA.util import checkNgen_folder, get_file_name


def gen_lmdb_dataset(pdb_path: str, lmdb_path: str):

    """
    Generate LMDB dataset from PDB dataset

    Args:
    - pdb_path, str: Path to the PDB files
    - lmdb_path, str: Path to directory to save LMDB dataset
    """

    # Load dataset from directory of PDB files
    pdb_file = da.load_dataset(pdb_path, 'pdb')
    # Create LMDB dataset from PDB dataset
    checkNgen_folder(lmdb_path)
    da.make_lmdb_dataset(pdb_file, lmdb_path)


def gen_all_lmdb(pdb_pattern: str = "data/*/*.pdb", lmdb_dir: str = "lmdb"):

    for pdb_path in sorted(glob(pdb_pattern)):

        print(f"Generating LMDB dataset for {pdb_path}...")

        protein_name = get_file_name(pdb_path)

        lmdb_path = os.path.join(lmdb_dir, protein_name)
        gen_lmdb_dataset(pdb_path, lmdb_path)



