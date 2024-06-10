"""
For generating CoVES based zs scores

NOTE: 
- have to use atom3d evn for atom3d gen
- have to be run in the coves env for coves
"""

import atom3d.datasets as da


# Load dataset from directory of PDB files
dataset = da.load_dataset("data/ParD3/ParD3.pdb", 'pdb')
# Create LMDB dataset from PDB dataset
da.make_lmdb_dataset(dataset, "lmdb")


