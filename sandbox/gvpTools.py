###########################################################################################
# This code is adapted from the Atom3d paper and the GVP paper
    # ATOM3D:
    # https://arxiv.org/pdf/2012.04035.pdf
    # https://github.com/drorlab/atom3d
    #
    # GVP:
    # https://arxiv.org/abs/2106.03843, https://openreview.net/forum?id=1YLJDvSx6J4
    # https://github.com/drorlab/gvp-pytorch
###########################################################################################

import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats as stats
import collections as col
import logging
import os
import re
import sys
import scipy.spatial
import parallel as par
import click

from functools import partial
import tqdm, torch, time, os
import sklearn.metrics as sk_metrics

import torch.nn as nn
import torch, random, scipy, math
import torch.nn.functional as F
from torch.utils.data import IterableDataset
import torch_cluster, torch_geometric, torch_scatter

import gvp
from gvp import GVP, GVPConvLayer, LayerNorm
from gvp.data import _normalize, _rbf
import gvp.atom3d
from gvp.atom3d import BaseTransform, _edge_features, _DEFAULT_E_DIM, _DEFAULT_V_DIM, _amino_acids, _element_mapping, _NUM_ATOM_TYPES

from atom3d.datasets import LMDBDataset
import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo
from atom3d.util import metrics

print = partial(print, flush=True)

label_res_dict={0:'HIS',1:'LYS',2:'ARG',3:'ASP',4:'GLU',5:'SER',6:'THR',7:'ASN',8:'GLN',9:'ALA',10:'VAL',11:'LEU',12:'ILE',13:'MET',14:'PHE',15:'TYR',16:'TRP',17:'PRO',18:'GLY',19:'CYS'}
res_label_dict={'HIS':0,'LYS':1,'ARG':2,'ASP':3,'GLU':4,'SER':5,'THR':6,'ASN':7,'GLN':8,'ALA':9,'VAL':10,'LEU':11,'ILE':12,'MET':13,'PHE':14,'TYR':15,'TRP':16,'PRO':17,'GLY':18,'CYS':19}
bb_atoms = ['N', 'CA', 'C', 'O']
allowed_atoms = ['C', 'O', 'N', 'S', 'P', 'SE']

# computed statistics from training set
res_wt_dict = {'HIS': 0.581391659111514, 'LYS': 0.266061611865989, 'ARG': 0.2796785729861747, 'ASP': 0.26563454667840314, 'GLU': 0.22814679094919596, 'SER': 0.2612916369563003, 'THR': 0.27832512315270935, 'ASN': 0.3477441570413752, 'GLN': 0.37781509139381086, 'ALA': 0.20421144813311043, 'VAL': 0.22354397064847012, 'LEU': 0.18395198072344454, 'ILE': 0.2631600545792168, 'MET': 0.6918305148744505, 'PHE': 0.3592224851905275, 'TYR': 0.4048964515721682, 'TRP': 0.9882874205355423, 'PRO': 0.32994186046511625, 'GLY': 0.2238561093317741, 'CYS': 1.0}

gly_CB_mu = np.array([-0.5311191 , -0.75842446,  1.2198311 ], dtype=np.float32)
gly_CB_sigma = np.array([[1.63731114e-03, 2.40018381e-04, 6.38361679e-04],
       [2.40018381e-04, 6.87853419e-05, 1.43898267e-04],
       [6.38361679e-04, 1.43898267e-04, 3.25022011e-04]], dtype=np.float32)

# to go from 3 letter amino acid code to one letter amino acid code
AA3_TO_AA1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

AA1_TO_AA3 = dict(zip(AA3_TO_AA1.values(), AA3_TO_AA1.keys()))

aa3_to_num = {
    'ALA': 0,
    'ARG': 1,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}

num_to_aa3 = dict(zip(aa3_to_num.values(), aa3_to_num.keys()))


class myResTransform(object):
    # pos_oi is for example: [('A', 60, 'TRP'),('A', 61, 'ASP'), ('A', 64, 'LYS'), ('A', 80, 'GLU')]
    # first entry is chain number, position in M1 indexing, and 3letter amino acid code

    def __init__(self, balance=False, pos_oi = []):
        self.balance = balance
        self.pos_oi = pos_oi

    def __call__(self, x):
        x['id'] = fi.get_pdb_code(x['id'])
        df = x['atoms']

        subunits = []
        # df = df.set_index(['chain', 'residue', 'resname'], drop=False)
        df = df.dropna(subset=['x','y','z'])
        #remove Hets and non-allowable atoms
        df = df[df['element'].isin(allowed_atoms)]
        df = df[df['hetero'].str.strip()=='']
        df = df.reset_index(drop=True)
        
        labels = []

        for chain_res, res_df in df.groupby(['chain', 'residue', 'resname']):
            # chain_res is something like ('A', 61, 'ASP')

            if chain_res not in self.pos_oi:
                continue
            print(chain_res)# s

            chain, res, res_name = chain_res
            # only train on canonical residues
            if res_name not in res_label_dict:
                continue
            # sample each residue based on its frequency in train data
            if self.balance:
                if not np.random.random() < res_wt_dict[res_name]:
                    continue

            if not np.all([b in res_df['name'].to_list() for b in bb_atoms]):
                # print('residue missing atoms...   skipping')
                continue
            CA_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

            CB_pos = CA_pos + (np.ones_like(CA_pos) * gly_CB_mu)

            # remove current residue from structure
            subunit_df = df[(df.chain != chain) | (df.residue != res) | df['name'].isin(bb_atoms)]
            
            # environment = all atoms within 10*sqrt(3) angstroms (to enable a 20A cube)
            kd_tree = scipy.spatial.KDTree(subunit_df[['x','y','z']].to_numpy())
            subunit_pt_idx = kd_tree.query_ball_point(CB_pos, r=10.0*np.sqrt(3), p=2.0)
            
            subunits.append(subunit_df.index[sorted(subunit_pt_idx)].to_list())
    
            sub_name = '_'.join([str(x) for x in chain_res])
            label_row = [sub_name, res_label_dict[res_name], CB_pos[0], CB_pos[1], CB_pos[2]]
            labels.append(label_row)

        assert len(labels) == len(subunits)
        print(len(labels))
        x['atoms'] = df
        x['labels'] = pd.DataFrame(labels, columns=['subunit', 'label', 'x', 'y', 'z'])
        x['subunit_indices'] = subunits

        return x


class myRESDataset(IterableDataset):
    '''
    A `torch.utils.data.IterableDataset` wrapper around a
    ATOM3D RES dataset.
    
    On each iteration, returns a `torch_geometric.data.Data`
    graph with the attribute `label` encoding the masked residue
    identity, `ca_idx` for the node index of the alpha carbon, 
    and all structural attributes as described in BaseTransform.
    
    Excludes hydrogen atoms.
    
    :param lmdb_dataset: path to ATOM3D dataset
    :param split_path: path to the ATOM3D split file
    '''
    def __init__(self, lmdb_dataset, chain_id_oi = 'A',split_path=None):
        self.dataset = LMDBDataset(lmdb_dataset) #load lmdb dataset as above
        self.idx = [0]
        self.transform = BaseTransform()
        self.chain_id_oi = chain_id_oi
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.idx))), 
                      shuffle=False)
        else:  
            per_worker = int(math.ceil(len(self.idx) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.idx))
            gen = self._dataset_generator(list(range(len(self.idx)))[iter_start:iter_end],
                      shuffle=False)
        return gen
    
    def _dataset_generator(self, indices, shuffle=False):
        if shuffle: random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                print('idx',idx)
                data = self.dataset[self.idx[idx]]
                atoms = data['atoms']
                for sub in data['labels'].itertuples():
                    _, num, aa_num = sub.subunit.split('_')
                    num, aa = int(num), _amino_acids(aa_num)
                    if aa == 20: 
                        print('aais20')
                        continue
                    my_atoms = atoms.iloc[data['subunit_indices'][sub.Index]].reset_index(drop=True)
                    ca_idx = np.where((my_atoms.residue == num) & (my_atoms.name == 'CA') &(my_atoms.chain  ==self.chain_id_oi))[0] # had to fix this
                    if len(ca_idx) != 1: 
                        print('len(ca_idx) is not 1')
                        continue
                        
                    with torch.no_grad():
                        graph = self.transform(my_atoms)
                        graph.label = aa
                        graph.ca_idx = int(ca_idx)
                        yield num, aa, graph

                        
def get_model(task):
    return {
        'RES' : gvp.atom3d.RESModel,
    }[task]()

def forward(model, batch, device):
    if type(batch) in [list, tuple]:
        batch = batch[0].to(device), batch[1].to(device)
    else:
        batch = batch.to(device)
    return model(batch)

def get_gvp_res_prefs(wt_seq='',
                      protein_name ='protein',
                     chain_number='',
                     pdb_din='',
                     lmdb_dout='',
                      model_weight_path = '../data/coves/res_weights/RES_1646945484.3030427_8.pt',
                      dout = './', 
                      max_pos_to_do = 1000,
                      n_ave = 15
                     ):
    
    # uses RES GVP to calculate residue preferences from structural environment
    # pdb_din: input directory of pdb file
    # lmdb_dout: output directory for making lmdb file

    ##############################################################################
    # create list of positions that are of interest
    pos_oi_all = list(zip([chain_number]* len(wt_seq),
                         range(1,len(wt_seq)+1),
                         [AA1_TO_AA3[aa] for aa in wt_seq]
                        )
                    )
    # Load dataset from directory of PDB files 
    # this is recursive, all pdb files in subdirectories will also be used
    dataset = da.load_dataset(pdb_din, 'pdb', 
                              transform = myResTransform(balance=False, pos_oi =pos_oi_all)) 

    # Create LMDB dataset from PDB dataset, and write to file
    da.make_lmdb_dataset(dataset, lmdb_dout)
    
    ########################## LOAD MODEL #######################################
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # push the model to cuda 
    model = get_model('RES').to(device)

    #load model
    if device == 'cuda':
        model.load_state_dict(torch.load(model_weight_path))
    else:
        model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))

    model = model.eval()
    
    ds_all = myRESDataset(lmdb_dout, chain_id_oi=chain_number)
    dl_all = torch_geometric.data.DataLoader(ds_all, num_workers=4, batch_size=1)
    
    ########################## predicting mutation preferences ##################
    df_result = pd.DataFrame()
    with torch.no_grad():
        c=0
        for d in tqdm.tqdm(dl_all):
            num, aa, b = d
            if c<max_pos_to_do:
                pos = num.numpy()[0]
                aa3 = num_to_aa3[aa.numpy()[0]]
                x= np.zeros([n_ave, 20])
                for i in range(n_ave):
                    out = forward(model, b, device)
                    m_out= out.cpu().detach().numpy().reshape(-1)

                    x[i,:] = m_out

                mean_x = x.mean(axis=0)
                std_x = x.std(axis=0)

                aa1 = AA3_TO_AA1[aa3]
                wt_pos = aa1+str(pos)

                muts = [wt_pos+AA3_TO_AA1[k] for k in aa3_to_num.keys()]

                zipped = list(zip(muts, mean_x, std_x))
                df_pos = pd.DataFrame(zipped, columns=['mut', 'mean_x', 'std_x'])

                df_result = pd.concat([df_result,df_pos], axis=0)
                c+=1
                print(c)
    df_result = df_result.reset_index()
    #print(df_result)
    df_result.to_csv(dout+'gvp_{}_m_{}_230523_.csv'.format(n_ave, protein_name))
    return df_result