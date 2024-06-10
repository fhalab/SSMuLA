"""
For generating CoVES based zs scores

NOTE: 
- have to use atom3d evn for atom3d gen
- have to be run in the coves env for coves
"""

import random
import math
import pandas as pd
import numpy as np

import time
import timeit

import tqdm
import scipy
import importlib

import torch, functools
from torch import nn, scatter_add
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch_cluster
from torch.utils.data import IterableDataset

from atom3d.datasets import LMDBDataset
import atom3d.datasets.datasets as da
import atom3d.splits.splits as spl
import atom3d.util.file as fi
import atom3d.util.formats as fo
from atom3d.util import metrics

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

_amino_acids = lambda x: {
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
}.get(x, 20)

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

_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    'H' : 0,
    'C' : 1,
    'N' : 2,
    'O' : 3,
    'F' : 4,
    'S' : 5,
    'Cl': 6, 'CL': 6,
    'P' : 7
}.get(x, 8)

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)


def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _edge_features(coords, edge_index, D_max=4.5, num_rbf=16, device='cpu'):
    
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), 
               D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num,
            (edge_s, edge_v))

    return edge_s, edge_v


class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x        

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)



class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s


class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)

class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


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


class BaseModel(nn.Module):
    '''
    A base 5-layer GVP-GNN for all ATOM3D tasks, using GVPs with 
    vector gating as described in the manuscript. Takes in atomic-level
    structure graphs of type `torch_geometric.data.Batch`
    and returns a single scalar.
    
    This class should not be used directly. Instead, please use the
    task-specific models which extend BaseModel. (Some of these classes
    may be aliases of BaseModel.)
    
    :param num_rbf: number of radial bases to use in the edge embedding
    '''
    def __init__(self, num_rbf=16):
        
        super().__init__()
        activations = (F.relu, None)
        
        self.embed = nn.Embedding(_NUM_ATOM_TYPES, _NUM_ATOM_TYPES)
        
        self.W_e = nn.Sequential(
            LayerNorm((num_rbf, 1)),
            GVP((num_rbf, 1), _DEFAULT_E_DIM, 
                activations=(None, None), vector_gate=True)
        )
        
        self.W_v = nn.Sequential(
            LayerNorm((_NUM_ATOM_TYPES, 0)),
            GVP((_NUM_ATOM_TYPES, 0), _DEFAULT_V_DIM,
                activations=(None, None), vector_gate=True)
        )
        
        self.layers = nn.ModuleList(
                GVPConvLayer(_DEFAULT_V_DIM, _DEFAULT_E_DIM, 
                             activations=activations, vector_gate=True) 
            for _ in range(5))
        
        ns, _ = _DEFAULT_V_DIM
        self.W_out = nn.Sequential(
            LayerNorm(_DEFAULT_V_DIM),
            GVP(_DEFAULT_V_DIM, (ns, 0), 
                activations=activations, vector_gate=True)
        )
        
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2*ns, 1)
        )
    
    def forward(self, batch, scatter_mean=True, dense=True):
        '''
        Forward pass which can be adjusted based on task formulation.
        
        :param batch: `torch_geometric.data.Batch` with data attributes
                      as returned from a BaseTransform
        :param scatter_mean: if `True`, returns mean of final node embeddings
                             (for each graph), else, returns embeddings seperately
        :param dense: if `True`, applies final dense layer to reduce embedding
                      to a single scalar; else, returns the embedding
        '''
        h_V = self.embed(batch.atoms)
        h_E = (batch.edge_s, batch.edge_v)
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        batch_id = batch.batch
        
        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)
        if scatter_mean: out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense: out = self.dense(out).squeeze(-1)
        return out




                        
class RESModel(BaseModel):
    '''
    GVP-GNN for the RES task.
    
    Extends BaseModel to output a 20-dim vector instead of a single
    scalar for each graph, which can be used as logits in 20-way
    classification.
    
    As noted in the manuscript, RESModel uses the final alpha
    carbon embeddings instead of the graph mean embedding.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ns, _ = _DEFAULT_V_DIM
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(2*ns, 20)
        )
    def forward(self, batch):
        out = super().forward(batch, scatter_mean=False)
        return out[batch.ca_idx+batch.ptr[:-1]]

                        
def get_model(task):
    return {
        'RES' : RESModel,
    }[task]()


class BaseTransform:
    '''
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs. This class should not be used directly; instead, use the
    task-specific transforms, which all extend BaseTransform. Node
    and edge features are as described in the EGNN manuscript.
    
    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -atoms      numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    
    Subclasses of BaseTransform will produce graphs with additional 
    attributes for the tasks-specific training labels, in addition 
    to the above.
    
    All subclasses of BaseTransform directly inherit the BaseTransform
    constructor.
    
    :param edge_cutoff: distance cutoff to use when drawing edges
    :param num_rbf: number of radial bases to encode the distance on each edge
    :device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, edge_cutoff=4.5, num_rbf=16, device='cpu'):
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
            
    def __call__(self, df):
        '''
        :param df: `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format
        
        :return: `torch_geometric.data.Data` structure graph
        '''
        with torch.no_grad():
            coords = torch.as_tensor(df[['x', 'y', 'z']].to_numpy(),
                                     dtype=torch.float32, device=self.device)
            atoms = torch.as_tensor(list(map(_element_mapping, df.element)),
                                            dtype=torch.long, device=self.device)

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff)

            edge_s, edge_v = _edge_features(coords, edge_index, 
                                D_max=self.edge_cutoff, num_rbf=self.num_rbf, device=self.device)

            return torch_geometric.data.Data(x=coords, atoms=atoms,
                        edge_index=edge_index, edge_s=edge_s, edge_v=edge_v)



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
                     lmdb_dout="",
                      model_weight_path = '/disk2/fli/ddingding-CoVES/data/coves/res_weights/RES_1646945484.3030427_8.pt',
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
    # dataset = da.load_dataset(pdb_din, 'pdb', 
    #                           transform = myResTransform(balance=False, pos_oi =pos_oi_all)) 

    # # Create LMDB dataset from PDB dataset, and write to file
    # da.make_lmdb_dataset(dataset, lmdb_dout)
    
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
    # dl_all = torch_geometric.data.DataLoader(ds_all, num_workers=4, batch_size=1)
    dl_all = torch_geometric.loader.DataLoader(ds_all, num_workers=4, batch_size=1)

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


# computing the antitoxin residue preferences from the strutural surrounding
wt_at = 'MANVEKMSVAVTPQQAAVMREAVEAGEYATASEIVREAVRDWLAKRELRHDDIRRLRQLWDEGKASGRPEPVDFDALRKEARQKLTEVPPNGR'
n_ave = 100

start = timeit.default_timer()
df_result = get_gvp_res_prefs(wt_seq=wt_at,
                                protein_name ='at',
                                chain_number='A',
                                # pdb_din=pdb_dir+'at/ta/',
                                # lmdb_dout=pdb_dir+'at/lmdb_at_all/',
                                # lmdb_dout = "/disk2/fli/SSMuLA/lmdb",
                                lmdb_dout = "/disk2/fli/ddingding-CoVES/data/coves/pdbs/pabp/lmdb",
                                model_weight_path = "/disk2/fli/ddingding-CoVES/data/coves/res_weights/RES_1646945484.3030427_8.pt",
                                dout = "/disk2/fli/SSMuLA/coves", 
                                n_ave = n_ave
                             )
end = timeit.default_timer()

print(f'Computing residue preferences for {len(wt_at)} amino acids took {end-start} seconds.')