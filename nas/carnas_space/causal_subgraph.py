import copy
import torch
import argparse
# from datasets import SPMotif
from torch_geometric.data import DataLoader


import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, BatchNorm, fps
from torch_geometric.nn.conv import GINConv, GCNConv, GATConv, SAGEConv, GraphConv, MessagePassing

# from utils.mask import set_masks, clear_masks

import os
import numpy as np
import os.path as osp
from torch.autograd import grad
# from utils.logger import Logger
from datetime import datetime
# from utils.helper import set_seed, args_print
from .get_subgraph import inv_split_graph, split_batch, causal_relabel
# from gnn import SPMotifNet

from .op_graph_classification import *


from torch_geometric.utils import add_self_loops,remove_self_loops
# from operations import *
from .op_graph_classification import *
from torch.autograd import Variable
from .genotypes import NA_PRIMITIVES, LA_PRIMITIVES, POOL_PRIMITIVES, READOUT_PRIMITIVES, ACT_PRIMITIVES
# from genotypes import Genotype
from torch_geometric.nn import  global_mean_pool,global_add_pool
from torch_geometric.data import Data
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder

class NaSingleOp(nn.Module):
  def __init__(self, in_dim, out_dim, with_linear):
    super().__init__()
    self._ops = nn.ModuleList()

    # self.op = NA_OPS['leconv'](in_dim, out_dim)
    self.op = NA_OPS['gin'](in_dim, out_dim)

    self.op_linear = torch.nn.Linear(in_dim, out_dim)

  def forward(self, x, edge_index, edge_weights, edge_attr, with_linear):
    mixed_res = []
    if with_linear:
        mixed_res.append(self.op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr)+self.op_linear(x))
        # print('with linear')
    else:
        mixed_res.append(self.op(x, edge_index, edge_weight=edge_weights, edge_attr=edge_attr))
        # print('without linear')
    return sum(mixed_res)

class NaDisenOp(nn.Module):
  def __init__(self, in_dim, out_dim, with_linear, k = 4):
    super().__init__()
    self.ops = nn.ModuleList()
    self.op_linear = nn.ModuleList()
    self.in_dim = in_dim
    self.k = k

    for i in range(k):
      # self.ops.append(NA_OPS['leconv'](in_dim // 4, out_dim // 4))
      self.ops.append(NA_OPS['gin'](in_dim // 4, out_dim // 4))
      self.op_linear.append(torch.nn.Linear(in_dim, out_dim))

  def forward(self, x, edge_index, edge_weights, edge_attr, with_linear):
    # x: node * d
    mixed_res = []
    xs = x.hsplit(self.k)
    for i in range(self.k):
      z = self.ops[i](xs[i], edge_index, edge_weight=edge_weights, edge_attr=edge_attr)
      if with_linear:
        z = z + self.op_linear[i](xs[i])
      mixed_res.append(z)
    res = torch.hstack(mixed_res)
    return res
  
class Disen3Head(nn.Module):
  def __init__(self, in_dim, k = 4):
    super().__init__()
    self.ops = nn.ModuleList()
    self.in_dim = in_dim
    self.k = k

    for i in range(3):
      self.ops.append(torch.nn.Linear(in_dim // 4, 1))

  def forward(self, x):
    # x: node * d
    mixed_res = []
    xs = x.hsplit(self.k)
    for i in range(3):
      z = self.ops[i](xs[i])
      z = 0.05 + 0.35 * torch.sigmoid(z)
      mixed_res.append(z)
    res = torch.hstack(mixed_res)
    return res

class CausalAttNet(nn.Module):
    
    def __init__(self, causal_ratio, in_channels, med_channels=float(32), use_causal_x=False):
        super(CausalAttNet, self).__init__()
        self.conv1 = NaSingleOp(in_dim=in_channels, out_dim=med_channels, with_linear=False)
        self.conv2 = NaDisenOp(in_dim=med_channels, out_dim=med_channels, with_linear=False)
        # self.conv1 = LEConv(in_channels=in_channels, out_channels=med_channels)
        # self.conv2 = LEConv(in_channels=med_channels, out_channels=med_channels)
        self.mlp = nn.Sequential(
            nn.Linear(med_channels*2, med_channels*4),
            nn.ReLU(),
            nn.Linear(med_channels*4, 1)
        )
        self.ratio = causal_ratio
        self.use_causal_x = use_causal_x
    
    def forward(self, data):
        # batch_norm
        x, edge_index = data.x, data.edge_index
        edge_weights = torch.ones(edge_index.size()[1], device=edge_index.device).float()
        edge_attr = getattr(data, 'edge_attr', None)
        x = F.relu(self.conv1(x, edge_index, edge_weights, edge_attr, with_linear=False))
        x = self.conv2(x, edge_index, edge_weights, edge_attr, with_linear=False)
        # x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        # x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight), num_nodes, cum_nodes, num_edges = inv_split_graph(data, edge_score, self.ratio)

        causal_edge_attr = causal_edge_attr.to(dtype=torch.long) #lpw
        conf_edge_attr = conf_edge_attr.to(dtype=torch.long) #lpw

        if not self.use_causal_x:
          causal_x, causal_edge_index, causal_batch, _ = causal_relabel('causal', data.x, causal_edge_index, data.batch, num_nodes, cum_nodes, num_edges) #lpw
          conf_x, conf_edge_index, conf_batch, _ = causal_relabel('conf', data.x, conf_edge_index, data.batch, num_nodes, cum_nodes, num_edges)
        else:
          causal_x, causal_edge_index, causal_batch, _ = causal_relabel('causal', x, causal_edge_index, data.batch, num_nodes, cum_nodes, num_edges) #lpw
          conf_x, conf_edge_index, conf_batch, _ = causal_relabel('conf', x, conf_edge_index, data.batch, num_nodes, cum_nodes, num_edges)
          causal_x = causal_x.to(dtype=torch.long) #lpw
          conf_x = conf_x.to(dtype=torch.long) #lpw
        # causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        # conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                edge_score


