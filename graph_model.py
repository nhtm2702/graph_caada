import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)        
        return x, feat
    
class EdgePredictor(nn.Module):

    def __init__(self, feat_dim, hidden=128):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self,x,edge_index):
        
        src = edge_index[0]
        dst = edge_index[1]
        pair = torch.cat([x[src],x[dst]],dim=1)
        w = self.mlp(pair)

        return torch.sigmoid(w).squeeze()

class BinaryGCN(nn.Module):
    """
    Binary node classifier for:
        1: category c
        0: not category c

    The hidden representation is also returned for diverse
    active-learning selection.
    """

    def __init__(
        self, nfeat, nhid=128, dropout=0.5):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.output_layer = GraphConvolution(nhid, 1)

        self.dropout = dropout

    def forward(self, x, adj):
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        graph_embedding = F.relu(self.gc2(h, adj))

        h = F.dropout(
            graph_embedding,
            p=self.dropout,
            training=self.training,
        )

        logits = self.output_layer(h, adj).squeeze(-1)

        return logits, graph_embedding