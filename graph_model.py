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

class BinaryGCN(nn.Module):

    def __init__(
        self, nfeat, nhid=128, dropout=0.5):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.output_layer = GraphConvolution(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h,p=self.dropout,training=self.training,)
        graph_embedding = F.relu(self.gc2(h, adj))
        h = F.dropout(graph_embedding,p=self.dropout,training=self.training)
        logits = self.output_layer(h, adj).squeeze(-1)

        return logits, graph_embedding
    
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
    
class WeightedSAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear_self = nn.Linear(in_features,out_features,bias=True)
        self.linear_neigh = nn.Linear(in_features,out_features,bias=False)

    def forward(self, x, adj):
        neigh = torch.sparse.mm(adj, x)
        out = (self.linear_self(x) + self.linear_neigh(neigh))
        return out

class BinaryGraphSAGE(nn.Module):
    def __init__(self,nfeat,nhid=128,dropout=0.5,):
        super().__init__()
        
        self.sage1 = WeightedSAGELayer(nfeat, nhid)
        self.sage2 = WeightedSAGELayer(nhid, nhid)
        self.output_layer = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x, adj):

        h = F.relu(self.sage1(x, adj))
        h = F.dropout(h,p=self.dropout,training=self.training,)
        graph_embedding = F.relu(self.sage2(h, adj))
        h = F.dropout(graph_embedding,p=self.dropout,training=self.training)
        logits = self.output_layer(h).squeeze(-1)

        return logits, graph_embedding
    
class WeightedPNALayer(nn.Module):

    def __init__(self,in_features,out_features,avg_degree):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("avg_log_degree",torch.tensor(float(torch.log(torch.tensor(avg_degree + 1.0)))))

        num_aggregators = 4
        num_scalers = 3

        pna_dim = (in_features * num_aggregators * num_scalers)
        self.linear = nn.Linear(pna_dim,out_features)
        self.self_linear = nn.Linear(in_features,out_features,bias=False)

    def forward(self, x, adj):
        adj = adj.coalesce()
        
        indices = adj.indices()
        weights = adj.values()
        
        row = indices[0]   
        col = indices[1]  
        
        num_nodes = x.size(0)
        feat_dim = x.size(1)

        neighbor_x = x[col]
        weighted_message = (neighbor_x * weights.unsqueeze(-1))

        weight_sum = torch.zeros(num_nodes,device=x.device,dtype=x.dtype)
        weight_sum.scatter_add_(0,row,weights)

        sum_message = torch.zeros(num_nodes,feat_dim,device=x.device,dtype=x.dtype)
        sum_message.index_add_(0,row,weighted_message)
        weighted_mean = (sum_message / weight_sum.clamp_min(1e-12).unsqueeze(-1))

        weighted_square = (neighbor_x.pow(2) * weights.unsqueeze(-1))
        square_sum = torch.zeros_like(sum_message)
        square_sum.index_add_(0,row,weighted_square)
        mean_square = (square_sum / weight_sum.clamp_min(1e-12).unsqueeze(-1))

        variance = (mean_square - weighted_mean.pow(2)).clamp_min(0.0)
        weighted_std = torch.sqrt(variance + 1e-6)

        max_message = torch.full((num_nodes, feat_dim), -torch.inf, device=x.device, dtype=x.dtype,)
        expanded_row = row.unsqueeze(-1).expand(-1,feat_dim)

        max_message.scatter_reduce_(0,expanded_row,weighted_message,reduce="amax",include_self=False)
        max_message[torch.isinf(max_message)] = 0.0

        min_message = torch.full((num_nodes, feat_dim),torch.inf,device=x.device,dtype=x.dtype)
        min_message.scatter_reduce_(0,expanded_row,weighted_message,reduce="amin",include_self=False)
        min_message[torch.isinf(min_message)] = 0.0

        aggregated = torch.cat([weighted_mean,weighted_std,max_message,min_message], dim=-1,)
        degree = torch.zeros(num_nodes,device=x.device,dtype=x.dtype)
        degree.scatter_add_(0,row,torch.ones_like(weights))
        log_degree = torch.log(degree + 1.0)

        avg_log_degree = (self.avg_log_degree.to(x.device).clamp_min(1e-6))
        amplification = (log_degree / avg_log_degree).unsqueeze(-1)
        attenuation = (avg_log_degree / log_degree.clamp_min(1e-6)).unsqueeze(-1)

        identity_scaled = aggregated
        amplification_scaled = (aggregated * amplification)
        attenuation_scaled = (aggregated * attenuation)

        pna_features = torch.cat([identity_scaled,amplification_scaled,attenuation_scaled],dim=-1)

        out = (self.linear(pna_features) + self.self_linear(x))
        return out
    
class BinaryPNA(nn.Module):
    def __init__(self,nfeat,nhid=128,dropout=0.5,avg_degree=10.0,):
        super().__init__()

        self.pna1 = WeightedPNALayer(in_features=nfeat,out_features=nhid,avg_degree=avg_degree)
        self.pna2 = WeightedPNALayer(in_features=nhid,out_features=nhid,avg_degree=avg_degree)
        self.output_layer = nn.Linear(nhid,1)
        self.dropout = dropout

    def forward(self, x, adj):

        h = F.relu(self.pna1(x,adj))
        h = F.dropout(h,p=self.dropout,training=self.training)
        graph_embedding = F.relu(self.pna2(h,adj))
        h = F.dropout(graph_embedding,p=self.dropout,training=self.training)
        logits = self.output_layer(h).squeeze(-1)
        return logits, graph_embedding