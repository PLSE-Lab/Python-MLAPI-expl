import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
 
# from algorithms.util import glorot, zeros
 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 
class GATConvLayer(nn.Module):
    def __init__(self,
                 node_in_fts,
                 node_out_fts,
                 ed_in_fts,
                 ed_out_fts,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 repeat_edge = 1
                 ):
        super(GATConvLayer, self).__init__()
 
        self.node_in_fts = node_in_fts
        self.node_out_fts = node_out_fts
        self.ed_out_fts = ed_out_fts
        self.ed_in_fts = ed_in_fts
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.repeat_edge = repeat_edge
 
        self.triplet_transform = Parameter(torch.zeros(size=(2*self.node_in_fts + self.ed_in_fts, self.node_out_fts)))
        self.edge_transform = Parameter(torch.zeros(size=(self.node_out_fts, self.ed_out_fts)))
        self.att = Parameter(torch.Tensor(self.node_out_fts, 1))
 
        if bias and concat:
            self.bias = Parameter(torch.Tensor(node_out_fts))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(node_out_fts))
        else:
            self.register_parameter('bias', None)
 
        self.reset_parameters()
 
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.triplet_transform.data, gain=1.414)
        nn.init.xavier_uniform_(self.edge_transform.data, gain=1.414)
        # glorot(self.att)
        # zeros(self.bias)
 
 
    def forward(self, x, edge_attr):
        return self.propagate( x=x, num_nodes=x.size(0), edge_attr=edge_attr)
 
    def propagate(self, **kwargs):
        edge_attr = kwargs['edge_attr']
        x = kwargs['x']
        N = kwargs['num_nodes']
 
        triplet = torch.cat([x.repeat(1, self.repeat_edge*N).view(self.repeat_edge*N*N, -1), edge_attr, x.repeat(self.repeat_edge*N, 1)], dim=-1)
        new_hij = torch.matmul(triplet, self.triplet_transform) # N^2-fts x fts-out_fts = N^2-out_fts
        new_edge_attr = torch.matmul(new_hij, self.edge_transform) # N^2-fts x edge_fts = N^2-edge_fts
        new_hij = new_hij.view(N, -1, self.node_out_fts) # N-N-fts 
 
        attention = torch.matmul(new_hij, self.att).squeeze(2) # N-N-fts x fxs-1 = N-N
        attention = F.leaky_relu(attention, self.negative_slope) # N-N
        attention = F.softmax(attention, dim=-1) # N-N
        attention = F.dropout(attention, self.dropout, training=self.training)
 
        h_prime = torch.matmul(attention.view(N, 1, self.repeat_edge*N), new_hij)
        new_x = h_prime.squeeze(1)
 
        return [new_x, new_edge_attr]