'''Models for static graphs'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graph_nn
import torch_geometric_temporal.nn as temporal_nn
import dgl

from .abstract import *

class GCN(GraphModule):
    def __init__(self, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.conv1 = graph_nn.GCNConv(num_features, 128)
        self.conv2 = graph_nn.GCNConv(128, num_classes)
        self.conv_layers = [self.conv1, self.conv2] # Для SAINT сэмплера
        self.loss = loss

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)

        return torch.sigmoid(x)


class GCN_DGL(GraphModule):
    def __init__(self, num_features, num_hidden_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(num_features, num_hidden_features)
        self.conv2 = dgl.nn.GraphConv(num_hidden_features, num_classes)
        self.loss = loss

    def forward(self, data):
        h = self.conv1(data, data.get_node_features())
        h = F.relu(h)
        h = F.dropout(h, p=0.1, training=self.training)
        h = self.conv2(data, h)
        return torch.sigmoid(h)


class EvolveGCN_H(GraphModule):
    def __init__(self, num_nodes, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = temporal_nn.EvolveGCNH(num_nodes, num_features)
        self.conv_layers = [self.rnn.conv_layer] # Для SAINT сэмплера
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, graph):
        h = self.rnn(graph.x, graph.edge_index, graph.edge_attr)
        h = F.relu(h)
        h = F.dropout(h, 0.2, training=self.training)

        h = self.linear(h)

        return F.log_softmax(h, dim=1)


class GCN_SAINT(GCN, GraphModuleSAINT):
    def __init__(self, *args, use_normalization=True):
        super().__init__(*args)
        self.use_normalization = use_normalization


class GCN_Cluster(GCN, GraphModuleBatched):
    pass


class GCN_NB_DGL(GraphModuleBatchedNB_DGL):
    def __init__(self, num_features, num_hidden_features, num_classes, loss=torch.nn.CrossEntropyLoss()):
        '''
        GCN с Neighbour Sampling
        '''
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(num_features, num_hidden_features)
        self.conv2 = dgl.nn.GraphConv(num_hidden_features, num_classes)
        self.loss = loss

    def forward(self, mfgs):
        x = mfgs[0].srcdata['feat']
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return torch.sigmoid(h)