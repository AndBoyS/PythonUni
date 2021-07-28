'''Models for dynamic graphs'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as graph_nn
import torch_geometric_temporal.nn as temporal_nn
import dgl

from .abstract import *
from .data import dgl_graph

class GCN(DynamicGraphModule):
    def __init__(self, num_classes, num_features, loss=nn.CrossEntropyLoss()):
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

        return torch.log_softmax(x, dim=1)


class EvolveGCN_H(DynamicGraphModule):
    def __init__(self, num_nodes, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        '''
        Примечание: в реализации слоя EvolveGCN-H на PyT Geometric Temporal, который используется здесь, требуется одинаковый размер матриц фичей для всех графов https://github.com/benedekrozemberczki/pytorch_geometric_temporal/issues/10
        '''
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = temporal_nn.EvolveGCNH(num_nodes, num_features)
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, graph):
        h = self.rnn(graph.x, graph.edge_index, graph.edge_attr)
        h = F.relu(h)
        h = F.dropout(h, 0.2, training=self.training)

        h = self.linear(h)

        return F.log_softmax(h, dim=1)


class EvolveGCN_O(DynamicGraphModule):
    def __init__(self, num_nodes, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = temporal_nn.EvolveGCNO(num_features, improved=False)
        self.linear = nn.Linear(num_features, num_classes)
    
    def forward(self, graph):
        h = self.rnn(graph.x, graph.edge_index, graph.edge_attr)
        h = F.relu(h)
        h = F.dropout(h, 0.2, training=self.training)

        h = self.linear(h)

        return F.log_softmax(h, dim=1)


class GConvGRU(DynamicGraphModule):
    def __init__(self, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = temporal_nn.GConvGRU(num_features, 10, K=5)
        self.linear = nn.Linear(10, num_classes)
    
    def forward(self, graph):
        h = self.rnn(graph.x, graph.edge_index, graph.edge_attr)
        h = F.relu(h)
        h = F.dropout(h, 0.2, training=self.training)

        h = self.linear(h)

        return F.log_softmax(h, dim=1)


class LRGCN(DynamicGraphModuleHidden):
    def __init__(self, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = temporal_nn.LRGCN(num_features, num_classes, num_relations=1, num_bases=None)
    
    def forward(self, graph, h=None):
        h,c = self.rnn(graph.x, graph.edge_index, graph.edge_attr, h)
        c = F.relu(c)
        c = F.dropout(c, 0.2, training=self.training)

        return F.log_softmax(c, dim=1), h


class GCLSTM(DynamicGraphModuleHidden):
    def __init__(self, num_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = temporal_nn.GCLSTM(num_features, num_classes, 5)
    
    def forward(self, graph, h=None):
        h,c = self.rnn(graph.x, graph.edge_index, graph.edge_attr, h)
        c = F.relu(c)
        c = F.dropout(c, 0.2, training=self.training)

        return F.log_softmax(c, dim=1), h


class GCN_Cluster(GCN, DynamicGraphModuleBatched):
    pass


class GCN_SAINT(GCN, DynamicGraphModuleSAINT):
    def __init__(self, *args, use_normalization=True):
        super().__init__(*args)
        self.use_normalization = use_normalization


class GConvGRU_SAINT(GConvGRU, DynamicGraphModuleSAINT):
    def __init__(self, *args, use_normalization=True):
        super().__init__(*args)
        self.use_normalization = use_normalization


class GConvGRU_Cluster(GConvGRU, DynamicGraphModuleBatched):
    pass


class GConvGRU_Cell_DGL(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, k: int,
                 bias: bool=True):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.k = k
        self.bias = bias
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = dgl.nn.ChebConv(in_feats=self.in_feats,
                                        out_feats=self.out_feats,
                                        k=self.k,
                                        bias=self.bias)

        self.conv_h_z = dgl.nn.ChebConv(in_feats=self.out_feats,
                                        out_feats=self.out_feats,
                                        k=self.k,
                                        bias=self.bias)

    def _create_reset_gate_parameters_and_layers(self):
        
        self.conv_x_r = dgl.nn.ChebConv(in_feats=self.in_feats,
                                        out_feats=self.out_feats,
                                        k=self.k,
                                        bias=self.bias)

        self.conv_h_r = dgl.nn.ChebConv(in_feats=self.out_feats,
                                        out_feats=self.out_feats,
                                        k=self.k,
                                        bias=self.bias)

    def _create_candidate_state_parameters_and_layers(self):
        
        self.conv_x_h = dgl.nn.ChebConv(in_feats=self.in_feats,
                                        out_feats=self.out_feats,
                                        k=self.k,
                                        bias=self.bias)

        self.conv_h_h = dgl.nn.ChebConv(in_feats=self.out_feats,
                                        out_feats=self.out_feats,
                                        k=self.k,
                                        bias=self.bias)
    
    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_feats).to(X.device)
        return H

    def _calculate_update_gate(self, graph, X, H):
        Z = self.conv_x_z(graph, X)
        Z = Z + self.conv_h_z(graph, H)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, graph, X, H):
        R = self.conv_x_r(graph, X)
        R = R + self.conv_h_r(graph, H)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, graph, X, H, R):
        H_tilde = self.conv_x_h(graph, X)
        H_tilde = H_tilde + self.conv_h_h(graph, H*R)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z*H + (1-Z)*H_tilde
        return H

    def forward(self, graph, X, H: torch.FloatTensor=None) -> torch.FloatTensor:
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(graph, X, H)
        R = self._calculate_reset_gate(graph, X, H)
        H_tilde = self._calculate_candidate_state(graph, X, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H
    

class GConvGRU_Cell_DGL_NB(GConvGRU_Cell_DGL):

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = dgl.nn.GraphConv(self.in_feats, self.out_feats, bias=self.bias, norm='none')
        self.conv_h_z = dgl.nn.GraphConv(self.out_feats, self.out_feats, bias=self.bias, norm='none')
        
    def _create_reset_gate_parameters_and_layers(self):
        
        self.conv_x_r = dgl.nn.GraphConv(self.in_feats, self.out_feats, bias=self.bias, norm='none')
        self.conv_h_r = dgl.nn.GraphConv(self.out_feats, self.out_feats, bias=self.bias, norm='none')

    def _create_candidate_state_parameters_and_layers(self):
        
        self.conv_x_h = dgl.nn.GraphConv(self.in_feats, self.out_feats, bias=self.bias, norm='none')
        self.conv_h_h = dgl.nn.GraphConv(self.out_feats, self.out_feats, bias=self.bias, norm='none')

    def forward(self, graph, X, H: torch.FloatTensor=None) -> torch.FloatTensor:

        H = self._set_hidden_state(list(X[0])[0], H)
        Z = self._calculate_update_gate(graph[0], X[0], H)
        R = self._calculate_reset_gate(graph[0], X[0], H)
        
        H = H[:graph[0].num_dst_nodes()]
        H_tilde = self._calculate_candidate_state(graph[1], X[1], H, R)
        
        Z = Z[:graph[1].num_dst_nodes()]
        H = H[:graph[1].num_dst_nodes()]
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class GConvGRU_DGL(DynamicGraphModule):
    def __init__(self, num_features, num_hidden_features, num_classes, loss=nn.CrossEntropyLoss()):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = GConvGRU_Cell_DGL(in_feats=num_features, 
                                     out_feats=num_hidden_features, k=5)
        self.linear = nn.Linear(num_hidden_features, num_classes)
    
    def forward(self, graph):
        h = self.rnn(graph, graph.get_node_features())
        h = F.relu(h)
        h = F.dropout(h, 0.2, training=self.training)

        h = self.linear(h)

        return F.log_softmax(h, dim=1)


class GConvGRU_NB_DGL(DynamicGraphModuleBatchedNB_DGL):
    def __init__(self, num_features, num_hidden_features, num_classes, loss=nn.CrossEntropyLoss()):
        '''
        GConvGRU с Neighbour Sampling
        '''
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.loss = loss

        self.rnn = GConvGRU_Cell_DGL_NB(in_feats=num_features, 
                                     out_feats=num_hidden_features, k=5)
        self.linear = nn.Linear(num_hidden_features, num_classes)
    
    def forward(self, mfgs):
        # x1 - матрица фич с узлами, нужными для вычисления эмбеддингов узлов на первой свертке
        # h_dst1 - та же матрица с узлами, эмбеддинги которых будут вычисляться
        x1 = mfgs[0].srcdata['feat']
        h_dst1 = x1[:mfgs[0].num_dst_nodes()]
        x2 = mfgs[1].srcdata['feat']
        h_dst2 = x2[:mfgs[1].num_dst_nodes()]

        h = self.rnn(mfgs, [(x1, h_dst1), (x2, h_dst2)])
        h = F.relu(h)
        h = F.dropout(h, 0.2, training=self.training)

        h = self.linear(h)
        return F.log_softmax(h, dim=1)