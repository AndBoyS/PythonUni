import warnings

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import dgl
from sklearn.model_selection import train_test_split

class torch_geometric_graph(Data):
    '''
    Граф torch_geometric.data.Data с улучшенным методом to (работает для кастомных аттрибутов) и get методами для совместимости
    '''
    def to(self, device):
        new_graph = super().to(device)
        for attr, _ in new_graph:
            new_val = getattr(new_graph, attr).to(device)
            setattr(new_graph, attr, new_val)
        return new_graph

    def get_node_features(self):
        return self.x
    def get_node_labels(self):
        return self.y
    def get_edge_index(self):
        return self.edge_index
        
class dgl_graph(dgl.DGLGraph):
    '''
    dgl.DGLGraph с get методами для совместимости 
    '''
    def __init__(self, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super().__init__(*args)

    def get_node_features(self):
        return self.ndata['feat']
    def get_node_labels(self):
        return self.ndata['label']
    def get_edge_index(self):
        return torch.vstack(self.all_edges())


class OrderedEncoder:
    dict_ = {}
    inverse_dict_ = {}
    counter = 0

    def fit(self, data):
        for el in data:
            self.dict_[el] = self.counter
            self.inverse_dict_[self.counter] = el
            self.counter += 1
        return None
    
    def transform(self, data):
        original_shape = data.shape
        data = data.reshape(-1)
        if type(data) == torch.Tensor:
            data = data.numpy()
        res = np.array([self.dict_[el] for el in data])
        return res.reshape(*original_shape)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        res = np.array([self.inverse_dict_[el] for el in data])
        return res.reshape(*data.shape)
        
        
def load_and_preprocess_elliptic_dataset(features_path, edges_path, classes_path):
    '''
    features_path: str - путь содержащий elliptic_txs_features.csv
    edges_path: str - путь содержащий elliptic_txs_edgelist.csv
    classes_path: str - путь содержащий elliptic_txs_classes.csv
    '''

    df_features = pd.read_csv(features_path, header=None)
    #df_features=df_features.loc[df_features[1]<=25]
    df_edges = pd.read_csv(edges_path)
    df_classes = pd.read_csv(classes_path)
    df_classes['class'] = df_classes['class'].map({'unknown':2, '1':1, '2':0})
    # добавляем номер класса в df_features
    df_features = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
    # удаляем unknown класс
    df_features = df_features[df_features['class'] != 2].drop('txId', axis=1)
    df_edges = df_edges[df_edges['txId1'].isin(df_features[0]) & df_edges['txId2'].isin(df_features[0])]

    nodes = df_features[0]
    nodes_in_edges = nodes[np.isin(nodes, df_edges['txId1']) | np.isin(nodes, df_edges['txId2'])]
    df_features = df_features[df_features[0].isin(nodes_in_edges)]

    node_encoder = OrderedEncoder()
    df_features[0] = node_encoder.fit_transform(df_features[0].values)
    df_features = df_features.sort_values(0)

    df_edges['txId1'] = node_encoder.transform(df_edges['txId1'].values)
    df_edges['txId2'] = node_encoder.transform(df_edges['txId2'].values)
    df_edges = df_edges.astype(int)
    
    return df_features, df_edges


def create_elliptic_dataset_graph(df_features, df_edges, df_features_orig=None, encode_nodes=False, create_masks=False, graph_type='pytorch-geometric'):
    '''
    df_features_orig - датафрейм фичей, который нужно использовать в качестве фичей в графе (для случая с EvolveGCN, когда требуется не меняющаяся матрица фичей в темпоральном датасете)
    '''
    assert graph_type in ['pytorch-geometric', 'dgl']

    df_features_orig = df_features if df_features_orig is None else df_features_orig

    df_edges = df_edges.copy()
    df_edges = df_edges[df_edges['txId1'].isin(df_features[0]) | df_edges['txId2'].isin(df_features[0])]
    
    edge_index = df_edges.to_numpy().T

    edge_index = torch.LongTensor(edge_index).contiguous()
    weights = None

    node_features = df_features_orig.copy()
    
    y = torch.LongTensor(node_features['class'].values)

    if encode_nodes:
        node_encoder = OrderedEncoder()
        node_encoder.fit(node_features[0])
        edge_index = torch.LongTensor(node_encoder.transform(edge_index))

    node_features = node_features.drop([0, 'class', 1], axis=1)
    node_features = torch.FloatTensor(node_features.values)

    if graph_type == 'pytorch-geometric':
        data = torch_geometric_graph(x=node_features, edge_index=edge_index, edge_attr=weights,
                y=y)
        num_nodes = data.num_nodes
    else:
        U, V = edge_index[0,:], edge_index[1,:] 
        data = dgl_graph((U,V))
        data.ndata['feat'] = node_features
        data.ndata['label'] = y
        data = dgl.add_self_loop(data)
        num_nodes = data.num_nodes()


    if create_masks:
        train_idx, test_idx = train_test_split(
            np.arange(num_nodes), test_size=0.15, random_state=42, stratify=data.get_node_labels())
        data.train_mask = torch.BoolTensor(
            [(node in train_idx) for node in np.arange(num_nodes)])
        data.test_mask = torch.BoolTensor(
            [(node in test_idx) for node in np.arange(num_nodes)])
    
    return data


def temporal_list_split(graph_list, train_ratio):
    split_idx = int(train_ratio * len(graph_list))
    return graph_list[:split_idx], graph_list[split_idx:]
    
    
def create_temporal_elliptic_dataset_graph(df_features, df_edges, graph_type='pytorch-geometric', temporal_feature_matrix=True):

    timestamps = sorted(df_features[1].unique())
    graph_list = []
    for timestamp in timestamps:
        df_features_snapshot = df_features.copy()
        df_features_snapshot = df_features_snapshot[df_features_snapshot[1] == timestamp]
        if temporal_feature_matrix:
            graph = create_elliptic_dataset_graph(df_features_snapshot, df_edges, df_features_snapshot, 
                                                    encode_nodes=True, create_masks=False, graph_type=graph_type)
        else:
            graph = create_elliptic_dataset_graph(df_features_snapshot, df_edges, df_features, 
                                                    encode_nodes=False, create_masks=False, graph_type=graph_type)

        node_mask = np.isin(np.arange(graph.get_node_features().shape[0]), graph.get_edge_index().reshape(-1))
        graph.node_mask = torch.BoolTensor(node_mask)

        graph_list.append(graph)

    return graph_list


def create_nb_dataloader_dgl(data, fanouts, node_mask, device):
    '''
    Создание Neighbour Sampling батчей

    Parameters
    fanouts: list[int] - сколько соседей используется для вычисления эмбеддинга узла для каждого слоя свертки
    node_mask: Tensor - булевая маска, которая определяет для каждого узла, участвует ли он в обучении
    '''
    nids = torch.LongTensor([el for el, cond
                             in zip(range(data.num_nodes()), node_mask)
                             if cond])
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    dataloader = dgl.dataloading.NodeDataLoader(
        data, nids, sampler, device=device,
        batch_size=128, shuffle=False, drop_last=False, num_workers=0
    )
    return dataloader