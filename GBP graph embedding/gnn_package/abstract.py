'''Abstract classes for static graphs'''
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score

class GraphModule(nn.Module):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric и DGL
    '''

    def train_(self, data, epochs=100, device='cpu', node_mask_attr='train_mask'):
        '''
        Обучить модель

        data: torch_geometric.data.Data - граф
        node_mask_attr: str или None - название атрибута data, являющегося маской узлов, которые будут использоваться при обучении
        '''
        assert node_mask_attr in ['train_mask', 'val_mask', 'test_mask', None]
        self.device = device
        self.to(device)
        data = data.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
        
        self.train()
        for epoch in tqdm(range(1,epochs+1)):
            self.prop_and_backprop(data, node_mask_attr)

    def prop_and_backprop(self, data, node_mask_attr='train_mask'):
        '''
        Propagate and backpropagate
        '''
        self.optimizer.zero_grad()
        cost = self.compute_cost(data, node_mask_attr)
        cost.backward()
        self.optimizer.step()

    def compute_cost(self, data, node_mask_attr):
        out = self(data)
        if node_mask_attr is not None:
            node_mask = getattr(data, node_mask_attr)
            cost = self.loss(out[node_mask], data.get_node_labels()[node_mask])
        else:
            cost = self.loss(out, data.get_node_labels())
        return cost

    def predict(self, data):
        self.eval()
        _, pred = self(data).max(dim=1)
        return pred

    def evaluate(self, data, node_mask_attr='test_mask'):
        assert node_mask_attr in ['train_mask', 'val_mask', 'test_mask', None]
        data = data.to(self.device)
        all_pred = self.predict(data)
        if node_mask_attr is not None:
            node_mask = getattr(data, node_mask_attr)
            pred = all_pred[node_mask]
            true = data.get_node_labels()[data.test_mask]
        else:
            pred = all_pred
            true = data.get_node_labels()

        return pred, f1_score(true.cpu(), pred.cpu(), average=None)
    

class GraphModuleBatched(GraphModule):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric и DGL, получающих батчи
    '''
    def train_(self, batch_loader, epochs=100, device='cpu', node_mask_attr='train_mask'):
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
        self.train()
        for epoch in tqdm(range(1,epochs+1)):
            for batch in batch_loader:
                try:
                    batch = batch.to(device)
                except AttributeError:
                    pass
                self.prop_and_backprop(batch, node_mask_attr)


class GraphModuleSAINT(GraphModuleBatched):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric, получающих батчи, созданные через torch_geometric.data.GraphSAINTSampler
    '''
    def train_(self, batch_loader, epochs=100, device='cpu', node_mask_attr='train_mask'):
        
        self.set_conv_aggr('add' if self.use_normalization else 'mean')
        super().train_(batch_loader, epochs, device, node_mask_attr)
        self.set_conv_aggr('mean')

    def set_conv_aggr(self, aggr):
        for conv_layer in self.conv_layers:
            conv_layer.aggr = aggr

# predict на батчах
#   def predict(self, batch_loader):
#       self.eval()
#       preds = []
#       for data in batch_loader:
#           out = self(data.to(self.device))
#           preds.append(out.detach().float().cpu())
#       pred = torch.cat(preds, dim=0)
#       return pred.max(dim=1).indices
                
# Abstract classes for dynamic graphs


class GraphModuleBatchedNB_DGL(GraphModuleBatched):
    '''
    Абстрактный класс для создания моделей DGL, получающих батчи при помощи Neighbour Sampling
    '''
    def compute_cost(self, data, _):
        input_nodes, output_nodes, mfgs = data
        labels = mfgs[-1].dstdata['label']
        out = self(mfgs)
        cost = self.loss(out, labels)
        return cost

    def evaluate(self, data, node_mask_attr=None):
        true, pred = [], []
        for input_nodes, output_nodes, mfgs in data:
            batch_pred = self.predict(mfgs)
            pred.append(batch_pred.cpu())
            true.append(mfgs[-1].dstdata['label'].cpu())
        true, pred = torch.cat(true), torch.cat(pred)
        return pred, f1_score(true, pred, average=None)


class DynamicGraphModule(GraphModule):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric и DGL, работающих с динамичными графами
    '''
    def train_(self, dataset, epochs=100, device='cpu', node_mask_attr='train_mask', evaluate=True):
        '''
        Обучить модель

        data: torch_geometric.data.Data - граф
        node_mask_attr: str или None - название атрибута data, являющегося маской узлов, которые будут использоваться при обучении. node_mask - обучение только на узлах, присутствующих в снапшоте
        '''
        assert node_mask_attr in ['train_mask', 'val_mask', 'test_mask', 'node_mask', None]
        self.device = device
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        for epoch in tqdm(range(1,epochs+1)):
            self.prop_and_backprop(dataset, node_mask_attr)

            if evaluate and epoch % 10 == 0:
                preds, metrics = self.evaluate(dataset, node_mask_attr)
                print(f' Epoch {epoch} - f1 minority class mean: {np.mean([metric[1] for metric in metrics])}')
    
    def compute_cost(self, dataset, node_mask_attr):
        cost = 0
        for snapshot in dataset:
            snapshot = snapshot.to(self.device)
            pred = self(snapshot)
            if node_mask_attr is not None:
                node_mask = getattr(snapshot, node_mask_attr)
                cost += self.loss(pred[node_mask], snapshot.get_node_labels()[node_mask])
            else:
                cost += self.loss(pred, snapshot.get_node_labels())

        cost = cost / len(dataset)
        return cost

    def evaluate(self, dataset, node_mask_attr='test_mask'):
        assert node_mask_attr in ['train_mask', 'val_mask', 'test_mask', 'node_mask', None]
        preds, metrics = [], []
        for snapshot in dataset:
            snapshot = snapshot.to(self.device)
            pred = self.predict(snapshot)
            if node_mask_attr is not None:
                node_mask = getattr(snapshot, node_mask_attr)
                metric = f1_score(snapshot.get_node_labels()[node_mask].cpu(), pred[node_mask].cpu(), average=None)
            else:
                metric = f1_score(snapshot.get_node_labels().cpu(), pred.cpu(), average=None)
            
            preds.append(pred)
            metrics.append(metric)
        return preds, metrics


class DynamicGraphModuleBatched(DynamicGraphModule):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric и DGL, работающих с динамичными графами и получающих батчи
    '''
    def compute_cost(self, batch_loader_list, node_mask_attr):
        cost = 0
        for batch_loader in batch_loader_list:
            for batch in batch_loader:
                batch = batch.to(self.device)
                pred = self(batch)
                if node_mask_attr is not None:
                    node_mask = getattr(batch, node_mask_attr)
                    cost += self.loss(pred[node_mask], batch.y[node_mask])
                else:
                    cost += self.loss(pred, batch.y)
        cost = cost / len(batch_loader_list)
        return cost

    def evaluate(self, batch_loader_list, node_mask_attr='test_mask'):
        assert node_mask_attr in ['train_mask', 'val_mask', 'test_mask', 'node_mask', None]
        preds, metrics = [], []
        for batch_loader in batch_loader_list:
            pred = []
            true = []
            for batch in batch_loader:
                batch = batch.to(self.device)
                batch_pred = self.predict(batch)
                if node_mask_attr is not None:
                    node_mask = getattr(batch, node_mask_attr)
                    true.append(batch.get_node_labels()[node_mask].cpu())
                    pred.append(batch_pred[node_mask].cpu())
                else:
                    true.append(batch.get_node_labels().cpu())
                    pred.append(batch_pred.cpu())

            true, pred = torch.cat(true), torch.cat(pred)

            preds.append(pred)
            metrics.append(f1_score(true, pred, average=None))

        return preds, metrics


class DynamicGraphModuleSAINT(DynamicGraphModuleBatched):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric, работающих с динамичными графами и получающих батчи, созданные через GraphSAINT
    '''
    set_conv_aggr = GraphModuleSAINT.__dict__['set_conv_aggr']
    
    def train_(self, *args, **kwargs):
        self.set_conv_aggr('add' if self.use_normalization else 'mean')
        super().train_(*args, **kwargs)
        self.set_conv_aggr('mean')


class DynamicGraphModuleBatchedNB_DGL(DynamicGraphModuleBatched):
    '''
    Абстрактный класс для создания моделей DGL, работающих с динамичными графами и получающих батчи при помощи Neighbour Sampling
    '''
    def compute_cost(self, batch_loader_list, _):
        cost = 0
        for batch_loader in batch_loader_list:
            for input_nodes, output_nodes, mfgs in batch_loader:
                labels = mfgs[-1].dstdata['label']
                out = self(mfgs)
                cost += self.loss(out, labels)

        cost = cost / len(batch_loader_list)
        return cost

    def evaluate(self, batch_loader_list, node_mask_attr=None):
        preds, metrics = [], []
        for batch_loader in batch_loader_list:
            true, pred = [], []
            for input_nodes, output_nodes, mfgs in batch_loader:
                batch_pred = self.predict(mfgs)
                pred.append(batch_pred.cpu())
                true.append(mfgs[-1].dstdata['label'].cpu())
            true, pred = torch.cat(true), torch.cat(pred)
            metrics.append(f1_score(true, pred, average=None))
            preds.append(pred)

        return preds, metrics 


class DynamicGraphModuleHidden(DynamicGraphModule):
    '''
    Абстрактный класс для создания моделей PyTorch Geometric, работающих с динамичными графами (для LRGCN и GCLSTM)
    '''

    def compute_cost(self, dataset, node_mask_attr):
        cost = 0
        h = None # Hidden state of rnn
        for snapshot in dataset:
            snapshot = snapshot.to(self.device)
            pred, h = self(snapshot, h)
            if node_mask_attr is not None:
                node_mask = getattr(snapshot, node_mask_attr)
                cost += self.loss(pred[node_mask], snapshot.y[node_mask])
            else:
                cost += self.loss(pred, snapshot.y)

        cost = cost / len(dataset)
        return cost

    def predict(self, snapshot, h=None):
        self.eval()
        pred, h = self(snapshot, h)
        _, pred = pred.max(dim=1)
        return pred, h

    def evaluate(self, dataset, node_mask_attr='test_mask'):
        preds, metrics = [], []
        h = None
        for snapshot in dataset:
            snapshot = snapshot.to(self.device)
            pred, h = self.predict(snapshot, h)
            if node_mask_attr is not None:
                node_mask = getattr(snapshot, node_mask_attr)
                metric = f1_score(snapshot.y[node_mask].cpu(), pred[node_mask].cpu(), average=None)
            else:
                metric = f1_score(snapshot.y.cpu(), pred.cpu(), average=None)

            preds.append(pred)
            metrics.append(metric)
        return preds, metrics