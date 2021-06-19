import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertModel, BertPreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

class BertProccessor:
    '''
    Класс для предобработки данных под работу берта + НН
    '''

    def _create_and_fit_encoder(self, data_to_encode, encoder, attr_name, train=True):
        '''
        Создает и тренирует енкодер если идет процесс обучения
        '''
        if train:

            setattr(self, attr_name, encoder())
            getattr(self, attr_name).fit(data_to_encode)


    def get_data(self, data, train=True, all_data=None):
        
        if all_data is None:
            all_data = data

        data.loc[:, 'purp'] = data.purp.str.lower()

        X = data.purp
        for d in '1234567890%[].,/"()-<>':
            X = X.str.replace(d, '')

        y = data['class'].values
        self._create_and_fit_encoder(all_data['class'], 
                                     encoder=LabelEncoder, attr_name='le_target', train=train)
        y = self.le_target.transform(y)

        #other features
        names_to_encode = np.unique(np.hstack((all_data.name_to.values, all_data.name_from.values)))
        self._create_and_fit_encoder(names_to_encode, 
                                       encoder=LabelEncoder, attr_name='le', train=train)
        
        name_to_fitted = self.le.transform(data.name_to.values)
        name_from_fitted = self.le.transform(data.name_from.values)

        other_features = np.zeros((data.shape[0], 3))
        other_features[:, 0] = name_to_fitted
        other_features[:, 1] = name_from_fitted

        sum_arr = data['sum'].to_numpy().reshape(-1,1)
        self._create_and_fit_encoder(sum_arr, 
                                     encoder=StandardScaler, attr_name='sum_scaler', train=train)
        other_features[:, 2] = self.sum_scaler.transform(sum_arr).ravel()

        # other_features[:, 3] = data.bic_same_y
        # other_features[:, 4] = data.inn_same_y

        return X, other_features, y


def check_data(values):
    assert pd.isnull(values).sum() == 0, 'data contains na target values'

class TextsLoader:
    '''
    Вспомогательный класс для подачи данных модели
    '''
    def __init__(self, text, other_features, target, mode='train'):

        super().__init__()
        self.mode = mode
        assert self.mode in ['train', 'test'], 'unknown mode'
        self.text = text
        self.target = target
        self.other_features = other_features

        check_data(self.text)
        check_data(self.other_features)
        if self.mode == 'train':
            check_data(self.target)

        self.tokenizer = BertTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

        self.inputs = None
        self.attention = None
        self.process_tokenizer()

    def process_tokenizer(self):
        self.inputs, _, self.attention = self.tokenizer(list(self.text), padding='longest',
                                                        return_tensors="pt").data.values()
        self.other_features = torch.tensor(self.other_features).type(torch.FloatTensor)
        if self.mode == 'train':
            self.target = torch.tensor(self.target).type(torch.LongTensor)

    def __getitem__(self, index):
        return self.inputs[index], self.attention[index], self.target[index], self.other_features[index]

    def __len__(self):
        return self.text.shape[0]


class BertForClassification(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.linear_1 = nn.Linear(768, 256)
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.linear_2 = nn.Linear(256 + 3, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, inputs, attention, other_features):
        output = self.bert(inputs, attention)[1]

        output = F.relu(self.dropout(self.batch_norm_1(self.linear_1(output))))
        output = torch.cat((output, other_features), dim=1)
        output = self.linear_2(output)

        return output


    def train_with_chunks(self, X_train, other_features_train, y_train,
                          X_val, other_features_val, y_val,
                          epochs, optimizer, loss_fn, batch_size=8, device=torch.device('cpu')):

        train_loader, val_loader = TextsLoader(X_train, other_features_train, y_train), TextsLoader(X_val, other_features_val, y_val)
        train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_loader, batch_size=batch_size, shuffle=False, drop_last=True)

        train_loss = []
        val_loss = []

        for epoch in tqdm(range(epochs)):

            train_epoch_losses = []
            val_epoch_losses = []

            self.train()
            for text, attention, labels, other_features in tqdm(train_dataloader):
        
                optimizer.zero_grad()

                text, attention, labels, other_features = text.to(device), attention.to(device), labels.to(device), other_features.to(device)

                preds = self(text, attention, other_features)

                loss = loss_fn(preds, labels)

                loss.backward()
                optimizer.step()
        
                train_epoch_losses.append(loss.item())

            train_loss.append(np.mean(train_epoch_losses))
    
            self.eval()
            true = []
            predicted = []
            for text, attention, labels, other_features in tqdm(val_dataloader):

                text, attention, labels, other_features = text.to(device), attention.to(device), labels.to(device), other_features.to(device)

                preds = self(text, attention, other_features)

                loss = loss_fn(preds, labels)
        
                val_epoch_losses.append(loss.item())

                true.extend(labels.cpu().numpy())
                predicted.extend(preds.max(dim=1).indices.detach().cpu().numpy())

            val_loss.append(np.mean(val_epoch_losses))
    
            print(f'Epoch: {epoch + 1}, train_loss: {train_loss[-1]}, val_loss: {val_loss[-1]}')
            print('f1: ', f1_score(true, predicted, average='micro'))


    def predict_with_chunks(self, X, other_features, y, batch_size=8, device=torch.device('cpu')):

        loader = TextsLoader(X, other_features, y, mode='test')
        dataloader = DataLoader(loader, batch_size=batch_size, shuffle=False)

        self.eval()
        self.to(device)
        
        predicted_proba = []

        for text, attention, labels, other_features in dataloader:

            text, attention, labels, other_features = text.to(device), attention.to(device), labels.to(device), other_features.to(device)

            preds = self(text, attention, other_features)

            predicted_proba.extend(F.softmax(preds, dim=1).detach().cpu().numpy())

        return np.array(predicted_proba)