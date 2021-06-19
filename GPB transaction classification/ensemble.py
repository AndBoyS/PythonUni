import numpy as np
import pandas as pd
import os
import sys
import pickle
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from nltk.corpus import stopwords

import tensorflow as tf
import torch

import preprocessing
import embeddings as eb
import conv
import bert
import random_forest

def blockPrint(): #игнор вывода при обучении моделей
    if globals().get('sys_stdout', None) is None:
        globals()['sys_stdout'] = sys.stdout
        
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys_stdout

class EnsembleTemplate:

    def __init__(self, n_splits, model_folder_path, model_name):
        '''
        Parameters:
        n_splits: int - количество разбиений для KFold
        model_folder_path: str - путь папки, в которую будут сохраняться модели
        model_name: str - название модели
        '''
        self.n_splits = n_splits
        self.model_folder_path = model_folder_path
        self.model_name = model_name

    def train(self, data, retrain=False):
        '''
        Тренирует модель на KFold разделе выборки data

        Parameters:
        data: pandas.DataFrame
        retrain: bool - игнорировать ли сохраненные модели
        '''

        object_path = os.path.join(self.model_folder_path, self.model_name+'.pkl') #для сохранения/загрузки объекта
        self.models = []
        
        if not retrain:
            print('Попытка загрузить обученные модели и аттрибуты')
            if self.load_models_and_attrs(object_path) == 'loaded':
                print('Загружено')
                return None
            print('Попытка не удалась')

        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True)

        if not os.path.isdir(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        data, data_for_meta_model = train_test_split(data, test_size=0.2, random_state=42)
        
        for i, idxs in enumerate(kf.split(data, data['class']), 1):
            
            train_index, val_index = idxs

            print(f'Обучение модели {i}')
            blockPrint()
            try:
                self.models.append(self.fit(data.iloc[train_index], data.iloc[val_index]))
                enablePrint()
            except Exception as s:
                enablePrint()
                raise s
            
            load_path = os.path.join(self.model_folder_path, self.model_name+str(i))
            self.save_model(self.models[-1], load_path)

        self.one_hot_encoder = OneHotEncoder()
        all_labels = pd.concat([data['class'], data_for_meta_model['class']]).unique().reshape(-1,1)
        self.one_hot_encoder.fit(all_labels)

        self.train_meta_model(data_for_meta_model)

        self.save(object_path)

    def train_meta_model(self, data):
        '''
        Обучить мета-модель для предсказаний
        '''

        self.lr_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        predictions_for_meta_model = np.hstack([self.one_hot_encoder.transform(self.model_predict(model, data)).toarray() for model in self.models])
        self.lr_model.fit(predictions_for_meta_model, data['class'])

    def predict(self, data, mode='mode'):
        '''
        Предсказать ансамблем

        Parameters:
        data: pd.DataFrame - данные для предсказания
        mode: str - режим предсказаний ('mode' - мода всех предсказаний, 'lr' - на основе лог регрессии, обученной на предсказаниях)
        '''

        if mode == 'mode':
            predictions = np.hstack([self.model_predict(model, data) for model in self.models])
            return stats.mode(predictions, axis=1).mode
        
        elif mode == 'lr':
            predictions = np.hstack([self.one_hot_encoder.transform(self.model_predict(model, data)).toarray() for model in self.models])
            return self.lr_model.predict(predictions)

    def save(self, path):
        '''
        Сохранить объект, наследующий EnsembleTemplate

        Parameters:
        path: str - путь, на который сохраняется объект
        '''
        models = self.models.copy()
        self.models = []

        with open(path, 'wb') as f:
            pickle.dump(self, f)

        self.models = models

    def load(self, path):
        '''
        Загрузить аттрибуты объекта, наследующего EnsembleTemplate

        Parameters:
        path: str - путь, с которого загружается объект
        '''
        with open(path, 'rb') as f:
            loaded_self = pickle.load(f)
            self.__dict__.update(loaded_self.__dict__)


    def load_models_and_attrs(self, object_path):
        '''
        Пытается загрузить модели в self.models и атрибуты self если такого сохранены

        Returns:
        'loaded' если загрузка произошла успешна, иначе 'not loaded'    
        '''
        try:
            self.load(object_path)

            for i in range(1, self.n_splits+1):
                load_path = os.path.join(self.model_folder_path, self.model_name+str(i))
                self.models.append(self.load_model(load_path))

            return 'loaded'

        except FileNotFoundError as s:
            pass

        return 'not loaded'

    def fit(self, data, val_data):
        '''
        Обучение модели

        Parameters:
        data: pd.DataFrame - данные для обучения модели (все столбцы)
        val_data : pd.DataFrame - данные для валидации

        Returns:
        model - обученная модель
        '''
        pass

    def model_predict(self, model, data):
        '''
        Parameters:
        model - обученная модель
        data: pd.DataFrame - данные для предсказания
        
        Returns:
        prediction: np.array - вектор-столбец, значения в принятой кодировке классов 
        '''
        pass

    def save_model(self, model, path):
        '''
        Сохранить обученную модель

        Parameters:
        model - обученная модель
        path: str - путь, на который сохраняется модель 
        '''
        pass

    def load_model(self, path):
        '''
        Parameters:
        path: str - путь, с которого загружается модель
        '''
        pass


class SvmEnsemble(EnsembleTemplate):

    def __init__(self, svm_params, **kwargs):

        super().__init__(**kwargs)
        self.svm_params = svm_params

    def fit(self, data, _):
        
        if not getattr(self, 'vectorizer', False):
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(data['purp'])
        
        train_vectors = self.vectorizer.transform(data['purp'])

        model = SVC(**self.svm_params)
        model.fit(train_vectors, data['class'])

        return model
    
    def model_predict(self, model, data):
        
        vectors = self.vectorizer.transform(data['purp'])
        prediction = model.predict(vectors)

        return prediction.reshape(-1,1)

    def save_model(self, model, path):

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, path):
        
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model


class ConvEnsemble(EnsembleTemplate):

    def __init__(self, full_corpus, cleaned_corpus_version, cleaned_corpus_path, path_to_shortenings_file, fix_spelling, stop_words, number_of_dimensions, embedding_path, checkpoint_path, **kwargs):
        
        super().__init__(**kwargs)

        self.full_corpus = full_corpus
        self.cleaned_corpus_version = cleaned_corpus_version
        self.cleaned_corpus_path = cleaned_corpus_path

        self.path_to_shortenings_file = path_to_shortenings_file
        self.fix_spelling = fix_spelling
        self.stop_words = stop_words

        self.number_of_dimensions = number_of_dimensions
        self.embedding_path = embedding_path
        self.checkpoint_path = checkpoint_path

    def fit(self, data, val_data):

        if not getattr(self, 'conv_processor', False):

            try:
                cleaned_corpus = eb.get_cleaned_corpus(self.cleaned_corpus_path, 
                                       self.full_corpus, 
                                       self.stop_words,
                                       path_to_shortenings_file=self.path_to_shortenings_file,
                                       fix_spelling=self.fix_spelling)
            except TypeError: #в случае если self.full_corpus == None
                cleaned_corpus = None

            self.conv_processor = conv.ConvPreprocessor(use_other_features=False, 
                                   path_to_shortenings_file=self.path_to_shortenings_file,
                                   fix_spelling=self.fix_spelling)

            X_train, y_train, ids_train = self.conv_processor.get_data(data, data, train=True)
            
            vocab_size = len(self.conv_processor.tokenizer.word_index)+1
            self.vocab_size = self.conv_processor.tokenizer.num_words if self.conv_processor.tokenizer.num_words < vocab_size else vocab_size
            self.padding_len = X_train.shape[1]
            self.output_number = y_train.shape[1]
            
            self.embedding_matrix = eb.load_fast_text_trained(self.embedding_path, self.conv_processor.tokenizer, self.number_of_dimensions, cleaned_corpus)

        else:

            X_train, y_train, ids_train = self.conv_processor.get_data(data, data, train=False)
        
        X_val, y_val, ids_val = self.conv_processor.get_data(val_data, val_data, train=False)
        model,history = conv.train_conv1d_model(X_train, X_val, y_train, y_val,
                                           self.vocab_size, self.number_of_dimensions, self.embedding_matrix, self.padding_len, self.output_number, self.checkpoint_path)

        model.load_weights(self.checkpoint_path)
        return model
    
    def model_predict(self, model, data):
        
        X, y, ids = self.conv_processor.get_data(data, train=False)
    
        prediction_binary = conv.round_prediction(model.predict(X))
        prediction = self.conv_processor.y_encoder.inverse_transform(prediction_binary)

        return prediction.reshape(-1,1)

    def save_model(self, model, path):

        model.save(path+'.h5')

    def load_model(self, path):
        
        model = conv.create_conv1d_model(self.vocab_size, self.number_of_dimensions, self.embedding_matrix, self.padding_len, self.output_number)
        model.load_weights(path+'.h5')

        return model

    def load_models_and_attrs(self, object_path):
        
        try:

            self.load(object_path)
            for i in range(1, self.n_splits+1):
                load_path = os.path.join(self.model_folder_path, self.model_name+str(i))
                self.models.append(self.load_model(load_path))

            return 'loaded'

        except FileNotFoundError:
            pass
        except RuntimeError:
            pass
        except tf.errors.NotFoundError:
            pass
        except OSError:
            pass
        except Exception as s:
            print(s)
            pass

        return 'not loaded'


class BertEnsemble(EnsembleTemplate):

    def __init__(self, num_classes, all_data, batch_size=8, **kwargs):

        self.num_classes = num_classes
        self.all_data = all_data
        self.batch_size = batch_size
        super().__init__(**kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, data, val_data):

        if not getattr(self, 'bert_processor', False):
            self.bert_processor = bert.BertProccessor()
            X_train, other_features_train, y_train = self.bert_processor.get_data(data, train=True, all_data=self.all_data)
        
        else:
            X_train, other_features_train, y_train = self.bert_processor.get_data(data, train=False)

        X_val, other_features_val, y_val = self.bert_processor.get_data(val_data, train=False)
        
        model = bert.BertForClassification(self.num_classes).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        epochs = 6

        model.train_with_chunks(X_train, other_features_train, y_train,
                                X_val, other_features_val, y_val,
                                epochs=epochs, optimizer=optimizer, loss_fn=loss_fn, device=self.device, batch_size=self.batch_size)
        
        return model
    
    def model_predict(self, model, data):

        X_test, other_features_test, y_test = self.bert_processor.get_data(data, train=False)

        predicted_proba = model.predict_with_chunks(X_test, other_features_test, y_test, device=self.device)
        pred_bert_encoded = np.argmax(predicted_proba, axis=1)
        prediction = self.bert_processor.le_target.inverse_transform(pred_bert_encoded)

        return prediction.reshape(-1,1)

    def save_model(self, model, path):

        torch.save(model.state_dict(), path)

    def load_model(self, path):
        
        model = bert.BertForClassification(self.num_classes)
        model.load_state_dict(torch.load(path, map_location=self.device))

        return model


class RFEnsemble(EnsembleTemplate):

    def __init__(self, params, all_data, **kwargs):

        super().__init__(**kwargs)
        self.params = params
        self.all_data = all_data

    def fit(self, data, _):
        
        if not getattr(self, 'rf_processor', False):
            self.rf_processor = random_forest.RandomForestProcessor()
            X, y = self.rf_processor.get_data(data, self.all_data)
        
        else:
            X, y = self.rf_processor.get_data(data, train=False)
        
        model = RandomForestClassifier(**self.params)
        model.fit(X, y)

        return model
    
    def model_predict(self, model, data):
        
        X, y = self.rf_processor.get_data(data, train=False)
        prediction = model.predict(X)

        return prediction.reshape(-1,1)

    def save_model(self, model, path):

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, path):
        
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model