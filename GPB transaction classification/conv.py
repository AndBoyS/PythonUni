import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import preprocessing
from preprocessing import get_drive_path

from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import scipy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.core import Dropout, Dense, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.optimizers import Adam
from tensorflow.keras import regularizers as reg
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
from keras.losses import CategoricalCrossentropy

from keras.layers import Input, Concatenate, Flatten, Dense
from keras.models import Model

import tensorflow as tf
import keras.backend as K

def create_conv1d_model(vocab_size, number_of_dimensions, embedding_matrix, padding_len, output_number):
    
    loss = 'categorical_crossentropy'
    activation = 'softmax'
    filters, kernel_size = 50, 5

    model = Sequential()
    embedding_layer = Embedding(input_dim=vocab_size, 
                            output_dim=number_of_dimensions, 
                            weights=[embedding_matrix], 
                            input_length=padding_len, 
                            trainable=True,
                            embeddings_regularizer=reg.l1(0.0005))
    model.add(embedding_layer)
    model.add(Conv1D(filters, 
                 kernel_size, 
                 activation='relu', 
                 kernel_regularizer=reg.l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(10))
    #model.add(GlobalMaxPooling1D())
    model.add(Flatten())
    model.add(Dense(output_number, activation=activation))

    model.compile(optimizer=Adam(learning_rate=0.01), 
                loss=loss, metrics=['acc', f1_micro, f1_weighted])

    return model

def train_conv1d_model(X_train, X_val, y_train, y_val,
                       vocab_size, number_of_dimensions, embedding_matrix, padding_len, output_number, checkpoint_path, model=None):

    if model is None:
        model = create_conv1d_model(vocab_size, number_of_dimensions, embedding_matrix, padding_len, output_number)
    
    batch_size = 128
    epochs = 50

    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_f1_micro', verbose=0, save_best_only=True, mode='max', save_weights_only=True)
    early_stop = EarlyStopping(monitor='val_f1_micro', mode='max', verbose=1, patience=10)
    callbacks = [checkpoint, early_stop] 

    history = model.fit(X_train,
                    y_train,
                    batch_size=batch_size, 
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(X_val, y_val)
                   )
    return model,history
	
def create_conv_model_with_two_inputs(other_features_dim, vocab_size, number_of_dimensions, embedding_matrix, padding_len, output_number):

    loss = CategoricalCrossentropy(label_smoothing=0.2)
    #loss = 'categorical_crossentropy'
    activation = 'softmax'
    filters, kernel_size = 10, 5

    text_input = Input(shape=(padding_len,))
    vector_input = Input(shape=(other_features_dim,))

    embedding_layer = Embedding(input_dim=vocab_size, 
                            output_dim=number_of_dimensions, 
                            weights=[embedding_matrix],
                            input_length=padding_len,
                            trainable=True,
                            embeddings_regularizer=reg.l1(0.0005))(text_input)
    conv_layer = Conv1D(filters, kernel_size, activation='relu', kernel_regularizer=reg.l2(0.005))(embedding_layer)
    conv_layer = Dropout(0.2)(conv_layer)
    conv_layer = MaxPooling1D(10)(conv_layer)
    #conv_layer = GlobalMaxPooling1D()(conv_layer)
    conv_layer = Flatten()(conv_layer)

    concat_layer = Concatenate()([vector_input, conv_layer])
    concat_layer = Dense(10, activation='relu')(concat_layer)
    concat_layer = Dropout(0.1)(concat_layer)

    output = Dense(output_number, activation=activation)(concat_layer)

    model = Model(inputs=[text_input, vector_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.01), loss=loss, metrics=['acc',f1_micro,f1_weighted])
    
    return model

def train_conv_model_with_two_inputs(X_train, X_val, other_features_train, other_features_val, y_train, y_val,
                                     vocab_size, number_of_dimensions, embedding_matrix, padding_len, output_number, model=None):
    
    if model is None:
        model = create_conv_model_with_two_inputs(other_features_train.shape[1], vocab_size, number_of_dimensions, embedding_matrix, padding_len, output_number)
    
    filepath = get_drive_path("saved models\\saved-model-conv-other-features.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_f1_micro', verbose=0, save_best_only=True, mode='max', save_weights_only=True)
    early_stop = EarlyStopping(monitor='val_f1_micro', mode='max', verbose=1, patience=20)

    callbacks = [checkpoint, early_stop]
    batch_size = 128
    epochs = 100

    history = model.fit([X_train,other_features_train],
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=([X_val,other_features_val], y_val)
                   )

    return model,history
	

#https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras
def f1_(true, pred, average='micro'): #shapes (batch, output_number) 
        
    predLabels = K.argmax(pred, axis=-1)
    output_number = true.shape[1]
    pred = K.one_hot(predLabels, output_number) 

    ground_positives = K.sum(true, axis=0)       # = TP + FN
    pred_positives = K.sum(pred, axis=0)         # = TP + FP
    true_positives = K.sum(true * pred, axis=0)  # = TP
    
    if average == 'micro':
        true_positives = K.sum(true_positives)
        pred_positives = K.sum(pred_positives)
        ground_positives = K.sum(ground_positives)
    
    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon()) 
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon()) 
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (output_number,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #not sure if this last epsilon is necessary
        #mathematically not, but maybe to avoid computational instability
        #still with shape (output_number,)
    
    if average == 'weighted':
        f1 = f1 * ground_positives / K.sum(ground_positives)
        f1 = K.sum(f1)
    
    return f1


def f1_micro(true, pred):
    return f1_(true, pred, average='micro')
    
def f1_weighted(true, pred):
    return f1_(true, pred, average='weighted')


def visualize(history):
	def visualize_evaluations(plot_data, scores, title, i):
		plt.subplot(2,2,i)
		[plt.plot(plot_data[score]) for score in scores]
		plt.title(title)
		plt.ylabel('score')
		plt.xlabel('epoch')
		plt.legend(scores, loc='upper left')

	plot_data = history.history

	plt.figure(figsize=(10,10))
	visualize_evaluations(plot_data, ['acc','val_acc'], 'Model accuracy', 1)
	visualize_evaluations(plot_data, ['f1_micro','val_f1_micro'], 'Model f1 micro score', 2)
	visualize_evaluations(plot_data, ['f1_weighted','val_f1_weighted'], 'Model f1 weighted score', 3)
	visualize_evaluations(plot_data, ['loss','val_loss'], 'Model loss', 4)


class ConvPreprocessor:
    '''
    Класс для предобработки данных под работу сверточной нейронной сети
    '''
    def __init__(self, use_other_features=False, stop_words = stopwords.words('russian'), path_to_shortenings_file=None, fix_spelling=False):
        '''
        use_other_features: bool - использовать ли дополнительные признаки из других столбцов
        stop_words: list - список стоп-слов (параметр для get_cleaned_features)
        path_to_shortenings_file: str - путь к таблице xlsx сокращений (параметр для get_cleaned_features)
        fix_spelling: bool - исправлять ли опечатки при помощи Яндекс.Спеллер API (параметр для get_cleaned_features)
        '''

        self.use_other_features = use_other_features
        self.stop_words = stop_words
        self.path_to_shortenings_file = path_to_shortenings_file
        self.fix_spelling = fix_spelling

    
    def get_data(self, data, all_data=None, train=True):
        '''
        data: pd.DataFrame - таблица со всеми признаками транзакций
        all_data: pd.DataFrame - таблица со всеми признаками транзакций (обучаяющая+тестовая выборка - для обучения енкодеров)
        train: bool - для обучения ли данные
        '''

        if all_data is None:
            all_data = data

        X, y, ids = data['purp'], data['class'], data.index

        if self.use_other_features:

            other_features = self.extract_features(data, encoder_dict=getattr(self, 'encoder_dict', False), all_data=all_data)

            other_features.index = ids
            other_features = other_features.dropna()

            ids = other_features.index
            X, y = X.loc[ids], y.loc[ids]

        X = preprocessing.get_cleaned_features(X, self.stop_words, self.path_to_shortenings_file, self.fix_spelling)

        if train:
            y, y_encoder = encode(y, OneHotEncoder, all_data['class'])
            self.y_encoder = y_encoder
        else:
            y = encode(y, self.y_encoder)

        if train:
            self.tokenizer = Tokenizer(num_words=20000)
            self.tokenizer.fit_on_texts(X)

        X = self.tokenizer.texts_to_sequences(X)

        if train:
            self.padding_len = len(max(X, key=len)) 
        
        X = pad_sequences(X, padding='post', maxlen=self.padding_len)
        
        if self.use_other_features:

            return X, y, other_features, ids

        else:

            return X, y, ids

    #encoder_dict - для получения доп фич тестового датасета
    def extract_features(self, data, encoder_dict=False, all_data=None):
        if all_data is None:
            all_data = data

        test = bool(encoder_dict)
        #name_pair = data['to_acc'].astype('str').str[:]+' '+data['from_acc'].astype('str').str[:]

        other_features = pd.DataFrame()
    
        features_dict = {}
        #not encoded features
        #features_dict['payer_ОКВЭД_main'] = data['payer_ОКВЭД_main']
        #features_dict['payee_ОКВЭД_main'] = data['payee_ОКВЭД_main']
        features_dict['is_sum_int'] = (data['sum'] % 1 == 0).astype('int')
        if not test:
            #encoded features
            encoder_dict = {}
        
            #features_dict['to_acc'], encoder_dict['to_acc'] = encode(data['to_acc'], LabelEncoder, all_data['to_acc'])
            #features_dict['from_acc'], encoder_dict['from_acc'] = encode(data['from_acc'], LabelEncoder, all_data['from_acc'])
            features_dict['sum'], encoder_dict['sum'] = encode(data['sum'], MinMaxScaler, all_data['sum'])
            #one-hot
            #features_dict['payer_ОКВЭД_main'], encoder_dict['payer_ОКВЭД_main'] = encode(data['payer_ОКВЭД_main'], OneHotEncoder, all_data['payer_ОКВЭД_main'])
            #features_dict['payee_ОКВЭД_main'], encoder_dict['payee_ОКВЭД_main'] = encode(data['payee_ОКВЭД_main'], OneHotEncoder, all_data['payee_ОКВЭД_main'])
            features_dict['to_acc'], encoder_dict['to_acc'] = encode(data['to_acc'], OneHotEncoder, all_data['to_acc'])
            features_dict['from_acc'], encoder_dict['from_acc'] = encode(data['from_acc'], OneHotEncoder, all_data['from_acc'])

            self.other_features_indices = [] #для перебора one-hot фич
        
        else:
            for col in encoder_dict:
                features_dict[col] = encoder_dict[col].transform(data[col].to_numpy().reshape(-1, 1))
                if type(features_dict[col]) == scipy.sparse.csr.csr_matrix:
                    features_dict[col] = features_dict[col].toarray()

        for col,feature in features_dict.items():
            #для one-hot фич
            if len(feature.shape) > 1 and feature.shape[1] > 1:
                #print(feature, type(feature))
                feature = pd.DataFrame(feature, columns=[col+str(i) for i in range(1,feature.shape[1]+1)])
                self.other_features_indices.append(list( range(other_features.shape[1], other_features.shape[1]+feature.shape[1]) ))
                other_features = pd.concat((other_features, feature), axis=1)
            else:
                other_features[col] = feature
                self.other_features_indices.append([other_features.shape[1]-1])

        self.encoder_dict = encoder_dict
        return other_features


def encode(feature, encoder, all_feature=None): #all_feature - полный набор признаков для енкодинга
    fitted = bool(all_feature is None)
    feature = feature.to_numpy().reshape(-1,1)

    if not fitted:
        all_feature = all_feature.dropna().to_numpy().reshape(-1,1)
        encoder = encoder()
        encoder.fit(all_feature)

    res = encoder.transform(feature)

    if type(res) == scipy.sparse.csr.csr_matrix:
        res = res.toarray()

    return (res, encoder) if not fitted else res


def round_prediction(pred_array):

    res = np.zeros(pred_array.shape)

    for i,max_idx in enumerate(np.argmax(pred_array, axis=1)):
        res[i,max_idx] = 1

    return res