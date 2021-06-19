import numpy as np
import pandas as pd

import transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import TFRobertaModel, BertTokenizer, RobertaConfig, RobertaTokenizerFast, RobertaTokenizer

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import softmax, sigmoid, relu

import os
import keras 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
import re

from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.preprocessing.sequence import pad_sequences

def smooth_labels(labels, factor=0.1):
    target = np.copy(labels)
    target *= (1 - factor)
    target += (factor / target.shape[1])
    return target

class RobertaProcessor:
    '''
    Класс для предобработки данных под работу роберты + НН
    '''
    def get_data(self, data, all_data=None, train=True, smoothing_label_factor=0.4):
        
        if all_data is None:
            all_data = data

        if train:
            self.tokenizer = RobertaTokenizerFast.from_pretrained('blinoff/roberta-base-russian-v0')
            self.encoder = OneHotEncoder()
            self.encoder.fit(np.array(all_data['class']).reshape(-1, 1))

        X = data['purp'].apply(lambda x: ' '.join(re.findall(r'[\w\d\+]+[\.,]*[\w\d\+]*', x)))
        result = self.tokenizer(list(X), padding='longest')

        tokens = np.array(result['input_ids'])
        attn_mask = np.array(result['attention_mask'])

        if train:
            self.padding_len = tokens.shape[1]
        else:
            tokens = pad_sequences(tokens, padding='post', maxlen=self.padding_len)
            attn_mask = pad_sequences(attn_mask, padding='post', maxlen=self.padding_len)

        y = self.encoder.transform(np.array(data['class']).reshape(-1, 1)).toarray()
        if train:
            y = smooth_labels(y, factor=smoothing_label_factor)

        return tokens, attn_mask, y

def create_roberta_model(tokens_train, attn_mask_train, num_classes):

    config = RobertaConfig(vocab_size=50021, hidden_size=1024,
                           num_hidden_layers=16, num_attention_heads=16, intermediate_size=2048, 
                           attention_probs_dropout_prob=0.3, hidden_dropout_prob=0.3)
    
    bert = TFRobertaModel(config)

    # dense1 = Dense(500, activation='relu')
    dense2 = Dense(368, activation='relu')
    dense3 = Dense(num_classes, activation='softmax')
    dropout = Dropout(0.3)
    
    tokens = Input(shape=(tokens_train.shape[1],), dtype=tf.int32)
    attn_mask = Input(shape=(attn_mask_train.shape[1],), dtype=tf.int32)

    pooled_output = bert(tokens, attn_mask).pooler_output

    med = dropout(dense2(pooled_output))

    final = dense3(pooled_output)

    model = Model(inputs=[tokens, attn_mask], outputs=final)
    
    return model

def train_roberta_model(tokens_train, tokens_val, attn_mask_train, attn_mask_val, y_train, y_val, num_classes, checkpoint_path, epochs):

    # TPU initialization code
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
           
    early_stopping_monitor = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max', save_weights_only=True)
    callbacks = [early_stopping_monitor, checkpoint]
           
    optimizer = tf.optimizers.Adam(learning_rate=1e-05, epsilon=1e-07)
    with strategy.scope():
        model = create_roberta_model(tokens_train, attn_mask_train, num_classes)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

        model.fit(x=[tokens_train, attn_mask_train], y=y_train, batch_size=80, 
                validation_data=([tokens_val, attn_mask_val], y_val), epochs=epochs, 
                callbacks=[early_stopping_monitor, checkpoint])

    return model