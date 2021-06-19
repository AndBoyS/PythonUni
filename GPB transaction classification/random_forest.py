import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class RandomForestProcessor:
    '''
    Класс для предобработки данных под работу случайного леса
    '''
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.c_vectorizer = CountVectorizer(min_df=0.005)

    def get_data(self, data, all_data=None, train=True):

        if all_data is None:
            all_data = data

        if train:
            self.label_encoder.fit(pd.concat((all_data['name_to'], all_data['name_from'])))
            self.scaler.fit(np.expand_dims(np.array(data['sum']), axis=1))
            self.c_vectorizer.fit(data['purp'])

        label_names = np.array(data[['name_to', 'name_from']].apply(self.label_encoder.transform))

        inn_bic = np.array(data[['inn_same', 'bic_same']])

        normalized_sum = self.scaler.transform(np.expand_dims(np.array(data['sum']), axis=1))
        is_int = np.array(data['sum'].apply(lambda x: float(x).is_integer()), dtype=int).reshape(-1, 1)

        accounts = np.array(data[['to_acc', 'from_acc']])
        trimmed_accounts = np.array(list(data[['to_acc', 'from_acc']].apply(lambda x: [str(x['to_acc'])[:2], str(x['from_acc'])[:2]], axis=1)), dtype=int)

        count_purps = self.c_vectorizer.transform(data['purp']).toarray()

        X = np.hstack((label_names, accounts, trimmed_accounts, count_purps, normalized_sum, is_int))
        y = data['class']

        return X, y