import numpy as np
import pandas as pd
import scipy.stats as st
import datetime
import os
from copy import deepcopy

from faker import Faker

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.randint(int_delta)
    return start + datetime.timedelta(seconds=random_second)

def normalize_probability(prob):
    return prob / prob.sum()

class GraphGenerator:
    '''Класс для генерация транзакций
    folder_path содержит файлы для каждого из классов (название_класса.xlsx), в которых указаны частоты и суммы пар типов агентов.
    Также folder_path содержит class_freqs.csv с частотами классов'''
    
    # Словарь из датафреймов, в каждом из них - вероятности пар типов агентов 
    # (для каждого класса отдельный датафрейм)
    classes_prob_dict = {}
    classes_sum_hist_dict = {}
    
    agent_set = set()
    # Словарь вида {тип агента : {инн : {'balance':сумма баланса, 'default':будет ли дефолт}}}
    agents_company_dict = {}
    # Список словарей по месяцам
    agents_company_dict_list = []

    def _generate_sum(self, hist_str, n=1):
        '''
        Сгенерировать n случайных сумм основываясь на hist_str
        hist_str - строка вида "bin_1, bin_2, ..., bin_5, value_1, value_2, ..., value_4"
        '''
        hist_str = np.array(list(map(float, hist_str.split(', '))))
        bins, hist = hist_str[:5], hist_str[5:]

        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1] # Нормализация

        values = np.random.rand(n)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = bin_midpoints[value_bins]
        return random_from_cdf
    
    # amount_of_companies - Колво компаний для одного типа агента
    def __init__(self, folder_path, start_time=datetime.datetime(2018, 1, 1), amount_of_companies=30):
        
        self.time = start_time

        self.class_freqs = pd.read_csv(os.path.join(folder_path, 'class_freqs.csv'))
        possible_classes = self.class_freqs['class']
        
        for class_ in possible_classes:
            try:
                df = pd.read_excel(
                    os.path.join(folder_path, class_+'.xlsx'), index_col=1
                ).iloc[:, 1:]
                
            except FileNotFoundError:
                self.class_freqs = self.class_freqs[self.class_freqs['class'] != class_]
                continue
            
            df_freq = df.iloc[::3,:]
            df_sum_hist = df.iloc[2::3,:]

            amount_of_transactions = df_freq.sum().sum()
            df_prob = df_freq / amount_of_transactions
            
            self.classes_prob_dict[class_] = df_prob.astype('float64')
            self.classes_sum_hist_dict[class_] = df_sum_hist
            
            self.agent_set = self.agent_set | set(df_freq.index) | set(df_freq.columns)
        
        self.class_probs = self.class_freqs.copy().rename(columns={'freq':'prob'})
        self.class_probs['prob'] = normalize_probability(self.class_freqs['freq'])
        
        # Генерация компаний
        fake = Faker()
        for agent in list(self.agent_set):
            self.agents_company_dict[agent] = {}

            for _ in range(amount_of_companies):
                inn = fake.unique.numerify(text='############')
                balance = st.norm.rvs(loc=10**9, scale=10**7)

                default_prob = 0.2
                default = np.random.choice([True, False], 
                                            p=[default_prob, 1-default_prob])
                    
                if default:
                    # Границы, внутри которых генерируется случайная дата
                    start = self.time
                    end = self.time + datetime.timedelta(days=30*8)
                    default_time = random_date(start, end)
                else:
                    default_time = None
                    
                self.agents_company_dict[agent][inn] = {'balance':balance,
                                                        'default time':default_time}
        
        self.agents_company_dict_orig = deepcopy(self.agents_company_dict)

    def generate_transactions(self, N=100):
        '''
        N: int - количество транзакций
        start_time: datetime.datetime - с какого момента начинают генерироваться транзакции
        '''
                
        current_month = self.time.month
        data = []
        
        for _ in range(N):
            self.time += datetime.timedelta(minutes=np.random.randint(1, 20))
            if self.time.month != current_month:
                current_month = self.time.month
                self.agents_company_dict_list.append(
                    deepcopy(self.agents_company_dict))

            data.append(self._generate_transaction())
        
        self.agents_company_dict_list.append(
            deepcopy(self.agents_company_dict))

        return pd.DataFrame(data)
            
    def _generate_transaction(self):
        sampled_class = np.random.choice(self.class_probs['class'], p=self.class_probs['prob'])
        
        df_prob = self.classes_prob_dict[sampled_class]
        df_sum_hist = self.classes_sum_hist_dict[sampled_class]
        
        sampled_indices = self._sample_from_prob_matrix(df_prob)
        
        sampled_sum = self._generate_sum(df_sum_hist.iloc[sampled_indices])[0]
        sampled_from_agent = df_prob.index[sampled_indices[0]]
        sampled_to_agent = df_prob.columns[sampled_indices[1]]

        # Во время сэмпла платящей компании может увеличиться сумма и класс (преддефолтное поведение)
        sampled_from, sampled_sum, sampled_class = self._sample_company(
            sampled_from_agent, sampled_sum, sampled_class, 'from')

        sampled_to, sampled_sum, sampled_class = self._sample_company(
            sampled_to_agent, sampled_sum, sampled_class, 'to')

        if sampled_from is None or sampled_to is None:
            return self._generate_transaction()

        return {
                'time':self.time,
                'from':sampled_from,
                'to':sampled_to,
                'class':sampled_class,
                'sum':sampled_sum,
                }
    
    def _sample_company(self, sampled_agent, sampled_sum, sampled_class, direction):
        '''Взять сэмпл компании на основе типа агента + изменить баланс этой компании'''
        assert direction in ['from', 'to']
        
        sampled_agent_companies_inn = list(self.agents_company_dict[sampled_agent].keys())
        sampled_agent_companies_dicts = list(self.agents_company_dict[sampled_agent].values())
        
        # Вероятности того, какая компания будет выбрана:
        # Чем больше баланс компании, тем больше вероятность
        # Для "to" доп логика: чем ближе компания к дефолту, тем меньше вероятность
        balances = np.array([company_dict['balance']
                             for company_dict in sampled_agent_companies_dicts])
        
        prob = normalize_probability(balances)
        
        # Веса, умножающиеся на вероятности компаний (преддефолтное поведение)
        weights = self._get_weights(sampled_agent_companies_dicts, direction)
        
        prob = normalize_probability(prob*weights)
        
        sampled_company_idx = np.random.choice(range(len(sampled_agent_companies_inn)), p=prob)
        sampled_company = sampled_agent_companies_inn[sampled_company_idx]
        weight = weights[sampled_company_idx]
        
        if direction == 'from': # Вывод средств со счета
            sampled_sum = -sampled_sum
            # Преддефолтное поведение
            sampled_sum *= weight 
            sampled_class = self._resample_class(
                sampled_class, sampled_agent_companies_dicts[sampled_company_idx])

            if abs(sampled_sum) > self.agents_company_dict[sampled_agent][sampled_company]['balance']:
                sampled_sum = -self.agents_company_dict[sampled_agent][sampled_company]['balance']
        
        self.agents_company_dict[sampled_agent][sampled_company]['balance'] += sampled_sum
        
        return sampled_company, abs(sampled_sum), sampled_class

    def _resample_class(self, sampled_class, company_dict):

        # Преддефолтное поведение: за полгода до дефолта вероятность типа транзакции возврат начинает увеличиваться
        # Максимальная вероятность 0.5
        half_year = 365 // 2
        prob = ((company_dict['default time']-self.time).days / half_year
                if company_dict['default time'] is not None else 0)
        prob = max(min(prob, 1), 0)
        prob = (1-prob) / 2

        return np.random.choice(['возврат', sampled_class], p=[prob, 1-prob])
    
    def _get_weights(self, sampled_agent_companies_dicts, direction):
        '''Веса, умножающиеся на вероятности выбора компаний (эмулируют преддефолтное поведение)'''
        
        # Преддефолтное поведение начинается за полгода до дефолта
        half_year = 365 // 2
        
        weights = [(company_dict['default time']-self.time).days / half_year
                    if company_dict['default time'] is not None else 1
                    for company_dict in sampled_agent_companies_dicts]
        
        # Преддефолтное поведение:
        # понижается вероятность того, что компания получает средства;
        # при этом вероятность выплат начинает увеличивается за полгода и возрастает вплоть до него
        if direction == 'from':
            for i,weight in enumerate(weights):
                if sampled_agent_companies_dicts[i]['default time'] is not None:
                    weights[i] = 1 + 100*(1-min(max(weight, 0), 1))
                
        elif direction == 'to':
            weights = np.array([min(max(weight, 0), 1) for weight in weights])
            
        return weights
    
    def _sample_from_prob_matrix(self, df_prob):
        '''
        На основе матрицы вероятностей возвращает случайные индексы
        '''
        i = np.random.choice(np.arange(df_prob.size), p=df_prob.values.flatten())
        return np.unravel_index(i, df_prob.shape)

def total_balance(generator):
    res = 0
    for agent in generator.agents_company_dict:
        for inn in generator.agents_company_dict[agent]:
            res += generator.agents_company_dict[agent][inn]['balance']
    return res