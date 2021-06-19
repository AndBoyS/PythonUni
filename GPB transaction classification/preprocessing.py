import numpy as np
import pandas as pd
import re 
import string
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from rutimeparser import get_clear_text
import requests
import multiprocessing as mp
import sys

#чтобы не менять пути к файлам в зависимости от того, в колабе ноутбук или нет
def get_drive_path(path):
    in_colab = 'google.colab' in sys.modules
    return '/content/drive/My Drive/GPB/'+re.sub(r'\\','/',path) if in_colab else path

def get_code_names(code_names_path):

    classes_codes = pd.read_csv(code_names_path)
    classes_codes = classes_codes.append(pd.DataFrame(['__прочее', -1],index=classes_codes.columns, columns=[classes_codes.index[-1]+1]).T)
    classes_codes = classes_codes.append(pd.DataFrame(['лизинг', classes_codes['0'].max()+1],index=classes_codes.columns, columns=[classes_codes.index[-1]+1]).T)
    code_to_class = pd.Series(classes_codes.iloc[:,0].values, index=classes_codes.iloc[:,1])
    class_to_code = pd.Series(classes_codes.iloc[:,1].values, index=classes_codes.iloc[:,0])

    return code_to_class, class_to_code

def remove_noise(tokens, stop_words = []):
    '''
    Очистка от мусора в описании транзакции
    '''
    cleaned_tokens = []
    
    for token in tokens:
        
        #номера карт (весь токен это 20 цифр)
        token = "карта" if re.search(r"^\d{20}$", token) else token
        #убираем числа
        #token = re.sub(r'\d', '', token)
        #очистка нелексигографических символов
        #token = re.sub(r'\W', '', token)
        token  = '' if re.search(r'[\W\d]', token) else token

        #убираем токены состояющие из одного символа или являющиеся стоп-словом
        if len(token) > 1 and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
                
    return cleaned_tokens

def replace_multiple_times(string, find, replacement):
    '''
    Замена подстроки (заменяются все подстроки find)
    '''
    
    while True: 
        old_string = string
        string = string.replace(find, replacement)
    
        if old_string == string:
            return string


def get_cleaned_features(features, stop_words, path_to_shortenings_file=None, fix_spelling=False, chunk_size=5):
    '''
    Предобработка описания транзакций, включающая: очистку от дат; токенизацию; очистку от токенов с числовыми и неликсикографическими символами, стоп-слов; перевод сокращений в полную форму; исправление опечаток

    Parameters:
    features: iterable - контейнер со строками
    stop_words: iterable - контейнер со стоп-словами
    path_to_shortenings_file: str - путь к файлу со списком сокращений
    fix_spelling: boolean - исправлять ли опечатки при помощи API Яндекс спеллера
    chunk_size: int - сколько за раз текстов мы посылаем API
    
    Returns:
    str[][]
    '''

    if path_to_shortenings_file is not None: #перевод сокращений в полную форму
        shortenings = pd.read_excel(path_to_shortenings_file)
        shortenings_dict = {}
        
        for i,sh_str in shortenings['Сокращение'].items():
            for sh in sh_str.split(';'):
                if sh != '':
                    shortenings_dict[sh.lower()] = shortenings['Полное значение'][i].lower()
      
        for sh in shortenings_dict:
            features = [replace_multiple_times(' '+sentence+' ', ' '+sh+' ', ' '+shortenings_dict[sh]+' ')[1:-1] for sentence in features]

    if fix_spelling:
        features = spell_fixer(features, chunk_size)

    sentences_tokens = [word_tokenize(get_clear_text(sentence)) for sentence in features]
    return [remove_noise(tokens, stop_words) for tokens in sentences_tokens]


#Исправление опечаток при помощи Яндекс спеллер API

def parse(string):
    return '+'.join(string.split())
  
def create_request(texts):
    return 'https://speller.yandex.net/services/spellservice.json/checkTexts?'+'&'.join('text='+parse(text) for text in texts)
  
def list_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:(i+n)]

def process_texts(texts):
    if type(texts) == pd.Series:
        texts = texts.tolist()
    elif type(texts) == list:
        texts = texts.copy()

    response = requests.get(create_request(texts))
    try:
        json_list = response.json()
    except:
        print(response.content)
        return texts

    for i,error_list in enumerate(json_list):
        changed_len = 0
        for error in error_list:
            texts[i], changed_len = texts[i][:error['pos']+changed_len] + error['s'][0] + texts[i][error['pos']+error['len']-changed_len:], len(error['s'][0])-error['len']

    return texts

def spell_fixer(all_texts, chunk_size=5):

    cpu_count = mp.cpu_count()
    fixed_texts = []
    amount_of_chunks = len(all_texts)//chunk_size+1


    for texts_list in list_chunks(np.array_split(all_texts, amount_of_chunks), cpu_count):
    
        with mp.Pool(cpu_count) as p: #multiprocessing
            texts_list_mapped = p.map(process_texts, texts_list)
    
        [fixed_texts.extend(texts) for texts in texts_list_mapped]
  
    return fixed_texts




