import numpy as np
from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText as FT_gensim
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models import Word2Vec
import pandas as pd
from preprocessing import get_cleaned_features 


def load_embedding(file, tokenizer):
    """
    Получить матрицу с эмбеддингами из файла

    Parameters:
    file: str - путь к файлу с векторами (каждая строка кроме первой имеет вид "слово число1 число2...")
    tokenizer: keras.preprocessing.text.Tokenizer - токенизатор имеющий информацию о тренировочном корпусе
    
    Returns:
    embedding_matrix: np.array - матрица ембеддингов для использования в моделях keras
    """
    embeddings_dictionary = {}
    with open(file, encoding='utf8') as f:
        for i,line in enumerate(f):
            if i == 0:
                number_of_dimensions = int(line.split()[1])
                continue
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
            
    return model_to_matrix(embeddings_dictionary.get, tokenizer, number_of_dimensions)

def model_to_matrix(get_word_func, tokenizer, number_of_dimensions):
    """
    Получить матрицу с ембеддингами из модели

    Parameters:
    get_word_func: function - функция принимающая слово и возвращающая ембеддинги
    tokenizer: keras.preprocessing.text.Tokenizer - токенизатор имеющий информацию о тренировочном корпусе
    number_of_dimensions: int - размерность ембеддингов
    
    Returns:
    embedding_matrix: np.array - матрица ембеддингов для использования в моделях keras
    """
    num_of_rows = tokenizer.num_words if tokenizer.num_words < len(tokenizer.word_index)+1 else len(tokenizer.word_index)+1
    embedding_matrix = np.zeros((num_of_rows, number_of_dimensions))
    for word, index in tokenizer.word_index.items():
        if index < num_of_rows:
            try:
                embedding_vector = get_word_func(word) 
            except:
                continue
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
    return embedding_matrix

def model_to_vec_file(corpus, output_file_path, get_word_func, number_of_dimensions):
    """
    Записать файл с ембеддингами, со словами из corpus, вектора получаются из get_word_func
    
    Parameters:
    corpus: array - корпус
    output_file_path: str - путь выходного файла 
    get_word_func: function - функция принимающая слово и возвращающая ембеддинги
    number_of_dimensions: int - размерность ембеддингов
    """
    tokenizer_all = Tokenizer(num_words=50000)
    tokenizer_all.fit_on_texts(corpus)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f'{len(tokenizer_all.word_index)+1} {number_of_dimensions}\n')
        for word in tokenizer_all.word_index.keys():
            try:
                vector = get_word_func(word)
                f.write(f"{word} {' '.join(map(str, vector))}\n")
            except:
                pass
            
def get_cleaned_corpus(cleaned_corpus_path, corpus, stop_words, path_to_shortenings_file=None, fix_spelling=False, chunk_size=5):
    """
    Получение очищенного корпуса

    Parameters:
    cleaned_corpus_path: str - путь, по которому будет располагаться/располагается очищенный корпус
    corpus: pd.Series - неочищенный корпус
  
    Return:
    str[][] - очищенный список предложений разделенных по токенам
    """
    try:
        cleaned_corpus = pd.read_csv(cleaned_corpus_path).iloc[:,0]
    except FileNotFoundError:
    
        cleaned_corpus = pd.Series([' '.join(sentence) for sentence in get_cleaned_features(corpus, stop_words, path_to_shortenings_file=None, fix_spelling=False, chunk_size=5)])

        cleaned_corpus.to_csv(cleaned_corpus_path, header=False, index=False)

    cleaned_corpus = cleaned_corpus.dropna()
    return [sentence.split() for sentence in cleaned_corpus]

def load_fast_text_trained(output_file_path, tokenizer, number_of_dimensions, cleaned_corpus=None):
    """
    Загрузить embedding_matrix из обученных на нашем корпусе векторов (либо через готовый файл с ембеддингами, либо обучив модель, при этом будет создан файл с векторами)
    
    Parameters:
    output_file_path: str - путь к файлу с векторами (при отсутствии файла будет создан таковой)
    tokenizer: keras.preprocessing.text.Tokenizer - токенизатор имеющий информацию о тренировочном корпусе
    number_of_dimensions: int - размерность векторов
	cleaned_corpus: pd.Series - очищенный корпус 

    Returns:
    embedding_matrix: np.array - матрица, где индекс строки соответствует индексу слова в токенизаторе, строка - вектор
    """
    try:
        embedding_matrix = load_embedding(output_file_path, tokenizer)
    except FileNotFoundError:
        ft_model = FT_gensim(size=number_of_dimensions)
        ft_model.build_vocab(cleaned_corpus)
    
        ft_model.train(
            cleaned_corpus, epochs=ft_model.epochs,
            total_examples=ft_model.corpus_count, total_words=ft_model.corpus_total_words
        )
    
        model_to_vec_file(cleaned_corpus, output_file_path, ft_model.wv.__getitem__, number_of_dimensions)
        embedding_matrix = model_to_matrix(ft_model.wv.__getitem__, tokenizer, number_of_dimensions)
    
    return embedding_matrix

def load_fast_text_pretrained_oov(output_file_path, path_to_model, tokenizer, number_of_dimensions, cleaned_corpus=None):
    """
    Загрузить embedding_matrix из предобученных векторов fasttext
    
    Parameters:
    output_file_path: str - путь к файлу с векторами (при отсутствии файла будет создан таковой)
    path_to_model: str - путь к файлу с моделью FastTextKeyedVectors
    tokenizer: keras.preprocessing.text.Tokenizer - токенизатор имеющий информацию о тренировочном корпусе
    number_of_dimensions: int - размерность векторов
	cleaned_corpus: pd.Series - очищенный корпус 

    Returns:
    embedding_matrix: np.array - матрица, где индекс строки соответствует индексу слова в токенизаторе, строка - вектор
    """
    try:
        embedding_matrix = load_embedding(output_file_path, tokenizer)
    except FileNotFoundError:
        ft_model = FastTextKeyedVectors.load(path_to_model)
        model_to_vec_file(cleaned_corpus, output_file_path, ft_model.__getitem__, number_of_dimensions)
        embedding_matrix = model_to_matrix(ft_model.__getitem__, tokenizer, number_of_dimensions)
    
    return embedding_matrix

def load_w2v_pretrained(path_to_original_vec, path_to_vec, tokenizer, number_of_dimensions):
    """
    Загрузить embedding_matrix из предобученных векторов word2vec (от rusvectores)
    
    Parameters:
    path_to_original_vec: str - путь к файлу с векторами от rusvectores
    path_to_original_vec: str - путь к файлу с исправленными векторами (при отсутствии файла будет создан таковой)
    tokenizer: keras.preprocessing.text.Tokenizer - токенизатор имеющий информацию о тренировочном корпусе
    number_of_dimensions: int - размерность векторов

    Returns:
    embedding_matrix: np.array - матрица, где индекс строки соответствует индексу слова в токенизаторе, строка - вектор
    """
    try:
        embedding_matrix = load_embedding(path_to_vec, tokenizer)
    except: #спасибо rusvectores за формат
        with open(path_to_original_vec, 'r', encoding='utf-8') as f, open(path_to_vec, 'w', encoding='utf-8') as g:
            for i,line in enumerate(f):
                if i == 0:
                    g.write(line)
                    continue
                records = line.split()
                word = records[0]
                word = word[:word.index("_")]
                g.write(word+" "+' '.join(records[1:])+'\n')
        embedding_matrix = load_embedding(path_to_vec, tokenizer)

    return embedding_matrix

def load_w2v_trained(output_file_path, tokenizer, number_of_dimensions, cleaned_corpus=None):
    """
    Загрузить embedding_matrix из обученных на нашем датасете векторов word2vec
    
    Parameters:
    output_file_path: str - путь к файлу с векторами (при отсутствии файла будет создан таковой)
    tokenizer: keras.preprocessing.text.Tokenizer - токенизатор имеющий информацию о тренировочном корпусе
    number_of_dimensions: int - размерность векторов
	cleaned_corpus: pd.Series - очищенный корпус

    Returns:
    embedding_matrix: np.array - матрица, где индекс строки соответствует индексу слова в токенизаторе, строка - вектор
    """
    try:
        embedding_matrix = load_embedding(output_file_path, tokenizer)
    except Exception as e:
        print(e)
    
        w2v_model = Word2Vec(cleaned_corpus, size=number_of_dimensions)

        model_to_vec_file(cleaned_corpus, output_file_path, w2v_model.wv.__getitem__, number_of_dimensions)
        embedding_matrix = load_embedding(output_file_path, tokenizer)

    return embedding_matrix