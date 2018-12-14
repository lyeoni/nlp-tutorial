import argparse
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import read_word_pair as read_wp

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-corpus_tk',
                   default='corpus.tk.txt',
                   help='Default=corpus.tk.txt')

    p.add_argument('-trained_word_vector',
                   default='corpus.tk.vec.txt',
                   help='Default=corpus.tk.vec.txt')

    p.add_argument('-score_corpus',
                   default='score_corpus.txt',
                   help='Default=score_corpus.txt')
    
    config = p.parse_args()
    
    return config

class DataLoader():
    def __init__(self,
                 tokenized_corpus=None,
                 trained_word_vector=None,
                 score_corpus=None,
                 max_word_num=12000,
                 min_corpus_len=3,
                 max_corpus_len=30,
                 embedding_dim=300):
        
        super(DataLoader, self).__init__()
        
        self.corpus = tokenized_corpus
        self.trained_word_vector = trained_word_vector
        self.score_corpus = score_corpus
        self.max_word_num = max_word_num
        self.min_corpus_len = min_corpus_len
        self.max_corpus_len = max_corpus_len
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        self.train = None
        self.test = None
        
    def token_to_index(self, tokenized_corpus, maximum_word_num):
        tokenizer = Tokenizer(num_words = maximum_word_num+1, oov_token='UNK')
        
        # token to index
        tokens = tokenized_corpus.apply(lambda i: i.split())
        tokenizer.fit_on_texts(tokens)
        
        tokenizer.word_index = {word:index for word, index in tokenizer.word_index.items() if index <= maximum_word_num}

        vocabulary = tokenizer.word_index

        print('number of unique tokens: {}'.format(len(vocabulary)))
        
        return vocabulary, tokenizer.texts_to_sequences(tokens)
    
    def load_data(self):
        
        data = pd.read_csv(self.score_corpus) # read pair of score-corpus
        data['corpus_tk'] = pd.read_csv(self.corpus, header=None).loc[:,0] # add tokenized coprus
        
        vocab, data['corpus_tk_index'] = self.token_to_index(data.corpus_tk, self.max_word_num)
        
        # only use corpus including more than 'min_corpus_len' words
        data['corpus_len'] = data.corpus_tk_index.apply(lambda i: len(i))
        # print('corpus length info\n{}'.format(data.corpus_len.describe()))
        data = data[data.corpus_len >= self.min_corpus_len]
        
        x = pad_sequences(data.corpus_tk_index, self.max_corpus_len)
        y = data.score.apply(lambda i: i-1)
        # print(x.shape, x, sep='\n')
        # print(y.shape, y[:5], sep='\n')
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        y_train, y_test = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test)
        self.train, self.test = (x_train, y_train), (x_test, y_test)
        print('x_train: {} / y_train: {}\nx_test: {} / y_test: {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        trained_wv = read_wp.read_word_pair(self.trained_word_vector) # read trained embedding word vectors
        print('number of trained word vector: {}'.format(len(trained_wv)))
        
        self.embedding_matrix = np.zeros((self.max_word_num+1, self.embedding_dim))
        for word, idx in vocab.items():
            embedding_wv = trained_wv.get(word)
            if embedding_wv is not None:
                self.embedding_matrix[idx] = embedding_wv
          # else:
              # print(word, idx, embedding_wv)
        print('embedding matrix shape: {}'.format(self.embedding_matrix.shape))

if __name__ == "__main__":
    config = argparser()
    
    dataloader = DataLoader(config.corpus_tk, config.trained_word_vector, config.score_corpus)
    dataloader.load_data()
