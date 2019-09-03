import argparse
import numpy as np
import pandas as pd
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
        
    p.add_argument('-label_corpus',
                  default='News_Category_Dataset_v2.json',
                  help='Default=News_Category_Dataset_v2.json')
    
    p.add_argument('-max_word_num',
                   type=int,
                   default=50000,
                   help='number of words to use. Default=50,000')
                   
    p.add_argument('-min_corpus_len',
                   type=int,
                   default=5,
                   help='minimum value of each corpus length. Default=5')
                  
    p.add_argument('-max_corpus_len',
                   type=int,
                   default=50,
                   help='maximum value of each corpus length. Default=50')
                  
    config = p.parse_args()
    
    return config

class DataLoader():
    def __init__(self,
                 tokenized_corpus=None,
                 trained_word_vector=None,
                 label_corpus=None,
                 max_word_num=50000,
                 min_corpus_len=5,
                 max_corpus_len=50):
        
        super(DataLoader, self).__init__()
        
        self.corpus_tk = tokenized_corpus
        self.trained_word_vector = trained_word_vector
        self.label_corpus = label_corpus
        self.max_word_num = max_word_num
        self.min_corpus_len = min_corpus_len
        self.max_corpus_len = max_corpus_len
        self.embedding_dim = 300
        self.embedding_matrix = None
        self.train = None
        self.text = None
        self.category_dict = None
    
    def token_to_index(self, tokenized_corpus, maximum_word_num):
        tokenizer = Tokenizer(num_words = maximum_word_num+1, oov_token='UNK')
        
        # token_to_index
        tokens = tokenized_corpus.apply(lambda i: i.split())
        tokenizer.fit_on_texts(tokens)
        
        tokenizer.word_index = {word:index for word, index in tokenizer.word_index.items() if index <= maximum_word_num}
        
        vocabulary = tokenizer.word_index
        
        print('number of unique tokens: {}'.format(len(vocabulary)))
        
        return vocabulary, tokenizer.texts_to_sequences(tokens)

    def load_data(self):
        # read pair of category-corpus + drop useless columns
        data = pd.read_json(self.label_corpus, lines=True).drop(['authors', 'date', 'link'], axis=1)
        
        # add tokenized corpus on data DataFrame
        data['corpus_tk'] = [ line.replace('\n', '').strip() for line in open(self.corpus_tk, 'r')]
        
        # token to word index
        print(self.max_word_num)
        vocab, data['corpus_tk_index'] = self.token_to_index(data.corpus_tk, self.max_word_num)
        
        # only use corpus including more than 'min_corpus_len' words
        data['corpus_len'] = data.corpus_tk_index.apply(lambda i: len(i))
        # print('corpus length info\n{}'.format(data.corpus_len.describe()))
        data = data[data.corpus_len >= self.min_corpus_len]
        
        x = pad_sequences(data.corpus_tk_index, self.max_corpus_len) # padding
        self.category_dict = dict()
        for i, category_name in enumerate(data.category.unique()): # category name to integer
            self.category_dict[i] = category_name
            data.category[data.category == category_name] = i
        y = data.category
        # print(x.shape, x[:5], sep='\n')
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
    
    loader = DataLoader(config.corpus_tk, config.trained_word_vector, config.label_corpus, config.max_word_num, config.min_corpus_len, config.max_corpus_len)
    
    loader.load_data()
