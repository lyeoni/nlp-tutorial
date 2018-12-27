import argparse
import pandas as pd
import data_loader
from nltk.tokenize.moses import MosesTokenizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

class DataGenerator():
    def __init__(self,
                 inputs=None,
                 pretrained_corpus=None,
                 max_word_num=None,
                 max_sequence_len=None):
        
        super(DataGenerator, self).__init__()
        
        self.data = None
        self.context_vector = None
        self.question_vector = None
        self.answer_text_vector = None
        self.vocabulary = None
        self.tokenizer = None
        self.context_vector, self.question_vector, self.answer_text_vector = self.get_vector(inputs, pretrained_corpus, max_word_num, max_sequence_len)
    
    def create_vocab(self, inputs, maximum_word_num):
        '''
        create vocabulary based on pre-trained corpus(previously used for fasttext)
        '''
        # make tokenizer
        tokenizer = Tokenizer(num_words = maximum_word_num+1, oov_token='UNK')
        
        # fit on input (tokenized) corpus
        f = open(inputs, 'r')
        corpus = [line for line in f]
        tokenizer.fit_on_texts(corpus)
        
        # create vocab
        tokenizer.word_index = {word:index for word, index in tokenizer.word_index.items() if index <= maximum_word_num}
        vocabulary = tokenizer.word_index
        
        print('number of unique tokens: {}'.format(len(vocabulary)))
        
        return tokenizer, vocabulary

    def get_vector(self, inputs, pretrained_corpus, max_word_num, max_sequence_len):
        loader = data_loader.DataLoader(inputs)
        self.data = pd.DataFrame({'title': loader.title, 'context': loader.context, 'question':loader.question,
                                 'answer_start':loader.answer_start, 'answer_end':loader.answer_end, 'answer_text':loader.answer_text})
            
        self.tokenizer, self.vocabulary = self.create_vocab(pretrained_corpus, max_word_num)
                            
        # tokenization & add results to columns
        nltk_tokenizer = MosesTokenizer()
        vectors = []
        for i, text_column in enumerate(['context' , 'question', 'answer_text']):
            self.data[text_column + '_tk'] = self.data[text_column].apply(lambda i: nltk_tokenizer.tokenize(i.replace('\n', '').strip(), escape=False))
        
            # token to index
            self.data[text_column+'_tk_index'] = self.tokenizer.texts_to_sequences(self.data[text_column + '_tk'].apply(lambda i: ' '.join(i)))
            
            # padding
            vectors.append(pad_sequences(self.data[text_column+'_tk_index'], max_sequence_len[i]))

        return vectors

if __name__ == "__main__":
    inputs = 'data/train-v1.1.json'
    pretrained_corpus = 'corpus.tk.txt'
    max_word_num = 100000
    max_sequence_len = [256, 32, 32]

    gen = DataGenerator(inputs, pretrained_corpus, max_word_num, max_sequence_len)
