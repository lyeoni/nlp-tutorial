# Movie Rating Classification
This repository provides a simple Keras implementation of TextCNN for Text Classification. Here we use the movie review corpus written in Korean. The model trained on this dataset identify the sentiment based on review text. 
Keyword: TextCNN, Text classification, Sentiment analysis

## Data
32,775 corpus(review)-score pairs.
<p align="left">
<img width="700" src="https://github.com/lyeoni/nlp-tutorial/blob/master/movie-rating-classification/images/corpus_sample.png">
</p>

## Usage
### 1. Preprocessing corpus
Just run preprocessing.sh. It runs tokenization_ko.py, build_vocab.py, fasttext in order.
```
$ ./preprocessing.sh
```
```
structure:
  preprocessing.sh
  ├── tokenization_ko.py
  ├── build_vocab.py (optional)
  └── fasttext
```
You may also need to change the argument parameters in code.

### 2. Make data loader
It returns train/test-set as well as embedding_matrix.
```
$ python data_loader.py -h
usage: data_loader.py [-h] [-corpus_tk CORPUS_TK]
                      [-trained_word_vector TRAINED_WORD_VECTOR]
                      [-score_corpus SCORE_CORPUS]
optional arguments:
  -h, --help            show this help message and exit
  -corpus_tk CORPUS_TK  Default=corpus.tk.txt
  -trained_word_vector TRAINED_WORD_VECTOR
                        Default=corpus.tk.vec.txt
  -score_corpus SCORE_CORPUS
                        Default=score_corpus.txt
```
```
structure:
  data_loader.py
  └── read_word_pair.py
```
You may also need to change the argument parameters in code.

### 3. Training
To visualize overall training process, I also provide ipython notebook version in [here](https://github.com/lyeoni/nlp-tutorial/blob/master/movie-rating-classification/ipython_notebook/train.ipynb).
```
$ python train.py -h
usage: train.py [-h] [-corpus_tk CORPUS_TK]
                [-trained_word_vector TRAINED_WORD_VECTOR]
                [-score_corpus SCORE_CORPUS] [-epoch EPOCH]
                [-batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -corpus_tk CORPUS_TK  Default=corpus.tk.txt
  -trained_word_vector TRAINED_WORD_VECTOR
                        Default=corpus.tk.vec.txt
  -score_corpus SCORE_CORPUS
                        Default=score_corpus.txt
  -epoch EPOCH          number of iteration to train model. Default=20
  -batch_size BATCH_SIZE
                        mini batch size for parallel inference. Default=64
```
```
structure:
  train.py
  └── data_loader.py
```
You may also need to change the argument parameters in code.
