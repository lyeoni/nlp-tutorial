## Usage
### 1. Preprocessing corpus
Just run preprocessing.sh. It runs tokenization_ko.py, build_vocab.py, fasttext in order.
```
$ ./preprocessing.sh
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
You may also need to change the argument parameters in code.
