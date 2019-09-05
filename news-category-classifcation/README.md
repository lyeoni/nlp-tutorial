# News Category Classification

PyTorch implementation of Text Classification, with simple annotation.

Here we use [HuffPost news corpus](https://www.kaggle.com/rmisra/news-category-dataset) including corresponding category. The classification model trained on this dataset identify the category of news article based on their headlines and descriptions.

## Dataset

The dataset contains 200k records. Each json record contains following attributes:

- category: Category article belongs to
- headline: Headline of the article
- authors: Person authored the article
- link: Link to the post
- short_description: Short description of the article
- date: Date the article was published

<p align="center">
<img width="600" src="https://github.com/lyeoni/nlp-tutorial/blob/master/news-category-classifcation/data/images/sample.png">
</p>

## Model Overview

### CBoW (Continuous Bag of Words)
CBoW is extremely simple, and effective in text classification. For instance, if there are many positive words, the review is likely positive.
CBoW can be defined as follows:
- Ignore the order of the tokens.
- Simply average the token vectors.
  - Averaging is a differentiable operator.
  - Just one operator node in the DAG(Directed Acyclic Graph).
- Generalizable to bag-of-n-grams.
  - N-gram: a phrase of N tokens.

<p align="center"><img width= 700 src="https://github.com/lyeoni/nlp-tutorial/blob/master/news-category-classifcation/data/images/cbow.png"></p>

### LSTM (Long-Short Term Memory Networks)
A Recurrent Neural Network, or RNN, is a network that operates on a sequence and uses its own output as input for subsequent steps. For example, its output could be used as part of the next input, so that information can propogate along as the network passes over the sequence.

In the case of an LSTM, for each element in the sequence, there is a corresponding hidden state, which in principle can contain information from arbitrary points earlier in the sequence. We can use all the hidden states to predict categories.

Here we use bidirectional LSTM for the task of text classification.
Model architecture can be defined as follows:

<p align="center"><img width= 600 src="https://github.com/lyeoni/nlp-tutorial/blob/master/news-category-classifcation/data/images/lstm.png"></p>

## Usage

### 1. Build Corpus
#### Download news category dataset from Kaggle
News Category Dataset contains around 200k news headlines/short descriptions from the year 2012 to 2018 obatined from HuffPost. Download this dataset from [Kaggle](https://www.kaggle.com/rmisra/news-category-dataset), unzip into the `corpus` directory.
```
$ mkdir corpus
$ unzip news-category-dataset.zip -d corpus
```

#### Build corpus
`corpus.txt` contains 200,853 news headline/short description with corresponding category.
```
$ python build_corpus.py --corpus corpus/News_Category_Dataset_v2.json > corpus/corpus.txt
$ wc -l corpus/corpus.txt
200853 corpus/corpus.txt
$ head -n 2 corpus/corpus.txt
CRIME   There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV. She left her husband. He killed their children. Just another day in America.
ENTERTAINMENT   Will Smith Joins Diplo And Nicky Jam For The 2018 World Cup's Official Song. Of course it has a song.
```

<br>

### 2. Preprocessing

#### Cleaning: remove noise
Clean HTML tag/speical symbols(e.g. <, >, ', ", ...), and replace url/email with \<url>, \<email> token.
```
$ python preprocessing.py --corpus corpus/corpus.txt > corpus/corpus.clean.txt
$ head -n 2 corpus/corpus.clean.txt
CRIME   There Were 2 Mass Shootings In Texas Last Week But Only 1 On TV. She left her husband. He killed their children. Just another day in America.
ENTERTAINMENT   Will Smith Joins Diplo And Nicky Jam For The 2018 World Cups Official Song. Of course it has a song.
```

#### Shuffle & Split corpus into train and valid subsets.
Now, we need to split this clean corpus to train-set, valid-set. We have 200,853 samples in corpus. Here it is divided by a ratio of 8:2.

```
$ cat corpus/corpus.clean.txt | shuf > corpus/corpus.shuf.txt
$ head -n 160682 corpus/corpus.shuf.txt > corpus/corpus.train.txt
$ tail -n 40171 corpus/corpus.shuf.txt > corpus/corpus.valid.txt
$ wc -l corpus/corpus.train.txt corpus/corpus.valid.txt
  160682 corpus/corpus.train.txt
   40171 corpus/corpus.valid.txt
```

#### Build Vocabulary
```
$ python build_vocab.py --corpus corpus/corpus.train.txt --vocab vocab.train.pkl --lower
Namespace(bos_token='<bos>', corpus='corpus/corpus.train.txt', eos_token='<eos>', is_sentence=False, lower=True, max_seq_length=1024, min_freq=3, pad_token='<pad>', pretrained_vectors=None, tokenizer='treebank', unk_token='<unk>', vocab='vocab.train.pkl')
Vocabulary size:  42765
Vocabulary saved to vocab.train.pkl
```

#### Optional: Build vocabulary with fastText pre-trained word representations

In order to build fastText, use the following. This will produce object files for all the classes as well as the main binary fasttext.
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

fastText provides two models for computing word representations: skipgram and cbow.
To train a skipgram/cbow model with fastText, run the following command:
```
$ cd ..
$ awk -F '\t' '{print tolower($2)}' corpus/corpus.train.txt > corpus/corpus.train.text.txt
$ fastText/fasttext skipgram -input corpus/corpus.train.text.txt -output skipgram
$ fastText/fasttext cbow -input corpus/corpus.train.text.txt -output cbow
```

Build vocabulary with fastText word representations by adding `--pretrained_vectors` flag.
```
$ python build_vocab.py --corpus corpus/corpus.train.txt --vocab skipgram.vocab.train.pkl --pretrained_vectors skipgram.vec --lower
Namespace(bos_token='<bos>', corpus='corpus/corpus.train.txt', eos_token='<eos>', is_sentence=False, lower=True, max_seq_length=1024, min_freq=3, pad_token='<pad>', pretrained_vectors='skipgram.vec', tokenizer='treebank', unk_token='<unk>', vocab='skipgram.vocab.train.pkl')
Loading 39452 word vectors(dim=100)...
#pretrained_word_vectors: 39452
Vocabulary size:  42765
Vocabulary saved to skipgram.vocab.train.pkl
```

<br>

### 3. Training
```
$ python trainer.py  -h
usage: trainer.py [-h] --train_corpus TRAIN_CORPUS --valid_corpus VALID_CORPUS
                  --vocab VOCAB --model_type MODEL_TYPE [--is_sentence]
                  [--tokenizer TOKENIZER] [--max_seq_length MAX_SEQ_LENGTH]
                  [--cuda CUDA] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                  [--learning_rate LEARNING_RATE] [--shuffle SHUFFLE]
                  [--embedding_trainable] [--embedding_size EMBEDDING_SIZE]
                  [--hidden_size HIDDEN_SIZE] [--dropout_p DROPOUT_P]
                  [--n_layers N_LAYERS]

optional arguments:
  -h, --help            show this help message and exit
  --train_corpus TRAIN_CORPUS
  --valid_corpus VALID_CORPUS
  --vocab VOCAB
  --model_type MODEL_TYPE
                        Model type selected in the list: cbow, lstm
  --is_sentence         Whether the corpus is already split into sentences
  --tokenizer TOKENIZER
                        Tokenizer used for input corpus tokenization:
                        treebank, mecab
  --max_seq_length MAX_SEQ_LENGTH
                        The maximum total input sequence length after
                        tokenization
  --cuda CUDA           Whether CUDA is currently available
  --epochs EPOCHS       Total number of training epochs to perform
  --batch_size BATCH_SIZE
                        Batch size for training
  --learning_rate LEARNING_RATE
                        Initial learning rate
  --shuffle SHUFFLE     Whether to reshuffle at every epoch
  --embedding_trainable
                        Whether to fine-tune embedding layer
  --embedding_size EMBEDDING_SIZE
                        Word embedding vector dimension
  --hidden_size HIDDEN_SIZE
                        Hidden size
  --dropout_p DROPOUT_P
                        Dropout rate used for dropout layer
  --n_layers N_LAYERS   Number of layers in LSTM
```

example:
```
$ python trainer.py --train_corpus corpus/corpus.train.txt --valid_corpus corpus/corpus.valid.txt --vocab vocab.train.pkl --model_type cbow --epochs 30 --learning_rate 5e-3 --embedding_trainable
```

## Evaluation

### Models

Model architecture snapshots are like as below. You may increase the performance with hyper-parameter optimization. (cf. Among Kaggle kernels for this dataset, 65% accuracy is the highest.)

#### CBoW
```
CBoWClassifier(
  (embedding): Embedding(42765, 100)
  (fc): Linear(in_features=100, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.5)
  (fc2): Linear(in_features=128, out_features=41, bias=True)
  (softmax): LogSoftmax()
)
```
#### LSTM
```
LSTMClassifier(
  (embedding): Embedding(42765, 100)
  (lstm): LSTM(100, 128, num_layers=2, batch_first=True, dropout=0.5, bidirectional=True)
  (fc): Linear(in_features=256, out_features=41, bias=True)
  (softmax): LogSoftmax()
)
```

### Results
The models were trained with NVIDIA Tesla K80, and the number of epochs was 30. The following table shows the evaluation results for the validation set.

|Model|Word representation|Loss|Accuracy|
|-|-|:-:|:-:|
|CBoW|-|1.904|48.96%|
||fastText - _cbow (freeze embedding layer)_|1.836|49.22%|
||fastText - _cbow (fine-tune all)_|1.547|57.00%|
||fastText - _skipgram (freeze embedding layer)_|1.733|52.02%|
||fastText - _skipgram (fine-tune all)_|1.499|58.99%|
||||
|LSTM|-|1.665|56.17%|
||fastText - _cbow (freeze embedding layer)_|1.378|61.43%|
||fastText - _cbow (fine-tune all)_|1.412|63.31%|
||fastText - _skipgram (freeze embedding layer)_|1.243|63.77%|
||fastText - _skipgram (fine-tune all)_|1.290|64.56%|
