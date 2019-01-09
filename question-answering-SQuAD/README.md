# Qusetion Answering system for SQUAD
This repo contains a simple source code for building question-answering system for SQuAD.

## Data
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
You can download this dataset [here](https://rajpurkar.github.io/SQuAD-explorer/).

- `SQuAD 2.0`: combines the 100,000 questions in SQuAD 1.1 with over 50,000 new, unanswerable questions written adversarially by crowdworkers to look similar to answerable ones.
- `SQuAD 1.1`: the previous version of the SQuAD dataset, contains 100,000+ question-answer pairs on 500+ articles.

Here I used SQuAD 1.1. Each article contains following structure:
```
structure:
  article (dict)
  ├── title (str)
  ├── paragraphs (dict)
      ├── context (str)
      └── qas (dict)        
          ├── question (str)
          ├── id (str)
          └── answers (dict)
              ├── answer_start (integer)
              └── text (str)
  ...
  └── paragraphs n
      ├── context
      └── qas 
```
example:
```
'title': 'University_of_Notre_Dame'
'paragraphs': [{
  'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
  'qas': [{
    'answers': [{
        'answer_start': 515,
        'text': 'Saint Bernadette Soubirous'
        }],
    'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
    'id': '5733be284776f41900661182'
    },
    {
    'answers': [{
        'answer_start': 188,
        'text': 'a copper statue of Christ'
    }],
    'question': 'What is in front of the Notre Dame Main Building?',
    'id': '5733be284776f4190066117f'
    }
  ]},
  {'context': ...., 'qas': ....},
  {'context': ...., 'qas': ....}]

'title': ...
'paragraphs': {...}
```

## Usage
### 1. Preprocessing corpus
First, just run preprocessing.sh. It creates tokenized corpus 'corpus.tk.txt', and it will be used as the input to data_generator.py later (especially, to create vocabulary).

You can adopt the pre-trained word embedding model in the below of your choice. I tested both Glove and Fasttext, and used Glove embedding here.
- `Glove(glove.6B.100d.txt)`:  trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download). 
- `Fasttext(wiki.en.bin or wiki.en.vec)`: trained on Wikipedia (2519370 vocab, 300d vectors).
 
example usage:
```
$ ./preprocessing.sh
```
```
structure:
  preprocessing.sh
  └── tokenization.py
      └── data_loader.py
  └── fasttext (optional)
```

Second, when you create DataGenerator instance in data_generator.py,
it will create vocabulary, context-question vectors, and embedding matrix used for MRC model training in order.

example usage:
```
>>> gen = DataGenerator(inputs = 'data/train-v1.1.json',
                             tokenized_corpus = 'corpus.tk.txt',
                             embedding_vectors = '/Users/hoyeonlee/glove.6B/glove.6B.100d.txt',
                             embedding_dim = 100,
                             max_word_num = 100000,
                             max_sequence_len = [300, 30] # [context, question])
```

### 2. Training

## Reference
### Word embeddings
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [facebookresearch/fastText](https://github.com/facebookresearch/fastText)
### MRC models
- [Rahulrt7/Machine-comprehension-Keras](https://github.com/Rahulrt7/Machine-comprehension-Keras)
- [rajpurkar/SQuAD-explorer](https://github.com/rajpurkar/SQuAD-explorer)
