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
Just run preprocessing.sh. It creates corpus.tk.txt, corpus.tk.vec.txt

To train your custom word embedding model instead of using the pre-trained model(wiki.en.bin), you need to use line 10 in preprocessing.sh rather than line 7.

example usage:
```
$ ./preprocessing.sh
```
```
structure:
  preprocessing.sh
  └── tokenization.py
      └── data_loader.py
  └── fasttext
```
## Reference
- [rajpurkar/SQuAD-explorer](https://github.com/rajpurkar/SQuAD-explorer)
