# Qusetion Answering system for SQUAD
This repo contains a simple source code for building question-answering system for SQuAD.

## Data
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.
You can download this dataset [here](https://rajpurkar.github.io/SQuAD-explorer/).

- `SQuAD 1.1`: the previous version of the SQuAD dataset, contains 100,000+ question-answer pairs on 500+ articles.
- `SQuAD 2.0`: combines the 100,000 questions in SQuAD 1.1 with over 50,000 new, unanswerable questions written adversarially by crowdworkers to look similar to answerable ones.

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
sample:
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

## Model Overview
### BERT: Bidirectional Encoder Representations from Transformers
**BERT** is a new language representation model, stands for Bidirectional Encoder Representations from Transformers.

BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create SOTA for 11 NLP tasks, including *Question Answering*.

<p align="center">
<img src="http://jalammar.github.io/images/bert-transfer-learning.png" />
</p>

#### Pre-training BERT
BERT is pre-trained by two unsupervised tasks, Masked Language Model and Next Sentence Prediction.

- Masked Language Model (Masked LM)
<p align="center">
<img src="http://jalammar.github.io/images/BERT-language-modeling-masked-lm.png" />
</p>

- Next Sentence Prediction (NSP)
<p align="center">
<img src="http://jalammar.github.io/images/bert-next-sentence-prediction.png" />
</p>


## Usage
### 1. Get data
```
$ download_squad.sh
```



## Reference
- [Google AI Language] [BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Jay Alammar] [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)
- [huggingface/pytorch-pretrained-BERT] [PyTorch Pretrained BERT: The Big & Extending Repository of pretrained Transformers](https://github.com/huggingface/pytorch-pretrained-BERT)
- [lyeoni/KorQuAD] [KorQuAD](https://github.com/lyeoni/KorQuAD)