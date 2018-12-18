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


## Usage

## Reference
- [rajpurkar/SQuAD-explorer](https://github.com/rajpurkar/SQuAD-explorer)
