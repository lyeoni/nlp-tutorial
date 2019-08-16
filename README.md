# NLP Tutorial
[![LICENSE](https://img.shields.io/github/license/lyeoni/nlp-tutorial?style=flat-square)](https://github.com/lyeoni/nlp-tutorial/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/lyeoni/nlp-tutorial?style=flat-square)](https://github.com/lyeoni/nlp-tutorial/issues)
[![GitHub stars](https://img.shields.io/github/stars/lyeoni/nlp-tutorial?style=flat-square&color=important)](https://github.com/lyeoni/nlp-tutorial/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lyeoni/nlp-tutorial?style=flat-square&color=blueviolet)](https://github.com/lyeoni/nlp-tutorial/network/members)

A list of NLP(Natural Language Processing) tutorials built on PyTorch and Keras.
<br><br>
<p align="center">
<img width="400" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"  align="middle">
<img width="320" src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" align="middle">
</p>

## Table of Contents
A step-by-step tutorial on how to implement and adapt to the simple real-word NLP task.

- [Neural Machine Translation](https://github.com/lyeoni/nlp-tutorial/tree/master/neural-machine-translation): This repo provides a simple PyTorch implementation of Neural Machine Translation, along with an intrinsic/extrinsic comparison of various sequence-to-sequence (seq2seq) models in translation.
<br>**_Keyword_:** sequence to seqeunce network(seq2seq), Attention, Auto-regressive, Teacher-forcing

- [Question-Answer Matching](https://github.com/lyeoni/nlp-tutorial/tree/master/question-answer-matching): This repo provides a simple PyTorch implementation of Question-Answer matching. Here we use the corpus from _Stack Exchange_ in English to build embeddings for entire questions. Using those embeddings, we find similar questions for a given question, and show the corresponding answers to those I found.
<br>**_Keyword_:** TF-IDF, LSTM with variable-length seqeucnes, CBoW, Text classification

- [News Category Classification](https://github.com/lyeoni/nlp-tutorial/tree/master/news-category-classifcation): This repo contains a simple source code for text-classification based on TextCNN. Corpus is _Huffpost_ news category dataset in English. Most open sources are a bit difficult to study & make text-classification model for beginners. So, I hope that this repo can be a good solution for people who want to have their own text-classification model.
<br>**_Keyword_:** TextCNN, Text classification, Text cateogrization

- [Movie Rating Classification (Korean NLP)](https://github.com/lyeoni/nlp-tutorial/tree/master/movie-rating-classification): This repo contains a simple source code for text-classification based on TextCNN. Corpus is _movie review_ dataset in the Korean language. Most open sources are a bit difficult to study & make text-classification model for beginners. So, I hope that this repo can be a good solution for people who want to have their own text-classification model.
<br>**_Keyword_:** TextCNN, Text classification, Sentiment analysis