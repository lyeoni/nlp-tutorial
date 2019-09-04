# NLP Tutorial
[![LICENSE](https://img.shields.io/github/license/lyeoni/nlp-tutorial?style=flat-square)](https://github.com/lyeoni/nlp-tutorial/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/lyeoni/nlp-tutorial?style=flat-square)](https://github.com/lyeoni/nlp-tutorial/issues)
[![GitHub stars](https://img.shields.io/github/stars/lyeoni/nlp-tutorial?style=flat-square&color=important)](https://github.com/lyeoni/nlp-tutorial/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/lyeoni/nlp-tutorial?style=flat-square&color=blueviolet)](https://github.com/lyeoni/nlp-tutorial/network/members)

A list of NLP(Natural Language Processing) tutorials built on PyTorch.
<br><br>
<p align="center">
<img width="350" src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png"  align="middle">
</p>

## Table of Contents
A step-by-step tutorial on how to implement and adapt to the simple real-word NLP task.

- [News Category Classification](https://github.com/lyeoni/nlp-tutorial/tree/master/news-category-classifcation): This repo provides a simple PyTorch implementation of Text Classification, with simple annotation. Here we use _Huffpost_ news corpus including corresponding category. The classification model trained on this dataset identify the category of news article based on their headlines and descriptions.
<br>**_Keyword_:** CBoW, LSTM, fastText, Text cateogrization, Text classification<br>

- [Question-Answer Matching](https://github.com/lyeoni/nlp-tutorial/tree/master/question-answer-matching): This repo provides a simple PyTorch implementation of Question-Answer matching. Here we use the corpus from _Stack Exchange_ to build embeddings for entire questions. Using those embeddings, we find similar questions for a given question, and show the corresponding answers to those I found.
<br>**_Keyword_:** CBoW, TF-IDF, LSTM with variable-length seqeucnes, Text classification<br>

- [Neural Machine Translation](https://github.com/lyeoni/nlp-tutorial/tree/master/neural-machine-translation): This repo provides a simple PyTorch implementation of Neural Machine Translation, along with an intrinsic/extrinsic comparison of various sequence-to-sequence (seq2seq) models in translation.
<br>**_Keyword_:** sequence to seqeunce network(seq2seq), Attention, Autoregressive, Teacher-forcing<br>

- [Neural Language Model](https://github.com/lyeoni/pretraining-for-language-understanding): This repo provides a simple PyTorch implementation of Neural Language Model for natural language understanding.
Here we implement unidirectional/bidirectional language models, and pre-train language representations from unlabeled text (_Wikipedia_ corpus).
<br>**_Keyword_:** Autoregressive Language Model, Perplexity, Natural language understanding<br>

- [Movie Rating Classification (Korean NLP)](https://github.com/lyeoni/nlp-tutorial/tree/master/movie-rating-classification): This repo provides a simple Keras implementation of TextCNN for Text Classification.
Here we use the _movie review_ corpus written in Korean. The model trained on this dataset identify the sentiment based on review text.
<br>**_Keyword_:** TextCNN, Text classification, Sentiment analysis<br>
