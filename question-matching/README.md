# Question Matching
This repository provides a simple PyTorch implementation of Question matching. Here we use the corpus from Stack Exchange in English to build embeddings for entire questions. Using those embeddings, we find similar questions for a given question, and show the corresponding answers to those I found.

## Model Overview

### TF-IDF (Term-Frequency - Inverse Documnet Frequency)
The _TF-IDF_ is usually used to find how relevant a term is in a document, and the _TF-IDF_ value is the product of two statistics, _Term-Frequency (TF) and Inverse Documnet Frequency (IDF)._ 
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?tfidf(t,&space;d,&space;D)&space;=tf(t,d)&space;\times&space;idf(t,&space;D)" title="tfidf(t, d, D) =tf(t,d) \times idf(t, D)" />
</p>

- `TF`: _Term Frequency_, which is a value that indicates how frequently a particular term occurs in a document. And the higher it is, the more relevant it is in the document. In other words, if a term occurs more times than other terms in a document, the term has more relevance than other terms for the document.
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?tf(t,&space;d)&space;=&space;\frac{Number\;&space;of\;&space;times\;&space;term\;&space;t\;&space;appears\;&space;in\;&space;a\;&space;document\;}{Total\;&space;number\;&space;of\;&space;terms\;&space;in\;&space;the\;&space;document}" title="tf(t, d) = \frac{Number\; of\; times\; term\; t\; appears\; in\; a\; document\;}{Total\; number\; of\; terms\; in\; the\; document}" />
</p>

- `IDF`: _Inverse Document Frequency_, which is the inverse of _Document Frequency (DF)_. _DF_ measures how much information a term provides, i.e., if it's common or rare across all documents. 
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?idf(t,&space;D)&space;=&space;\log_{&space;}\frac{Total\;&space;number\;&space;of\;&space;documents\;&space;D}{Number\;&space;of\;&space;documents\;&space;with\;&space;term\;&space;t\;&space;in\;&space;it}" title="idf(t, D) = \log_{ }\frac{Total\; number\; of\; documents\; D}{Number\; of\; documents\; with\; term\; t\; in\; it}" />
</p>

**The higher the term frequency within a particular document, and the smaller the document containing the term in the whole document, the higher the TF-IDF value.** Therefore, this value provides the effect of filtering out common words from all documents. The more terms a document contains, the closer the value of the log function is to 1, in which case the IDF value and TF-IDF value are closer to 0.

#### Example
Consider that there are two documents as follows:
- document1 = "a new car, used car, car review"
- documnet2 = "a friend in need is a friend indeed"

|word|TF|TF|IDF|TF-IDF|TF-IDF|
|:-:|:-:|:-:|:-:|:-:|:-:|
||document1|document2||document1|document2|
|a|1/7|2/8|log(2/2) = 0|0|0|
|new|1/7|0|log(2/1) = 0.3|0.04|0|
|car|3/7|0|log(2/1) = 0.3|0.13|0|
|used|1/7|0|log(2/1) = 0.3|0.04|0|
|review|1/7|0|log(2/1) = 0.3|0.04|0|
|friend|0|2/8|log(2/1) = 0.3|0|0.08|
|in|0|1/8|log(2/1) = 0.3|0|0.04|
|need|0|1/8|log(2/1) = 0.3|0|0.04|
|is|0|1/8|log(2/1) = 0.3|0|0.04|
|indeed|0|1/8|log(2/1) = 0.3|0|0.04|

## References
- [[DOsinga/deep_learning_cookbook](https://github.com/DOsinga/deep_learning_cookbook/blob/master/06.1%20Question%20matching.ipynb)] 06.1 Question matching
- [[Minsuk Heo](https://www.youtube.com/watch?v=meEchvkdB1U)] [딥러닝 자연어처리] TF-IDF

