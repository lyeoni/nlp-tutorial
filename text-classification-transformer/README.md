# Text Classification with Transformer
This text classification tutorial trains a Transformer model on the IMDb movie review dataset for sentiment analysis.

## Setup input pipeline

### Building vocab based on WikiText-103 corpus
```
python vocab.py --corpus .data/wikitext-103/wiki.train.tokens --prefix wiki
```