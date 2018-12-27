# tokenization
python tokenization.py data/train-v1.1.json > corpus.tk.txt

# get word embedding vectors based on pre-trained fasttext model
# wiki.en.bin is a binary file containing the parameters of the model along with the dictionary and all hyperparameters.
# The binary file can be used later to compute word vectors or to restart the optimization.
cat corpus.tk.txt | /Users/hoyeonlee/fastText-0.1.0/fasttext print-word-vectors /Users/hoyeonlee/pretrained-word-vector/wiki.en.bin > corpus.tk.vec.txt
#
# train your custom embedding model instead of using the pre-trained model, wiki.en.bin above.
# /Users/hoyeonlee/fastText-0.1.0/fasttext skipgram -input corpus.tk.txt -output corpus.tk -dim 512 -epoch 100 -minCount 5

