# tokenization
python tokenization.py data/train-v1.1.json > corpus.tk.txt

# fasttext - using pre-trained model(wiki.en.bin)
# wiki.en.bin is a binary file containing the parameters of the model along with the dictionary and all hyperparameters.
# The binary file can be used later to compute word vectors or to restart the optimization.
#cat corpus.tk.txt | /Users/hoyeonlee/fastText-0.1.0/fasttext print-word-vectors /Users/hoyeonlee/pretrained-word-vector/wiki.en.bin > corpus.tk.vec.txt

# fasttext - train your custom embedding model
# /Users/hoyeonlee/fastText-0.1.0/fasttext skipgram -input corpus.tk.txt -output corpus.tk -dim 512 -epoch 100 -minCount 5
