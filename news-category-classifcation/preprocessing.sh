# tokenization-ko
python tokenization_en.py -input News_Category_Dataset_v2.json -column short_description -output corpus.tk.txt

# get word embedding vectors based on pre-trained fasttext model
# wiki.en.bin is a binary file containing the parameters of the model along with the dictionary and all hyperparameters.
# The binary file can be used later to compute word vectors or to restart the optimization.
cat corpus.tk.txt | /Users/hoyeonlee/fastText-0.1.0/fasttext print-word-vectors /Users/hoyeonlee/pretrained-word-vector/wiki.en.bin > corpus.tk.vec.txt

