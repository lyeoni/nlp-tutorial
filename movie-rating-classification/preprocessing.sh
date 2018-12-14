# tokenization-ko
python tokenization_ko.py < corpus.txt > corpus.tk.txt

# optional: build vocabulary
python build_vocab.py -input corpus.tk.txt -output corpus.tk.vocab.txt -word_num 12000

# get word embedding vectos based on pre-trained fasttext model
# wiki.ko.bin is a binary file containing the parameters of the model along with the dictionary and all hyperparameters.
# The binary file can be used later to compute word vectors or to restart the optimization.
cat corpus.tk.txt | /Users/hoyeonlee/fastText-0.1.0/fasttext print-word-vectors /Users/hoyeonlee/pretrained-word-vector/wiki.ko.bin > corpus.tk.vec.txt

