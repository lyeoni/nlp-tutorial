import pickle
import argparse
from collections import Counter

from prenlp.tokenizer import NLTKMosesTokenizer, SentencePiece

TOKENIZER_CLASSES = {
    'moses': NLTKMosesTokenizer,
    'sentencepiece': SentencePiece
}

class Vocab:
    def __init__(self, counter, specials = ['[]']):
        self.freqs = counter
        self.itos = []
                unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
        # self.vocab_size = vocab_size

        for token, freq in self.freqs.most_common():
            self.itos.append(token)
        self.stoi = {token:i for i, token in enumerate(self.itos)}
        print(len(self.itos), len(self.stoi))

    def __len__(self):
        return len(self.itos)

def build(args, corpus):
    if args.tokenizer == 'sentencepiece':
        tokenizer = SentencePiece()
        tokenizer.train(input=args.corpus,
                        model_prefix=args.prefix,
                        vocab_size=args.vocab_size,
                        model_type=args.model_type)

    else:
        # Load corpus
        with open(args.corpus, 'r', encoding='utf-8') as reader:
            corpus = [line.replace('\n', '') for line in reader.readlines()]
        corpus = list(filter(lambda x: len(x.strip()) > 0, corpus))
        
        # Build vocabulary
        counter = Counter()
        for text in corpus:
            tokens = tokenizer(text)
            for token in tokens:
                counter[token] += 1
        vocab = Vocab(counter=counter)

        # Save vocabulary
        with open(args.prefix+'.vocab', 'wb') as f:
            pickle.dump(vocab, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, type=str)
    parser.add_argument('--prefix', required=True, type=str)
    parser.add_argument('--tokenizer', default='sentencepiece', type=str)
    parser.add_argument('--vocab_size', default=16000, type=int)
    parser.add_argument('--model_type', default='bpe', type=str)
    # parser.add_argument('--max_sentence_length')
    args = parser.parse_args()

    build(args, corpus)