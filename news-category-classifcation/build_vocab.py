import argparse
import pickle
from tokenization import Vocab, Tokenizer

TOKENIZER = ('treebank', 'mecab')

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--corpus', default=None, type=str, required=True)
    p.add_argument('--vocab', default=None, type=str, required=True)
    
    # Other parameters
    p.add_argument('--pretrained_vectors', default=None, type=str)
    p.add_argument('--is_sentence', action='store_true',
                   help='Whether the corpus is already split into sentences')
    p.add_argument('--tokenizer', default='treebank', type=str,
                   help='Tokenizer used for input corpus tokenization: ' + ', '.join(TOKENIZER))
    p.add_argument('--max_seq_length', default=1024, type=int,
                   help='The maximum total input sequence length after tokenization')
    p.add_argument('--unk_token', default='<unk>', type=str,
                   help='The representation for any unknown token')
    p.add_argument('--pad_token', default='<pad>', type=str,
                   help='The representation for the special token of padding token')
    p.add_argument('--bos_token', default='<bos>', type=str,
                   help='The representation for the special token of beginning-of-sequence token')
    p.add_argument('--eos_token', default='<eos>', type=str,
                   help='The representation for the special token of end-of-sequence token')
    p.add_argument('--min_freq', default=3, type=int,
                   help='The minimum frequency required for a token')
    p.add_argument('--lower', action='store_true',
                   help='Whether to convert the texts to lowercase')

    config = p.parse_args()
    return config

def load_pretrained(fname):
    """
    Load pre-trained FastText word vectors

    :param fname: text file containing the word vectors, one per line.
    """

    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split()) 
    print('Loading {} word vectors(dim={})...'.format(n, d))

    word2vec_dict = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word2vec_dict[tokens[0]] = list(map(float, tokens[1:]))
    print('#pretrained_word_vectors:', len(word2vec_dict))

    return word2vec_dict

if __name__=='__main__':
    config = argparser()
    print(config)
    
    # Select tokenizer
    config.tokenizer = config.tokenizer.lower()
    if config.tokenizer==TOKENIZER[0]:
        from nltk.tokenize import word_tokenize
        tokenization_fn = word_tokenize
    elif config.tokenizer ==TOKENIZER[1]:
        from konlpy.tag import Mecab
        tokenization_fn = Mecab().morphs
    
    tokenizer = Tokenizer(tokenization_fn=tokenization_fn,
                          is_sentence=config.is_sentence,
                          max_seq_length=config.max_seq_length)

    # Tokenization & read tokens
    list_of_tokens = []
    with open(config.corpus, 'r', encoding='-utf-8', errors='ignore') as reader:
        for li, line in enumerate(reader):
            text = ' '.join(line.split('\t')[1:]).strip()
            list_of_tokens += tokenizer.tokenize(text)

    # Build vocabulary
    vocab = Vocab(list_of_tokens=list_of_tokens,
                  unk_token=config.unk_token,
                  pad_token=config.pad_token,
                  bos_token=config.bos_token,
                  eos_token=config.eos_token,
                  min_freq=config.min_freq,
                  lower=config.lower)
    vocab.build()
    if config.pretrained_vectors:
        pretrained_vectors = load_pretrained(fname=config.pretrained_vectors)
        vocab.from_pretrained(pretrained_vectors=pretrained_vectors)
    print('Vocabulary size: ', len(vocab))

    # Save vocabulary
    with open(config.vocab, 'wb') as writer:
        pickle.dump(vocab, writer)
    print('Vocabulary saved to', config.vocab)