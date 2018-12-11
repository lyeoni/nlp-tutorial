import argparse
import nltk

def argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('-input', required=True)
    p.add_argument('-output', required=True)
    p.add_argument('-word_num',
                   type=int,
                   required=True,
                   help='how many words to use. the words are sorted by decreasing frequency.')

    config = p.parse_args()

    return config

if __name__ == "__main__":
    config = argparser()
    
    f = open(config.input, 'r')

    vocabulary = []
    for line in f:
        if line.replace('\n', '').strip() != '':
            vocabulary += line.replace('\n', '').strip().split()

    vocabulary = nltk.Text(vocabulary)

    print('build_vocab.py: number of tokens = {}'.format(len(vocabulary.tokens)))
    print('build_vocab.py: number of unique tokens = {}'.format(len(set(vocabulary.tokens))))
    print('build_vocab.py: frequency of vocabulary(top 10)\n{}'.format(vocabulary.vocab().most_common(10)))

    f_out = open(config.output, 'w')
    for idx, words in enumerate(dict(vocabulary.vocab().most_common(config.word_num)).keys()):
        f_out.write(words + ' ' + str(idx) + '\n')
