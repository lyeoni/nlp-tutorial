import pandas as pd
import konlpy
from konlpy.tag import Mecab
import argparse

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--data',
                   required=True,
                   help='recommended file formats= .txt, .csv')

    config = p.parse_args()

    return config

if __name__ == "__main__":
    config = argparser()

    corpus = pd.read_csv(config.data, error_bad_lines=False, warn_bad_lines=False)

    # tokenizing
    tokenizer = Mecab()
    corpus['tokenization'] = [' '.join(tokenizer.morphs(i)) for i in corpus]

    # save
    corpus.to_csv('new_'+config.data, encoding='utf-8', index=False)
