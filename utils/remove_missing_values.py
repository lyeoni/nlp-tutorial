import pandas as pd
import numpy as np
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

    # Show which index in a DataFrame has NA, and drop
    index = np.where(corpus.isna().any(axis='columns'))[0]
    corpus = corpus.drop(index, axis='rows')

    # save
    corpus.to_csv('new_'+config.data, encoding='utf-8', index=False)