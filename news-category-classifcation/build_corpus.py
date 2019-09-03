import re
import sys
import argparse
import pandas as pd

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--corpus', default=None, type=str, required=True)

    config = p.parse_args()
    return config

if __name__=='__main__':
    config = argparser()
    
    # Load data
    data = pd.read_json(config.corpus, lines=True)
    data = data.loc[:, ['category', 'headline', 'short_description']]
    
    # Extract news text(headline + short description) and corresponding category
    corpus = data['headline'].str.strip() + '. ' + data['short_description'].str.strip()
    labels = data['category'].str.strip()

    for i, (text, label) in enumerate(zip(corpus, labels)):
        line = '{}\t{}'.format(label, re.sub(r'\n', ' ', text))
        sys.stdout.write(line+'\n')