import sys, argparse
import pandas as pd
from nltk.tokenize.moses import MosesTokenizer
import remove_emoji

def argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('-input',
                   required=True,
                   help='data file name to use')
        
    p.add_argument('-column',
                  required=True,
                  help='column name to use. headline or short_description')
    
    p.add_argument('-output',
                   required=True,
                   help='data file name to write')
                   
    config = p.parse_args()
                   
    return config

if __name__ == "__main__":
    config = argparser()
    
    corpus = pd.read_json(config.input, lines=True).loc[:,config.column]
    corpus = remove_emoji.remove(corpus)
    
    tokenizer = MosesTokenizer()

    sys.stdout = open(config.output, 'w')
    
    for line in corpus:
        if line.replace('\n', '').strip() != '':
            # tokenization
            tokens = tokenizer.tokenize(line.replace('\n', '').strip(), escape=False)
            sys.stdout.write(' '.join(tokens) + '\n')
        else:
            sys.stdout.write('\n')

