import sys, fileinput
from konlpy.tag import Mecab

if __name__ == "__main__":
    for line in fileinput.input():
        if line.replace('\n', '').strip() != '':
            # make tokenizer
            tokenizer = Mecab()
            
            # tokenization
            tokens = tokenizer.morphs(line.replace('\n', '').strip())
            sys.stdout.write(' '.join(tokens) + '\n')
        else:
            sys.stdout.write('\n')
