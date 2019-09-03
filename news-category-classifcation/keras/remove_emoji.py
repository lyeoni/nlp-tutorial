import sys, re
import pandas as pd

def remove(corpus):
    # emoji regex
    p_emoji = re.compile('[\U00010000-\U0010ffff][\u20000000-\u2fffffff][\U0001f000-\U0001ffff]', flags=re.UNICODE)
    
    for i, sentence in enumerate(corpus):
        if p_emoji.match(sentence): # if there is emoji, remove
            corpus[i] = p_emoji.sub('', sentence)

    return corpus

if __name__ == "__main__":
    remove(sys.argv[1])
