import sys
import numpy as np

def read_word_pair(input):
    t = {}
    f = open(input, 'r')
    
    for line in f:
        key_val = line.rstrip().rsplit(' ')
        
        if len(key_val[1:]) == 1: # for vocabulary read (key=word, value=index)
            t[key_val[0]] = int(key_val[1])
        else: # for embedding vector read (key=word, value=embedding vector)
            t[key_val[0]] = np.asarray(key_val[1:], dtype='float32')
    f.close()
    
    return t

if __name__ == "__main__":
    pass
