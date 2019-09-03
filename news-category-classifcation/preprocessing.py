import re
import sys
import argparse

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--corpus', default=None, type=str, required=True)

    config = p.parse_args()
    return config

def cleaning(text):
    text = re.sub(r'[\U00010000-\U0010ffff][\u20000000-\u2fffffff][\U0001f000-\U0001ffff]', '', text) # Clean emoji
    text = re.sub(r'<.*?>', '', text) # Clean HTML tag
    text = re.sub(r'http\S+', '<url>', text) # url -> <url> token
    text = re.sub(r'[\w._-]+[@]\w+[.]\w+', '<email>', text) # email -> <email> token
    text = re.sub(r'\d+[-.]\d{3,4}[-.]\d{3,4}', '<pnum>', text) # phone number -> <pnum> token
    text = re.sub(r'[!]{2,}', '!', text) # multiple !s -> !
    text = re.sub(r'[!]{2,}', '?', text) # multiple ?s -> ?
    text = re.sub(r'[-=+,#:^$@*\"※~&%ㆍ』┘\\‘|\(\)\[\]\`\'…》]','', text) # Clean special symbols

    return text

if __name__=='__main__':
    config = argparser()
    
    with open(config.corpus, 'r', encoding='-utf-8', errors='ignore') as reader:
        for li, line in enumerate(reader):
            _line = line.split('\t')
            label, text = _line[0], ' '.join(_line[1:])
            
            # Cleaning
            text = cleaning(text)

            if len(text) > 0:
                line = '{}\t{}'.format(label, re.sub(r'\n', ' ', text))
                sys.stdout.write(line+'\n')