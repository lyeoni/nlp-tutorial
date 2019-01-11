import os
import re
import json
import argparse
import nltk
from tqdm import tqdm
from collections import Counter
from utils.utils import get_word_span, get_word_idx, process_tokens

def main():
    config = argparser()
    load(config)

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('-mode', type=str,
                   help='select train or dev')
    
    p.add_argument('-glove_vec_size', default=100, type=int,
                   help='embedding vector size')
        
    return p.parse_args()
       
def get_word2vec(config, word_counter):
    glove_path = os.path.join(os.getcwd(), 'glove', 'glove.6B.{}d.txt'.format(config.glove_vec_size))

    total = int(4e5)
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
            elif word.capitalize() in word_counter:
                word2vec_dict[word.capitalize()] = vector
            elif word.lower() in word_counter:
                word2vec_dict[word.lower()] = vector
            elif word.upper() in word_counter:
                word2vec_dict[word.upper()] = vector

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
    return word2vec_dict


def save(config, data, shared):
    data_path = os.path.join(os.getcwd(), "data_{}.json".format(config.mode))
    shared_path = os.path.join(os.getcwd(), "shared_{}.json".format(config.mode))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))

def load(config):
    # load train/dev json file
    # data: dict, dict_keys = ['version', 'data']
    data_path = os.path.join(os.getcwd(), 'squad', '{}-v1.1.json'.format(config.mode))
    data = json.load(open(data_path, 'r'))['data']
    
    # create tokenizer
    s_tokenizer = nltk.sent_tokenize
    s_tokenizer = lambda para: [para]
    def word_tokenize(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    
    '''
    structure:
        data > title
             > paragraphs > context
                          > qas    > question
                                   > answers    > text
                                                > answer_start
    article: dict, dict_keys['title', 'paragraphs']
    paragraph = dict, dict_keys['context', 'qas']
    qas = dict, dict_keys['question', 'answers']
    answers = dict, dict_keys['text', 'answer_start']
    '''
    q, cq, y, cy, rx, rcx, ids, idxs = [], [], [], [], [], [], [], []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    for a_i, article in enumerate(tqdm(data)):
        pp, xp, cxp = [], [], [] # context, word-tokenized context, character-tokenized context
        p.append(pp)
        x.append(xp)
        cx.append(cxp)
        for p_i, paragraph in enumerate(article['paragraphs']):
            # context = Architecturally, the school has a Catholic character....
            context = paragraph['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            
            # s_tokenizer(context) = ['Architecturally, the school has a Catholic character.',...]
            # list(map(word_tokenize),s_tokenizer..) = [['Architecturally', ',', 'the', 'school', 'has', 'a', 'Catholic', 'character', '.'], ...]]
            tk_context = list(map(word_tokenize, s_tokenizer(context))) # sentence + word tokenize
            tk_context = [process_tokens(tokens) for tokens in tk_context] # denoising
            # ctk_context = [[['A', 'r', 'c', 'h', 'i', 't', 'e', 'c', 't', 'u', 'r', 'a', 'l', 'l', 'y'], ...]]]
            ctk_context =  [[list(tk) for tk in tk_s] for tk_s in tk_context]
            
            pp.append(context)
            xp.append(tk_context)
            cxp.append(ctk_context)
            
            # word_counter = Counter({'Architecturally': 5, 'character': 5, '.': 5, 'school': 5, 'a': 5, ',': 5, 'has': 5, 'Catholic': 5, 'the': 5})
            # lower_word_counter = Counter({'school': 5, 'character': 5, 'catholic': 5, 'architecturally': 5, 'a': 5, ',': 5, 'has': 5, '.': 5, 'the': 5})
            # char_counter = Counter({'c': 30, 'a': 30, 'h': 30, 't': 25, 'l': 20, 'r': 20, 'o': 15, 'e': 15, 'i': 10, 's': 10, '.': 5, 'C': 5, 'A': 5, 'y': 5, ',': 5, 'u': 5})
            for tk_s in tk_context:
                for tk in tk_s:
                    word_counter[tk] += len(paragraph['qas'])
                    lower_word_counter[tk.lower()] += len(paragraph['qas'])
                    for c in tk:
                        char_counter[c] += len(paragraph['qas'])
            
            rxi = [a_i, p_i]
            
            # original = To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
            # tk_question = ['To', 'whom', 'did', 'the', 'Virgin', 'Mary', 'allegedly', 'appear', 'in', '1858', 'in', 'Lourdes', 'France', '?']
            # ctk_question [['T', 'o'], ['w', 'h', 'o', 'm'], ['d', 'i', 'd'], ['t', 'h', 'e'], ['V', 'i', 'r', 'g', 'i', 'n'], ['M', 'a', 'r', 'y'], ['a', 'l', 'l', 'e', 'g', 'e', 'd', 'l', 'y'], ['a', 'p', 'p', 'e', 'a', 'r'], ['i', 'n'], ['1', '8', '5', '8'], ['i', 'n'], ['L', 'o', 'u', 'r', 'd', 'e', 's'], ['F', 'r', 'a', 'n', 'c', 'e'], ['?']]
            for q_a in paragraph['qas']:
                tk_question = word_tokenize(q_a['question'])
                ctk_question = [list(tk) for tk in tk_question]
                
                # answer start/end span location, first/last index(answer word base) of answer start/end word(span), answer_text
                yi, cyi, answers = [], [], [] 
                
                # answer_text = Saint Bernadette Soubirous = context[answer_start:answer_stop]
                # answer_start, answet_stop = 515, 541
                for answer in q_a['answers']:
                    answer_text = answer['text']
                    answers.append(answer_text)
                    answer_start = answer['answer_start']
                    answer_stop = answer_start + len(answer_text)
                    
                    # get answer start/end span location
                    # yi0, yi1 = (0, 108) (0, 111) = 108th token, 111th token in context
                    yi0, yi1 = get_word_span(context, tk_context, answer_start, answer_stop)
                    
                    # get answer start/end word(span)
                    # w0, w1 = Saint, Soubirous
                    w0 = tk_context[yi0[0]][yi0[1]] 
                    w1 = tk_context[yi1[0]][yi1[1]-1]
                    
                    # get first index(context base) of answer start/end word(span)
                    # i0, i1 = 515, 532
                    i0 = get_word_idx(context, tk_context, yi0)
                    i1 = get_word_idx(context, tk_context, (yi1[0], yi1[1]-1))
                    
                    # get first/last index(answer word base) of answer start/end word(span)
                    # cyi0 = 515-515, cyi1 = 541-532-1
                    cyi0 = answer_start - i0
                    cyi1 = answer_stop - i1 - 1
                    
                    yi.append([yi0, yi1])
                    cyi.append([cyi0, cyi1])
                    break
                for tk in tk_question:
                    word_counter[tk] += 1
                    lower_word_counter[tk.lower()] += 1
                    for c in tk:
                        char_counter[c] += 1
                    
                q.append(tk_question)
                cq.append(ctk_question)
                y.append(yi)
                cy.append(cyi)
                rx.append(rxi)
                rcx.append(rxi)
                ids.append(q_a['id'])
                idxs.append(len(idxs))
                answerss.append(answers)
                
    word2vec_dict = get_word2vec(config, word_counter)
    lower_word2vec_dict = get_word2vec(config, lower_word_counter)   
    
    # add context here
    
    data = {'q': q, 'cq': cq, 'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p, 'word_counter': word_counter,
              'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
              'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
    
    print("saving...")
    save(config, data, shared)

if __name__ == "__main__":
    main()
