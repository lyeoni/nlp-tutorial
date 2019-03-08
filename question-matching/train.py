import argparse
from sklearn.model_selection import train_test_split
import torch
import dataLoader as loader
import preprocessing as pproc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--filename',
                    default = 'Posts.xml')
    
    config = p.parse_args()

    return config

# def train():

def tensorsFromSentences(sentences):
    indexes = []
    for sentence in sentences.splitlines():
        sentence = tokenizer.normalizeString(sentence)
        indexes += [tokenizer.word2index[word] for word in sentence.split(' ')]

    return  torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def dataGenerator(input, random_seed=0):
    # Here, 'posttypeid==1' means that both title(question) and body(answer) exist.
    # trimmed to 33,413 question-answer pairs
    input = input[input.posttypeid==1]

    questions = [tensorsFromSentences(sents) for sents in input.title]
    answers = [tensorsFromSentences(sents) for sents in input.body]
    pairs = list(zip(questions, answers))

    train, test = train_test_split(pairs, test_size=0.15, random_state=random_seed)
    # |train|, |test| = (n_pairs, 2, sequence_length)

if __name__=='__main__':
    config = argparser()

    data = loader.to_dataframe('data/'+config.filename).reset_index(drop=True)
    data, word_emb_matirx, tfidf_matrix, tokenizer = pproc.preprocessing(input = data,
                                                                         clean_drop = False)
    # |data| = (n_pairs, n_columns) = (91517, 5)
    # |word_emb_matrix| = (tokenizer.n_words, 100)
    # |tfidf_matrix| = (tokenizer.n_words, 1)
    print(data.shape)
    dataGenerator(data)