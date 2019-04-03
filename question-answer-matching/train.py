import argparse
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import dataLoader as loader
import preprocessing as pproc
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--filename',
                    default='Posts.xml')
    
    p.add_argument('--clean_drop',
                    default=False,
                    help='Drop if either title or body column is NaN')
    
    p.add_argument('--epochs',
                    type=int,
                    default=7,
                    help='Number of epochs to train. Default=7')
    
    p.add_argument('--batch_size',
                    type=int,
                    default=2,
                    help='Mini batch size for gradient descent. Default=2')    
                    
    p.add_argument('--learning_rate',
                    type=float,
                    default=.001,
                    help='Learning rate. Default=.001')

    p.add_argument('--hidden_size',
                    type=int,
                    default=64,
                    help='Hidden size of LSTM. Default=64')

    p.add_argument('--n_layers',
                    type=int,
                    default=1,
                    help='Number of layers. Default=1')

    p.add_argument('--dropout_p',
                    type=float,
                    default=.1,
                    help='Dropout ratio. Default=.1')

    config = p.parse_args()

    return config

class QuestionAnswerDataset(Dataset):
    # If the data that comes out of the pytorch dataset is unpadded (if samples are of
    # different lengths), then pytorch dataloader returns a python list instead of 
    # pytorch tensor with samples truncated to minimum length of the sample in the batch.
    def __init__(self, input, tokenizer, maxlen=32, negative_sampling=True):
        self.input = input[input.posttypeid==1]
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.questions = np.array([self.indexesFromSentences(sentences) for sentences in self.input.title])
        self.answers = np.array([self.indexesFromSentences(sentences) for sentences in self.input.body])
        self.labels = np.ones(len(self.questions))
        if negative_sampling:
            self.n_questions, self.n_answers = self.negativeSampling(questions=self.questions,
                                                                     answers=self.answers)
            self.n_labels = np.zeros(len(self.n_questions))
            self.questions = np.concatenate((self.questions, self.n_questions))
            self.answers = np.concatenate((self.answers, self.n_answers))
            self.labels = np.concatenate((self.labels, self.n_labels))
        self.questions_len = np.array([np.count_nonzero(q) for q in self.questions])
        self.answers_len = np.array([np.count_nonzero(a) for a in self.answers])
        self.labels = self.labels.reshape(-1, 1)
        
    def __getitem__(self, idx):
        # |self.questions|, |self.answers| = (n_samples, maxlen)
        # |self.questions_len|, |self.answers_len| = (n_samples, 1)
        # |self.labels| = (n_samples, 1)
        return self.questions[idx], self.answers[idx], self.questions_len[idx], self.answers_len[idx], self.labels[idx]

    def __len__(self):
        return len(self.questions)
    
    def indexesFromSentences(self, sentences):
        indexes = []
        for sentence in sentences.splitlines():
            sentence = tokenizer.normalizeString(sentence)
            indexes += [self.tokenizer.word2index[word] for word in sentence.split(' ')]
            padded_indexes = self.padSequences(indexes) # padding
        
        return padded_indexes

    def padSequences(self, indexes):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(indexes) > self.maxlen:
            padded = indexes[:self.maxlen]
        else:
            padded[:len(indexes)] = indexes
        
        return padded
        
    def negativeSampling(self, questions, answers):
        indexes = list(range(len(questions)))

        random.shuffle(indexes)
        negative_questions = [questions[i] for i in indexes]
        
        random.shuffle(indexes)
        negative_answers = [answers[i] for i in indexes]

        return np.array(negative_questions), np.array(negative_answers)

def sort_by_len(sequences, sequence_length):
    sequence_length, si = sequence_length.sort(0, descending=True)
    return sequences[si], sequence_length

def train_model(epoch):
    model.train()
    losses, accs = 0, 0
    for i, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        qus, ans, qus_len, ans_len, labels = data
        qus, ans, labels = qus.to(device), ans.to(device), labels.to(device=device, dtype=torch.float32)
        # |qus|, |ans| = (batch_size, maxlen)
        # |qus_len|, |ans_len| = (batch_size)
        # |labels| = (batch_size, 1)
        
        # sort by sequence length in descending order
        qus, qus_len = sort_by_len(qus, qus_len)
        ans, ans_len = sort_by_len(ans, ans_len)
        
        # get loss
        output = model(qus, ans, qus_len, ans_len)
        loss = criterion(output, labels)
        losses += loss.item()
        # |output| = (batch_size, 1)
        # |loss| = (1)
        
        # get accuracy
        acc = (torch.round(output) == labels).sum().item()/len(qus)       
        accs += acc

        loss.backward()
        optimizer.step()
           
        # if i % 300 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
        #         epoch, i * len(qus), len(train_loader.dataset), 100. * i / len(train_loader), loss.item(), 100*acc))

    print('====> Train Epoch: {} Average loss: {:.4f}\tAverage accuracy: {:.2f}%'.format(
            epoch, losses / len(train_loader), 100*accs/len(train_loader)))        

def test_model():
    model.eval()
    losses, accs = 0, 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            qus, ans, qus_len, ans_len, labels = data
            qus, ans, labels = qus.to(device), ans.to(device), labels.to(device=device, dtype=torch.float32)

            # sort by sequence length in descending order
            qus, qus_len = sort_by_len(qus, qus_len)
            ans, ans_len = sort_by_len(ans, ans_len)
            
            # get loss
            output = model(qus, ans, qus_len, ans_len)
            loss = criterion(output, labels)
            losses += loss.item()

            # get accuracy
            acc = (torch.round(output) == labels).sum().item()/len(qus)       
            accs += acc
            
    print('====> Test Epoch: {} Average loss: {:.4f}\tAverage accuracy: {:.2f}%\n'.format(
            epoch, losses / len(test_loader), 100*accs/len(test_loader)))        

if __name__=='__main__':
    config = argparser()

    # data load
    data = loader.to_dataframe('data/'+config.filename)
    
    # preprocessing
    data, word_emb_matirx, tfidf_matrix, tokenizer = pproc.preprocessing(input = data,
                                                                         clean_drop = config.clean_drop)
    # |data| = (n_pairs, n_columns) = (91,517, 5)
    # |word_emb_matrix| = (tokenizer.n_words, 100)
    # |tfidf_matrix| = (tokenizer.n_words, 1)
    
    # build dataset & data loader
    train, test = train_test_split(data, test_size=0.1)    

    qa_train = QuestionAnswerDataset(train, tokenizer, negative_sampling=True)
    train_loader = DataLoader(dataset=qa_train, batch_size=config.batch_size, shuffle=True, num_workers=4)

    qa_test = QuestionAnswerDataset(test, tokenizer, negative_sampling=True)
    test_loader = DataLoader(dataset=qa_test, batch_size=len(qa_test), shuffle=False, num_workers=4)
    print('Total batches - train: {}, test: {}'.format(len(train_loader), len(test_loader)))
    
    # build model
    model = models.Model(input_size = tokenizer.n_words,
                         embedding_size = 100,
                         hidden_size = config.hidden_size,
                         n_layers = config.n_layers,
                         dropout_p = config.dropout_p,
                         word_embedding_matrix = word_emb_matirx,
                         tfidf_matrix = tfidf_matrix
                         ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    print(model)

    # train
    for epoch in range(1, config.epochs+1):
        train_model(epoch)
        test_model()
    
    # save model
    torch.save(model.state_dict(), 'model.pth')