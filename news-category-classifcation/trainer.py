import argparse
import pickle

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tokenization import Vocab, Tokenizer
from dataset_utils import Corpus
from models import CBoWClassifier, LSTMClassifier

TOKENIZER = ('treebank', 'mecab')
MODEL = ('cbow', 'lstm')

def argparser():
    p = argparse.ArgumentParser()

    # Required parameters
    p.add_argument('--train_corpus', default=None, type=str, required=True)
    p.add_argument('--valid_corpus', default=None, type=str, required=True)
    p.add_argument('--vocab', default=None, type=str, required=True)
    p.add_argument('--model_type', default=None, type=str, required=True,
                   help='Model type selected in the list: ' + ', '.join(MODEL))

    # Input parameters
    p.add_argument('--is_sentence', action='store_true',
                   help='Whether the corpus is already split into sentences')
    p.add_argument('--tokenizer', default='treebank', type=str,
                   help='Tokenizer used for input corpus tokenization: ' + ', '.join(TOKENIZER))
    p.add_argument('--max_seq_length', default=64, type=int,
                   help='The maximum total input sequence length after tokenization')

    # Train parameters
    p.add_argument('--cuda', default=True, type=bool,
                   help='Whether CUDA is currently available')
    p.add_argument('--epochs', default=30, type=int,
                   help='Total number of training epochs to perform')
    p.add_argument('--batch_size', default=128, type=int,
                   help='Batch size for training')
    p.add_argument('--learning_rate', default=5e-3, type=float,
                   help='Initial learning rate')
    p.add_argument('--shuffle', default=True, type=bool, 
                   help='Whether to reshuffle at every epoch')
    
    # Model parameters
    p.add_argument('--embedding_trainable', action='store_true',
                   help='Whether to fine-tune embedding layer')
    p.add_argument('--embedding_size', default=100, type=int,
                   help='Word embedding vector dimension')
    p.add_argument('--hidden_size', default=128, type=int,
                   help='Hidden size')
    p.add_argument('--dropout_p', default=.5, type=float,
                   help='Dropout rate used for dropout layer')
    p.add_argument('--n_layers', default=2, type=int,
                   help='Number of layers in LSTM')

    config = p.parse_args()
    return config

def train():
    n_batches, n_samples = len(train_loader), len(train_loader.dataset)

    model.train()
    losses, accs = 0, 0
    for iter_, batch in enumerate(train_loader):
        inputs, targets = batch
        # |inputs|, |targets| = (batch_size, max_seq_length), (batch_size)
        
        preds = model(inputs)
        # |preds| = (batch_size, n_classes)

        loss = loss_fn(preds, targets)
        losses += loss.item()
        acc = (preds.argmax(dim=-1) == targets).sum()
        accs += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_ % (n_batches//5) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.4f} \tAccuracy: {:.3f}%'.format(
                    epoch, iter_, n_batches, 100.*iter_/n_batches, loss.item(), 100.*acc.item()/config.batch_size))

    print('====> Train Epoch: {} Average loss: {:.4f} \tAccuracy: {:.3f}%'.format(
            epoch, losses/n_batches, 100.*accs/n_samples))

def validate():
    n_batches, n_samples = len(valid_loader), len(valid_loader.dataset)

    model.eval()
    losses, accs = 0, 0
    with torch.no_grad():
        for iter_, batch in enumerate(valid_loader):
            inputs, targets = batch

            preds = model(inputs)

            loss = loss_fn(preds, targets)
            losses += loss.item()
            acc = (preds.argmax(dim=-1) == targets).sum()
            accs += acc.item()

    print('====> Validate Epoch: {} Average loss: {:.4f} \tAccuracy: {:.3f}%'.format(
            epoch, losses/n_batches, 100.*accs/n_samples))

if __name__=='__main__':
    config = argparser()
    print(config)

    # Load vocabulary
    with open(config.vocab, 'rb') as reader:
        vocab = pickle.load(reader)

    # Select tokenizer
    config.tokenizer = config.tokenizer.lower()
    if config.tokenizer==TOKENIZER[0]:
        from nltk.tokenize import word_tokenize
        tokenization_fn = word_tokenize
    elif config.tokenizer ==TOKENIZER[1]:
        from konlpy.tag import Mecab
        tokenization_fn = Mecab().morphs
        
    tokenizer = Tokenizer(tokenization_fn=tokenization_fn, vocab=vocab,
                          is_sentence=config.is_sentence, max_seq_length=config.max_seq_length)

    # Build dataloader
    train_dataset = Corpus(corpus_path=config.train_corpus, tokenizer=tokenizer, cuda=config.cuda)
    valid_dataset = Corpus(corpus_path=config.valid_corpus, tokenizer=tokenizer, cuda=config.cuda)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    
    # Build Model : CBoW, LSTM
    config.model_type = config.model_type.lower()
    if config.model_type==MODEL[0]:
        model = CBoWClassifier(input_size=len(vocab), # n_tokens
                               embedding_size =config.embedding_size,
                               hidden_size=config.hidden_size,
                               output_size=len(train_dataset.ltoi), # n_classes
                               dropout_p=config.dropout_p,
                               embedding_weights=vocab.embedding_weights,
                               embedding_trainable=config.embedding_trainable)
    elif config.model_type==MODEL[1]:
        model = LSTMClassifier(input_size=len(vocab), # n_tokens
                               embedding_size=config.embedding_size,
                               hidden_size=config.hidden_size,
                               output_size=len(train_dataset.ltoi), # n_classes
                               n_layers=config.n_layers,
                               dropout_p=config.dropout_p,
                               embedding_weights=vocab.embedding_weights,
                               embedding_trainable=config.embedding_trainable)
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    if config.cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
    print('=========MODEL=========\n',model)

    # Train & validate
    for epoch in range(1, config.epochs+1):
        train()
        validate()

    # Save model
    torch.save(model.state_dict(), '{}.pth'.format(config.model_type))