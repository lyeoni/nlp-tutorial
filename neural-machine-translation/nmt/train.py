import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import argparse
import time
import math
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
import dataLoader as loader
import seq2seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--n_iters',
                required=True,
                type=int,
                help='Number of iterations to train')

    p.add_argument('--embedding_size',
                    type=int,
                    default=300,
                    help='Word embedding vector dimension size. Default=300')

    p.add_argument('--hidden_size',
                    type=int,
                    default=600,
                    help='Hidden size of LSTM. Default=600')
    
    p.add_argument('--teacher_forcing_ratio',
                    type=float,
                    default=1,
                    help='Teacher forcing ratio. Default=1')
    
    p.add_argument('--n_layers',
                    type=int,
                    default=2,
                    help='Number of layers. Default=2')
    
    p.add_argument('--dropout_p',
                    type=float,
                    default=.1,
                    help='Dropout ratio. Default=.1')

    config = p.parse_args()
    return config

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(loader.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def merge_encoder_hiddens(encoder_hiddens):
    new_hiddens, new_cells = [], []
    hiddens, cells = encoder_hiddens
    # |hiddens|, |cells| = (num_layers*num_directions, 1, hidden_size/2)

    # i-th and (i+1)-th layer is opposite direction.
    # Also, each direction of layer is half hidden size.
    # Therefore, we concatenate both directions to 1 hidden size layer.
    for i in range(0, hiddens.size(0), 2):
        new_hiddens += [torch.cat([hiddens[i], hiddens[i+1]], dim=-1)]
        new_cells += [torch.cat([cells[i], cells[i+1]], dim=-1)]
    
    new_hiddens, new_cells = torch.stack(new_hiddens), torch.stack(new_cells)
    # |new_hiddens|, |new_cells| = (num_layers, 1, hidden_size)
    return (new_hiddens, new_cells)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion, max_length=loader.MAX_LENGTH):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length, target_length = input_tensor.size(0), target_tensor.size(0)
    # |input_length|, |target_length| = (sentence_length)

    encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
    # |encoder_hidden[0]|, |encoder_hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)
    # |encoder_outputs| = (max_length, hidden_size)

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # |encoder_output| = (batch_size, sequence_length, num_directions*(hidden_size/2))
        # |encoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size/2)
        # 2: respectively, hidden state and cell state.
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[loader.SOS_token]]).to(device)
    # |decoder_input| = (1, 1)
    decoder_hidden = merge_encoder_hiddens(encoder_hidden)
    # |decoder_hidden|= (2, num_layers*num_directions, batch_size, hidden_size)
    # 2: respectively, hidden state and cell state
    # Here, the lstm layer in decoder is uni-directional.

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: feed the target as the next input.
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                         decoder_hidden, encoder_outputs)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (2, num_layers, batch_size, hidden_size)
            # 2: respectively, hidden state and cell state.
            # Here, the lstm layer in decoder is uni-directional.
            # |decoder_attention| = (sequence_length, max_length)
            
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di] # teacher forcing
            # |decoder_input|, |target_tensor[di]] = (1)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                         decoder_hidden, encoder_outputs)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (2, num_layers, batch_size, hidden_size)
            # 2: respectively, hidden state and cell state.
            # Here, the lstm layer in decoder is uni-directional.
            # |decoder_attention| = (sequence_length, max_length)
            
            topv, topi = decoder_output.topk(1) # top-1 value, index
            # |topv|, |topi| = (1, 1)

            decoder_input = topi.squeeze().detach() # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            # |target_tensor[di]| = (1)
            if decoder_input.item() == loader.EOS_token:
                # |decoder_input| = (1)
                break
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()/target_length

def trainiters(pairs, encoder, decoder, n_iters,
                train_pairs_seed=0, print_every=1000, plot_every=1000, learning_rate=.01):
    start = time.time()
    plot_losses = []
    print_loss_total, plot_loss_total = 0, 0

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    train_pairs *= n_iters//len(train_pairs)
    train_pairs += [random.choice(train_pairs) for i in range(n_iters%len(train_pairs))]
    train_pairs = [tensorsFromPair(pair) for pair in train_pairs]
    # |train_pairs| = (n_iters, 2, sentence_length, 1) # eng, fra

    encoder_optimizer = optim.SGD(encoder.parameters(), lr= learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr= learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        pair = train_pairs[iter-1]
        # |pair| = (2) # eng, fra
        input_tensor, target_tensor = pair[0], pair[1]
        # |input_tensor|, |target_tensor| = (sentence_length, 1)

        loss = train(input_tensor, target_tensor, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss

        # print loss
        if iter % print_every==0:
            print_loss_avg = print_loss_total/print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter/n_iters),
                                        iter, iter/n_iters*100, print_loss_avg))
        
        # plot loss
        if iter % plot_every==0:
            plot_loss_avg = plot_loss_total/print_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

    plt.savefig('nmt-loss')
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')

if __name__=="__main__":
    config = argparser()

    embedding_size = config.embedding_size
    hidden_size = config.hidden_size
    teacher_forcing_ratio = config.teacher_forcing_ratio
    n_iters = config.n_iters

    input_lang, output_lang, pairs = loader.prepareData('eng', 'fra', True)
    
    input_emb_matrix, output_emb_matrix= np.load('input_emb_matrix.npy'), np.load('output_emb_matrix.npy')
    print('Embedding-matrix shape: {}, {}'.format(input_emb_matrix.shape, output_emb_matrix.shape))

    encoder = seq2seq.Encoder(input_size = input_lang.n_words,
                              embedding_size = embedding_size,
                              hidden_size = hidden_size,
                              embedding_matrix = input_emb_matrix,
                              n_layers = config.n_layers,
                              dropout_p = config.dropout_p
                              ).to(device)

    decoder = seq2seq.AttnDecoder(output_size = output_lang.n_words,
                                  embedding_size = embedding_size,
                                  hidden_size = hidden_size,
                                  embedding_matrix = output_emb_matrix,
                                  n_layers = config.n_layers,
                                  dropout_p = config.dropout_p
                                  ).to(device)

    trainiters(pairs, encoder, decoder, n_iters)