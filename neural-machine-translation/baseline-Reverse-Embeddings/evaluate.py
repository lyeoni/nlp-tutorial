import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import dataLoader as loader
import seq2seq
import train
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--encoder',
                    required=True,
                    help='Encoder file path to load trained_encoder\'s learned parameters.')

    p.add_argument('--decoder',
                    required=True,
                    help='Decoder file path to load trained_decoder\'s learned parameters.')

    p.add_argument('--embedding_size',
                    type=int,
                    default=300,
                    help='Word embedding vector dimension size. Default=300')

    p.add_argument('--hidden_size',
                    type=int,
                    default=300,
                    help='Hidden size of RNN. Default=300')

    config = p.parse_args()
    return config

def translate(pair, output):
    print('Source:\t{}\nAnswer:\t{}'.format(pair[0], pair[1])) 
    print('Translate: {}'.format(output), end='\n\n')
    # print('{}|{}'.format(pair[1], output))

def evaluate(sentence, encoder, decoder, max_length=loader.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = train.tensorFromSentence(input_lang, sentence)
        # |input_tensor| = (sentence_length, 1)
        input_length = input_tensor.size(0)
        
        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        # |encoder_hidden|= (2, num_layers*num_directions, batch_size, hidden_size/2)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size).to(device)
        # |encoder_outputs| = (max_length, hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # |encoder_output| = (batch_size, sequence_length, num_directions*(hidden_size/2))
            # |encoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size/2)
            # 2: respectively, hidden state and cell state.
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[loader.SOS_token]]).to(device)
        # |decoder_input| = (1, 1)
        decoder_hidden = train.merge_encoder_hiddens(encoder_hidden)
        # |decoder_hidden|= (2, num_layers*num_directions, batch_size, hidden_size)
        # 2: respectively, hidden state and cell state
        # Here, the lstm layer in decoder is uni-directional.

        decoded_words=[]
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # |decoder_output| = (sequence_length, output_lang.n_words)
            # |decoder_hidden| = (2, num_layers*num_directions, batch_size, hidden_size)
            # 2: respectively, hidden state and cell state.
            # Here, the lstm layer in decoder is uni-directional.
            
            topv, topi = decoder_output.data.topk(1) # top-1 value, index
            # |topv|, |topi| = (1, 1)

            if topi.item() == loader.EOS_token:
                # decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            
    return decoded_words

def evaluateiters(pairs, encoder, decoder, train_pairs_seed=0):
    start = time.time()
    cc = SmoothingFunction()
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.15, random_state=train_pairs_seed)
    # |test_pairs| = (n_pairs, 2, sentence_length, 1) # eng, fra

    scores = []
    for pi, pair in enumerate(test_pairs):
        output_words = evaluate(pair[0], encoder, decoder)
        output_sentence = ' '.join(output_words)

        # for print
        translate(pair, output_sentence)
        
        # for nltk.bleu
        ref = pair[1].split()
        hyp = output_words
        scores.append(sentence_bleu([ref], hyp, smoothing_function=cc.method3) * 100.)
        
    print('BLEU: {:.4}'.format(sum(scores)/len(test_pairs)))

if __name__ == "__main__":
    '''
    Evaluation is mostly the same as training,
    but there are no targets so we simply feed the decoder's predictions back to itself for each step.
    Every time it predicts a word, we add it to the output string,
    and if it predicts the EOS token we stop there.
    '''
    config = argparser()

    input_lang, output_lang, pairs = loader.prepareData('eng', 'fra', True)
    
    input_emb_matrix, output_emb_matrix= np.load('input_emb_matrix.npy'), np.load('output_emb_matrix.npy')
    print('Embedding-matrix shape: {}, {}'.format(input_emb_matrix.shape, output_emb_matrix.shape))

    encoder = seq2seq.BiLSTMEncoder(input_size = input_lang.n_words,
                                     embedding_size = config.embedding_size,
                                     hidden_size = config.hidden_size,
                                     embedding_matrix = input_emb_matrix
                                     ).to(device)

    decoder = seq2seq.BiLSTMDecoder(output_size = output_lang.n_words,
                                     embedding_size = config.embedding_size,
                                     hidden_size = config.hidden_size,
                                     embedding_matrix = output_emb_matrix
                                     ).to(device)

    encoder.load_state_dict(torch.load(config.encoder))
    encoder.eval()
    decoder.load_state_dict(torch.load(config.decoder))
    decoder.eval()

    evaluateiters(pairs, encoder, decoder)