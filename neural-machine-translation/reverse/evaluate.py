import random
import dataLoader as loader
import seq2seq
import train
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(encoder, decoder, sentence, max_length=loader.MAX_LENGTH):
    with torch.no_grad():
        input_tensor = train.tensorFromSentence(input_lang, sentence)
        # |input_tensor| = (sentence_length, 1)
        input_length = input_tensor.size(0)

        encoder_hidden = (encoder.initHidden().to(device), encoder.initHidden().to(device))
        # |encoder_hidden[0]|, |encoder_hidden[1]| = (2, 1, hidden_size/2)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        # |encoder_outputs| = (max_length, hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            # |encoder_output| = (1, 1, hidden_size)
            # |encoder_hidden[0]|, |encoder_hidden[1]| = (2, 1, hidden_size/2)
            encoder_outputs[ei] += encoder_output[0, 0]
        
        decoder_input = torch.tensor([[loader.SOS_token]], device=device)
        decoder_hidden = train.merge_encoder_hiddens(encoder_hidden)
        # |decoder_input| = (1, 1)
        # |decoder_hidden[0]|, |decoder_hidden[1]| = (1, 1, hidden_size)

        decoded_words=[]
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # |decoder_output| = (1, output_lang.n_words)
            # |decoder_hidden[0]|, |decoder_hidden[1]| = (1, 1, hidden_size)
            topv, topi = decoder_output.data.topk(1) # decoder_output.data == decoder_output
            # |topv| = (1, 1)
            # |topi| = (1, 1)
            if topi.item() == loader.EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            
            decoder_input = topi.squeeze().detach()
        
    return decoded_words

def evaluateRandomly(encoder, decoder, pairs, n=10):
    cc = SmoothingFunction()
    scores = []
    random.shuffle(pairs)
    for i in range(n):
        pair = pairs[i]
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        print('From(source):\t{}\n To(answer):\t{}'.format(pair[0], pair[1])) 
        print('To(predict):\t{}'.format(output_sentence), end='\n\n')

        # for nltk.bleu
        ref = pair[1].split()
        hyp = output_words[:-1]
        scores.append(sentence_bleu([ref], hyp) * 100.)

    print('BLEU: {:.4}'.format(sum(scores)/n))

if __name__ == "__main__":
    '''
    Evaluation is mostly the same as training,
    but there are no targets so we simply feed the decoder’s predictions back to itself for each step.
    Every time it predicts a word we add it to the output string,
    and if it predicts the EOS token we stop there.
    '''
    hidden_size = 256

    input_lang, output_lang, pairs = loader.prepareData('eng', 'fra', True)
    
    encoder = seq2seq.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seq.Decoder(hidden_size, output_lang.n_words).to(device)

    encoder.load_state_dict(torch.load('encoder.pth'))
    encoder.eval()
    decoder.load_state_dict(torch.load('decoder.pth'))
    decoder.eval()

    evaluateRandomly(encoder, decoder, pairs, int(len(pairs)*0.1))
