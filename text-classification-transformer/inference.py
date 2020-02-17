import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
import torch

from tokenization import Tokenizer, PretrainedTokenizer
from model import TransformerEncoder

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

def main(args):
    print(args)

    # Load tokenizer
    if args.tokenizer == 'sentencepiece':
        tokenizer = PretrainedTokenizer(pretrained_model = args.pretrained_model, vocab_file = args.vocab_file)
    else:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
        tokenizer = Tokenizer(tokenizer = tokenizer, vocab_file = args.vocab_file)

    # Load model
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    model = torch.load(args.model).to(device)
    model.eval()

    # Make input
    text = 'I have to admit, I got so emotional all throughout the movie.\
            And some parts brought me to tears. The cast was phenomenal and I think every superhero got to have their spotlight.'
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    padding_length = args.max_seq_len - len(input_ids)
    input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    print('--------------------------------------------------------')
    print('tokens: {}'.format(tokens))
    print('input_ids: {}'.format(input_ids))
    print('|input_ids|: {}'.format(input_ids))
    print('--------------------------------------------------------')

    # Inference
    output, attention_weights = model(input_ids)
    print('class: {}'.format(output.argmax(dim=1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',               required=True,            type=str, help='transformer model to load')

    parser.add_argument('--dataset',             default='imdb',           type=str, help='dataset')
    parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--tokenizer',           default='sentencepiece',  type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER_CLASSES.keys()))
    parser.add_argument('--pretrained_model',    default='wiki.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    
    # Input parameters
    parser.add_argument('--max_seq_len',    default=512,  type=int,   help='the maximum size of the input sequence')
    parser.add_argument('--no_cuda',        action='store_true')
    
    args = parser.parse_args()
    
    main(args)