import argparse
import torch

from tokenization import Tokenizer, PretrainedTokenizer

def main(args):
    print(args)
    
    # Load tokenizer
    tokenizer_src = PretrainedTokenizer(pretrained_model = args.pretrained_model_src, vocab_file = args.vocab_file_src)
    tokenizer_tgt = PretrainedTokenizer(pretrained_model = args.pretrained_model_tgt, vocab_file = args.vocab_file_tgt)

    # Load model
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    model = torch.load(args.model).to(device)
    model.eval()

    # Make input
    text = 'Je ferai n\'importe quoi pour lui.'
    tokens = tokenizer_src.tokenize(text)
    tokens = tokens[:args.max_seq_len]
    
    src_ids = tokenizer_src.convert_tokens_to_ids(tokens)
    padding_length = args.max_seq_len - len(src_ids)
    src_ids = src_ids + ([tokenizer_src.pad_token_id] * padding_length)
    src_ids = torch.tensor(src_ids).unsqueeze(0).to(device)
    tgt_ids = torch.tensor([tokenizer_tgt.bos_token_id]).unsqueeze(0).to(device)

    print('--------------------------------------------------------')
    print('tokens: {}'.format(tokens))
    print('src_ids: {}'.format(src_ids))
    print('tgt_ids: {}'.format(tgt_ids))
    print('--------------------------------------------------------')

    # Inference
    for i in range(args.max_seq_len):
        outputs, encoder_attns, decoder_attns, enc_dec_attns = model(src_ids, tgt_ids)
        output_token_id = outputs[:,-1,:].argmax(dim=-1).item()
        
        if output_token_id == tokenizer_tgt.eos_token_id:
            break
        else:
            tgt_ids = torch.cat((tgt_ids, torch.tensor([output_token_id]).unsqueeze(0).to(device)), dim=-1)
    
    ids = tgt_ids[0].tolist()
    tokens = tokenizer_tgt.convert_ids_to_tokens(ids)
    print(tokenizer_tgt.detokenize(tokens)) # I will do anything for him.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model',               required=True,            type=str, help='transformer model to load')

    parser.add_argument('--vocab_file_src',          default='fra.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--vocab_file_tgt',          default='eng.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--pretrained_model_src',    default='fra.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--pretrained_model_tgt',    default='eng.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')    
    # Input parameters
    parser.add_argument('--max_seq_len',    default=80,  type=int,   help='the maximum size of the input sequence')
    parser.add_argument('--no_cuda',        action='store_true')

    args = parser.parse_args()
    
    main(args)