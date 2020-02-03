import argparse
from prenlp.tokenizer import NLTKMosesTokenizer
from torch.utils.data import DataLoader

from data_utils import create_examples
from tokenization import Tokenizer, PretrainedTokenizer
from trainer import Trainer

TOKENIZER_CLASSES = {'nltk_moses': NLTKMosesTokenizer}

def main(args):
    print(args)

    # Load tokenizer
    if args.tokenizer == 'sentencepiece':
        tokenizer = PretrainedTokenizer(pretrained_model = args.pretrained_model, vocab_file = args.vocab_file)
    else:
        tokenizer = TOKENIZER_CLASSES[args.tokenizer]()
        tokenizer = Tokenizer(tokenizer = tokenizer, vocab_file = args.vocab_file)

    # Build DataLoader
    train_dataset = create_examples(args, tokenizer, mode='train')
    test_dataset = create_examples(args, tokenizer, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Build Trainer
    trainer = Trainer(args, train_loader, test_loader, tokenizer)
    
    # Train & Validate
    for epoch in range(1, args.epochs+1):
        trainer.train(epoch)
        trainer.validate(epoch)
        trainer.save(epoch, args.output_model_prefix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',             default='imdb',           type=str, help='dataset')
    parser.add_argument('--vocab_file',          default='wiki.vocab',     type=str, help='vocabulary path')
    parser.add_argument('--tokenizer',           default='sentencepiece',  type=str, help='tokenizer to tokenize input corpus. available: sentencepiece, '+', '.join(TOKENIZER_CLASSES.keys()))
    parser.add_argument('--pretrained_model',    default='wiki.model',     type=str, help='pretrained sentencepiece model path. used only when tokenizer=\'sentencepiece\'')
    parser.add_argument('--output_model_prefix', default='model',          type=str, help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size',     default=32,   type=int,   help='batch size')
    parser.add_argument('--max_seq_len',    default=512,  type=int,   help='the maximum size of the input sequence')
    # Train parameters
    parser.add_argument('--epochs',         default=7,   type=int,   help='the number of epochs')
    parser.add_argument('--lr',             default=1e-4, type=float, help='learning rate')
    parser.add_argument('--no_cuda',        action='store_true')
    # Model parameters
    parser.add_argument('--hidden',         default=256,  type=int,   help='the number of expected features in the transformer')
    parser.add_argument('--n_layers',       default=6,    type=int,   help='the number of heads in the multi-head attention network')
    parser.add_argument('--n_attn_heads',   default=8,    type=int,   help='the number of multi-head attention heads')
    parser.add_argument('--dropout',        default=0.1,  type=float, help='the residual dropout value')
    parser.add_argument('--ffn_hidden',     default=1024, type=int,   help='the dimension of the feedforward network')
    
    args = parser.parse_args()
    
    main(args)