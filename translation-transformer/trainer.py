from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from model import Transformer
from optimization import ScheduledOptim

class Trainer:
    def __init__(self, args, train_loader, test_loader, tokenizer_src, tokenizer_tgt):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.src_vocab_size = tokenizer_src.vocab_size
        self.tgt_vocab_size = tokenizer_tgt.vocab_size
        self.pad_id = tokenizer_src.pad_token_id # pad_token_id in tokenizer_tgt.vocab should be the same with this.
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

        self.model = Transformer(src_vocab_size = self.src_vocab_size,
                                 tgt_vocab_size = self.tgt_vocab_size,
                                 seq_len        = args.max_seq_len,
                                 d_model        = args.hidden,
                                 n_layers       = args.n_layers,
                                 n_heads        = args.n_attn_heads,
                                 p_drop         = args.dropout,
                                 d_ff           = args.ffn_hidden,
                                 pad_id         = self.pad_id)
        if args.multi_gpu:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.optimizer = ScheduledOptim(optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-9),
                                        init_lr=2.0, d_model=args.hidden)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def train(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            encoder_inputs, decoder_inputs, decoder_outputs = map(lambda x: x.to(self.device), batch)
            # |encoder_inputs| : (batch_size, seq_len), |decoder_inputs| : (batch_size, seq_len-1), |decoder_outputs| : (batch_size, seq_len-1)

            outputs, encoder_attns, decoder_attns, enc_dec_attns = self.model(encoder_inputs, decoder_inputs)
            # |outputs| : (batch_size, seq_len-1, tgt_vocab_size)
            # |encoder_attns| : [(batch_size, n_heads, seq_len, seq_len)] * n_layers
            # |decoder_attns| : [(batch_size, n_heads, seq_len-1, seq_len-1)] * n_layers
            # |enc_dec_attns| : [(batch_size, n_heads, seq_len-1, seq_len)] * n_layers
            
            loss = self.criterion(outputs.view(-1, self.tgt_vocab_size), decoder_outputs.view(-1))
            losses += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.update_learning_rate()
            self.optimizer.step()

            if i % (n_batches//5) == 0 and i != 0:
                print('Iteration {} ({}/{})\tLoss: {:.4f}\tlr: {:.4f}'.format(i, i, n_batches, losses/i, self.optimizer.get_current_lr))
        
        print('Train Epoch: {}\t>\tLoss: {:.4f}'.format(epoch, losses/n_batches))
            
    def validate(self, epoch):
        losses = 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                encoder_inputs, decoder_inputs, decoder_outputs = map(lambda x: x.to(self.device), batch)
                # |encoder_inputs| : (batch_size, seq_len), |decoder_inputs| : (batch_size, seq_len-1), |decoder_outputs| : (batch_size, seq_len-1)

                outputs, encoder_attns, decoder_attns, enc_dec_attns = self.model(encoder_inputs, decoder_inputs)
                # |outputs| : (batch_size, seq_len-1, tgt_vocab_size)
                # |encoder_attns| : [(batch_size, n_heads, seq_len, seq_len)] * n_layers
                # |decoder_attns| : [(batch_size, n_heads, seq_len-1, seq_len-1)] * n_layers
                # |enc_dec_attns| : [(batch_size, n_heads, seq_len-1, seq_len)] * n_layers
                
                loss = self.criterion(outputs.view(-1, self.tgt_vocab_size), decoder_outputs.view(-1))
                losses += loss.item()

        print('Valid Epoch: {}\t>\tLoss: {:.4f}'.format(epoch, losses/n_batches))

    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        
        torch.save(self.model, path)