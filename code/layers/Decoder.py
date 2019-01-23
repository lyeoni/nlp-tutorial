import torch
import torch.nn as nn

'''
refer to:
    - https://github.com/kh-kim/nlp_with_pytorch/blob/master/neural-machine-translation/seq2seq.md
    - https://github.com/kh-kim/nlp_with_pytorch/blob/master/neural-machine-translation/seq2seq.mdhttps://github.com/kh-kim/simple-nmt/blob/master/simple_nmt/seq2seq.py
'''


class Decoder(nn.Module):
    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()
        
        # Unlike encoder, 'bidirectional' parameter set False
        self.rnn = nn.LSTM(word_vec_dim+hidden_size,
                              hidden_size,
                              num_layer=n_layers,
                              dropout=dropout_p,
                              bidirectional=False,
                              batch_first=True)
    
    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        # |emb_t| = (batch_size, 1, word_vec_dim) , current time-step vector
        # |h_t_1_tilde| = (batch_size, 1, hidden_size) , previous time-step vector after softmax 
        # |h_t_1[0]| = (n_layers, batch_size, hidden_size) , previous time-step vector before softmax
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)
        
        if h_t_1_tilde is None:
            # If this is the first time-step(In other words, 'BOS' token), 
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()
        
        # Input feeding trcik - conactenate 'current embedding token' with 'hidden-state of previous time-step'
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)
        
        # Unlike encoder, decoder must take an input for sequentially.
        y, h = self.rnn(x, h_t_1)
        
        return y, h
