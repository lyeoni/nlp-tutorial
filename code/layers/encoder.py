import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

'''
refer to:
    - https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
    - https://github.com/kh-kim/nlp_with_pytorch/blob/master/neural-machine-translation/attention.md
'''

class Encoder(nn.Module):
    def __init__(self, word_vec_dim, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()
        
        # Be aware of value of 'batch_first' parameter
        # Also, its hidden_size is half of original hidden_size, because it is bi-directional.
        self. rnn = nn.LSTM(word_vec_dim, int(hidden_size/2),
                                   num_layers = n_layers,
                                   dropout = dropout_p,
                                   bidirectional = True,
                                   batch_first = True)
    
    def forward(self, emb):
        if isinstance(emb, tuple):
            # emb = (mini-batch, length of the sample in the mini-batch)
            
            x, lengths = emb
            # |x| = (batch_size, length, word_vec_dim)
            x = pack_padded_sequence(x, lengths.tolist(), batch_first= True)
            
            '''
            Below is how pack_padded_sequence works.
            As you can see, PackedSequence object has information about mini-batch-wise information, not time-step-wise information.
            
            a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
            b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
            >>> tensor([[1, 2, 3],
                            [3, 4, 0]])
            torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2])
            >>> PackedSequence(data=tensor([1, 3, 2, 4, 3]), batch_sizes = tensor([2, 2, 1]))
            '''
        else:
            x = emb
        
        y, h = self.rnn(x)
        # |y| = (batch_size, length, hidden_size)
        # |h[0]| = (num_layers*2, batch_size, hidden_size/2)
        
        if isinstance(emb, tuple):
            y, _ = pad_packed_sequence(y, batch_first= True)
        
        return y, h