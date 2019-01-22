import torch

# refered to https://github.com/kh-kim/nlp_with_pytorch/blob/master/neural-machine-translation/attention.md

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax == nn.Softmax(dim=-1)
        
    def forward(self, h_src, h_t_tgt, mask=None):
        # |h_src| = (batch_size, length, hidden_size)
        # |h_t_tgt| = (batch_size, 1, hidden_size)
        # |mask| = (batch_size, length)
        
        query = self.linear(h_t_tgt.squeeze(1). unsqueeze(-1))
        # |query| = (batch_size, hidden_size, 1)
        
        weight = torch.bmm(h_src, query).squeeze(-1)  # query-key 'similarity' estimation
        # |weight| = (batch_size, length) 
        
        if mask is not None:
            '''
            Set each weight as -inf, if the mask value equals to 1.
            Since the softmax operation makes -inf to 0, masked weights would be set to 0 after softmax operation.
            Thus, if the sample is shorter than other samples in mini-batch, the weight for empty time-stp would be set to 0.
            '''
            weight.masked_fill_(mask, -float('inf'))
        weight = self.softmax(weight)
        
        context_vector = torch.bmm(weight.unsqueeze(1), h_src)
        # |context_vector| = (batch_size, 1, hidden_size)
        
        return context_vector