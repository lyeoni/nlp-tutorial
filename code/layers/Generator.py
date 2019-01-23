import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()
        
        super.output = nn.Linear(hidden_size, output_size)
        super.sotmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, x)
    # |x| = (batch_size, length, hidden_size)
    
    y = self.softmax(self.output(x))
    # |y| = (batch_size, length, output_size)
    
    return y