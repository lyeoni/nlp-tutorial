import torch
import torch.nn as nn
import torch.nn.functional as F
import dataLoader as loader

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_matrix):
        '''
        When the LSTM is Bi-directional, num_directions should be 2, else it should be 1.
            Inputs: input, (h_0, c_0)
            Outputs: output, (h_n, c_n)

            - |input| = features of the input sequence.
                      = (seq_len, batch_size, input_size)
            - |h_0| = initial hidden state for each element in the batch.
                    = (num_layers*num_directions, batch_size, hidden_size)
            - |c_0| = initial cell state for each element in the batch.
                    = (num_layers*num_directions, bathc_size, hidden_size)
            - |output| = output features (h_t) from the last layer of the RNN.
                    = (seq_len, batch_size, num_directions*hidden_size)
            - |h_n| = hidden state for t = seq_len.
                    = (num_layers*num_directions, batch_size, hidden_size)
            - |c_n| = cell state for t = seq_len.
                    = (num_layers*num_directions, batch_size, hidden_size)
        '''
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # |input_size| = (input_lang.n_words)
        self.lstm = nn.LSTM(hidden_size,
                            int(hidden_size/2),
                            bidirectional=True)

    def forward(self, input, hidden):
        # |input| = (1)
        # |hidden| = (2, 1, hidden_size/2)
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        # |output| = (1, 1, hidden_size)
        output, hidden = self.lstm(output, hidden)
        # |output| = (1, 1, hidden_size)
        # |hidden[0]|, |hidden[1]| = (2, 1, hidden_size/2)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(2, 1, int(self.hidden_size/2))

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, embedding_matrix, dropout_p=0.1, max_length=loader.MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # |input| = (1, 1)
        # |hidden| = (2, 1, 1, hidden_size) # tuple. hidden, cell state respectively.
        # |encoder_outputs| = (max_length, hidden_size)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # |embedded| = (1, 1, hidden_size)
        
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0]), 1)), dim=1)
        # |torch.cat((embedded[0], hidden[0][0]), 1)| = (1, hidden_size*2)
        # |self.attn(torch.cat((embedded[0], hidden[0][0]), 1))| = (1, max_length)
        # |attn_weights| = (1, max_length)        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # |attn_weights.unsqueeze(0)| = (1, 1, max_length)
        # |encoder_outputs.unsqueeze(0)| = (1, max_length, hidden_size)
        # |attn_applied| = (1, 1, hidden_size)
        
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # |output| = (1, hidden_size*2)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        # |output| = (1, 1, hidden_size)
        output, hidden = self.lstm(output, hidden)
        # |output| = (1, 1, hidden_size)
        # |hidden| = (2, 1, 1, hidden_size) # tuple. hidden, cell state respectively.
        output = F.log_softmax(self.out(output[0]), dim=1)
        # |output| = (1, output_size)
        
        return output, hidden, attn_weights