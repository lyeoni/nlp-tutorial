import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        '''
        Inputs: input, (h_0, c_0)
        Outputs: output, (h_n, c_n)
            - |input| = features of the input sequence.
                      = (seq_len, batch_size, input_size)
            - |h_0| = initial hidden state for each element in the batch.
                    = (num_layers*num_directions, batch_size, hidden_size)
            - |c_0| = initial cell state for each element in the batch.
                    = (num_layers*num_directions, bathc_size, hidden_size)
            - |output| = output features (h_t) from the last layer of the LSTM, for each t.
                       = (seq_len, batch_size, num_directions*hidden_size)
            - |h_n| = hidden state for t = seq_len.
                    = (num_layers*num_directions, batch_size, hidden_size)
            - |c_n| = cell state for t = seq_len.
                    = (num_layers*num_directions, batch_size, hidden_size)
        '''
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size) 
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            bidirectional=False,
                            batch_first=True)

    def forward(self, input, hidden):
        # |input| = (1)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        # |output| = (1, 1, hidden_size)
        output, hidden = self.lstm(output, hidden)
        # |output| = (batch_size, sequence_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        return output, hidden

    def initHidden(self):
        # |hidden| = (num_layers*num_directions, batch_size, hidden_size)
        return torch.zeros(1, 1, self.hidden_size)

class LSTMDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        super(LSTMDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            bidirectional=False,
                            batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # |input| = (1)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        # |output| = (1, 1, hidden_size)

        output, hidden = self.lstm(output, hidden)
        # |output| = (batch_size, sequence_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        output = self.softmax(self.out(output[0]))
        # |output| = (1, output_lang.n_words)
        return output, hidden