import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
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
        super(BiLSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size) 
        self.lstm = nn.LSTM(embedding_size, int(hidden_size/2),
                            bidirectional=True,
                            batch_first=True)

    def forward(self, input, hidden):
        # |input| = (1)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
        embedded = self.embedding(input).view(1,1,-1)
        output = embedded
        # |output| = (1, 1, hidden_size)
        output, hidden = self.lstm(output, hidden)
        # |output| = (batch_size, sequence_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
        return output, hidden

    def initHidden(self):
        # |hidden|, |cell| = (num_layers*num_directions, batch_size, hidden_size/2)
        return torch.zeros(1*2, 1, int(self.hidden_size/2))

class BiLSTMDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        super(BiLSTMDecoder, self).__init__()
        
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
        # Here, the lstm layer in decoder is uni-directional.
        output = self.embedding(input).view(1,1,-1)
        output = F.relu(output)
        # |output| = (1, 1, hidden_size)

        output, hidden = self.lstm(output, hidden)
        # |output| = (batch_size, sequence_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        # Here, the lstm layer in decoder is uni-directional.
        output = self.softmax(self.out(output[0]))
        # |output| = (1, output_lang.n_words)
        return output, hidden