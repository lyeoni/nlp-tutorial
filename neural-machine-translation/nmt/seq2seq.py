import torch
import torch.nn as nn
import torch.nn.functional as F
import dataLoader as loader

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,
                 embedding_matrix, n_layers=3, dropout_p=.1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm = nn.LSTM(embedding_size, int(hidden_size/2),
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)

    def forward(self, input, hidden):
        # |input| = (1)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # |output| = (1, 1, embedding_size)
        output, hidden = self.lstm(output, hidden)
        # |output| = (batch_size, sequence_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size/2)
        return output, hidden

    def initHidden(self):
        # |hidden|, |cell| = (num_layers*num_directions, batch_size, hidden_size/2)
        return torch.zeros(self.n_layers*2, 1, int(self.hidden_size/2))
        

class AttnDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size,
                 embedding_matrix, n_layers=3, dropout_p=.1, max_length=loader.MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        # layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=False,
                            batch_first=True)
        self.out = nn.Linear(self.hidden_size*2, self.output_size)
        self.softmax =nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        # |input| = (1, 1)
        # |hidden| = (2, Enocder.n_layers, batch_size, hidden_size) # 2: respectively, hidden state and cell state
        # |encoder_outputs| = (max_length, hidden_size)
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        # |output| = (1, 1, embedding_size)

        output, hidden = self.lstm(output, hidden)
        # |output| = (batch_size, sequence_length, hidden_size)
        # |hidden| = (2, AttnDecoder.n_layers, batch_size, hidden_size)
        
        attn_weights = torch.bmm(output, encoder_outputs.unsqueeze(0).permute(0, 2, 1)).squeeze(0)
        attn_weights = F.softmax(attn_weights, dim=1)
        # |attn_weights| = (sequence_length, max_length)

        context = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        # |context| = (batch_size, sequence_length, hidden_size)

        output = torch.cat((output, context), -1)
        # |output| = (batch_size, sequence_length, hidden_size*2)

        output= self.softmax(self.out(output[0]))
        # |output| = (1, output_lang.n_words)

        return output, hidden, attn_weights
