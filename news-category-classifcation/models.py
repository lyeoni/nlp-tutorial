import torch
import torch.nn as nn

class CBoWClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout_p,
                 embedding_weights, embedding_trainable):
        super(CBoWClassifier, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        if not embedding_trainable:
            self.embedding.weight.requires_grad = False
        self.fc = nn.Linear(embedding_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input):
        # |input| = (batch_size, max_seq_length)
        batch_size = input.size(0)

        embeds = self.embedding(input)
        # |embeds| = (bathc_size, max_seq_length, embedding_size)
        mean_embeds = torch.mean(embeds, dim=1)
        # |mean_embeds| = (batch_size, embedding_size)
        
        fc_out = self.dropout(self.relu(self.fc(mean_embeds)))
        # |fc_out| = (batch_size, hidden_size)
        output = self.softmax(self.fc2(fc_out))
        # |output| = (batch_size, output_size)
        
        return output

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, dropout_p, 
                 embedding_weights, embedding_trainable):
        super(LSTMClassifier, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        if embedding_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_weights))
        if not embedding_trainable:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(self.hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, input):
        # |input| = (batch_size, max_seq_length)
        batch_size = input.size(0)

        embeds = self.embedding(input)
        # |embeds| = (bathc_size, max_seq_length, embedding_size)
      
        lstm_out, hidden = self.lstm(embeds)
        # If bidirectional=True, num_directions is 2, else it is 1.
        # |lstm_out| = (batch_size, max_seq_length, num_directions*hidden_size)
        # |hidden[0]|, |hidden[1]| = (num_layers*num_directions, batch_size, hidden_size)
        
        mean_lstm_out = torch.mean(lstm_out, dim=1)
        # |lstm_out| = (batch_size, hidden_size*2)
        output = self.softmax(self.fc(mean_lstm_out))
        # |output| = (batch_size, output_size)

        return output