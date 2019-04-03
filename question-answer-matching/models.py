import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers, dropout_p,
                  word_embedding_matrix, tfidf_matrix, evaluation_mode=False):
        super(Model, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.eval_mode = evaluation_mode

        # layers
        self.word_embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.word_embedding.weight.data.copy_(torch.from_numpy(word_embedding_matrix))
        self.tfidf_embedding = nn.Embedding(self.input_size, 1)
        self.tfidf_embedding.weight.data.copy_(torch.from_numpy(tfidf_matrix))
        self.tfidf_embedding.weight.requires_grad = False
        self.lstm_q = nn.LSTM(self.embedding_size, self.hidden_size,
                              num_layers= self.n_layers,
                              bidirectional= True,
                              dropout= self.dropout_p,
                              batch_first= True)
        self.lstm_a = nn.LSTM(self.embedding_size, self.hidden_size,
                              num_layers= self.n_layers,
                              bidirectional= True,
                              dropout= self.dropout_p,
                              batch_first= True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, qus, ans, qus_len, ans_len):
        # |qus|, |ans| = (batch_size, maxlen)
        # |qus_len|, |ans_len| = (batch_size)
        batch_size = qus.size(0)
        hidden_qus = (self.initHidden(batch_size), self.initHidden(batch_size))
        hidden_ans = (self.initHidden(batch_size), self.initHidden(batch_size))
        # |hidden_qus|, |hidden_ans| = (2, num_layers*num_directions, batch_size, hidden_size)

        embed_qus, embed_ans = self.word_embedding(qus), self.word_embedding(ans)
        # |embed_qus|, |embed_ans| = (batch_size, maxlen, embedding_size)

        tfidf_qus, tfidf_ans = self.tfidf_embedding(qus), self.tfidf_embedding(ans)
        # |tfidf_qus|, |tfidf_ans| = (batch_size, maxlen, 1)

        # Element-wise multiplication
        embed_qus = embed_qus * tfidf_qus
        embed_ans = embed_ans * tfidf_ans
        # |embed_qus|, |embed_ans| = (batch_size, maxlen, embedding_size)

        # Pack to feed the variable-length sequences to LSTM
        pack_qus = pack_padded_sequence(embed_qus, qus_len.tolist(), batch_first=True)
        pack_ans = pack_padded_sequence(embed_ans, ans_len.tolist(), batch_first=True)
        # |pack_qus|, |pack_ans| = PackedSequence object holding the data and list of batch_sizes of a packed sequence

        # Feed to LSTM
        pack_output_qus, (h_qus, c_qus) = self.lstm_q(pack_qus, hidden_qus)
        pack_output_ans, (h_ans, c_ans) = self.lstm_a(pack_ans, hidden_ans)
        # |pack_output_qus|, |pack_output_ans| = PackedSequence object holding the data and list of batch_sizes of a packed sequence
        # |h_qus|, |h_ans| = (num_layers*num_directions, batch_size, hidden_size)
        # |c_qus|, |c_ans| = (num_layers*num_directions, batch_size, hidden_size)
        
        # Unpack the packed sequence
        output_qus, output_qus_len = pad_packed_sequence(pack_output_qus, batch_first=True)
        output_ans, output_ans_len = pad_packed_sequence(pack_output_ans, batch_first=True)
        # |output_qus|, |output_ans| = (batch_size, batch_maxlen, num_directions*hidden_size)
        # |output_qus_len|, |output_ans_len| = (batch_size)

        # CBoW: sum up outputs in each sequence
        output_qus = torch.sum(output_qus, dim = 1) 
        output_ans = torch.sum(output_ans, dim = 1)
        # |output_qus|, |output_ans| = (batch_size, num_directions*hidden_size)
        if self.eval_mode: return output_qus

        # Compute the cosine similarity between question and answer representation
        output = torch.matmul(output_qus.unsqueeze(1), output_ans.unsqueeze(-1))
        output = output.squeeze(-1)
        output /= (torch.norm(output_qus, p=2, dim=1) * torch.norm(output_ans, p=2, dim=1)).unsqueeze(-1)
        # |output| = (batch_size, 1)
        
        output = self.sigmoid(output)
        # |output| = (batch_size, 1)

        return output   
    
    def initHidden(self, batch_size):
        # |hidden|, |cell| = (num_layers*num_directions, batch_size, hidden_size)
        return torch.zeros((self.n_layers*2, batch_size, self.hidden_size)).to(device)