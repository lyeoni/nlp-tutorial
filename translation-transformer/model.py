import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, q, k, v, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k), |v| : (batch_size, n_heads, v_len, d_v)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_score.masked_fill_(attn_mask, -1e9)
        # |attn_score| : (batch_size, n_heads, q_len, k_len)

        attn_weights = nn.Softmax(dim=-1)(attn_score)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)
        
        output = torch.matmul(attn_weights, v)
        # |output| : (batch_size, n_heads, q_len, d_v)
        
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model//n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        
    def forward(self, Q, K, V, attn_mask):
        # |Q| : (batch_size, q_len, d_model), |K| : (batch_size, k_len, d_model), |V| : (batch_size, v_len, d_model)
        # |attn_mask| : (batch_size, seq_len(=q_len), seq_len(=k_len))
        batch_size = Q.size(0)
        
        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2) 
        # |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
        
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
        # |attn| : (batch_size, n_heads, q_len, d_v)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        # |attn| : (batch_size, q_len, n_heads * d_v)
        output = self.linear(attn)
        # |output| : (batch_size, q_len, d_model)

        return output, attn_weights

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len, d_model)

        output = self.relu(self.linear1(inputs))
        # |output| : (batch_size, seq_len, d_ff)
        output = self.linear2(output)
        # |output| : (batch_size, seq_len, d_model)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |attn_mask| : (batch_size, seq_len, seq_len)
        
        attn_outputs, attn_weights = self.mha(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len, d_model)
        # |attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=seq_len))

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        return ffn_outputs, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table):
        super(TransformerEncoder, self).__init__()
        self.pad_id = pad_id

        # layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def forward(self, inputs):
        # |inputs| : (batch_size, seq_len)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)

        outputs = self.embedding(inputs) + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)

        attention_weights = []
        for layer in self.layers:
            outputs, attn_weights = layer(outputs, attn_pad_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            attention_weights.append(attn_weights)

        return outputs, attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, p_drop, d_ff):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)

        self.mha2 = MultiHeadAttention(d_model, n_heads)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout3 = nn.Dropout(p_drop)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, inputs, encoder_outputs, attn_mask, enc_dec_attn_mask):
        # |inputs| : (batch_size, seq_len, d_model)
        # |encoder_outputs| : (batch_size, encoder_outputs_len, d_model)
        # |attn_mask| : (batch_size, seq_len ,seq_len)
        # |enc_dec_attn_mask| : (batch_size, seq_len, encoder_outputs_len)

        attn_outputs, attn_weights = self.mha1(inputs, inputs, inputs, attn_mask)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(inputs + attn_outputs)
        # |attn_outputs| : (batch_size, seq_len, d_model)
        # |attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=seq_len))

        enc_dec_attn_outputs, enc_dec_attn_weights = self.mha2(attn_outputs, encoder_outputs, encoder_outputs, enc_dec_attn_mask)
        enc_dec_attn_outputs = self.dropout2(enc_dec_attn_outputs)
        enc_dec_attn_outputs = self.layernorm2(attn_outputs + enc_dec_attn_outputs)
        # |enc_dec_attn_outputs| : (batch_size, seq_len, d_model)
        # |enc_dec_attn_weights| : (batch_size, n_heads, q_len(=seq_len), k_len(=encoder_outputs_len))
        
        ffn_outputs = self.ffn(enc_dec_attn_outputs)
        ffn_outputs = self.dropout3(ffn_outputs)
        ffn_outputs = self.layernorm3(enc_dec_attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)

        return ffn_outputs, attn_weights, enc_dec_attn_weights

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table):
        super(TransformerDecoder, self).__init__()
        self.pad_id = pad_id

        # layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def forward(self, inputs, encoder_inputs, encoder_outputs):
        # |inputs| : (batch_size, seq_len)
        # |encoder_inputs| : (batch_size, encoder_inputs_len)
        # |encoder_outputs| : (batch_size, encoder_outputs_len(=encoder_inputs_len), d_model)
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).repeat(inputs.size(0), 1) + 1
        position_pad_mask = inputs.eq(self.pad_id)
        positions.masked_fill_(position_pad_mask, 0)
        # |positions| : (batch_size, seq_len)
        
        outputs = self.embedding(inputs) + self.pos_embedding(positions)
        # |outputs| : (batch_size, seq_len, d_model)

        attn_pad_mask = self.get_attention_padding_mask(inputs, inputs, self.pad_id)
        # |attn_pad_mask| : (batch_size, seq_len, seq_len)
        attn_subsequent_mask = self.get_attention_subsequent_mask(inputs).to(device=attn_pad_mask.device)
        # |attn_subsequent_mask| : (batch_size, seq_len, seq_len)
        attn_mask = torch.gt((attn_pad_mask.to(dtype=attn_subsequent_mask.dtype) + attn_subsequent_mask), 0)
        # |attn_mask| : (batch_size, seq_len, seq_len)

        enc_dec_attn_mask = self.get_attention_padding_mask(inputs, encoder_inputs, self.pad_id)
        # |enc_dec_attn_mask| : (batch_size, seq_len, encoder_inputs_len)

        attention_weights, enc_dec_attention_weights = [], []
        for layer in self.layers:
            outputs, attn_weights, enc_dec_attn_weights = layer(outputs, encoder_outputs, attn_mask, enc_dec_attn_mask)
            # |outputs| : (batch_size, seq_len, d_model)
            # |attn_weights| : (batch_size, n_heads, seq_len, seq_len)
            # |enc_dec_attn_weights| : (batch_size, n_heads, seq_len, encoder_outputs_len)
            attention_weights.append(attn_weights)
            enc_dec_attention_weights.append(enc_dec_attn_weights)

        return outputs, attention_weights, enc_dec_attention_weights

    def get_attention_padding_mask(self, q, k, pad_id):
        attn_pad_mask = k.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask
    
    def get_attention_subsequent_mask(self, q):
        bs, q_len = q.size()
        subsequent_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)
        # |subsequent_mask| : (batch_size, q_len, q_len)
        
        return subsequent_mask

class Transformer(nn.Module):
    """Transformer is a stack of N encoder/decoder layers.

    Args:
        src_vocab_size (int)    : encoder-side vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        tgt_vocab_size (int)    : decoder-side vocabulary size (vocabulary: collection mapping token to numerical identifiers)
        seq_len        (int)    : input sequence length
        d_model        (int)    : number of expected features in the input
        n_layers       (int)    : number of sub-encoder-layers in the encoder
        n_heads        (int)    : number of heads in the multiheadattention models
        p_drop         (float)  : dropout value
        d_ff           (int)    : dimension of the feedforward network model
        pad_id         (int)    : pad token id
    
    Examples:
    >>> model = Transformer(src_vocab_size=1000, tgt_vocab_size=1000, seq_len=512)
    >>> enc_input, dec_input = torch.arange(512).repeat(2, 1), torch.arange(512).repeat(2, 1)
    >>> model(enc_input, dec_input)
    """

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 seq_len,
                 d_model=512,
                 n_layers=6,
                 n_heads=8,
                 p_drop=0.1,
                 d_ff=2048,
                 pad_id=0):
        super(Transformer, self).__init__()
        sinusoid_table = self.get_sinusoid_table(seq_len+1, d_model) # (seq_len+1, d_model)
        
        self.encoder = TransformerEncoder(src_vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, n_layers, n_heads, p_drop, d_ff, pad_id, sinusoid_table)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, encoder_inputs, decoder_inputs):
        # |encoder_inputs| : (batch_size, encoder_inputs_len(=seq_len))
        # |decoder_inputs| : (batch_size, decoder_inputs_len(=seq_len-1))
        
        encoder_outputs, encoder_attns = self.encoder(encoder_inputs)
        # |encoder_outputs| : (batch_size, encoder_inputs_len, d_model)
        # |encoder_attns| : [(batch_size, n_heads, encoder_inputs_len, encoder_inputs_len)] * n_layers

        decoder_outputs, decoder_attns, enc_dec_attns = self.decoder(decoder_inputs, encoder_inputs, encoder_outputs)
        # |decoder_outputs| : (batch_size, decoder_inputs_len, d_model)
        # |decoder_attns| : [(batch_size, n_heads, decoder_inputs_len, decoder_inputs_len)] * n_layers
        # |enc_dec_attns| : [(batch_size, n_heads, decoder_inputs_len, encoder_inputs_len)] * n_layers
        
        outputs = self.linear(decoder_outputs)
        # |outputs| : (batch_size, decoder_inputs_len, tgt_vocab_size)
        
        return outputs, encoder_attns, decoder_attns, enc_dec_attns
    
    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i//2)) / d_model)
        
        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i%2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)