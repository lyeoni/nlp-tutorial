# Text Classification with Transformer
This text classification tutorial trains a Transformer model on the IMDb movie review dataset for sentiment analysis.

## Model Overview: Transformer
<p align="center"><img width= 400 src="https://pytorch.org/tutorials/_images/transformer_architecture.jpg"></p>

The transformer model(based on the paper Attention is All You Need) follows the same general pattern as a standard sequence to sequence with attention model.

The input sentence is passed through N encoder layers that generates an output for each word/token in the sequence. The decoder attends on the encoder's output and its own input (self-attention) to predict the next word.

The transformer model has been proved to be superior in quality for many sequence-to-sequence problems while being more parallelizable.

Here, we are going to do sentiment analysis that is not sequence-to-sequence problem. So, only use transformer encoder.

### Positional Encoding
Since this model doesn't contain any recurrence or convolution, positional encoding is added to give the model some information about the relative position of the words in the sentence.

The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be closer to each other. But the embeddings do not encode the relative position of words in a sentence. So after adding the positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sentence, in the d-dimensional space.

The formula for calculating the positional encoding is as follows:
<p align="center"><img width= 450 src="https://2.bp.blogspot.com/-qQM2StMXbnI/XfgQvt0kJwI/AAAAAAAAB1U/vK0l9KdK-1wQWBimKTTxasjRFQSmmylhwCLcBGAsYHQ/s1600/pose.png"></p>

```python
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
```

### Encoder
The input sentence is passed through encoder(consists of N encoder layers) that generates an output for each word/token in the sequence.

Each encoder layer consists of two sub-layers:
- Multi-head attention (with padding mask)
- Pointwise feed forward networks

### Multi-Head Attention
<p align="center"><img width= 450 src="https://3.bp.blogspot.com/-mnlTQLXKuiU/XfgOSZ2eBsI/AAAAAAAAB0w/6jjXEtzO6_M1IlPkNVzR_wcmP62u0jI0ACLcBGAsYHQ/s1600/attention.png"></p>
Multi-head attention consists of four parts:

1. Linear layers and split into heads
2. Scaled dot-product attention (with padding mask)
3. Concatenation of heads
4. Final linear layer

#### 1. Linear layers and split into heads
```python
q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) 
v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2) 
# |q_heads| : (batch_size, n_heads, q_len, d_k), |k_heads| : (batch_size, n_heads, k_len, d_k), |v_heads| : (batch_size, n_heads, v_len, d_v)
```

#### 2. Scaled dot-product attention (with padding mask)
```python
attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
# |attn_mask| : (batch_size, n_heads, seq_len(=q_len), seq_len(=k_len))
attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads, attn_mask)
# |attn| : (batch_size, n_heads, q_len, d_v)
# |attn_weights| : (batch_size, n_heads, q_len, k_len)
```

#### 3. Concatenation of heads
```python
attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
# |attn| : (batch_size, q_len, n_heads * d_v)
```

#### 4. Final linear layer
```python
output = self.linear(attn)
# |output| : (batch_size, q_len, d_model)
```

You can see full multi-head attention code [here](https://github.com/lyeoni/nlp-tutorial/blob/430010c826439d03677d5498406c3477b0a4e678/text-classification-transformer/model.py#L26).

### Pointwise feed forward networks
<p align="center"><img width= 300 src="https://pozalabs.github.io/assets/images/ffn.png"></p>

Pointwise feed forward network consists of two fully-connected layers with a ReLU activation in between.

```python
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
```

### Encoder Layer
Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections help in avoiding the vanishing gradient problem in deep networks.

The output of each sublayer is `LayerNorm(x + Sublayer(x))`. The normalization is done on the d_model (last) axis. There are N encoder layers in the transformer.
```python
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
        # |attn_outputs| : (batch_size, seq_len(=q_len), d_model)
        # |attn_weights| : (batch_size, n_heads, q_len, k_len)

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        # |ffn_outputs| : (batch_size, seq_len, d_model)
        
        return ffn_outputs, attn_weights
```

## Usage

### 0. Install PreNLP library
PreNLP is Preprocessing Library for Natural Language Processing. Using this, we will load and preprocess our IMDb dataset.
```shell
$ pip install prenlp
```

### 1. Setup input pipeline

#### Building vocab based on WikiText-103 corpus
You can easily download WikiText-103 corpus using below command.
```shell
$ python -c "import prenlp; prenlp.data.WikiText103()"
$ ls .data/wikitext-103/
wiki.test.tokens  wiki.train.tokens  wiki.valid.tokens
```

Build Vocabulary based on WikiText-103 corpus, using sentencepiece subword tokenizer.
```shell
$ python vocab.py --corpus .data/wikitext-103/wiki.train.tokens --prefix wiki --tokenizer sentencepiece --vocab_size 16000
```

### 2. Train
```shell
$ python main.py --dataset imdb --vocab_file wiki.vocab --tokenizer sentencepiece --pretrained_model wiki.model
```
Out:
```
Namespace(batch_size=32, dataset='imdb', dropout=0.1, epochs=7, ffn_hidden=1024, hidden=256, lr=0.0001, max_seq_len=512, n_attn_heads=8, n_layers=6, no_cuda=False, output_model_prefix='model', pretrained_model='wiki.model', tokenizer='sentencepiece', vocab_file='wiki.vocab')
Iteration 156 (156/782) Loss: 0.6533 Acc: 62.680288%
Iteration 312 (312/782) Loss: 0.6050 Acc: 69.691506%
Iteration 468 (468/782) Loss: 0.5790 Acc: 72.863248%
Iteration 624 (624/782) Loss: 0.5619 Acc: 74.969952%
Iteration 780 (780/782) Loss: 0.5488 Acc: 76.426282%
Train Epoch: 1  >       Loss: 0.5479 / Acc: 76.3%
Valid Epoch: 1  >       Loss: 0.4983 / Acc: 81.8%
Iteration 156 (156/782) Loss: 0.4754 Acc: 85.837340%
Iteration 312 (312/782) Loss: 0.4713 Acc: 85.987580%
Iteration 468 (468/782) Loss: 0.4689 Acc: 85.990919%
Iteration 624 (624/782) Loss: 0.4686 Acc: 85.737179%
Iteration 780 (780/782) Loss: 0.4677 Acc: 85.633013%
Train Epoch: 2  >       Loss: 0.4672 / Acc: 85.5%
Valid Epoch: 2  >       Loss: 0.4665 / Acc: 85.0%
Iteration 156 (156/782) Loss: 0.4405 Acc: 89.883814%
Iteration 312 (312/782) Loss: 0.4359 Acc: 89.793670%
Iteration 468 (468/782) Loss: 0.4381 Acc: 89.189370%
Iteration 624 (624/782) Loss: 0.4387 Acc: 88.967348%
Iteration 780 (780/782) Loss: 0.4394 Acc: 88.766026%
Train Epoch: 3  >       Loss: 0.4389 / Acc: 88.7%
Valid Epoch: 3  >       Loss: 0.4600 / Acc: 85.5%
```

You may need to change below argument parameters.
```
$ python main.py -h
usage: main.py [-h] [--dataset DATASET] [--vocab_file VOCAB_FILE]
               [--tokenizer TOKENIZER] [--pretrained_model PRETRAINED_MODEL]
               [--output_model_prefix OUTPUT_MODEL_PREFIX]
               [--batch_size BATCH_SIZE] [--max_seq_len MAX_SEQ_LEN]
               [--epochs EPOCHS] [--lr LR] [--no_cuda] [--hidden HIDDEN]
               [--n_layers N_LAYERS] [--n_attn_heads N_ATTN_HEADS]
               [--dropout DROPOUT] [--ffn_hidden FFN_HIDDEN]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     dataset
  --vocab_file VOCAB_FILE
                        vocabulary path
  --tokenizer TOKENIZER
                        tokenizer to tokenize input corpus. available:
                        sentencepiece, nltk_moses
  --pretrained_model PRETRAINED_MODEL
                        pretrained sentencepiece model path. used only when
                        tokenizer='sentencepiece'
  --output_model_prefix OUTPUT_MODEL_PREFIX
                        output model name prefix
  --batch_size BATCH_SIZE
                        batch size
  --max_seq_len MAX_SEQ_LEN
                        the maximum size of the input sequence
  --epochs EPOCHS       the number of epochs
  --lr LR               learning rate
  --no_cuda
  --hidden HIDDEN       the number of expected features in the transformer
  --n_layers N_LAYERS   the number of heads in the multi-head attention
                        network
  --n_attn_heads N_ATTN_HEADS
                        the number of multi-head attention heads
  --dropout DROPOUT     the residual dropout value
  --ffn_hidden FFN_HIDDEN
                        the dimension of the feedforward network
```

### 3. Inference
```shell
$ python inference.py --model .model/model.ep7
```

Out:
```
...
class: tensor([1], device='cuda:0') # [0: neg, 1: pos]
```

### References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Sequence-to-Sequence Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer?hl=en)