# Neural Machine Translation
Translation with a sequence-to-sequence (seq2seq) network and attention.

## Model Overview
As you already know, a Recurrent Neural Network, or RNN, is a network that operates on a sequence and uses its own output as input for subsequent steps.
A seq2seq network(model), or Encoder-Decoder network, is a model consisting of two RNNs called the encoder and decoder. The encoder reads an input sequence and outputs a single vector, and the decoder reads that vector to produce an output sequence.

<p align="left">
<img width="700" src="https://github.com/lyeoni/nlp-tutorial/blob/master/neural-machine-translation/images/seq2seq.png">
</p>

Unlike sequence prediction with a single RNN, where every input corresponds to an output, the seq2seq model frees us from sequence length and order, which makes it ideal for translation between two languages.

Consider a pair of sentences "Je ne suis pas le chat noir" and "I am not the black cat". Most of the words in the input sentence have a direct translation in the output sentence, but are in slightly different orders. Because of the language construction, there is also one more word in the input sentence. It would be difficult to produce a correct translation directly from the sequence of input words.

With a seq2seq model, the encoder creates a single vector which, in the ideal case, encodes the "meaning" of the input sequence into a single vector - a single point in some N dimensional space of sentences.

That is, to find model parameters that maximize the probability of returning target sentence Y by receiving source sentence X. And we can find the model parameters(theta) that maximize P(Y|X,theta) using Maximum Likelihood Estimation(MLE). The formula for seq2seq model is as follows.
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\fn_cm&space;\theta&space;^{*}&space;\approx&space;argmaxP(Y|X;\theta)&space;\;&space;where&space;\;&space;X&space;=&space;\left&space;\{&space;x_{1},&space;x_{2},&space;...,&space;x_{n}&space;\right&space;\},&space;Y&space;=&space;\left&space;\{y_{1},&space;y_{2},&space;...,&space;y_{n}&space;\right&space;\}" title="\theta ^{*} \approx argmaxP(Y|X;\theta) \; where \; X = \left \{ x_{1}, x_{2}, ..., x_{n} \right \}, Y = \left \{y_{1}, y_{2}, ..., y_{n} \right \}" />
</p>


### Encoder
The encoder of seq2seq network is a RNN that outputs some value for every word from the input sentence. For every input word, the encoder outputs a `(context) vector` and `hidden state`. The last hidden state of the encoder would be a initial hidden state of decoder.

### Decoder
The decoder is another RNN that takes the encoder output vector(s) and outputs a sequence of words to create the translation.

This last output is sometimes called the context vector as it encodes context from the entire sequence. **This context vector is used as the initial hidden state of the decoder**. At every step of decoding, the decoder is given an input token and hidden state. The initial input token is the start of string <SOS> token, and the first hidden state is the context vector (= the encoder's last hidden state). The decoder continues generating words until it outputs an <EOS> token, representing the end of the sentence.

We are also going to use an `Attention Mechanism` in our decoder to help it to pay attention to certain parts of the input when generating the output.

### Auto-regressive and Teacher Forcing
The training method of sequence-to-sequence network is different from the way of inference. The difference between training and inference is fundamentally due to the property **auto-regressive**.

#### Auto-regressive
`Auto-regressive`: infers (or predicts) the present value by reffering to its own values in the past. As shown below, the current time-step output value *y_t* is determined by both the encoder's input sentence (or sequence) *X* and the previous time-step output *(y_1,...y_t-1)*.
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\fn_cm&space;y_{i}&space;=&space;argmax_{y}P(y|X,&space;y_{<i})\;&space;where\;&space;y_{0}&space;=&space;SOS\;&space;token" title="y_{i} = argmax_{y}P(y|X, y_{<i})\; where\; y_{0} = SOS\; token" />
</p>
If our decoder makes false prediction in the past, it can lead to a larger false prediction. (e.g. incorrect sentence structure or length)

#### Teacher Forcing
So we train using a method called Teacher Forcing, which is the concept of using the real target outputs as each next input, instead of using the decoer's guess as the next input.
- `In training mode`, seq2seq network considers that we already know all answers (= the outputs of the last time-step).
- `In inference mode`, seq2seq network takes the input from the output of the last time-step.
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\fn_cm&space;\hat{y}&space;=&space;argmax_{y}P(y|X,&space;y_{<t};\theta)\;&space;where\;&space;X&space;=&space;{x_{1},...,x_{n}}\;&space;and\;&space;Y=&space;{y_{0},...,y_{n}}" title="\hat{y} = argmax_{y}P(y|X, y_{<t};\theta)\; where\; X = {x_{1},...,x_{n}}\; and\; Y= {y_{0},...,y_{n}}" />
</p>
<p align="center">
<img width="600" src="https://github.com/lyeoni/nlp-tutorial/blob/master/neural-machine-translation/images/teacher-forcing.png" />
</p>

### Attention
If only the context vector is passed between the encoder and decoder, that single vector carries the burden of encoding the entire sentence.
Attention allows the decoder network to "focus" on a different part of the encoder's output for every step of the decoder's own outputs.

First, we calculate a set of attention weights. These will be multiplied by the encoder output vecotrs to create a weighted combination. The result should contain information about that specific part of the input sequence, and thus help the decoder choose the right output words.

Calculating the attention weights is done with another feed-forward layer, using the `decoder's input` and `hidden state` as inputs. Because there are sentences of all sizes in the training data, to actually create and train this layer, we have to choose a maximum sentence length (input length, for encoder outputs) that it can apply to. Sentences of the maximum length will use all the attention weights, while shorter sentences will only use the first few.

## Usage

## Evaluation

### Evaluation Overview

#### Dataset
- `training-set`
  - 130,143 sentence pairs.
  - Counted words: 20,391 (french), 12,362 (english)
- `test-set`
  - 39,042 sentence pairs.

#### Models
The model was trained with NVIDA Tesla K80, and the number of epochs was 7 (i.e. ~ 10 hours).

- Baseline(base): Simple Sequence to Sequence model
- Reverse: Apply Bi-directional LSTM to the encoder part
- Embeddings: Apply Fasttext word embeddings (300D)
- Attention: Apply attention mechanisms to the decoder part

### Extrinsic Evaluation

Below table shows the BLEU from various models in French-English translation task.

cf. Our dataset includes a small amount of sentences that is relatively short(maximum length is 15 words, including ending punctuation),
so it is recommended that the BLEU be considered as a reference only because it is excssively higher than the other experiments.

|MODEL|BLEU|
|:------|:-----:|
|Base-GRU|60.48|
|Base-LSTM|66.62|
|Reverse|66.12|
|Reverse + Embeddings|69.61|
|Reverse + Embeddings + Attention|74.47|

### Intrinsic Evaluation

Below table shows the results from various models in French-English translation task.

|Target|Base-GRU|Base-LSTM|Reverse|Reverse + Embeddings|Reverse + Embeddings + Attention|
|:------|:------|:------|:------|:------|:------|
|Go.|Go to|Go..|Go.|Go.|Go.|
|I have to go.|I must go.|I must leave.|I have to go.|I must go.|I need to go.|
|Calm down son.|Get it to my son.|Get my son.|Lie down.|Hold to my son.|Let's son my son.|
|I can't feel it.|I don't buy that.|I can't do it.|I don't deserve it.|I can't stand that.|I can't feel it.|
|How're you doing?|How are you doing?|How are you going?|How are you going?|How're you doing?|How are you?|
|Do you want to go?|Do you want to go there|Do you want to go?|Do you want to go?|Do you want to go?|Do you want to go?|
|Where's your coat?|Where's your coat?|where's your coat?|Where's your coat?|Where's your coat?|Where's your coat?|
|That isn't complex.|This isn't all.|It's not complicated.|It's not complicated.|It's not complicated.|It's not complex.|
|I pretended to work.|I pretended to work.|I work from work.|I've been working to work.|I pretended to work.|I pretended to work.|
|Come back in an hour.|Get an hour in a.|Get back into a hour.|come on a a!|Come back in an hour.|Come back an hour.|

## References
- [[Loung et al.2015](https://arxiv.org/pdf/1508.04025.pdf)] Effective Approaches to Attention-based Neural Machine Translation
- [[카카오 AI리포트](https://brunch.co.kr/@kakao-it/155)] 신경망 번역 모델의 진화 과정
- [[kh-kim/simple-nmt](https://github.com/kh-kim/simple-nmt)] Simple Neural Machine Translation (Simple-NMT)
- [[spro/practical-pytorch](https://github.com/spro/practical-pytorch)] Practical PyTorch
- [[spro/practical-pytorch](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model)] Translation with a Sequence to Sequence Network and Attention

