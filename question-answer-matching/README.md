# Question Answer Matching
This repository provides a simple PyTorch implementation of Question Answer Matching. Here we use the corpus from Stack Exchange in English to build embeddings for entire questions. Using those embeddings, we find similar questions for a given question, and show the corresponding answers to those I found.

## Model Overview

### TF-IDF (Term-Frequency - Inverse Documnet Frequency)
The _TF-IDF_ is usually used to find how relevant a term is in a document, and the _TF-IDF_ value is the product of two statistics, _Term-Frequency (TF) and Inverse Documnet Frequency (IDF)._ 
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?tfidf(t,&space;d,&space;D)&space;=tf(t,d)&space;\times&space;idf(t,&space;D)" title="tfidf(t, d, D) =tf(t,d) \times idf(t, D)" />
</p>

- `TF`: _Term Frequency_, which is a value that indicates how frequently a particular term occurs in a document. And the higher it is, the more relevant it is in the document. In other words, if a term occurs more times than other terms in a document, the term has more relevance than other terms for the document.
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?tf(t,&space;d)&space;=&space;\frac{Number\;&space;of\;&space;times\;&space;term\;&space;t\;&space;appears\;&space;in\;&space;a\;&space;document\;}{Total\;&space;number\;&space;of\;&space;terms\;&space;in\;&space;the\;&space;document}" title="tf(t, d) = \frac{Number\; of\; times\; term\; t\; appears\; in\; a\; document\;}{Total\; number\; of\; terms\; in\; the\; document}" />
</p>

- `IDF`: _Inverse Document Frequency_, which is the inverse of _Document Frequency (DF)_. _DF_ measures how much information a term provides, i.e., if it's common or rare across all documents. 
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?idf(t,&space;D)&space;=&space;\log_{&space;}\frac{Total\;&space;number\;&space;of\;&space;documents\;&space;D}{Number\;&space;of\;&space;documents\;&space;with\;&space;term\;&space;t\;&space;in\;&space;it}" title="idf(t, D) = \log_{ }\frac{Total\; number\; of\; documents\; D}{Number\; of\; documents\; with\; term\; t\; in\; it}" />
</p>

**The higher the term frequency within a particular document, and the smaller the document containing the term in the whole document, the higher the TF-IDF value.** Therefore, this value provides the effect of filtering out common words from all documents. The more terms a document contains, the closer the value of the log function is to 1, in which case the IDF value and TF-IDF value are closer to 0.

#### Example
Consider that there are two documents as follows:
- document1 = "a new car, used car, car review"
- documnet2 = "a friend in need is a friend indeed"

|word|TF|TF|IDF|TF-IDF|TF-IDF|
|:-:|:-:|:-:|:-:|:-:|:-:|
||document1|document2||document1|document2|
|a|1/7|2/8|log(2/2) = 0|0|0|
|new|1/7|0|log(2/1) = 0.3|0.04|0|
|car|3/7|0|log(2/1) = 0.3|0.13|0|
|used|1/7|0|log(2/1) = 0.3|0.04|0|
|review|1/7|0|log(2/1) = 0.3|0.04|0|
|friend|0|2/8|log(2/1) = 0.3|0|0.08|
|in|0|1/8|log(2/1) = 0.3|0|0.04|
|need|0|1/8|log(2/1) = 0.3|0|0.04|
|is|0|1/8|log(2/1) = 0.3|0|0.04|
|indeed|0|1/8|log(2/1) = 0.3|0|0.04|

### LSTM with variable-length seqeucnes
To feed the variable-length sequences to recurrent network such as GRU, LSTM in PyTorch, we need to follow below step. 

`padding -> pack sequence -> recurrent network -> unpack sequence`

And to pack/unpack the sequence easily, PyTorch provides us with two useful methods: `pack_padded_sequence`, `pad_packed_sequence`.

- `pack_padded_sequence`: *Packs a tensor containing padded_sequences of variable length*. The sequences should be sorted by length in a decreasing order, i.e. `input[:,0]` should be the longest sequence, and `input[:,-1]` the shortest one.
  - Input: a tensor of size `T x B x *`, where `T` is the length of the longest sequence(equal to first element of list containing sequence length), `B` is the patch size, and `*` is any number of dimensions (including 0). If `batch_first` argument is True, the input is expected in `B x T x *` format.
  - Returns: `PackedSequence` object.

[<p align="center"><img width= 500 src="https://cdn-images-1.medium.com/max/800/1*XmYVloKMe17nwf747z_CPQ.jpeg"></p>](https://medium.com/@sunwoopark/show-attend-and-tell-with-pytorch-e45b1600a749)

- `pad_packed_sequence`: *Pads a packed batch of variable length sequences*. It's an inverse operation to `pack_padded_sequence`. 
  - Input: `PackedSequence` object.
  - Returns: Tuple of tensor containing the padded sequence, and a tensor containing the list of lengths of each sequence in the batch. The returned tensor's data will be of size `T x B x *`, where `T` is the length of the longest sequence and `B` is the batch size. If `batch_first` argument is True, the data will be transposed into `B x T x *` format. Batch elements will be ordered decreasingly by their length.

- `PackedSequence`: Holds the `data` and list of `batch_sizes` of a packed sequence. All RNN moduels accept packed sequences as inputs. The data tensor contains packed seqeunce, and the batch_sizes tensor contains integers holding information about the batch size at each seqeunce step.
  - For instance, given data 'abc' and 'x', the PackedSequence would contain 'axbc' with batch_sizes=[2,1,1].

### CBoW (Continuous Bag of Words)
To represent a sentence, here we used CBoW. CBoW can be defined as follows:

- Ignore the order of the tokens.
- Simply average the token vectors.
  - Averaging is a differentiable operator.
  - Just one operator node in the DAG(Directed Acyclic Graph).
- Generalizable to bag-of-n-grams.
  - N-gram: a phrase of N tokens.

<p align="center"><img width= 700 src="https://github.com/lyeoni/nlp-tutorial/blob/master/question-answer-matching/data/images/cbow.png"></p>

CBoW is extremely effective in text classification. For instance, if there are many positive words, the review is likely positive.

## Usage

### 1. Data loading
Because the data from Stack Exchange has been saved as xml, we first install **beautifulsoup4** and **lxml parser**. You can easily install by running following commands.
```
$ pip install beautifulsoup4
$ pip install lxml
```

To load the question/answer text from xml file, run the dataLoader python script below.

example usage:
```
$ python dataLoader.py
```

The dataset contains 91,517 records, and each record contains 5 attributes: **title(question)**, **body(answer)**, tags , post type id, view count. Here we will mainly use title, body. Below table shows that the first 5 lines of our dataset.

<p align="left">
<img src="https://github.com/lyeoni/nlp-tutorial/blob/master/question-answer-matching/data/images/result-dataloader.png" />
</p>

### 2. Preprocessing
Preprocessing consists of largely 3 steps: 
- Text cleaning/normalization
- Tokenization
- Build TF-IDF and word embedding matrix with pre-trained word representations

The tokenization and building TF-IDF/embedding matrix used here, is not much different from that used other nlp tasks. 
But, as shown in the results of above data loading step, the html tags and urls (i.e. \<p>, \<a href=https://~>) exist in title and body columns. To clean it up and normalize our data, run preprocessing script below.

example usage:
```
$ python preprocessing.py
```

Below table shows that the first 5 lines of preprocessing results. We can see that all the html tags have disappeared.

<p align="left">
<img src="https://github.com/lyeoni/nlp-tutorial/blob/master/question-answer-matching/data/images/result-preprocessing.png" />
</p>

### 3. Training

```
$ python train.py -h
usage: train.py [-h] [--filename FILENAME] [--clean_drop CLEAN_DROP]
                [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--learning_rate LEARNING_RATE] [--hidden_size HIDDEN_SIZE]
                [--n_layers N_LAYERS] [--dropout_p DROPOUT_P]

optional arguments:
  -h, --help            show this help message and exit
  --filename FILENAME
  --clean_drop CLEAN_DROP
                        Drop if either title or body column is NaN
  --epochs EPOCHS       Number of epochs to train. Default=7
  --batch_size BATCH_SIZE
                        Mini batch size for gradient descent. Default=2
  --learning_rate LEARNING_RATE
                        Learning rate. Default=.001
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM. Default=64
  --n_layers N_LAYERS   Number of layers. Default=1
  --dropout_p DROPOUT_P
                        Dropout ratio. Default=.1
```

example usage:
```
$ python train.py --epochs 15 --batch_size 2 --learning_rate .001 --hidden_size 64 --n_layers 1 --dropout_p .1
```
You may need to change the argument parameters.

### 4. Evaluation
```
$ python evaluate.py -h
usage: evaluate.py [-h] --model MODEL [--filename FILENAME]
                   [--clean_drop CLEAN_DROP] [--hidden_size HIDDEN_SIZE]
                   [--n_layers N_LAYERS] [--dropout_p DROPOUT_P]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model file(.pth) path to load trained model's learned
                        parameters
  --filename FILENAME
  --clean_drop CLEAN_DROP
                        Drop if either title or body column is NaN
  --hidden_size HIDDEN_SIZE
                        Hidden size of LSTM. Default=64
  --n_layers N_LAYERS   Number of layers. Default=1
  --dropout_p DROPOUT_P
                        Dropout ratio. Default=.1
```

example usage:
```
$ python evaluate.py --model model.pth
```
You may need to change the argument parameters.

### Evaluation

### Dataset
 - `training-set`
   - 60,114 question-answer pairs.
   - With sampling, we doubled it: 30,057 pairs to 60,114 pairs.
 - `test-set`
   - 3,356 question-answer pairs

### Models
The models were trained with NVIDIA Tesla K80, and the number of epochs was 10.

### Intrinsic Evaluation

Below table shows the results from the model in question matching task. If a sample question is given, question matching model tries to find the closest question based on cosine similarity (cf. If we have corresponding answers to the closest questions given, we can answer all the questions.).

|Given sample question|nearest 1|nearest 2|nearest 3|
|------|------|------|------|
|Can you travel on an expired passport?|How serious is an expired passport? _(similarity=.96)_|What should I do with my expired passport? _(similarity=.96)_|Can I use a study visa in an expired passport? _(similarity=.92)_|
|Can I carry comics with me while traveling from the USA to india?|Can I take two laptops to india from united states? one bought in india and one in US. _(similarity=.86)_|Can I carry mobile phones to US from india? _(similarity=.85)_|Can I bring laptops to india from the US? _(similarity=.85)_|
|Do I need transit visa if I have to recheck my checked in baggage for a layover in dubai?|requirements for the transit visa and baggage information for layover in dubai. _(similarity=.96)_|Do I need a transit visa to collect and re-check my luggage at istanbul ataturk airport? _(similarity=.96)_|Do I need a transit visa for an hour layover in dubai? _(similarity=.94)_|
|What happens if I'm forced to overstay in the U.S. Because my flight is delayed or cancelled?|Returning plane ticket if delayed in the USA. _(similarity=.87)_|Is airline obliged to refund cost of flight if the passenger is unable to fly because his travel visa has been rejected? _(similarity=.88)_|What rights do I have if my flight is cancelled? _(similarity=.86)_|
|Are campsites always free on canary islands?|Where can I find authentic areas in canary islands? _(similarity=.93)_|Safe typical dishes to order away from the tourist trail in thailand when english is not supported? _(similarity=.89)_|Is there any region island mountain or village in japan known for hot spicy local cuisine? _(similarity=.89)_|
|Renting a car in israel when under?|Car rental in israel without additional insurance _(similarity=.86)_|What are the rules for renting a car in italy? _(similarity=.85)_|What should I be aware of when hiring a car from one of the cheaper rental firms in spain? _(similarity=.87)_|
|What does a technical stop mean in air travel?|What does it mean when a flight is delayed due to a tail swap if anything? _(similarity=.92)_|Is there a way to check that the conditions on your plane ticket are actually what your travel agent said they are? _(similarity=.92)_|Can you earn miles with different airlines for the same one flight if they are part of the same loyalty program? _(similarity=.91)_|
|Do I need a transit visa for paris when travelling to italy with a schengen visa?|Do I need a transit visa through italy from romania to algeria? _(similarity=.93)_|Do I need a transit visa for paris if i have a schengen visa issued by portugal? _(similarity=.92)_|Do I need a transit visa for frankfurt if i have a schengen italian visa? _(similarity=.92)_
|Travel to italy via germany using us refugee travel document|Travel to italy with romanian travel document. _(similarity=.79)_|Do I need transit visa for germany travelling from india to poland via germany with polish d type national visa? _(similarity=.79)_|Schengen visa from italy without visiting italy. _(similarity=.77)_|
|Need to travel to the USA by ship from europe.|Do you need visa to connect in USA from Russia? _(similarity=.74)_|Can I travel to USA from a foreign country _(similarity=.73)_|Should I convert US dollars to euros while in USA or in Europe? _(similarity=.72)_|

## References
- [[Himanshu](https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130)] Sentiment Analysis with Variable length sequences in Pytorch
- [[William Falcon](https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e)] Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health
- [[PyTorch](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence)] PyTorch official document - package reference - torch.nn
- [[Sunwoo Park](https://medium.com/@sunwoopark/show-attend-and-tell-with-pytorch-e45b1600a749)] Show, Attend, and Tell with Pytorch
- [[ediwth, Kyunghyun Cho](https://www.edwith.org/deepnlp/lecture/29208/)] CBoW & RN & CNN
- [[Minsuk Heo](https://www.youtube.com/watch?v=meEchvkdB1U)] [딥러닝 자연어처리] TF-IDF
- [[DOsinga/deep_learning_cookbook](https://github.com/DOsinga/deep_learning_cookbook/blob/master/06.1%20Question%20matching.ipynb)] 06.1 Question matching

