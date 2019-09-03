class Vocab(object):
    def __init__(self, list_of_tokens, unk_token=None, bos_token=None, eos_token=None,
                 pad_token=None, min_freq=1, lower=True):
        self.list_of_tokens = list_of_tokens
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.min_freq = min_freq
        self.lower = lower
        self.embedding_weights = None
        self.stoi, self.itos, self.freqs = {}, {}, {}

        # Initialize with special tokens
        for sti, special_token in enumerate([self.unk_token, self.bos_token, self.eos_token, self.pad_token]):
             if special_token: 
                 self.stoi[special_token] = sti
                 self.itos[sti] = special_token

    def build(self):
        # If the token doesn't appear in the vocabulary at least once
        for ti, token in enumerate(self.list_of_tokens):
            # Lowercase the token
            if self.lower:
                token = token.lower()
            
            # Count the frequencies of tokens in whole list of tokens
            if token not in self.freqs.keys():
                self.freqs[token] = 1
            else:
                self.freqs[token] += 1
        
        # Sort by frequency in 'descending' order
        self.freqs = dict(sorted(self.freqs.items(), key=lambda x: x[1], reverse=True))
        
        # Minimum frequency required for a token
        unique_tokens = []
        for token, freq in self.freqs.items():
            if freq >= self.min_freq:
                unique_tokens.append(token)

        # Build vocab mapping tokens to numerical index
        for token in unique_tokens:
            self.itos[self.__len__()] = token
            self.stoi[token] = self.__len__()

    def from_pretrained(self, pretrained_vectors):
        import numpy as np

        vector_size = len(list(pretrained_vectors.values())[0])
        self.embedding_weights = np.zeros((len(self.stoi), vector_size))
        for token, ti in self.stoi.items():
            vector = pretrained_vectors.get(token)
            if vector is not None:
                self.embedding_weights[ti] = vector

    def __len__(self):
        return len(self.stoi)

class Tokenizer(object):
    def __init__(self, tokenization_fn, vocab=None, is_sentence=False, max_seq_length=512):
        self.tokenization_fn = tokenization_fn
        self.vocab = vocab
        self.is_sentence = is_sentence
        self.max_seq_length = max_seq_length
        if not self.is_sentence:
            from nltk.tokenize import sent_tokenize
            self.sent_tokenization_fn = sent_tokenize
        
    def _vocab_fn(self, tokens):
        # Add beginning of sentence token
        if self.vocab.bos_token:
            tokens = [self.vocab.bos_token] + tokens 

        # Add end of sentence token
        if self.vocab.eos_token:    
            tokens = tokens + [self.vocab.eos_token]

        return tokens

    def tokenize(self, text):
        # Tokenize
        if self.is_sentence: 
            tokens = self.tokenization_fn(text)
            if self.vocab:
                tokens = self._vocab_fn(tokens)
        else:
            sentences = self.sent_tokenization_fn(text)
            tokens = []
            for sentence in sentences:
                _tokens = self.tokenization_fn(sentence)
                if self.vocab:
                    _tokens = self._vocab_fn(_tokens)
                tokens += _tokens

        # Truncate to the maximum sequence length
        if len(tokens) > self.max_seq_length:    
            tokens = tokens[:self.max_seq_length]

        if self.vocab:
            # Add padding token
            if self.vocab.pad_token and len(tokens) < self.max_seq_length:
                tokens += [self.vocab.pad_token] * (self.max_seq_length-len(tokens))
                
            # Lowercase the token
            if self.vocab.lower:
                tokens = [token.lower() for token in tokens]

        return tokens
    
    def transform(self, tokens):
        if self.vocab:
            return [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi[self.vocab.unk_token] for token in tokens]
    
    def inverse_transform(self, indices):
        if self.vocab:
            return [self.vocab.itos[index] for index in indices]

    def tokenize_and_transform(self, text):
        if self.vocab:
            return self.transform(self.tokenize(text))