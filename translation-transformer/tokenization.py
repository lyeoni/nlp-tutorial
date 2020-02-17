from typing import List
from collections import OrderedDict

from prenlp.tokenizer import SentencePiece

class Tokenizer:
    def __init__(self, tokenizer, vocab_file: str,
                 pad_token: str = '[PAD]',
                 unk_token: str = '[UNK]',
                 bos_token: str = '[BOS]',
                 eos_token: str = '[EOS]'):
        self.tokenizer = tokenizer
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab = OrderedDict()
        self.ids_to_tokens = OrderedDict()

        # Build vocab and ids_to_tokens
        with open(vocab_file, 'r', encoding='utf-8') as reader:
            for i, line in enumerate(reader.readlines()):
                token =line.split()[0]
                self.vocab[token] = i
        for token, id in self.vocab.items():
            self.ids_to_tokens[id] = token

    def tokenize(self, text: str) -> List[str]:
        """Tokenize given text.
        """
        return self.tokenizer(text)

    def convert_token_to_id(self, token: str) -> int:
        """Convert a token (str) in an id (integer) using the vocab.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def convert_id_to_token(self, id: int) -> str:
        """Convert an id (integer) in a token (str) using the vocab.
        """
        return self.ids_to_tokens.get(id, self.unk_token)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens in list of ids using the vocab.
        """
        return [self.convert_token_to_id(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert list of ids in list of tokens using the vocab.
        """
        return [self.convert_id_to_token(id) for id in ids]
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size.
        """
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        """Id of pad_token in the vocab.
        """
        return self.convert_token_to_id(self.pad_token)
    
    @property
    def unk_token_id(self) -> int:
        """Id of unk_token in the vocab.
        """
        return self.convert_token_to_id(self.unk_token)
    
    @property
    def bos_token_id(self) -> int:
        """Id of bos_token in the vocab.
        """
        return self.convert_token_to_id(self.bos_token)
    
    @property
    def eos_token_id(self) -> int:
        """Id of eos_token in the vocab.
        """
        return self.convert_token_to_id(self.eos_token)

class PretrainedTokenizer(Tokenizer):
    def __init__(self, pretrained_model: str, vocab_file: str,
                 pad_token: str = '[PAD]',
                 unk_token: str = '[UNK]',
                 bos_token: str = '[BOS]',
                 eos_token: str = '[EOS]'):
        tokenizer = SentencePiece.load(pretrained_model)

        super(PretrainedTokenizer, self).__init__(tokenizer, vocab_file, pad_token, unk_token, bos_token, eos_token)

    def detokenize(self, tokens: List[str]) -> str:
        return self.tokenizer.detokenize(tokens)