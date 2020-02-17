from typing import List
import random

import torch
from torch.utils.data import TensorDataset

class InputExample:
    """A single training/test example for machine translation.
    """
    def __init__(self, src_text: str, tgt_text: str):
        self.src_text = src_text
        self.tgt_text = tgt_text

class InputFeatures:
    """A single set of features of data.
    """
    def __init__(self, encoder_input_ids: List[int], decoder_input_ids: List[int], decoder_output_ids: List[int]):
        self.encoder_input_ids = encoder_input_ids
        self.decoder_input_ids = decoder_input_ids
        self.decoder_output_ids = decoder_output_ids

def convert_examples_to_features(examples, tokenizer_src, tokenizer_tgt, max_seq_len):
    # pad_token_id in tokenizer_tgt.vocab should be the same with this.
    pad_token_id = tokenizer_src.pad_token_id
    bos_token_id = tokenizer_tgt.bos_token_id
    eos_token_id = tokenizer_tgt.eos_token_id

    features = []
    for i, example in enumerate(examples):
        src_tokens = tokenizer_src.tokenize(example.src_text)
        tgt_tokens = tokenizer_tgt.tokenize(example.tgt_text)

        src_tokens = src_tokens[:max_seq_len]
        tgt_tokens = tgt_tokens[:max_seq_len-2] # BOS, EOS
        
        src_ids = tokenizer_src.convert_tokens_to_ids(src_tokens)
        tgt_ids = [bos_token_id] + tokenizer_tgt.convert_tokens_to_ids(tgt_tokens) + [eos_token_id]
        
        padding_length = max_seq_len - len(src_ids)
        src_ids = src_ids + ([pad_token_id] * padding_length)
        padding_length = max_seq_len - len(tgt_ids)
        tgt_ids = tgt_ids + ([pad_token_id] * padding_length)

        feature = InputFeatures(encoder_input_ids = src_ids,
                                decoder_input_ids = tgt_ids[:-1],
                                decoder_output_ids = tgt_ids[1:])
        features.append(feature)

    return features

def create_examples(args, tokenizer_src, tokenizer_tgt, mode = 'train', split_ratio=0.1, random_seed=42):
    random.seed(random_seed)

    dataset = []
    with open(args.dataset, 'r', encoding='utf-8') as reader:
        for line in reader.readlines():
            text_pair = list(map(lambda x: x.strip(), line.split('\t')))
            text_pair = list(reversed(text_pair)) # French to English
            dataset.append(text_pair)
    
    random.shuffle(dataset)
    if mode == 'train':
        dataset = dataset[:int(len(dataset)*(1-split_ratio))]
    elif mode == 'test':
        dataset = dataset[int(len(dataset)*(1-split_ratio)):]
        
    examples =[]
    for src_text, tgt_text in dataset:
        example = InputExample(src_text, tgt_text)
        examples.append(example)
    
    features = convert_examples_to_features(examples, tokenizer_src, tokenizer_tgt, args.max_seq_len)
    
    encoder_input_ids = torch.tensor([feature.encoder_input_ids for feature in features], dtype=torch.long)
    decoder_input_ids = torch.tensor([feature.decoder_input_ids for feature in features], dtype=torch.long)
    decoder_output_ids = torch.tensor([feature.decoder_output_ids for feature in features], dtype=torch.long)

    dataset = TensorDataset(encoder_input_ids, decoder_input_ids, decoder_output_ids)
    
    return dataset