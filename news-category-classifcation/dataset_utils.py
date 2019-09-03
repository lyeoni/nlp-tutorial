import pandas as pd
import torch
from torch.utils.data import Dataset

class Corpus(Dataset):
    def __init__(self, corpus_path, tokenizer, cuda):
        self.corpus = []
        self.ltoi, self.itol = {}, {} # {label_name : index}, {index : label_name}
        self.tokenizer = tokenizer
        self.cuda = cuda

        # Build Corpus dataset
        with open(corpus_path, 'r', encoding='utf8') as reader:
            for li, line in enumerate(reader):
                _line = line.split('\t')
                self.corpus.append([_line[0], ' '.join(_line[1:]).strip()])
        self.corpus = pd.DataFrame(self.corpus, columns=['label', 'text'])
        
        # Convert string label into int
        for i, label in enumerate(sorted(self.corpus['label'].unique())):
            self.ltoi[label] = i
            self.itol[i] = label
        self.corpus['label'] = self.corpus['label'].map(self.ltoi)
        
    def __getitem__(self, index):
        """
        Return inputs, targets tensors used for model training.
        """

        labels = self.corpus.iloc[index]['label']
        tokens_indices = self.tokenizer.tokenize_and_transform(self.corpus.iloc[index]['text'])
        
        labels = torch.tensor(labels)
        tokens_indices = torch.tensor(tokens_indices)
        if self.cuda:
            labels = labels.cuda()
            tokens_indices = tokens_indices.cuda()

        return tokens_indices, labels

    def __len__(self):
        return len(self.corpus)