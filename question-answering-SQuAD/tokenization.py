import sys
import pandas as pd
from nltk.tokenize.moses import MosesTokenizer
import data_loader

if __name__ == "__main__":
    loader = data_loader.DataLoader(sys.argv[1])
    data = pd.DataFrame({'title': loader.title, 'context': loader.context, 'question':loader.question,
                                                   'answer_start':loader.answer_start, 'answer_end':loader.answer_end, 'answer_text':loader.answer_text})
    
    # make tokenizer
    tokenizer = MosesTokenizer()
    
    # tokenization
    for text_column in data.columns:
        if text_column in ['context' , 'question', 'answer_text']:
            # drop duplicated context, question, answer text
            for line in data[text_column].drop_duplicates(): 
                tokens = tokenizer.tokenize(line.replace('\n', '').strip(), escape=False)
                sys.stdout.write(' '.join(tokens) +'\n')
            else:
                sys.stdout.write('\n')