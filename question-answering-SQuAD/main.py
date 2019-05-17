import json
import argparse
from konlpy.tag import Mecab
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, whitespace_tokenize
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
import run_squad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--bert',
                   default='bert-base-multilingual-cased',
                   help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased,"
                   "bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese.")

    p.add_argument('--train_file',
                   default='KorQuAD_v1.0_train.json',
                   help='json file for training.')

    p.add_argument('--dev_file',
                   default='KorQuAD_v1.0_dev.json',
                   help='json file for verification.')

    p.add_argument('--do_lower_case',
                   action='store_false',
                   default=False,
                   help="Whether to lower case the input text. True for uncased models, False for cased models.")

    p.add_argument("--max_seq_length",
                   type=int,
                   default=512, 
                   help="The maximum total input sequence length after WordPiece tokenization. "
                   "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")

    p.add_argument("--max_query_length",
                   type=int,
                   default=64,
                   help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.")
    
    config = p.parse_args()
    return config

class KorquadSample(object):
    # A single training/dev sample for the KOrquad dataset
    def __init__(self, paragraph_text, question_id, question_text, answer_text, answer_start, answer_end):
        self.paragraph_text = paragraph_text
        self.question_id = question_id
        self.question_text = question_text
        self.answer_text = answer_text
        self.answer_start = answer_start
        self.answer_end = answer_end

class SampleFeatures(object):
    # A single set of features of a sample
    def __init__(self, sample_index, tokens, answer_token_start, answer_token_end):
        self.sample_index = sample_index
        self.tokens = tokens
        self.answer_token_start = answer_token_start
        self.answer_token_end = answer_token_end

def read_samples(input_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    
    samples = []
    for i, data in enumerate(input_data): # data.keys(): dict_keys(['paragraphs', 'title'])
        for paragraph in data['paragraphs']: # paragraph.keys(): dict_keys(['qas', 'context'])
            paragraph_text = paragraph['context'] 
        
            for qa in paragraph['qas']: # qa.keys(): dict_keys(['answers', 'id', 'question'])
                question_id = qa['id']
                question_text = qa['question']
                answer_text = qa['answers'][0]['text']
                answer_start = qa['answers'][0]['answer_start']
                answer_end = answer_start + len(answer_text)-1
            
                sample = KorquadSample(paragraph_text=paragraph_text,
                                       question_id=question_id,
                                       question_text=question_text,
                                       answer_text=answer_text,
                                       answer_start=answer_start,
                                       answer_end=answer_end)
                samples.append(sample)
    return samples        

def get_exact_answer(sample, tokenizer, answer_orig_start, answer_orig_end, orig2token, tokens):
    '''
    simple approach using 'orig2token[token_index]' can't provide us exact answer span
    To deal with this, We should get the sub-token based index (such as token2orig input).
    example:
        answer_token_text: ['교', '##향', '##곡']
        orig2token-based: ['교', '##향', '##곡', '##을']
        token2orig-based: ['교', '##향', '##곡']
    promble: 현재 아래와 같이 조사?가 paragraph_text 내, answer_text에 붙어있는 경우는 분리시킬 수 없었음.
             word-piece tokenization 자체의 문제
        ['소', '##나', '##타', '형', '##식'] 소나타 형식
        ['소', '##나', '##타', '형', '##식으로']
    '''
    answer_token_text = tokenizer.tokenize(sample.answer_text)
    answer_token_start = orig2token[answer_orig_start]

    if answer_orig_end+1 < len(orig2token): 
        answer_token_end = orig2token[answer_orig_end+1]
        for new_start in range(answer_token_start, answer_token_end):
            for new_end in range(answer_token_end, answer_token_start, -1):
                if tokens[new_start:new_end] == answer_token_text:
                    answer_token_start, answer_token_end = new_start, new_end-1
    else:
        # answer token is located at the last of paragraph_text
        for ti, token in enumerate(answer_token_text):
            if tokens[answer_token_start:orig2token[answer_orig_end]+ti+1] == answer_token_text:
                answer_token_end = orig2token[answer_orig_end]+ti
                break
        else: # 12 partially exact examples
            answer_token_end = len(tokens)-1
    
    return answer_token_start, answer_token_end
    
def map_orig_to_token_index(sample, tokenizer):
    # map original - token index and save tokens tokenized by BertTokenizer
    answer_start, answer_end = 0, 0 # character-based index. similar to sample.answer_text, sample.answer_end
    answer_orig_start, answer_orig_end = 0, 0 # orig-based index. similar to element of orig2token
    orig2token, token2orig, tokens = [], [], []

    for ti, token in enumerate(sample.paragraph_text.split()):
        orig2token.append(len(tokens))
        for sub_token in tokenizer.tokenize(token):
            token2orig.append(ti)
            tokens.append(sub_token)  
        
        # get the answer orig-token index of start/end token
        if answer_start < sample.answer_start:
            answer_start += len(token) + 1
            answer_end = answer_start
            answer_orig_start = ti + 1
            continue

        if answer_end < sample.answer_end:
            answer_end += len(token) + 1 
            answer_orig_end = ti
    
    # get the token-based index corresponding answer start, end.
    # It sometimes occurs indexError
    answer_token_start, answer_token_end = get_exact_answer(sample, tokenizer, 
        answer_orig_start, answer_orig_end, orig2token, tokens)

    return orig2token, token2orig, tokens, answer_token_start, answer_token_end

def convert_samples_to_features(samples, tokenizer, max_seq_length, max_query_length):
    features = []
    for si, sample in enumerate(samples):

        # for query
        query_tokens = tokenizer.tokenize(sample.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]
       
        # for context
        try:
            orig2token, token2orig, tokens, answer_token_start, answer_token_end = map_orig_to_token_index(sample, tokenizer)
        except IndexError:
            continue
        
        # -3 accounts for '[CLS]', '[SEP]', '[SEP]'
        max_tokens_for_context = max_seq_length - len(query_tokens) - 3

        feature = SampleFeatures(sample_index = si,
                                 tokens = tokens,
                                 answer_token_start = answer_token_start,
                                 answer_token_end = answer_token_end)

        features.append(feature)
    
    print(len(features))
if __name__=='__main__':
    config = argparser()
    print(config)

    # read samples
    train_samples = read_samples(config.train_file)

    # preprocessing: convert samples to features
    tokenizer = BertTokenizer.from_pretrained(config.bert, do_lower_case=config.do_lower_case)
    convert_samples_to_features(samples=train_samples,
                                tokenizer=tokenizer,
                                max_seq_length=config.max_seq_length,
                                max_query_length=config.max_query_length)
    

    # build model
    # bert-base-multilingual-cased: (New, recommended) 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
    # model = BertForQuestionAnswering.from_pretrained(config.bert).to(device)
    # print(model.eval())
