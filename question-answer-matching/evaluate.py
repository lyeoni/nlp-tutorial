import argparse
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import dataLoader as loader
import preprocessing as pproc
import train as trainer
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model',
                    required=True,
                    help='Model file(.pth) path to load trained model\'s learned parameters')
    
    p.add_argument('--filename',
                    default='Posts.xml')
    
    p.add_argument('--clean_drop',
                    default=False,
                    help='Drop if either title or body column is NaN')

    p.add_argument('--hidden_size',
                    type=int,
                    default=64,
                    help='Hidden size of LSTM. Default=64')

    p.add_argument('--n_layers',
                    type=int,
                    default=1,
                    help='Number of layers. Default=1')

    p.add_argument('--dropout_p',
                    type=float,
                    default=.1,
                    help='Dropout ratio. Default=.1')

    config = p.parse_args()
    
    return config

def sentencesFromInexes(indexes, tokenizer):
    output_words = [tokenizer.index2word[i.item()] for i in indexes if i.item() != 0]
    return ' '.join(output_words)

def evaluate(sample_qus, output_qus, n_topk):
    # Compute the cosine similarity between sample question and remaining questions.
    output = torch.matmul(output_qus, sample_qus.unsqueeze(-1))
    output /= (torch.norm(output_qus, p=2, dim=1)* torch.norm(sample_qus, p=2, dim=0)).unsqueeze(-1)
    # |output| = (batch_size, 1)
    
    # Returns the k largest elements of the given input tensor.
    topv, topi = torch.topk(output.squeeze(-1), n_topk)
    
    return topv, topi

def evaluateiters(data, model, tokenizer, n_topk=3, iteration=5, evaluation_bs=64):
    # build dataset & data loader
    qa_data = trainer.QuestionAnswerDataset(data, tokenizer, negative_sampling=False)
    data_loader = DataLoader(dataset=qa_data, batch_size=evaluation_bs, shuffle=False, num_workers=4)
    
    model.eval()
    with torch.no_grad():
        for iter in range(iteration):
            # random index to extract one sample question
            ri = random.randrange(evaluation_bs)
            
            max_similarity, max_similarity_qus = [0]*n_topk, [0]*n_topk
            for i, data in enumerate(data_loader, 0):
                qus, ans, qus_len, ans_len, labels = data
                qus, ans, labels = qus.to(device), ans.to(device), labels.to(device=device, dtype=torch.float32)

                # sort by sequence length in descending order
                qus, qus_len = trainer.sort_by_len(qus, qus_len)
                ans, ans_len = trainer.sort_by_len(ans, ans_len)
                
                output_qus = model(qus, ans, qus_len, ans_len)
                # |output_qus| = (batch_size, num_directions*hidden_size)
                
                # sampling
                if i==0:
                    sample_output_qus = output_qus[ri]
                    # |sample_output_qus| = (num_directions*hidden_size)
                    sample_qus, sample_ans = qus[ri], ans[ri]
                
                # evaluation
                topv, topi = evaluate(sample_output_qus, output_qus, n_topk)

                for v, i in zip(topv, topi):
                    if v.item() >= min(max_similarity):
                        min_i = max_similarity.index(min(max_similarity))
                        
                        max_similarity[min_i] = v.item()
                        max_similarity_qus[min_i] = qus[i.item()]
            
            print('\nITER: #{}'.format(iter+1))
            print('sample question: {}'.format(sentencesFromInexes(sample_qus, tokenizer)))
            for v, qus in zip(max_similarity, max_similarity_qus):                
                print('==================================================================')
                print('Question: {}'.format(sentencesFromInexes(qus, tokenizer)))
                print('Similarity(with given sample question):\t{:.4f}'.format(v))
                    
if __name__=='__main__':
    config = argparser()

    # data load
    data = loader.to_dataframe('data/'+config.filename)

    # preprocessing
    data, word_emb_matirx, tfidf_matrix, tokenizer = pproc.preprocessing(input = data,
                                                                         clean_drop = config.clean_drop)
    # |data| = (n_pairs, n_columns) = (91,517, 5)
    # |word_emb_matrix| = (tokenizer.n_words, 100)
    # |tfidf_matrix| = (tokenizer.n_words, 1)

    # build model
    model = models.Model(input_size = tokenizer.n_words,
                         embedding_size = 100,
                         hidden_size = config.hidden_size,
                         n_layers = config.n_layers,
                         dropout_p = config.dropout_p,
                         word_embedding_matrix = word_emb_matirx,
                         tfidf_matrix = tfidf_matrix,
                         evaluation_mode=True
                         ).to(device)
    
    # load trained model parameters
    model.load_state_dict(torch.load(config.model))

    # evaluate
    evaluateiters(data, model, tokenizer)