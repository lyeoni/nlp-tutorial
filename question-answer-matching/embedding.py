import numpy as np

def word2vec(embedding_path):
    # Map word to fixed-length dense and continuous-valued vector,
    # with pre-trained word representation.
    word2vec_dict = {}

    with open(embedding_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            array = line.lstrip().rstrip().split(' ')
            word2vec_dict[array[0]] = list(map(float, array[1:])) # {word: vectors}
    
    return word2vec_dict

def word_embedding_matrix(word2index, embedding_path, embedding_size):
    trained_word2vec = word2vec(embedding_path) # pre-trained embedding word vectors
    print('number of trained word vectors of {}: {}'.format(embedding_path, len(trained_word2vec)))

    embedding_matrix = np.zeros((len(word2index), embedding_size))
    # |embedding_matrix| = (n_words, embedding_size)

    for word, idx in word2index.items():
        word_vec = trained_word2vec.get(word)
        if word_vec is not None:
            embedding_matrix[idx] = word_vec
        # else:
        #     print(word, idx, word_vec)

    return embedding_matrix

if __name__=='__main__':
    pass