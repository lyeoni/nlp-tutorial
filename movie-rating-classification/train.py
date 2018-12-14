import matplotlib
matplotlib.use('Agg')
import argparse
import pandas as pd
import numpy as np
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dropout, Concatenate, Flatten, Dense
from keras.models import Model
from keras.initializers import Constant
import matplotlib.pyplot as plt
import scikitplot as skplt
import data_loader
import datetime

def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-corpus_tk',
                   default='corpus.tk.txt',
                   help='Default=corpus.tk.txt')

    p.add_argument('-trained_word_vector',
                   default='corpus.tk.vec.txt',
                   help='Default=corpus.tk.vec.txt')

    p.add_argument('-score_corpus',
                   default='score_corpus.txt',
                   help='Default=score_corpus.txt')
    
    p.add_argument('-epoch',
                   type=int,
                   default=20,
                   help='number of iteration to train model. Default=20')

    p.add_argument('-batch_size',
                   type=int,
                   default=64,
                   help='mini batch size for parallel inference. Default=64')
    
    config = p.parse_args()
    
    return config

def build_model():
    inputs = Input(shape=(loader.max_corpus_len,), dtype='int32')
    embedding = Embedding(loader.max_word_num+1, loader.embedding_dim,
                         embeddings_initializer = Constant(loader.embedding_matrix),
                         input_length=loader.max_corpus_len, trainable=False)(inputs)

    stacks = []
    for kernel_size in [2, 3, 4]:
        conv = Conv1D(64, kernel_size, padding='same', activation='relu', strides=1)(embedding)
        pool = MaxPooling1D(pool_size=3)(conv)
        drop = Dropout(0.5)(pool)
        stacks.append(drop)

    merged = Concatenate()(stacks)
    flatten = Flatten()(merged)
    drop = Dropout(0.5)(flatten)
    outputs = Dense(y_train.shape[1], activation='softmax')(drop)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return model

if __name__ == "__main__":
    config = argparser()
    
    current =datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    loader = data_loader.DataLoader(config.corpus_tk, config.trained_word_vector, config.score_corpus)
    loader.load_data()

    x_train, y_train = loader.train
    x_test, y_test = loader.test
    
    # build model
    model = build_model()
    
    # training
    hist = model.fit(x_train, y_train,
                     epochs = config.epoch,
                     batch_size = config.batch_size,
                     validation_data=(x_test, y_test), verbose=2)
    
    # evaluation: confusion matrix & roc curve
    pred = model.predict(x_test)
    skplt.metrics.plot_confusion_matrix(np.argmax(y_test, axis=1), np.argmax(pred, axis=1))
    plt.savefig('cf_matrx_'+current+'.png')
    plt.close()
    skplt.metrics.plot_roc(np.argmax(y_test, axis=1), pred)
    plt.savefig('roc_curve_'+current+'.png')
    plt.close()
    
    # evaluation: summarize history for accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc_'+current+'.png')

    # evaluation: summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_'+current+'.png')
        
    # save model
    model.save('model_'+current+'.h5')
    print('MODEL SAVED')