#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gc
import logging
import datetime
import warnings
from tqdm import tqdm
tqdm.pandas()
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.layers as L
from keras.models import Model, load_model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Bidirectional
from keras.layers import LSTM, Lambda, concatenate, BatchNormalization, Embedding
from keras.layers import TimeDistributed

import nltk
EMBED_SIZE = 300
MAX_LEN = 220
MAX_FEATURES = 100000
BATCH_SIZE = 64
NUM_EPOCHS = 20
OUTPUT_PATH = '../input/output/'
JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
EMB_PATH = '../input/domain-embedding/ft_skip_300_D'
JIGSAW_PATH_TRAIN = '../input/preprocessed-jigsaw/toxic_train_preprocessed.csv'
JIGSAW_PATH_TEST='../input/preprocessed-jigsaw/toxic_test_preprocessed.csv'

def get_logger():
    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    return logger
    
logger = get_logger()
def load_data():
    logger.info('Loading kaggle toxic data..')
    train=pd.read_csv(JIGSAW_PATH_TRAIN)
    test=pd.read_csv(JIGSAW_PATH_TEST)
    logger.info('Training data shape:{}, Test data shape:{}'.format(train.shape,test.shape))
    return train, test

def remove_stopwords(data):
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    data['comment_text'] = data['comment_text'].progress_apply(lambda x: str(x).lower())
    data['comment_text'] = data['comment_text'].progress_apply(lambda x: " ".join(x for x in str(x).split() 
        if x not in stop or not x.isdigit()))
    return data

def run_tokenizer(train, test):
    logger.info('Fitting tokenizer')
    tokenizer = Tokenizer(num_words=MAX_FEATURES) 
    tokenizer.fit_on_texts(list(train['comment_text']))# + list(test['comment_text'])
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(list(train['comment_text']))
    y_train = train['target'].values
    X_test = tokenizer.texts_to_sequences(list(test['comment_text']))
    
    X_train = pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = pad_sequences(X_test, maxlen=MAX_LEN)
    
    del tokenizer
    gc.collect()
    return X_train, X_test, y_train, word_index

def build_embedding_matrix(word_index,EMBED_SIZE, embed_dir=EMB_PATH):
    logger.info('Loading and preparing pre-trained word embedding ...')
    embedding_model = KeyedVectors.load(embed_dir)
    embedding_matrix = np.zeros((len(word_index) + 1,EMBED_SIZE))
    uknown=[]
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_model[word]
        except:
            embedding_matrix[i] = np.random.random(EMBED_SIZE)
            uknown.append(word)
    del embedding_model
    print('No. of known words:',len(uknown))
    return embedding_matrix

def get_global_embedding(word_index,EMBED_SIZE):
    import collections
    uknown=[]
    word2emb=collections.defaultdict(int)
    fglove=open('../input/glove840b300dtxt/glove.840B.300d.txt',"rb")
    for line in fglove:
        cols=line.strip().split()
        word=cols[0]
        embedding=np.array(cols[1:],dtype='float32')
        word2emb[word]=embedding
    fglove.close()
    vocab_size=len(word_index)+1
    emb_matrix=np.zeros((vocab_size,EMBED_SIZE))
    for w, i in word_index.items():
        vect = word2emb.get(w)
        if vect is not None:
            emb_matrix[i] = vect
        else:
            emb_matrix[i]=np.random.random(EMBED_SIZE)
            uknown.append(word)
    
    del word2emb
    gc.collect()
    print('No. of known words:',len(uknown))
    return emb_matrix
def build_model(emb_matrix,word_index):
	inputs = Input(shape=(MAX_LEN,), dtype='int32')
	embedding_layer=Embedding(len(word_index)+1,EMBED_SIZE,
		weights=[emb_matrix],input_length=MAX_LEN,
		trainable=True)
	embedded_sequences = embedding_layer(inputs)
	l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
	l_pool1 = MaxPooling1D(5)(l_cov1)
	l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
	l_pool2 = MaxPooling1D(5)(l_cov2)
	l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
	#l_pool3 = MaxPooling1D(35)(l_cov3)  
	l_flat = GlobalMaxPool1D()(l_cov3)# global max pooling
	l_dense = Dense(128, activation='relu')(l_flat)
	preds = Dense(1, activation='sigmoid')(l_dense)
	model = Model(inputs, preds, name="jigsaw")
	return model

def submit(preds):
    print('Prepare submission')
    submission = pd.read_csv(os.path.join(JIGSAW_PATH,'sample_submission.csv'), index_col='id')
    submission['prediction'] = preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)
    print('saved to ')


def main():
    train, test = load_data()
    logger.info('Stop word removal ....')
    train=remove_stopwords(train)
    test=remove_stopwords(test)
#    toxic_words=[line.rstrip('\n') for line in open('Data/detected_toxic_words.txt')]
#    import gensim
#    model=gensim.models.Word2Vec.load('Models/embeddings/toxic_w2v_300D')
#    train_comment_text=train.comment_text
#    for comment in train_comment_text:
#        words=nltk.word_tokenize(comment)
#        for w in words:
#            for toxic in toxic_words:
#                print(w,'-',toxic,':',model.wv.similarity(w.lower(),toxic))
#            break
#        break
    X_train, X_test, y_train, word_index = run_tokenizer(train, test)
    embedding_matrix = build_embedding_matrix(word_index,300)
    logger.info('########## Data Statistics ############')
    logger.info('X-train shape:{}'.format(X_train.shape))
    logger.info('Y-train shape:{}'.format(y_train.shape))
    logger.info('X-test shape:{}'.format(X_test.shape))
    logger.info('Vocabulary size:{}'.format(len(word_index)+1))
    logger.info('Embedding shape:{}'.format(embedding_matrix.shape))
    model=build_model(embedding_matrix, word_index)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, validation_split=0.2, epochs=10,batch_size=64)
    algorithm='simple_cnn'
    model.save('../input/Output/MODEL_{}.h5'.format(algorithm))
    pred=model.predict(X_test)[:,0]
    submit(pred)
    # logger.info('loading embeding matrix')
    # embedding_matrix=np.load('fast_text_domain_trained.npy')
#    preds = run_model(X_train, X_test, y_train, embedding_matrix, word_index)
    # model = load_model('mod_0.hdf5')
    # sub_preds = model.predict(X_test)[:, 0]
#    submit(preds)
    
if __name__ == "__main__":
    main()
    


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input/output"))
OUTPUT_PATH = '../input/output/'
JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
def load_data():
    print('Load train and test data')
    train = pd.read_csv(os.path.join(OUTPUT_PATH,'domain_ft.csv'))
#     test = pd.read_csv(os.path.join(JIGSAW_PATH,'test.csv'), index_col='id')
    return train
def submit(preds):
    print('Prepare submission')
    submission = pd.read_csv(os.path.join(JIGSAW_PATH,'sample_submission.csv'), index_col='id')
    submission['prediction'] = preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)
    print('saved to ')
sub=load_data()
pred=sub['prediction'].values
submit(pred)
# Any results you write to the current directory are saved as output.

