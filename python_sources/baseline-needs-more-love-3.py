import os
import gc
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D,concatenate,Dropout,MaxPooling1D,AveragePooling1D
from keras.layers import GlobalMaxPooling1D,GlobalAveragePooling1D,BatchNormalization,SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D

"""
Paper:
https://arxiv.org/pdf/1805.09843.pdf
"""





def get_emb0():
    #Prepare Embedding:
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    return embedding_matrix
        


def get_emb1():
    # Load Embedding:
    EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
    
def get_emb2():
    # Load Embedding:
    EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
    

def get_model(EMB_MAT):
    
    inp = Input(shape=(maxlen,))
    emb = Embedding(max_features, embed_size, weights=[EMB_MAT], trainable=True)(inp)
    emb = SpatialDropout1D(0.1)(emb)
    mp = MaxPooling1D(pool_size =(int(emb.shape[1]//2),) )(emb)
    ap = AveragePooling1D(pool_size =(int(emb.shape[1]//2),) )(emb)
    

    x = concatenate([mp, ap])

    mp2 = MaxPooling1D(pool_size =(int(emb.shape[1]//4),) )(emb)
    ap2 = GlobalAveragePooling1D()(mp2)
    ap2 = Flatten()(ap2)
    
    x = concatenate([x, ap2])
    
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="sigmoid")(x)
    # Train:
    model = Model(inputs=inp, outputs=x)
    return model




if __name__ == "__main__":
    """
    Load Data:
    """
    train_df = pd.read_csv("../input/train.csv")
    train_df['question_text'] = train_df['question_text'].str.lower()
    test_df = pd.read_csv("../input/test.csv")
    test_df['question_text'] = test_df['question_text'].str.lower()
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    """
    Processing:
    """
    ## split to train and val
    train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=2018)
    ## some config values 
    embed_size = 300 # how big is each word vector
    max_features = 95000 # 
    maxlen = 70
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    val_X = val_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values
    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    val_X = tokenizer.texts_to_sequences(val_X)
    test_X = tokenizer.texts_to_sequences(test_X)
    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    val_X = pad_sequences(val_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)
    ## Get the target values
    train_y = train_df['target'].values
    val_y = val_df['target'].values
    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))
    val_idx = np.random.permutation(len(val_X))
    train_X = train_X[trn_idx]
    val_X = val_X[val_idx]
    train_y = train_y[trn_idx]
    val_y = val_y[val_idx]
    
    
    
    
    #glove embedding:
    emb0 = get_emb0()
    emb1 = get_emb1()
    emb2 = get_emb2()
    
    
    all_preds_test = []
    all_preds_val = []
    for e in [emb0,emb1,emb2]:
        model = get_model(e)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        model.fit(train_X, train_y, batch_size=512, epochs=5, validation_data=(val_X, val_y),verbose=0)
        #Evaluate
        pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
        # Search for F1 Score:
        max_f1 = 0
        best_thres = 0
        for thresh in np.arange(0.2, 0.501, 0.01):
            thresh = np.round(thresh, 2)
            temp_f1 = metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))
            # print("thres: {0} -- f1: {1}".format(thresh,temp_f1))
            if temp_f1 > max_f1:
                max_f1 = temp_f1
                best_thres = thresh
    
        print("Best thres: {0} -- f1: {1}".format(best_thres,max_f1))
        # Pred and Sub:
        pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
        # pred_test_y = (pred_test_y > best_thres).astype(int)
        all_preds_test.append(pred_test_y)
        all_preds_val.append(pred_val_y)
        del model
        del pred_val_y
        del max_f1
        del best_thres
        gc.collect()
       
    
    
    
    
    # Find best thres:
    val_pred = np.mean(all_preds_val,axis=0)
    
    max_f1 = 0
    best_thres = 0
    for thresh in np.arange(0.2, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        temp_f1 = metrics.f1_score(val_y, (val_pred>thresh).astype(int))
        print("thres: {0} -- f1: {1}".format(thresh,temp_f1))
        if temp_f1 > max_f1:
            max_f1 = temp_f1
            best_thres = thresh
    
    
    
    print("Best thres: {0} -- f1: {1}".format(best_thres,max_f1))
       
        
    out_df = pd.DataFrame({"qid":test_df["qid"].values})
    final_pred = np.mean(all_preds_test,axis=0)
    final_pred = (final_pred > best_thres).astype(int)
    out_df['prediction'] = final_pred
    out_df.to_csv("submission.csv", index=False)