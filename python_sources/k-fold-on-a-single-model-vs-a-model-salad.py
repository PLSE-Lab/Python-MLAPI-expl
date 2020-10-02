#!/usr/bin/env python
# coding: utf-8

# Have come across a lot of interesting blended models in the public kernels. Most of these models have used a single train-val split. I guess because of the 2-hour time limit, people have been focusing on blending multiple models together and not on K-fold cross validation which is expensive. I wanted to explore how much additional juice can be extracted by using a K-fold vcross-validation on a single simple model. K = 5 in this example.
# 
# Also I want to acknowledge[ Shujian Liu](https://www.kaggle.com/shujian),  [SRK](https://www.kaggle.com/sudalairajkumar/), [Dieter](https://www.kaggle.com/christofhenkel) and [Khoi Nguyen ](https://www.kaggle.com/suicaokhoailang) for their excellent public kernels. They have been inspiring and motivating, and extremely useful.
# 
# The model is based on SRK's original public kernel, with my own minor modifications  : 
# https://www.kaggle.com/sudalairajkumar/a-look-at-different-embeddings
# 
# 
#     input = Input(shape=(max_len,))
#         embed = Embedding(max_words, embed_size, weights=[embedding_matrix])(input)
#     
#     x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(embed)
#     x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
#     x = GlobalMaxPool1D()(x)
#     x = Dense(16, activation="relu")(x)
#     x = Dropout(0.1)(x)
#     
#     y = Bidirectional(CuDNNGRU(64, return_sequences=True))(embed)
#     y = Bidirectional(CuDNNLSTM(64, return_sequences=True))(y)
#     y = GlobalMaxPool1D()(y)
#     y = Dense(16, activation="relu")(y)
#     y = Dropout(0.1)(y)
#     
#     z= Concatenate()([x,y])
#     
#     output = Dense(1, activation="sigmoid")(z)

# In[ ]:


import os
import time
import numpy as np 
import pandas as pd
import gc
import tensorflow as tf
import pickle
import glob

np.random.seed(7418880)
tf.set_random_seed(7418880)
from collections import defaultdict

from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import  StratifiedKFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Bidirectional, Embedding, GlobalMaxPool1D, Input
from keras.layers import CuDNNLSTM, CuDNNGRU, Concatenate, Dense,  Dropout


# Let us load the train and test sets. Set max_words to be 50000 and max_len to be 70 

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test_shape : ", test_df.shape)

train_X = train_df["question_text"].fillna("_na_").values
test_X = test_df["question_text"].fillna("_na_").values
train_y = train_df['target'].values

embed_size = 300
max_words = 50000 # number of unique words
max_len = 70


# Tokenize the sentences

# In[ ]:


start_time = time.time()

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(train_X))
train_X=tokenizer.texts_to_sequences(train_X)
test_X=tokenizer.texts_to_sequences(test_X)

train_X = pad_sequences(train_X, maxlen=max_len)
test_X = pad_sequences(test_X, maxlen=max_len)

print (train_X.shape, test_X.shape)
print("Total time for tokenizing = {:.0f} s".format(time.time()-start_time))


# Tokenizing takes about 65 s on average in my experience. Interesting to know these numbers since we have a 7200 s time limit for these competitiom

# In[ ]:


start_time = time.time()

EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_words, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print("Total time for embedding = {:.0f} s".format(time.time()-start_time))


# Embedding the glove vectors takes > 3 mins on average. 
# Let us define the model now. Nothing fancy. A GRU/LSTM bilayer concated with a LSTM/GRU layer (don't ask me why) with a few pooling, dense and dropout layers thrown in to the mix. Based on SRK's kernel.

# In[ ]:


def generate_model ():
    input = Input(shape=(max_len,))
    embed = Embedding(max_words, embed_size, weights=[embedding_matrix])(input)
    
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(embed)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    
    y = Bidirectional(CuDNNGRU(64, return_sequences=True))(embed)
    y = Bidirectional(CuDNNLSTM(64, return_sequences=True))(y)
    y = GlobalMaxPool1D()(y)
    y = Dense(16, activation="relu")(y)
    y = Dropout(0.1)(y)
    
    z= Concatenate()([x,y])
    
    output = Dense(1, activation="sigmoid")(z)
    
    model = Model (inputs=input, outputs=output)
    return model


# Let us train this model using 5-fold stratified validation. Through trial and error,  running 2 epochs  seemed like a reasonable choice for this model, to prevent overfitting.

# In[ ]:


start_time = time.time()

uid = 1
version =1 
n_splits = 5
n_epochs =2
batch_size =1024

skf = StratifiedKFold(n_splits=n_splits, random_state = 7418880, shuffle=False)
val_preds = defaultdict(list)
test_preds = {}
historyD={}
for ii, (train_index, val_index)  in enumerate(skf.split(train_X, train_y)):
    
    X_train, X_val = train_X[train_index], train_X[val_index]
    y_train, y_val = train_y[train_index], train_y[val_index]
    
    model = generate_model()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs, 
              validation_data=(X_val, y_val), verbose= True)
    
    historyD["fold{}".format(ii+1)] = hist.history
    
    pred_val_y = model.predict([X_val], batch_size=batch_size, verbose=1)
    val_preds['fold{}'.format(ii+1)] = [pred_val_y.ravel(),y_val]
    
    pred_test_y = model.predict([test_X], batch_size=batch_size, verbose=1)
    test_preds['fold{}'.format(ii+1)] =  pred_test_y.ravel()
    
    del model
    gc.collect()
    
with open('train_history_uid{}_v{}.pkl'.format(uid, version),'wb') as pklfile:
    pickle.dump(historyD,pklfile)
with open('val_preds_uid{}_v{}.pkl'.format(uid, version),'wb') as pklfile:
    pickle.dump(val_preds,pklfile)
with open('test_preds_uid{}_v{}.pkl'.format(uid, version),'wb') as pklfile:
    pickle.dump(test_preds,pklfile)
print("Total time for training = {:.0f} s".format(time.time()-start_time))    
    


# So total training time is ~ 65 minutes.
# 
# For each fold, a validation prediction and a test prediction are made. The threshold value to use for each of these test predictions is  obtained by a threshold scan  that yields the maximum  F1 score on the validation set. Basically what others have been using. Ad-hoc, but works.
# 
# In separate kernels, I submitted the test prediction from each of the 5 folds. The public LB scores I got were 
# 0.662, 0.663, 0.664, 0.665, 0.665. An average of ~0.664. Now, how does the average prediction from these five folds perform?

# In[ ]:


test01_df = pd.DataFrame()
#print(val_preds['fold6'])
for ii in range(len(val_preds)):
    threshL = []
    y_preds, y_actual = (val_preds['fold{}'.format(ii+1)])
    for idx, thresh in enumerate(np.arange(0.1, 0.51, 0.01)):
        thresh = np.round(thresh,2)
        y_01 = [ 0 if x <thresh else 1 for x in y_preds]
        threshL.append((metrics.f1_score(y_actual,y_01), thresh))
    threshL=sorted(threshL)
    best_F1, opt_thresh = threshL[-1]
    print ("For fold {0}, best validation F1 of {1:.5f} at threshold {2:.2f}".format(ii+1, best_F1,opt_thresh))
    test01_df['fold{}'.format(ii)] = [ 0 if x<opt_thresh else 1 for x in test_preds['fold{}'.format(ii+1)]]


# In[ ]:


print(test01_df.sum(axis=1).value_counts())


# So,  the models disagree on (594+470+354+354) = 1772 samples . 
# 
# A pearson corelation coefficient between the predictions is also an useful number to know to see how correlated the differnt folds are. 

# In[ ]:


test01_df.corr(method='pearson')


# We get a correlation coefficient of ~ 0.88 between prediction from the various folds. Heavily correlated, but still some variance.  So averaging these slightly different predictions should give us a better score than each of the  individual predictions.
# 
#  Let us create the submission file:

# In[ ]:


out_df = pd.DataFrame({"qid":test_df["qid"].values})
out_df['prediction'] = [ 0 if x<2.5 else 1 for x in test01_df.sum(axis=1)]
out_df.to_csv("submission.csv", index=False)


# On submission, this should yields an LB score of 0.671or thereabouts, which is a reasonable improvement of 0.007 or so. One should expect a similar improvement on other "single models" while using K-fold validation. Hope this kernel was useful and gives an idea of how to balance K-fold validation of a single model  vs. adding additional models to the soup.
