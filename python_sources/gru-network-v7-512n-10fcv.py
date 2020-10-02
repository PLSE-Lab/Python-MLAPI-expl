#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, classification_report
from sklearn.utils import class_weight
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import CuDNNGRU, Dense, Bidirectional, Activation, Dropout, CuDNNLSTM
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

tqdm.pandas()


# In[ ]:


with open('../input/ndsc-beginner/categories.json', 'rb') as handle:
    cat_details = json.load(handle)
    
category_mapper = {}
product_type_mapper = {}

# change the blouse category name, something wrong with the category name

cat_details['Fashion']['Blouse'] = cat_details['Fashion']['Blouse\xa0']
del cat_details['Fashion']['Blouse\xa0']

for category in cat_details.keys():
    for key, value in cat_details[category].items():
        category_mapper[value] = key
        product_type_mapper[value] = category
        
df_train = pd.read_csv('../input/ndsc-beginner/train.csv')
df_test = pd.read_csv('../input/ndsc-beginner/test.csv')


# In[ ]:


w2v = KeyedVectors.load_word2vec_format('../input/gensim-train-custom-embeddings-v7/custom_glove_250d_no_processing.txt')


# In[ ]:


def sentence_to_wordvec(sentence):
    
    num_words = len(sentence.split())
    
    word_vector = np.zeros((30,250))  # Change to matched the dimension of word vector
    
    for index, word in enumerate(sentence.split()):
        try:
            word_vector[index] = w2v.get_vector(word)
        except:
            continue
            
    return word_vector


# In[ ]:


def batch_gen(x, y, batch_size):
    
    n_batches = len(x) // batch_size
    
    while True: 
        
        train = pd.concat([x,y], axis=1)
        
        train = train.sample(frac=1.)  # Shuffle the data.
        
        for i in range(n_batches):
            
            texts = train.iloc[i * batch_size : (i+1) * batch_size, 0]
            x_batch = np.array([sentence_to_wordvec(text) for text in texts])
            batch_labels = train.iloc[i * batch_size: (i+1) * batch_size, 1]
            y_batch = np.zeros((batch_size, 58))
            y_batch[np.arange(batch_size), batch_labels] = 1
                        
            yield x_batch, y_batch


# In[ ]:


def test_batch_gen(x, batch_size):
    
    n_batches = len(x) // batch_size + 1  # to make sure all the test data are predicted
    
    for i in range(n_batches):

        texts = x.iloc[i * batch_size : (i+1) * batch_size]
        x_batch = np.array([sentence_to_wordvec(text) for text in texts])

        yield x_batch


# In[ ]:


X = df_train['title']
y = df_train['Category']


# In[ ]:


X_test = df_test['title']


# In[ ]:


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(df_train['Category']),
                                                 df_train['Category'])
d_class_weights = dict(enumerate(class_weights))


# In[ ]:


skf = StratifiedKFold(n_splits=10, random_state=123)

for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    print(f'Fold #{i+1}')
    
    model = Sequential()
    model.add(Bidirectional(CuDNNGRU(512, return_sequences=True),
                            input_shape=(30, 250)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(CuDNNGRU(512)))
    model.add(Dropout(0.5))
    model.add(Dense(58, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    batch_size = 1024

    train_step = int(len(X_train) // batch_size)

    val_step = int(len(X_val) // batch_size)


    es = EarlyStopping(monitor='val_acc', 
                       mode='max', 
                       patience=10, restore_best_weights=True)

    train_data = batch_gen(X_train, y_train, batch_size)
    val_data = batch_gen(X_val, y_val, batch_size)


    model.fit_generator(train_data, 
                        epochs=100,
                        steps_per_epoch = train_step,
                        validation_data = val_data,
                        validation_steps = val_step,
                        verbose=2, 
                        class_weight = class_weights,
                        callbacks=[es])
            
    # Predict test set & save raw probability
    
    y_test_pred = []
    for x in test_batch_gen(X_test, batch_size = batch_size):
        y_test_pred.extend(model.predict(x))
            
    df_test[f'model_{i + 1}_prediction'] = y_test_pred


# In[ ]:


p10,p9,p8,p7,p6,p5,p4,p3,p2,p1 = [df_test.iloc[:,-i].values for i in range(1,11,1)]


# In[ ]:


df_test['combined_prob'] = [(s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10)/5 for s1,s2,s3,s4,s5,s6,s7,s8,s9,s10 in zip(p10,p9,p8,p7,p6,p5,p4,p3,p2,p1)]   # Get normalized probabilities of all 5 models


# In[ ]:


all(df_test['combined_prob'][0] == (p1[0] + p2[0] + p3[0] + p4[0] + p5[0] + p6[0] + p7[0] + p8[0] + p9[0] + p10[0])/10)


# In[ ]:


df_test['Category'] = df_test['combined_prob'].progress_apply(np.argmax)


# In[ ]:


submission = df_test.loc[:,['itemid','Category']]
submission.to_csv('10fold_GRU_v7_512neurons_250d_predictions.csv', index=False)


# In[ ]:


model_prediction = df_test.loc[:,['itemid','combined_prob','Category']]
model_prediction.to_csv('10fold_GRU_v7_512neurons_250d_raw_probabilities.csv', index=False)


# In[ ]:




