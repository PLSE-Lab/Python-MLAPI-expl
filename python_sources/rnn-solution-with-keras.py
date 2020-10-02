#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import math


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0/len(y))) ** 0.5
#Source: https://www.kaggle.com/marknagelberg/rmsle-function


# In[ ]:


#LOAD DATA
print("Loading data...")
start_time = time.time()
train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')

train['target'] = np.log1p(train['price']) 

print(train.shape)
print(test.shape)

test_len = test.shape[0]


# In[ ]:


#HANDLE MISSING VALUES
print("Handling missing values...")
def handle_missing(dataset):
    dataset.category_name.fillna(value="", inplace=True)
    dataset.brand_name.fillna(value="", inplace=True)
    dataset.item_description.fillna(value="", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)


# In[ ]:


train.head(3)


# In[ ]:


#PROCESS CATEGORICAL DATA
print("Handling categorical variables...")
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train['category'] = le.transform(train.category_name)
test['category'] = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train['brand'] = le.transform(train.brand_name)
test['brand'] = le.transform(test.brand_name)
del le

train.head(3)


# In[ ]:


#PROCESS TEXT: RAW
print("Text to seq process...")
from keras.preprocessing.text import Tokenizer
raw_text = np.hstack([train.category_name.str.lower(), 
                      train.item_description.str.lower(), 
                      train.name.str.lower()])

print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)
print("   Transforming text to seq...")

train["seq_category_name"] = tok_raw.texts_to_sequences(train.category_name.str.lower())
test["seq_category_name"] = tok_raw.texts_to_sequences(test.category_name.str.lower())
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
train.head(3)


# In[ ]:


#SEQUENCES VARIABLES ANALYSIS
max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(test.seq_name.apply(lambda x: len(x)))])
max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x)))
                                   , np.max(test.seq_item_description.apply(lambda x: len(x)))])
max_seq_category_name = np.max([np.max(train.seq_category_name.apply(lambda x: len(x)))
                                   , np.max(test.seq_category_name.apply(lambda x: len(x)))])
print("max name seq "+str(max_name_seq))
print("max item desc seq "+str(max_seq_item_description))
print("max category name seq "+str(max_seq_category_name))


# In[ ]:


train.seq_name.apply(lambda x: len(x)).hist()


# In[ ]:


train.seq_item_description.apply(lambda x: len(x)).hist()


# In[ ]:


train.seq_category_name.apply(lambda x: len(x)).hist()


# In[ ]:


#DEV TESTS
from sklearn.model_selection import train_test_split
dtrain, dvalid = train_test_split(train, random_state=233, train_size=0.99) 
print(dtrain.shape)
print(dvalid.shape)


# In[ ]:


#EMBEDDINGS MAX VALUE
MAX_NAME_SEQ = max_name_seq
MAX_ITEM_DESC_SEQ = max_seq_item_description
MAX_CATEGORY_NAME_SEQ = max_seq_category_name
#MAX_TEXT = len(tok_raw.word_index) + 1
MAX_TEXT = np.max([np.max(train.seq_name.max())
                    , np.max(test.seq_name.max())
                    , np.max(train.seq_category_name.max())
                    , np.max(test.seq_category_name.max())
                    , np.max(train.seq_item_description.max())
                    , np.max(test.seq_item_description.max())])+2
MAX_CATEGORY = np.max([train.category.max(), test.category.max()])+1
MAX_BRAND = np.max([train.brand.max(), test.brand.max()])+1
MAX_CONDITION = np.max([train.item_condition_id.max(),
                        test.item_condition_id.max()])+1


# In[ ]:


#KERAS DATA DEFINITION
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ)
        ,'item_desc': pad_sequences(dataset.seq_item_description
                                    , maxlen=MAX_ITEM_DESC_SEQ)
        ,'brand': np.array(dataset.brand)
        ,'category': np.array(dataset.category)
        ,'category_name': pad_sequences(dataset.seq_category_name
                                        , maxlen=MAX_CATEGORY_NAME_SEQ)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)


# In[ ]:


#KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization,     Activation, concatenate, GRU, Embedding, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras import initializers

dr = 0.25 

def get_model():
    #params
    dr_r = dr
    
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]], 
                          name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    #Embeddings layers
    emb_size = 64
    emb_name = Embedding(MAX_TEXT, emb_size)(name)
    emb_item_desc = Embedding(MAX_TEXT, emb_size)(item_desc)
    emb_category_name = Embedding(MAX_TEXT, emb_size)(category_name)
    emb_brand = Embedding(MAX_BRAND, emb_size)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_category_name)
    rnn_layer3 = GRU(8) (emb_name)
    #rnn_layer4 = GRU(8) (emb_brand)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_category)
        , Flatten() (emb_item_condition)
        , rnn_layer1
        , rnn_layer2
        , rnn_layer3
        #, rnn_layer4
        , num_vars
    ])
    main_l = Dropout(dr_r) (Dense(128) (main_l)) 
    #main_l = Dropout(dr_r) (Dense(64) (main_l))
    main_l = Dropout(dr_r) (Dense(32) (main_l))
    
    #output
    output = Dense(1, activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand
                   , category, category_name
                   , item_condition, num_vars], output)
    #optimizer = optimizers.RMSprop()
    optimizer = optimizers.Adam()
    model.compile(loss="mse", 
                  optimizer=optimizer)
    return model


# In[ ]:


def eval_model(model):
    val_preds = model.predict(X_valid)
    val_preds = np.expm1(val_preds)
    
    y_true = np.array(dvalid.price.values)
    y_pred = val_preds[:, 0]
    v_rmsle = rmsle(y_true, y_pred)
    return v_rmsle


# In[ ]:


exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1

del train

gc.collect()


# In[ ]:


#FITTING THE MODEL
epochs = 1
BATCH_SIZE = 512 * 3
steps = int(len(X_train['name'])/BATCH_SIZE) * epochs

lr_init, lr_fin = 0.013, 0.009
lr_decay = exp_decay(lr_init, lr_fin, steps)

model = get_model()
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

history = model.fit(X_train, dtrain.target
                    , epochs=epochs
                    , batch_size=BATCH_SIZE
                    , validation_split=0.01
                    , verbose=1
                    )


# In[ ]:


#EVALUATE MODEL ON DEV TESTS
v_rmsle = eval_model(model)
print(" RMSLE error on dev test: "+str(v_rmsle))
log_subname = '_'.join(['ep', str(epochs),
                    'bs', str(BATCH_SIZE),
                    'lrI', str(lr_init),
                    'lrF', str(lr_fin),
                    'dr', str(dr)])


# In[ ]:


#CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE) 
preds = np.expm1(preds)

submission = test[["test_id"]][:test_len] 
submission["price"] = preds[:test_len]


# In[ ]:


submission.to_csv("sample_submission.csv", index=False)
print('[{}] Finished submission...'.format(time.time() - start_time))


# 
#  
#     
