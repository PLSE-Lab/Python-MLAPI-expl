#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# forked from "A simple nn solution with Keras (~0.48611 PL)" by noobhound

from datetime import datetime
start_time = datetime.now()
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# read data
train = pd.read_table("../input/train.tsv")
test = pd.read_table("../input/test.tsv")
print(test.head())


# In[ ]:


# Handling missing values
def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)

train = handle_missing(train)
test = handle_missing(test)
print(train.shape)
print(test.shape)


# In[ ]:


# transfer category/brand to encode
le = LabelEncoder()

le.fit(np.hstack([train.category_name, test.category_name]))
train.category_name = le.transform(train.category_name)
test.category_name = le.transform(test.category_name)

le.fit(np.hstack([train.brand_name, test.brand_name]))
train.brand_name = le.transform(train.brand_name)
test.brand_name = le.transform(test.brand_name)

print(train.head())


# In[ ]:


# Stack arrays in sequence horizontally
raw_text = np.hstack([train.item_description.str.lower(), train.name.str.lower()])

# Tokenizer(), Split a sentence into a list of words
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)


# In[ ]:


# create new features with seq-text
train["seq_item_description"] = tok_raw.texts_to_sequences(train.item_description.str.lower())
test["seq_item_description"] = tok_raw.texts_to_sequences(test.item_description.str.lower())
train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())
print(train.head())


# In[ ]:


# get max len of name/item
max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), 
                       np.max(test.seq_name.apply(lambda x: len(x)))])

max_seq_item_description = np.max([np.max(train.seq_item_description.apply(lambda x: len(x)))
                                   , np.max(test.seq_item_description.apply(lambda x: len(x)))])

print("max name seq "+str(max_name_seq))
print("max item desc seq "+str(max_seq_item_description))


# In[ ]:


max_name_seq_set = 10
max_seq_item_description_set = 75
max_text = np.max([np.max(train.seq_name.max()), 
                   np.max(test.seq_name.max()), 
                   np.max(train.seq_item_description.max()), 
                   np.max(test.seq_item_description.max())])+2
max_category = np.max([train.category_name.max(), test.category_name.max()])+1
max_brand = np.max([train.brand_name.max(), test.brand_name.max()])+1
max_cond_id = np.max([train.item_condition_id.max(), test.item_condition_id.max()])+1


# In[ ]:


# choose max one in 4 features
max_text = np.max([np.max(train.seq_name.max()), 
                   np.max(test.seq_name.max()), 
                   np.max(train.seq_item_description.max()), 
                   np.max(test.seq_item_description.max())]) + 2

max_category = np.max([train.category_name.max(), test.category_name.max()]) + 1
max_brand = np.max([train.brand_name.max(), test.brand_name.max()]) + 1
max_cond_id = np.max([train.item_condition_id.max(), test.item_condition_id.max()]) + 1


# In[ ]:


# scale
train["target"] = np.log(train.price + 1)
target_scaler = MinMaxScaler(feature_range=(-1, 1)) # default (0, 1)
train["target"] = target_scaler.fit_transform(train.target.reshape(-1,1))
#pd.DataFrame(train.target).hist()


# In[ ]:


# train/val data sets
dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.90)
print(dtrain.shape)
print(dvalid.shape)


# In[ ]:


# create keras data set
from keras.preprocessing.sequence import pad_sequences

def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=max_name_seq_set),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=max_seq_item_description),
        'brand_name': np.array(dataset.brand_name),
        'category_name': np.array(dataset.category_name),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]])
    }
    return X

X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(test)

print(X_train)


# In[ ]:


# Keras model 
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None)+1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None)+1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def get_model():
    # drop rate
    dr_r = 0.2
    
    # input
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    
    # embeddings layers
    emb_name = Embedding(max_text, 50)(name)
    emb_item_desc = Embedding(max_text, 50)(item_desc)
    emb_brand_name = Embedding(max_brand, 10)(brand_name)
    emb_category_name = Embedding(max_category, 10)(category_name)
    emb_item_condition = Embedding(max_cond_id, 5)(item_condition)
    
    # rnn layer
    rnn_layer1 = GRU(16) (emb_item_desc) #16
    rnn_layer2 = GRU(8) (emb_name) #8
    
    # main layer
    main_l = concatenate([
        Flatten() (emb_brand_name),
        Flatten() (emb_category_name),
        Flatten() (emb_item_condition), 
        rnn_layer1,
        rnn_layer2,
        num_vars
    ])
    main_l = Dropout(dr_r) (Dense(64) (main_l)) #128
    main_l = Dropout(dr_r) (Dense(32) (main_l)) #64
    
    # output
    output = Dense(1, activation="linear") (main_l)
    
    # model
    model = Model([name, item_desc, brand_name
                   , category_name, item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])
    
    return model

    
model = get_model()
model.summary()


# In[ ]:


# model parameter tuning
model = get_model()
model.fit(X_train, dtrain.target, epochs=2, batch_size=5000, #25000
          validation_data=(X_valid, dvalid.target), verbose=1)


# In[ ]:


# predict for val set
#val_preds = model.predict(X_valid)
#val_preds = target_scaler.inverse_transform(val_preds)
#val_preds = np.exp(val_preds)+1

# mean_absolute_error, mean_squared_log_error
#y_true = np.array(dvalid.price.values)
#y_pred = val_preds[:,0]
#v_rmsle = rmsle_cust(y_true, y_pred)
#print(" RMSLE error on dev test: "+str(v_rmsle))


# In[ ]:


# predict for price by test set
preds = model.predict(X_test, batch_size=3000) # 15000
preds = target_scaler.inverse_transform(preds)
preds = np.exp(preds)-1


# In[ ]:


# output
submission = test[["test_id"]]
submission["price"] = preds
submission.to_csv("output_submission.csv", index=False)


# In[ ]:


end_time = datetime.now()
execution_time = start_time - end_time 
print(execution_time)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




