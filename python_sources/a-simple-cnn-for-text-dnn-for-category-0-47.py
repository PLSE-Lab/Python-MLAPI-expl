#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from keras import backend as K
import gc
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.preprocessing.sequence import pad_sequences

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")
 

def handle_missing_inplace(dataset):
    dataset['general_cat'].fillna(value='No Label', inplace=True)
    dataset['subcat_1'].fillna(value='No Label', inplace=True)
    dataset['subcat_2'].fillna(value='No Label', inplace=True)
    dataset['brand_name'].fillna(value='missing', inplace=True)
    dataset['item_description'].fillna(value='No description yet', inplace=True)


def cutting(dataset):
    pop_brand = dataset['brand_name'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    dataset.loc[~dataset['brand_name'].isin(pop_brand), 'brand_name'] = 'missing'
    pop_category1 = dataset['general_cat'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category2 = dataset['subcat_1'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    pop_category3 = dataset['subcat_2'].value_counts().loc[lambda x: x.index != 'missing'].index[:NUM_CATEGORIES]
    dataset.loc[~dataset['general_cat'].isin(pop_category1), 'general_cat'] = 'missing'
    dataset.loc[~dataset['subcat_1'].isin(pop_category2), 'subcat_1'] = 'missing'
    dataset.loc[~dataset['subcat_2'].isin(pop_category3), 'subcat_2'] = 'missing'


def to_categorical(dataset):
    dataset['general_cat'] = dataset['general_cat'].astype('category')
    dataset['subcat_1'] = dataset['subcat_1'].astype('category')
    dataset['subcat_2'] = dataset['subcat_2'].astype('category')


def raw_text(dataset):   
    raw_text = np.hstack([dataset.item_description.str.lower(), dataset.name.str.lower()])
    tok_raw = Tokenizer(num_words=20000,
                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                    lower=True,
                    split=" ",
                    char_level=False)
    tok_raw.fit_on_texts(raw_text)
    dataset["seq_item_description"] = tok_raw.texts_to_sequences(dataset.item_description.str.lower())
    dataset["seq_name"] = tok_raw.texts_to_sequences(dataset.name.str.lower())
    dataset["Raw Text Combined"] = dataset.seq_name + dataset.seq_item_description


def get_keras_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=10)
        ,'item_desc': pad_sequences(dataset.seq_item_description, maxlen=75)
        ,'brand_name': np.array(dataset.brand_name)
        ,'general_cat': np.array(dataset.general_cat)
        ,'subcat_1': np.array(dataset.subcat_1)
        ,'subcat_2': np.array(dataset.subcat_2)
        ,'item_condition': np.array(dataset.item_condition_id)
        ,'num_vars': np.array(dataset.shipping)
    }
    return X

NUM_BRANDS = 4000
NUM_CATEGORIES = 1000


# In[ ]:


#LOAD DATA
train = pd.read_table("../input/train.tsv")
test = pd.read_table("../input/test.tsv")

print(train.shape)
print(test.shape)
train.head(3)


# In[ ]:


start_time = time.time()

nrow_train = train.shape[0]
merge: pd.DataFrame = pd.concat([train, test])
submission: pd.DataFrame = test[['test_id']]

del train
del test
gc.collect()

merge['general_cat'], merge['subcat_1'], merge['subcat_2'] = zip(*merge['category_name'].apply(lambda x: split_cat(x)))
merge.drop('category_name', axis=1, inplace=True)
print('[{}] Split categories completed.'.format(time.time() - start_time))

handle_missing_inplace(merge)
print('[{}] Handle missing completed.'.format(time.time() - start_time))

cutting(merge)
print('[{}] Cut completed.'.format(time.time() - start_time))

to_categorical(merge)
print('[{}] Convert categorical completed'.format(time.time() - start_time))

raw_text(merge)
print('[{}] Raw text completed'.format(time.time() - start_time))

le = LabelEncoder()
merge.brand_name = le.fit_transform(merge.brand_name)
merge.general_cat = le.fit_transform(merge.general_cat)
merge.subcat_1 = le.fit_transform(merge.subcat_1)
merge.subcat_2 = le.fit_transform(merge.subcat_2)
print('[{}] category variable labelled completed'.format(time.time() - start_time))

merge.head(3)


# In[ ]:


#EXTRACT DEVELOPTMENT TEST
dtest = merge.iloc[nrow_train:, ]
dtrain, dvalid = train_test_split(merge.iloc[:nrow_train, ], random_state=123, train_size=0.7)
print(dtrain.shape)
print(dvalid.shape)


X_train = get_keras_data(dtrain)
X_valid = get_keras_data(dvalid)
X_test = get_keras_data(dtest)

Y_train =  np.log1p(np.array(dtrain.price))
Y_valid =  np.log1p(np.array(dvalid.price))


# **Model Part **

# In[ ]:


MAX_TEXT = np.max([np.max(merge.seq_name.max()), np.max(merge.seq_item_description.max())])+2
MAX_general_cat = np.max([merge.general_cat.max()])+1
MAX_subcat_1 = np.max([merge.subcat_1.max()])+1
MAX_subcat_2 = np.max([merge.subcat_2.max()])+1
MAX_BRAND = np.max([merge.brand_name.max()])+1

print(MAX_TEXT)
print(MAX_general_cat)
print(MAX_subcat_1)
print(MAX_subcat_2)
print(MAX_BRAND)


# In[ ]:


#KERAS MODEL DEFINITION
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, Conv1D, GlobalMaxPooling1D, Embedding, Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K

def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]

def rmsle_cust(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))

def get_model():
    #params
    dr_r = 0.5
    
    #Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    general_cat = Input(shape=[1], name="general_cat")
    subcat_1 = Input(shape=[1], name="subcat_1")
    subcat_2 = Input(shape=[1], name="subcat_2")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[1], name="num_vars")
    
    #Embeddings layers
    emb_name = Embedding(MAX_TEXT, 10)(name)
    emb_item_desc = Embedding(MAX_TEXT, 10)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 50)(brand_name)
    emb_general_cat = Embedding(MAX_general_cat, 10)(general_cat)
    emb_subcat_1 = Embedding(MAX_subcat_1, 20)(subcat_1)
    emb_subcat_2 = Embedding(MAX_subcat_2, 30)(subcat_2)
 
    #rnn layer
    cnn_layer1 = Conv1D(filters=16, kernel_size=3, activation='relu') (emb_item_desc)
    cnn_layer2 = Conv1D(filters=8, kernel_size=3, activation='relu')(emb_name)
    
    cnn_layer1 = GlobalMaxPooling1D()(cnn_layer1)
    cnn_layer2 = GlobalMaxPooling1D()(cnn_layer2)
    
    #main layer
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_general_cat)
        , Flatten() (emb_subcat_1)
        , Flatten() (emb_subcat_2)
        , cnn_layer1
        , cnn_layer2
        , num_vars
        , item_condition
    ])
    
    main_l = Dropout(dr_r) (Dense(256, activation="relu") (main_l))
    main_l = Dropout(dr_r) (Dense(128, activation="relu") (main_l))
    main_l = Dropout(dr_r) (Dense(64, activation="relu") (main_l))
    
    
    #output
    output = Dense(1, activation="linear") (main_l)
    
    #model
    model = Model([name, item_desc, brand_name, general_cat, subcat_1, subcat_2, item_condition, num_vars], output)
    
    model.compile(loss="mse", optimizer="adam", metrics=["mae", rmsle_cust])
    
    return model

    
model = get_model()
model.summary()


# In[ ]:


#FITTING THE MODEL
BATCH_SIZE = 20000
epochs = 25

model = get_model()
model.fit(X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE
          , validation_data=(X_valid, Y_valid)
          , verbose=1)


# In[ ]:


#CREATE PREDICTIONS
preds = model.predict(X_test, batch_size=BATCH_SIZE)
submission["price"] = np.expm1(preds)
submission.to_csv("./myNNsubmission.csv", index=False)


# In[ ]:




