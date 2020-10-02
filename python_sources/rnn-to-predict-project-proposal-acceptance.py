#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import keras
import keras.backend as K


# In[ ]:


K.tensorflow_backend._get_available_gpus()


# In[ ]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[ ]:


train_df = pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df["project_essay_1"].head()


# In[ ]:


train_df["project_essay_2"].head()


# In[ ]:


def get_proj_essay(df):
    return  df["project_essay_1"].fillna('')+" "+ df["project_essay_2"].fillna('')+ " "+ df["project_essay_3"].fillna('')+" "+ df["project_essay_4"].fillna('')+" "


# In[ ]:


get_proj_essay(train_df)


# In[ ]:


def get_text(df):
    return df["project_title"].fillna('')+' '+get_proj_essay(df)


# In[ ]:


get_text(train_df)


# In[ ]:


train, dev = train_test_split(train_df, random_state=123, shuffle=True, test_size=0.1)
print("Training data shape:", train.shape)
print("Test data shape:", dev.shape)


# In[ ]:


tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(get_text(train))


# In[ ]:


def preprocess_target(df):
    return df[["project_is_approved"]].copy()
def preprocess_data(df):
    processed_df = df[["teacher_number_of_previously_posted_projects"]].copy()
    processed_df["project_title"] = tokenizer.texts_to_sequences(df["project_title"])
    processed_df["project_essay"] = tokenizer.texts_to_sequences(get_proj_essay(df))    
    return processed_df


# In[ ]:


processed_train  = preprocess_data(train)
processed_target = preprocess_target(train)
processed_target.shape, processed_train.shape


# In[ ]:


processed_dev  = preprocess_data(dev)
processed_dev_target = preprocess_target(dev)
processed_dev.shape, processed_dev_target.shape


# In[ ]:


processed_train["project_essay"].apply(lambda x: len(x)).hist(bins=10)


# In[ ]:


processed_train["project_essay"].apply(lambda x: max(x) if len(x) > 0 else 0)


# In[ ]:


MAX_PROJECT_TITLE_SEQ_LEN = 12
MAX_PROJECT_TITLE = processed_train["project_title"].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1

MAX_PROJECT_ESSAY_SEQ_LEN = 450
MAX_PROJECT_ESSAY = processed_train["project_essay"].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1

MAX_TEXT = max([MAX_PROJECT_TITLE, MAX_PROJECT_ESSAY])


# In[ ]:


MAX_TEXT


# In[ ]:


def get_keras_data(df):
    return {
        "teacher_number_of_previously_posted_projects": np.array(df["teacher_number_of_previously_posted_projects"]),
        "project_title": keras.preprocessing.sequence.pad_sequences(df["project_title"], maxlen=MAX_PROJECT_TITLE_SEQ_LEN),
        "project_essay": keras.preprocessing.sequence.pad_sequences(df["project_essay"], maxlen=MAX_PROJECT_ESSAY_SEQ_LEN),
    }

X_train = get_keras_data(processed_train)
X_dev = get_keras_data(processed_dev)


# In[ ]:


def create_rnn_model():
    
    #Input Layers
    teacher_previous_projects = keras.layers.Input(shape=(1,), name = "teacher_number_of_previously_posted_projects")
    proj_title = keras.layers.Input(shape=(MAX_PROJECT_TITLE_SEQ_LEN,), name="project_title")
    proj_essay = keras.layers.Input(shape=(MAX_PROJECT_ESSAY_SEQ_LEN,), name = "project_essay")
    
    emb_layer = keras.layers.Embedding(MAX_TEXT,50)
    emb_project_title = emb_layer(proj_title)
    emb_project_essay = emb_layer(proj_essay)
    
    #RNN Layers
    rnn_project_title = keras.layers.GRU(8, activation = 'relu')(emb_project_title)
    rnn_project_essay = keras.layers.GRU(16, activation = 'relu')(emb_project_essay)
    
    all_layers = keras.layers.concatenate([teacher_previous_projects,
                                              rnn_project_title,
                                              rnn_project_essay])
    # Output layer
    rnn_output = keras.layers.Dense(1, activation = 'sigmoid')(all_layers)
    
    return keras.models.Model(
        inputs=[teacher_previous_projects,
                proj_title,
                proj_essay,
               ],
        output = rnn_output
    )
rnn_model = create_rnn_model()
rnn_model.summary()


# In[ ]:


optimizer = keras.optimizers.Adam(lr=0.001)
rnn_model.compile(optimizer=optimizer,
                  loss=keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

for i in range(3):
    rnn_model.fit(X_train, processed_target,
                 batch_size=(2 ** (i + 8)),
                 epochs=1,
                 validation_data=(X_dev, processed_dev_target))


# In[ ]:


preds = rnn_model.predict(X_dev, batch_size=512)
auc_score = roc_auc_score(processed_dev_target, preds)
print("AUC for validation data: %.4f" % (auc_score,))


# ### References 
#   [Simple RNN with keras](https://www.kaggle.com/nvhbk16k53/simple-rnn-with-keras)
