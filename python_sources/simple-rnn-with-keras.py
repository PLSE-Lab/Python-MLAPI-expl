#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook will build a simple RNN model using keras to solve DonorsChoose.org Application Screening problem.

# # Load data
# 
# Firstly, we need to read data into memory then process it.

# In[1]:


import os; os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import keras
import keras.backend as K


# In[2]:


train_df = pd.read_csv("../input/train.csv", sep=",")
print(train_df.shape)


# In[3]:


train, dev = train_test_split(train_df, random_state=123, shuffle=True, test_size=0.1)
print("Training data shape:", train.shape)
print("Test data shape:", dev.shape)


# # Process Data
# In this section, we will dive deeper into the data and process it for RNN model.

# In[4]:


get_ipython().run_cell_magic('time', '', '\ndef get_project_essay(df):\n    return (df["project_essay_1"].fillna(\'\') +\n            \' \' + df["project_essay_2"].fillna(\'\') +\n            \' \' + df["project_essay_3"].fillna(\'\') +\n            \' \' + df["project_essay_4"].fillna(\'\'))\n\ndef get_text(df):\n    return df["project_title"].fillna(\'\') + \' \' + get_project_essay(df)\n\n#project_title_tokenizer = keras.preprocessing.text.Tokenizer()\n#project_title_tokenizer.fit_on_texts(train["project_title"])\n\n#project_essay_tokenizer = keras.preprocessing.text.Tokenizer()\n#project_essay_tokenizer.fit_on_texts(get_project_essay(train))\n\ntokenizer = keras.preprocessing.text.Tokenizer()\ntokenizer.fit_on_texts(get_text(train))\n\ndef preprocess_target(df):\n    return df[["project_is_approved"]].copy()\n\ndef preprocess_data(df):\n    processed_df = df[["teacher_number_of_previously_posted_projects"]].copy()\n\n    #processed_df["project_title"] = project_title_tokenizer.texts_to_sequences(df["project_title"])\n    processed_df["project_title"] = tokenizer.texts_to_sequences(df["project_title"])\n    \n    #processed_df["project_essay"] = project_essay_tokenizer.texts_to_sequences(get_project_essay(df))\n    processed_df["project_essay"] = tokenizer.texts_to_sequences(get_project_essay(df))\n    \n    return processed_df\n\nprocessed_train = preprocess_data(train)\ny_train = preprocess_target(train)\nprint(processed_train.shape, y_train.shape)\n\nprocessed_dev = preprocess_data(dev)\ny_dev = preprocess_target(dev)\nprint(processed_dev.shape, y_dev.shape)')


# We now can plot histogram for project_title and project_essay by its length to pick appropriate maximun values:

# In[5]:


processed_train["project_title"].apply(lambda x: len(x)).hist(bins=10)


# In[6]:


processed_train["project_essay"].apply(lambda x: len(x)).hist(bins=10)


# Get data so that keras RNN model can deal with it.

# In[7]:


MAX_PROJECT_TITLE_SEQ_LEN = 12
MAX_PROJECT_TITLE = processed_train["project_title"].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1

MAX_PROJECT_ESSAY_SEQ_LEN = 450
MAX_PROJECT_ESSAY = processed_train["project_essay"].apply(lambda x: max(x) if len(x) > 0 else 0).max() + 1

MAX_TEXT = max([MAX_PROJECT_TITLE, MAX_PROJECT_ESSAY])

def get_keras_data(df):
    return {
        "teacher_number_of_previously_posted_projects": np.array(df["teacher_number_of_previously_posted_projects"]),
        "project_title": keras.preprocessing.sequence.pad_sequences(df["project_title"], maxlen=MAX_PROJECT_TITLE_SEQ_LEN),
        "project_essay": keras.preprocessing.sequence.pad_sequences(df["project_essay"], maxlen=MAX_PROJECT_ESSAY_SEQ_LEN),
    }

X_train = get_keras_data(processed_train)
X_dev = get_keras_data(processed_dev)


# ## RNN Model
# 
# We now can define a RNN model, train, and then evaluate the model.

# In[11]:


def create_rnn_model():
    # Input layers
    teacher_number_of_previously_posted_projects = keras.layers.Input(shape=(1,), name="teacher_number_of_previously_posted_projects")
    project_title = keras.layers.Input(shape=(MAX_PROJECT_TITLE_SEQ_LEN,), name="project_title")
    project_essay = keras.layers.Input(shape=(MAX_PROJECT_ESSAY_SEQ_LEN,), name="project_essay")
    #project_resource_summary = keras.layers.Input(shape=(MAX_PROJECT_RESOURCE_SUMMARY_SEQ_LEN,), name="project_resource_summary")
    
    # Embedding layers
    #emb_project_title = keras.layers.Embedding(MAX_PROJECT_TITLE, 25)(project_title)
    #emb_project_essay = keras.layers.Embedding(MAX_PROJECT_ESSAY, 50)(project_essay)
    emb_layer = keras.layers.Embedding(MAX_TEXT, 50)
    emb_project_title = emb_layer(project_title)
    emb_project_essay = emb_layer(project_essay)
    
    # RNN layers
    rnn_project_title = keras.layers.GRU(8, activation="relu")(emb_project_title)
    rnn_project_essay = keras.layers.GRU(16, activation="relu")(emb_project_essay)
    #rnn_project_resource_summary = keras.layers.GRU(16, activation="relu")(emb_project_resource_summary)
    
    # Merge all layers into one
    x = keras.layers.concatenate([teacher_number_of_previously_posted_projects,
                                 rnn_project_title,
                                 rnn_project_essay,
                                 #rnn_project_resource_summary,
                                 ])
    
    # Dense layers
    #x = keras.layers.Dense(128, activation="relu")(x)

    # Output layers
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    
    return keras.models.Model(
        inputs=[teacher_number_of_previously_posted_projects,
                project_title,
                project_essay,
                #project_resource_summary,
               ],
        outputs=output)

rnn_model = create_rnn_model()
rnn_model.summary()


# In[12]:


optimizer = keras.optimizers.Adam(lr=0.001)
rnn_model.compile(optimizer=optimizer,
                  loss=keras.losses.binary_crossentropy,
                  metrics=["accuracy"])

for i in range(3):
    rnn_model.fit(X_train, y_train,
                 batch_size=(2 ** (i + 8)),
                 epochs=1,
                 validation_data=(X_dev, y_dev))


# In[13]:


preds = rnn_model.predict(X_dev, batch_size=512)
auc_score = roc_auc_score(y_dev, preds)
print("AUC for validation data: %.4f" % (auc_score,))


# ## Submission

# In[14]:


test_df = pd.read_csv("../input/test.csv", sep=',')

processed_test = preprocess_data(test_df)

X_test = get_keras_data(processed_test)

preds = rnn_model.predict(X_test, batch_size=512)

submission = pd.DataFrame({
    "id": test_df["id"],
    "project_is_approved": preds.reshape(-1),
})

submission.to_csv("submission.csv", index=False)


# ## Something can be tried to improve the model
# 
# - Add more features for input data
# - Increase Embedding output size
# - Try difference learning rate
# - Add BatchNormalization layer
# - Add Dropout layer
# - Add more Des
