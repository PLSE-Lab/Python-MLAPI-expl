#!/usr/bin/env python
# coding: utf-8

# # RoBERTa with mean and max pooling
# 
# This notebook shows you how to set up a classifier based on RoBERTa, with mean and max pooling.
# 
# ## Read the data
# 
# We load the train and test datasets.

# In[ ]:


import pandas as pd

train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv").sample(frac=1.)
test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")


# ## Tokenize the tweets
# 
# We clean the tweets a little by removing tokens that are shorter than 2 characters or longer than 15 characters.

# In[ ]:


from transformers import *
from gensim.utils import simple_preprocess
import numpy as np

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
X_train = np.array([tokenizer.encode(" ".join(simple_preprocess(text, min_len=2, max_len=15)),
                                     add_special_tokens=True, 
                                     max_length=40, 
                                     pad_to_max_length=True) for text in train["text"]])
X_test = np.array([tokenizer.encode(" ".join(simple_preprocess(text, min_len=2, max_len=15)), 
                           add_special_tokens=True, 
                           max_length=40, 
                           pad_to_max_length=True) for text in test["text"]])


# ## RoBERTa with mean and max pooling
# 
# We embed the words using RoBERTa, we perform mean and max pooling over the embedded sequences 

# In[ ]:


from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow import int32
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

# Input layer
input_ids = Input((40,), dtype=int32)
# RoBERTa layer
lm = TFRobertaModel.from_pretrained('roberta-base')
sequence, cls = lm(input_ids) 
# Parallel mean and max global pooling
mean_pooling = GlobalAveragePooling1D()(sequence)
max_pooling = GlobalMaxPooling1D()(sequence)
pooling = concatenate([mean_pooling, max_pooling])
# Dropout layer
dropout = Dropout(0.5, name="dropout")(pooling)
# Classification layer
classification = Dense(1, activation="sigmoid", kernel_initializer=glorot_normal(seed=1), bias_initializer=glorot_normal(seed=1), name="classification")(pooling)
model = Model(input_ids, classification)
plot_model(model)


# We train the model, with a linearly decaying learning rate.

# In[ ]:


from tensorflow.keras.callbacks import LearningRateScheduler

def rate(epoch):
    return 1.5e-5/(epoch + 1)

scheduler = LearningRateScheduler(rate)
model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999), 
              loss="binary_crossentropy", 
              metrics=["accuracy"])
log = model.fit(X_train, train["target"].values, 
                callbacks=[scheduler],
                batch_size=12, 
                epochs=3, 
                verbose=1)


# ## Submit predictions

# In[ ]:


test["proba"] = model.predict(X_test)
test["target"] = test["proba"].apply(lambda p: int(p > 0.5))
test[["id", "target"]].to_csv("/kaggle/working/submission.csv", index=False)

