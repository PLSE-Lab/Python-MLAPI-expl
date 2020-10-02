#!/usr/bin/env python
# coding: utf-8

# $$
# \huge\text{Modeling Online Toxicity with LSTMs}\\
# \large\text{An interactive lab}\\
# \text{Andrew Riberio @ https://github.com/Andrewnetwork}
# $$
# [ Skill Level : Beginner ] [ Interactive ]
# 
# Welcome to this interactive laboratory for exploring the application of LSTM's to modeling toxicity in online dialog.
# 
# Outline
# 
# 1. Libraries and Environment Setup
# 2. Preprocessing 
# 3. Modeling 
# 4. Visualizing and Interpreting Results
# 
# Resources and Sources 
# * https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
# * http://www.deeplearningbook.org/contents/rnn.html
# 
# **NOTE**: In order for the interactive componnents of this kernel to function, you must either fork this kernel or download it to your local machine which has the required environment and dependencies. The simplest method is to fork the kernel here on kaggle. 

# **$\large\text{1. Libraries and Environment Setup}$**

# In[ ]:


import sys, os, re, csv, codecs
import numpy as np 
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

from ipywidgets import interact,interact_manual
from IPython.display import display

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# $\large\text{2. Preprocessing }$

# Load our training and test data using pandas into dataframe variables amicably named. 
# 
# *Note:* If you are using this kernel as a fork on kaggle, you will not need to change the training and test paths. If you have a local version of this kernel, you must download the training data and change the paths below to your local copy of the data. 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# Basic data information.

# In[ ]:


print("Shape of the training data: {0}.".format(train.shape) )
print("Shape of the test data: {0}.".format(test.shape))

# View some of the training data. 
train.head()


# Interactive data visualization

# In[ ]:


# Cashed filter computations for the visualization below. 
cashedFilters = {}


# In[ ]:


# An interactive widget for exploring the data. 
def visRow(row):
    print( row["comment_text"] )
    print ("{0} - Toxic( {1} ) - Severe Toxic ( {2} ) - Obscene( {3} ) - Threat ( {4} ) - Insult ( {5} ) - Identity Hate ( {6} )"
           .format(row["id"],row["toxic"],row["severe_toxic"],
                   row["obscene"],row["threat"],row["insult"],row["identity_hate"]))
    
def categoryVis(toxic,severeToxic, obscene, threat, insult, identHate,idx=0):
    try:
        visRow(cashedFilters[toxic,severeToxic, obscene, threat, insult, identHate].iloc[idx])
        
    except KeyError:
        print("Computing filter...")
        filterRes = train[train.apply(lambda x: x["toxic"] == toxic and x["severe_toxic"] == severeToxic and x["obscene"] == obscene 
                                  and x["threat"] == threat and x["insult"] == insult and x["identity_hate"] == identHate , axis=1)]
        cashedFilters[toxic,severeToxic, obscene, threat, insult, identHate] =  filterRes
        print("Done.\n---------")
        
        visRow(filterRes.iloc[idx])
    
interact(categoryVis,toxic=False,severeToxic=False, obscene=False, threat=False, insult=False, identHate=False,idx=(0,100))


# Split training data into data and label vectors. 

# In[ ]:


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y_train = train[list_classes].values
list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]


# Tokenize the words. 

# In[ ]:


max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


# The following shows the result of the tokenizing process. Our text comments are now sequences of token ids.

# In[ ]:


print(list_tokenized_train[:1])


# Padd our sequences so we can have a fixed training input size. 

# In[ ]:


maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)


# In[ ]:


totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords,bins = np.arange(0,410,10))
plt.show()


# $\large\text{3. Modeling }$

# In[ ]:


embed_size = 128

inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


# In[ ]:


batch_size = 32
epochs = 2
model.fit(X_t,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[ ]:


model.save('lstm_toxic.h5')


# $\large\text{4. Visualizing and Interpreting Results }$

# In[ ]:


def testModel(text):
    textSeq = tokenizer.texts_to_sequences([text])
    t = pad_sequences(textSeq, maxlen=maxlen)
    prediction = model.predict(t)[0]
    print("Toxic( {0:f} ) - Severe Toxic ( {1:f} ) - Obscene( {2:f} ) - Threat ( {3:f} ) - Insult ( {4:f} ) - Identity Hate ( {5:f} )"
          .format(prediction[0],prediction[1],prediction[2],prediction[3],prediction[3],prediction[4],prediction[5]))
interact(testModel,text="Testing 1 2 3")


# In[ ]:


testPred = model.predict(X_te)


# In[ ]:


["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
data_to_submit = pd.DataFrame.from_items([
    ('id',test["id"]),
    ('toxic',testPred[:,0]),
    ('severe_toxic',testPred[:,1]),
    ('obscene',testPred[:,2]),
    ('threat',testPred[:,3]),
    ('insult',testPred[:,4]),
    ('identity_hate',testPred[:,5])
])


# In[ ]:


data_to_submit.to_csv('csv_to_submit.csv', index = False)


# In[ ]:


data_to_submit.head()

