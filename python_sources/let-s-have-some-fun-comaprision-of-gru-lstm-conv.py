#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from dataprep.eda import plot
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_data=pd.read_csv("../input/nlp-getting-started/train.csv")
test_data=pd.read_csv("../input/nlp-getting-started/test.csv")
sample=pd.read_csv("../input/nlp-getting-started/sample_submission.csv")


# # dot head() will show dataframe upto five rows, niiiiceeee

# In[ ]:


train_data.head()


# ## Ahan, from Dataframe to Numpy array.
# > dot values will have all the credit
# 

# In[ ]:


train_X=train_data.text.values
train_y=train_data.target.values
test_X=test_data.text.values


# *Okie, remove all bullshit from text*

# In[ ]:


import re
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def preprocess(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text
    text = REPLACE_BY_SPACE_RE.sub('',text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([x for x in text.split() if x and x not in STOPWORDS]) # delete stopwords from text
    return text
train_X=[preprocess(x) for x in train_X]


# In[ ]:


train_X,val_X,train_y,val_y=train_test_split(train_X,train_y,shuffle=True)
print(train_X[:5])
print("--------------------------")
print(train_y[:5])


# In[ ]:


train_X=np.array(train_X)
train_X.shape,train_X.shape


# **We have words but our NN can't really understand them , ughh**
# 
# > WHYYYYYYYYYY ???, so much drama NN.
# 
# **so we do need to tokenize them and convert into sequences**
# 
# *I am TENSORFLOW, i got your back..*
# 

# In[ ]:


vocab_size=1000
trun='post'
max_length=150
tokenizer=Tokenizer(num_words=vocab_size,oov_token='<OOV>')
tokenizer.fit_on_texts(train_X)
word_index=tokenizer.word_index
seq=tokenizer.texts_to_sequences(train_X)
padded=pad_sequences(seq,maxlen=max_length,truncating=trun,padding='post')

val_seq=tokenizer.texts_to_sequences(val_X)
val_padded=pad_sequences(val_seq,maxlen=max_length,truncating=trun,padding='post')


# ## Uncomment below and see majiiicccc 

# In[ ]:


#word_index.items()


# # Embedding network

# In[ ]:


model_emb=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,20,input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D() ,
    tf.keras.layers.Dense(64,activation='elu'),
    tf.keras.layers.Dense(128,activation='elu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(264,activation='elu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()


# In[ ]:


model_emb.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history=model_emb.fit(padded,train_y,epochs=20,validation_data=(val_padded,val_y))


# In[ ]:


def want_plot_call_me(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string,'val_'+string])
    plt.show()
    
want_plot_call_me(history,'accuracy')
want_plot_call_me(history,'loss')


# In[ ]:


def predict_and_sub(testt,model,name):
    '''
        This guy will save model into csv.
        testt: test set
        model: NN model
        name: file name
    '''
    test=[preprocess(x) for x in testt]
    test=np.array(test)
    test_seq=tokenizer.texts_to_sequences(test)
    test_padded=pad_sequences(test_seq,maxlen=max_length,truncating=trun,padding='post')  
    pred=model.predict(test_padded).squeeze()
    pred=[1 if x>0.5 else 0 for x in pred]
    pred=np.array(pred)
    sample.target=pred
    sample.to_csv(name,index=False)
    


# # Lets do submission part

# In[ ]:


predict_and_sub(test_X,model_emb,"model_emb.csv")


# # Turn for CONV1d boi

# In[ ]:


model_cnv=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,20,input_length=max_length),
    tf.keras.layers.Conv1D(64,5,activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D() ,
    tf.keras.layers.Dense(64,activation='elu'),
    tf.keras.layers.Dense(128,activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(264,activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()


# In[ ]:


model_cnv.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history=model_cnv.fit(padded,train_y,epochs=20,validation_data=(val_padded,val_y))


# In[ ]:


want_plot_call_me(history,'accuracy')
want_plot_call_me(history,'loss')


# In[ ]:


predict_and_sub(test_X,model_cnv,"model_cnv.csv")


# # why not The Great LLLLLSSSSTTTMMMMM

# In[ ]:


model_lstm=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,25,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()


# In[ ]:


model_lstm.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history=model_lstm.fit(padded,train_y,epochs=20,validation_data=(val_padded,val_y))


# In[ ]:


want_plot_call_me(history,'accuracy')
want_plot_call_me(history,'loss')


# In[ ]:


predict_and_sub(test_X,model_lstm,"model_lstm.csv")


# In[ ]:


model_gru=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,20,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Dense(64,activation='elu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(264,activation='elu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(264,activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()


# In[ ]:


model_gru.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history=model_gru.fit(padded,train_y,epochs=20,validation_data=(val_padded,val_y))


# In[ ]:


want_plot_call_me(history,'accuracy')
want_plot_call_me(history,'loss')


# In[ ]:


predict_and_sub(test_X,model_gru,"model_gru.csv")


# In[ ]:




