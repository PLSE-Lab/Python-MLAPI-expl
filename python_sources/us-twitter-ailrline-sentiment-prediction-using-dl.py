#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install keras -U


# In[ ]:


import nltk
nltk.download('stopwords')


# ## **Import Library**

# In[ ]:


from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np 
import keras.utils as ku
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input,Embedding, LSTM, Dense
from keras.layers import Dense
from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt


# ## **Import Dataframe**

# In[ ]:


df = pd.read_csv("../input/twitter-airline-sentiment/Tweets.csv" ,encoding="utf-8")


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.head()


# ## **Data Preprocessing**

# In[ ]:


df['text'] = df['text'].str.strip()
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
df['text'] = df['text'].apply(lambda x: re.sub("\s+", " ", x))


# In[ ]:


x = df['text']
y = df['airline_sentiment']


# In[ ]:


words = []
for i in range(len(x)):
    words.extend(x[i].split())

words = list(set(words))
    
stop_words = set(stopwords.words('english'))

filtered_words = [w for w in words if not w in stop_words] 

max_words = len(filtered_words)

print("Max Words =  ", max_words)


# In[ ]:


y = y.astype('category')
y = y.cat.codes
y                                 # 1  --> "neutral", 2 --> "positive", 3 -->"negative"


# In[ ]:


max_len = len(max(x,key = len))
print("Max Length of String = ",max_len)


# In[ ]:


tokenizer = Tokenizer( num_words=max_words )
tokenizer.fit_on_texts(x)
x_seq = tokenizer.texts_to_sequences(x)
x_seq = pad_sequences(x_seq,maxlen = max_len, padding='pre')
x_seq[1]


# In[ ]:


y = y.values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(x_seq, y, test_size=0.40, random_state=42)
print("Training Data = ",len(x_train))
print("Testing Data = ",len(x_test))


# ### **Import Model & Layer using Functional API**

# In[ ]:


input = Input(shape=(max_len,))
embed = Embedding(input_dim= max_words,output_dim=512,input_length=max_len )(input)
lstm_1 = LSTM(128, return_sequences=True , dropout=0.4 )(embed)
lstm_2 = LSTM(64,dropout=0.3 )(lstm_1)
dense_1 = Dense(32 , activation='relu')(lstm_2)
predictions = Dense(3,activation='softmax')(dense_1)
model = Model(inputs=input, outputs=predictions)


# In[ ]:


print(model.summary())


# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model.fit(x_train,y_train ,verbose=1,epochs=12 , validation_data=(x_test[:2000],y_test[:2000]))


# In[ ]:


# plot graph
plot_model(model, to_file='multiple_outputs.png')


# In[ ]:


score,acc = model.evaluate(x_test, y_test,verbose = 1)
print('Test score:', score)
print('Test accuracy:', acc)
print()
score,acc = model.evaluate(x_train, y_train,verbose = 1)
print('Training score:', score)
print('Training accuracy:', acc)


# ## **Thank You**
