#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


# In[ ]:


data = pd.read_csv('../input/Sentiment.csv')

# we check how much null entries in each columns

print("data_is_null \n",data.isnull().sum())

# Keeping only the neccessary columns
data = data[['text','sentiment']]


# In[ ]:


data[:5]


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.columns


# In[ ]:


data.dtypes


# In[ ]:


data.isnull().sum()


# In[ ]:


print(data.size)
data[:5]


# In[ ]:


# it will remove all Neutral values from data
data = data[data.sentiment != "Neutral"]
print(data[:2])

# it will remove all the eg:-  RT @NancyLeeGrahn:  
data['text'] = data['text'].apply(lambda x: x.lower())
print(data[:2])
#print(data)
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
print(data[:2])
print(data.dtypes)
print(data[ data['sentiment'] == 'Positive'].size)
print(data[ data['sentiment'] == 'Negative'].size)
print(data[data['sentiment']!='Neutral'].size)

for idx,row in data.iterrows():
    #print(idx,row)
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
print(X[:5])


# pad: to make all input of same length
X = pad_sequences(X)
print(X[:5])


# In[ ]:


# Keras offers an Embedding layer that can be used for neural networks on text data.
'''
source from: machinelearningmastery.com
The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments:

It must specify 3 arguments:

input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.
For example, below we define an Embedding layer with a vocabulary of 200 (e.g. integer encoded words from 0 to 199, inclusive), a vector space of 32 dimensions in which words will be embedded, and input documents that have 50 words each.


e = Embedding(200, 32, input_length=50)
1
e = Embedding(200, 32, input_length=50)


'''
embed_dim = 128
lstm_out = 196
import time
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
start=time.time()
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print("Time to compile model:",time.time()-start)
print(model.summary())


# In[ ]:


print((data['sentiment']).values)
Y = pd.get_dummies(data['sentiment']).values
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.15, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


from tqdm import tqdm
batch_size = 32
tqdm(model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2))


# In[ ]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[ ]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")


# In[ ]:


twt = ['Meetings: ram is a good man.']
#vectorizing the tweet by the pre-fitted tokenizer instance
twt = tokenizer.texts_to_sequences(twt)
#padding the tweet to have exactly the same shape as `embedding_2` input
twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
print(twt)
sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
if(np.argmax(sentiment) == 0):
    print("negative")
elif (np.argmax(sentiment) == 1):
    print("positive")

