#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[ ]:


from sklearn.utils import shuffle

data = pd.read_csv("../input/training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None)

# In[3]:


data.columns = ['sentiment', 'id', 'date', 'q', 'user', 'text']


# In[4]:



data = data.drop(['id', 'date', 'q', 'user'], axis=1)

# In[5]:


data = shuffle(data)
data = data[data.sentiment != 2] 
data.sentiment = data.sentiment /4
data = shuffle(data)
data = data[:300000]


# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
num = 0
def clean_text(text):
    text = text.lower().replace('rt', '')
    text = "".join([ch for ch in text if ch not in string.punctuation])
    text = ' '.join(re.sub(r"(@[A-Za-z0-9]+( tweeted:)?)|([^0-9A-Za-z \t])|(https?\S*)|(\w+:\/\/\S+)"
                           , "", text).split())
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    stems = [stemmer.stem(item) for item in tokens]
    global num
    num += 1
    if num % 30000 == 0:
        print(num)
    return ' '.join(stems)


# In[ ]:


data = data[['text','sentiment']]


# In[ ]:


data['text'] = data['text'].apply(clean_text)
print(data.text)

max_fatures = 5000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)


# In[ ]:


from keras import regularizers

embed_dim = 128
lstm_out = 256

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2, kernel_regularizer=regularizers.l2(0.0004),
                activity_regularizer=regularizers.l1(0.0002)))
model.add(Dense(2,activation='softmax', kernel_regularizer=regularizers.l2(0.0004),
                activity_regularizer=regularizers.l1(0.0002)))
model.compile(loss = 'categorical_crossentropy', optimizer='adadelta',metrics = ['accuracy'])
print(model.summary())


# In[ ]:


Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test,Y_test, test_size = 0.8, random_state = 13)

print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


batch_size = 100
model.fit(X_train, Y_train, epochs = 20, batch_size=batch_size, verbose = 2)


# In[ ]:


model.save('sentiment_big_regularized_27')
print('model saved')


# In[ ]:


import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
print ('tokenizer saved')   


# In[ ]:



score,acc = model.evaluate(X_validate, Y_validate, verbose = 2, batch_size = batch_size)
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





# In[ ]:





# In[ ]:





# In[ ]:




