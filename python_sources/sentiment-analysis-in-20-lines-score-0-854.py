#!/usr/bin/env python
# coding: utf-8

# ## <center> Predict Sentiment Analysis over Movie Reviews

# In[ ]:


#Import necessary packages
import pandas as pd
import numpy as np
import re
import nltk
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
# nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import warnings
warnings.filterwarnings('ignore')

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Import dataset
df_train = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv', delimiter='\t')

df_train.head()


# ### Function to clean text.

# In[ ]:


#Function to clean text
def clean_text(input_text):
    """
    Processes the give text and removes all non words, digits, single letters and extra spaces.

    Parameters
    -----------
    1. input_text = Text to clean.
    2. token = 'word' or 'sentence'

    Returns: Text.

    """

    text = re.sub(r'\W',' ', input_text) #Remove all non words
    text = re.sub(r'\d+',' ', text) #Remove all digits
    text = text.lower() #Converting text into lowercase
    text = re.sub(r'\s+[a-z]\s+',' ', text) #Remove all single letters
    text = re.sub(r'^\s+','', text) #Remove space from start of text
    text = re.sub(r'\s+$','', text) #Remove space from end of text
    text = re.sub(r'\s+',' ', text) #Remove all multi space    
    text = text.split(' ') #Split the words into tokens
    text = [word for word in text if word not in stop_words] #Remove stopwords
    text = [WordNetLemmatizer().lemmatize(word) for word in text] #Lemmatize the words(get root form)
    text = ' '.join(text)

    return text


# ### Validate the clean text function

# In[ ]:


#Actual Text
temp = df_train.loc[0,'review']
temp


# In[ ]:


#Cleaned text
clean_text(temp)


# In[ ]:


#Apply clean text over movie reviews
df_train['processed_reviews'] = df_train['review'].apply(lambda x: clean_text(x))


# In[ ]:


#Review dataset
df_train.head()


# In[ ]:


#Finding average total length of words in review
df_train['processed_reviews'].apply(lambda x: len(x.split(' '))).mean()


# ## Preprocessing the data

# In[ ]:


max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df_train['processed_reviews'])
list_tokens = tokenizer.texts_to_sequences(df_train['processed_reviews'])


# In[ ]:


maxlen = 130 #Selected from mean of text length
X_train = pad_sequences(list_tokens, maxlen=maxlen)
y_train = df_train['sentiment']


# ## Building Model

# In[ ]:


embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


batch_size = 100
epochs = 3
model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


# ## Read the Test Data

# In[ ]:


df_test = pd.read_csv('/kaggle/input/word2vec-nlp-tutorial/testData.tsv', delimiter='\t')
df_test.head()


# ### Clean Test data

# In[ ]:


df_test['processed_review'] = df_test['review'].apply(lambda x: clean_text(x))


# ### Preprocess Test Data

# In[ ]:


list_sentences_test = df_test['processed_review']
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)


# ### Make Prediction

# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


#Set the prediction value
p=[]
for val in prediction:
    if val > 0.5:
        p.append(1)
    else:
        p.append(0)
        
sr_pred = pd.Series(data=p, name='sentiment')
sr_pred[:5]


# ### Preparing Submission File

# In[ ]:


submission = pd.DataFrame(columns=['id', 'sentiment'])
submission['id'] = df_test['id']
submission['sentiment'] = sr_pred
submission.head()


# In[ ]:


submission.to_csv('first_submission.csv', index=False)


# ### Secured score - 0.85456

# **Credits:** This work is insipired this notebook from Nilan - https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
