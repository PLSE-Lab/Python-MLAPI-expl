#!/usr/bin/env python
# coding: utf-8

# <img src="https://aquariuschannelings.files.wordpress.com/2017/06/sarcasm.jpg?w=930&h=450&crop=1" width="750px"> 
# 
# ## What is Sarcasm?
# 
# According to [Wikipedia](https://simple.wikipedia.org/wiki/Sarcasm), Sarcasm is a figure of speech or speech comment which is extremely difficult to define. It is a statement or comment which means the opposite of what it says. It may be made with the intent of humour, or it may be made to be hurtful.

# ## Objective
# The goal of this work is to make a classifier model to detect the headlines wether it's a sarcasm or not.

# ## Load the data
# 
# The very beginning of everything is loading the data.

# In[ ]:


import pandas as pd

df = pd.read_json("../input/Sarcasm_Headlines_Dataset.json", lines=True)
df.head()


# Next I want to check wether the data has balanced number for each category by using a piechart.

# In[ ]:


import plotly as py
from plotly import graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

# Make pie chart to compare the numbers of sarcastic and not-sarcastic headlines
labels = ['Sarcastic', 'Not Sarcastic']
count_sarcastic = len(df[df['is_sarcastic']==1])
count_notsar = len(df[df['is_sarcastic']==0])
values = [count_sarcastic, count_notsar]
# values = [20,50]

trace = go.Pie(labels=labels,
               values=values,
               textfont=dict(size=19, color='#FFFFFF'),
               marker=dict(
                   colors=['#DB0415', '#2424FF'] 
               )
              )

layout = go.Layout(title = '<b>Sarcastic vs Not Sarcastic</b>')
data = [trace]
fig = go.Figure(data=data, layout=layout)

iplot(fig)


# Well it's seems that our data has unbalance number on each category. It has 13% percent difference (about 3300 data) with *Not Sarcastic* category has more numbers of data. I won't remove some *Not Sarcastic* data to make it balance, instead I will compare the accuracy of each category prediction later.

# ## Text Preprocessing
# 
# <img src="https://www.mememaker.net/api/bucket?path=static/img/memes/full/2017/Apr/22/19/dirty-data-dirty-data-is-everywhere33.jpg" width="300px">

# First let's see how our headlines look like to decide what kind of text preprocess that we need to do.

# In[ ]:


for i,headline in enumerate (df['headline'], 1):
    if i > 20:
        break
    else:
        print(i, headline)


# It's shown that the data has already in the lowercase form. So we just need to clean the data, tokenize it, and then do lemmatization process.

# **1. Text Cleansing**
# 
# The purpose of text cleansing is to remove unnecessary character from our texts. Here we gonna remove *digits* and *punctuations* since they are not needed in our sarcasm detection.

# In[ ]:


import string
from string import digits, punctuation

hl_cleansed = []
for hl in df['headline']:
#     Remove punctuations
    clean = hl.translate(str.maketrans('', '', punctuation))
#     Remove digits/numbers
    clean = clean.translate(str.maketrans('', '', digits))
    hl_cleansed.append(clean)
    
# View comparison
print('Original texts :')
print(df['headline'][37])
print('\nAfter cleansed :')
print(hl_cleansed[37])


# **2. Tokenization**
# 
# Tokenization is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. It's an important process to do in natural language processing since tokenized words will help for words checking or convertion process.

# In[ ]:


# Tokenization process
hl_tokens = []
for hl in hl_cleansed:
    hl_tokens.append(hl.split())

# View Comparison
index = 100
print('Before tokenization :')
print(hl_cleansed[index])
print('\nAfter tokenization :')
print(hl_tokens[index])


# **3. Lemmatization**
# 
# Lemmatization is a process to converting the words of a sentence to its dictionary form, which is known as the *lemma*. Unlike stemming, lemmatization depends on correctly identifying the intended part of speech and meaning of a word in a sentence, as well as within the larger context surrounding that sentence, such as neighboring sentences or even an entire document.
# 
# Here is an example of lemmatization result.

# In[ ]:


# Lemmatize with appropriate POS Tag
# Credit : www.machinelearningplus.com/nlp/lemmatization-examples-python/

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Init Lemmatizer
lemmatizer = WordNetLemmatizer()

hl_lemmatized = []
for tokens in hl_tokens:
    lemm = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    hl_lemmatized.append(lemm)
    
# Example comparison
word_1 = ['skyrim','dragons', 'are', 'having', 'parties']
word_2 = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_1]
print('Before lemmatization :\t',word_1)
print('After lemmatization :\t',word_2)


# # Training Process

# **1. Preparing the Data**
# 
# Before we start the training process with Keras, we need to convert our data so Keras can read and process it. First we should vectorize our data and convert them into sequences.

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Vectorize and convert text into sequences
max_features = 2000
max_token = len(max(hl_lemmatized))
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(hl_lemmatized)
sequences = tokenizer.texts_to_sequences(hl_lemmatized)
X = pad_sequences(sequences, maxlen=max_token)


# Below is an example result of what we did before.

# In[ ]:


index = 10
print('Before :')
print(hl_lemmatized[index],'\n')
print('After sequences convertion :')
print(sequences[index],'\n')
print('After padding :')
print(X[index])


# Next we need to split our data into *training data* and *testing data*. Here we use **X** as a list of data value and **Y** to list of prediction value.

# In[ ]:


from sklearn.model_selection import train_test_split

Y = df['is_sarcastic'].values
Y = np.vstack(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, random_state = 42)


# **2. Building the Model**
# 
# In this project we will using LSTM model. There are also some variables that called *hyperparameters* which we must set before we train our model and their values are somehow intuitive, no strict rules or standards to set the values. They are *embed dim, number of neurons,* and *dropout rate*.

# In[ ]:


from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

embed_dim = 64

model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = max_token))
model.add(LSTM(96, dropout=0.2, recurrent_dropout=0.2, activation='relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


# **3. Training Process**
# 
# Now we are ready to train our model.<br>*yeay!!!*

# In[ ]:


epoch = 10
batch_size = 128
model.fit(X_train, Y_train, epochs = epoch, batch_size=batch_size, verbose = 2)


# **4. Test the Model**
# 
# After we trained our model now we can test our model by count it's accuracy.

# In[ ]:


loss, acc = model.evaluate(X_test, Y_test, verbose=2)
print("Overall scores")
print("Loss\t\t: ", round(loss, 3))
print("Accuracy\t: ", round(acc, 3))


# Next, I want to check the accuracy of each categories since we have unbalanced numbers of data on both categories.

# In[ ]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_test)):
    
    result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.around(result) == np.around(Y_test[x]):
        if np.around(Y_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.around(Y_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1


# In[ ]:


print("Sarcasm accuracy\t: ", round(pos_correct/pos_cnt*100, 3),"%")
print("Non-sarcasm accuracy\t: ", round(neg_correct/neg_cnt*100, 3),"%")


# <img src="https://media.giphy.com/media/3otPoUkg3hBxQKRJ7y/giphy.gif"> 
