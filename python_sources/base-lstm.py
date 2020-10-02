#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import os
import pandas as pd
import re


# In[ ]:


pwd


# In[ ]:


news_category = ["business", "entertainment", "politics", "sport", "tech"]

row_doc = "../input/bbc-news-summary/BBC News Summary/News Articles/"
summary_doc = "../input/bbc-news-summary/BBC News Summary/Summaries/"

data={"articles":[], "summaries":[]}


# In[ ]:


directories = {"news": row_doc, "summary": summary_doc}
row_dict = {}
sum_dict = {}

for path in directories.values():
    if path == row_doc:
        file_dict = row_dict
    else:
        file_dict = sum_dict
    dire = path
    for cat in news_category:
        category = cat
        files = os.listdir(dire + category)
        file_dict[cat] = files


# In[ ]:



row_data = {}
for cat in row_dict.keys():
    cat_dict = {}
    # row_data_frame[cat] = []
    for i in range(0, len(row_dict[cat])):
        filename = row_dict[cat][i]
        path = row_doc + cat + "/" + filename
        with open(path, "rb") as f:                
            text = f.read()
            cat_dict[filename[:3]] = text
    row_data[cat] = cat_dict


# In[ ]:


sum_data = {}
for cat in sum_dict.keys():
    cat_dict = {}
    # row_data_frame[cat] = []
    for i in range(0, len(sum_dict[cat])):
        filename = sum_dict[cat][i]
        path = summary_doc + cat + "/" + filename
        with open(path, "rb") as f:                
            text = f.read()
            cat_dict[filename[:3]] = text
    sum_data[cat] = cat_dict


# In[ ]:


news_business = pd.DataFrame.from_dict(row_data["business"], orient="index", columns=["row_article"])
news_business.head()


# In[ ]:


news_category = ["business", "entertainment", "politics", "sport", "tech"]
news_entertainment = pd.DataFrame.from_dict(row_data["entertainment"], orient="index", columns=["row_article"])
news_politics = pd.DataFrame.from_dict(row_data["politics"], orient="index", columns=["row_article"])
news_sport = pd.DataFrame.from_dict(row_data["sport"], orient="index", columns=["row_article"])
news_tech = pd.DataFrame.from_dict(row_data["tech"], orient="index", columns=["row_article"])


# In[ ]:


# summary data
summary_business = pd.DataFrame.from_dict(sum_data["business"], orient="index", columns=["summary"])
summary_entertainment = pd.DataFrame.from_dict(sum_data["entertainment"], orient="index", columns=["summary"])
summary_politics = pd.DataFrame.from_dict(sum_data["politics"], orient="index", columns=["summary"])
summary_sport = pd.DataFrame.from_dict(sum_data["sport"], orient="index", columns=["summary"])
summary_tech = pd.DataFrame.from_dict(sum_data["tech"], orient="index", columns=["summary"])


# In[ ]:


summary_business.head()


# In[ ]:


business = news_business.join(summary_business, how='inner')
entertainment = news_entertainment.join(summary_entertainment, how='inner')
politics = news_politics.join(summary_politics, how='inner')
sport = news_sport.join(summary_sport, how='inner')
tech = news_tech.join(summary_tech, how='inner')


# In[ ]:


business = news_business.join(summary_business, how='inner')


# In[ ]:


business.head()


# In[ ]:


print("row", len(business.iloc[0,0]))
print("sum", len(business.iloc[0,1]))


# In[ ]:


list_df = [business, entertainment, politics, sport, tech]
length = 0
for df in list_df:
    length += len(df)


# In[ ]:


print("length of all data: ", length)


# In[ ]:


bbc_df = pd.concat([business, entertainment, politics, sport, tech], ignore_index=True)
len(bbc_df)


# Step 2. Preprocessing Text Data.
# 
#     Clean Text
#     Tokenize
#     Vocabrary
#     Padding
#     One-Hot Encoding
#     Reshape to (MAX_LEN, One-Hot Encoding DIM)

# In[ ]:


def cleantext(text):
    text = str(text)
    text=text.split()
    words=[]
    for t in text:
        if t.isalpha():
            words.append(t)
    text=" ".join(words)
    text=text.lower()
    text=re.sub(r"what's","what is ",text)
    text=re.sub(r"it's","it is ",text)
    text=re.sub(r"\'ve"," have ",text)
    text=re.sub(r"i'm","i am ",text)
    text=re.sub(r"\'re"," are ",text)
    text=re.sub(r"n't"," not ",text)
    text=re.sub(r"\'d"," would ",text)
    text=re.sub(r"\'s","s",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"can't"," cannot ",text)
    text=re.sub(r" e g "," eg ",text)
    text=re.sub(r"e-mail","email",text)
    text=re.sub(r"9\\/11"," 911 ",text)
    text=re.sub(r" u.s"," american ",text)
    text=re.sub(r" u.n"," united nations ",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r":"," ",text)
    text=re.sub(r"-"," ",text)
    text=re.sub(r"\_"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[$#@%&*!~?%{}()]"," ",text)
    
    return text


# In[ ]:


for col in bbc_df.columns:
    bbc_df[col] = bbc_df[col].apply(lambda x: cleantext(x))


# In[ ]:


bbc_df.head()


# In[ ]:


df.head()


# In[ ]:


len_list =[]
for article in df.row_article:
    words = article.split()
    length = len(words)
    len_list.append(length)
max(len_list)


# # 2-2. Tokenizer
#     Tokenize and One-Hot : Tokenizer
#     Vocabraly: article and summary 15000 words
#     Padding: pad_sequences 1000 max_len
#     Reshape: manual max_len * one-hot matrix

# In[ ]:


import numpy as np
import os
import pandas as pd
import re


# In[ ]:


articles = list(bbc_df.row_article)
summaries = list(bbc_df.summary)


# In[ ]:


articles


# In[ ]:


# from sklearn.model_selection import train_test_split
# art_train, art_test, sum_train, sum_test = train_test_split(pad_art_sequences, pad_sum_sequences, test_size=0.2)


# In[ ]:


from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 1999
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(articles)
article_sequences = tokenizer.texts_to_sequences(articles)
art_word_index = tokenizer.word_index
len(art_word_index)


# In[ ]:



print(article_sequences[0][:20])
print(article_sequences[1][:20])
print(article_sequences[2][:20])


# In[ ]:


# Vocabraly: article and summary 15000 words
art_word_index_1500 = {}
counter = 0
for word in art_word_index.keys():
    if art_word_index[word] == 0:
        print("found 0!")
        break
    if art_word_index[word] > VOCAB_SIZE:
        continue
    else:
        art_word_index_1500[word] = art_word_index[word]
        counter += 1


# In[ ]:


counter


# In[ ]:


tokenizer.fit_on_texts(summaries)
summary_sequences = tokenizer.texts_to_sequences(summaries)
sum_word_index = tokenizer.word_index
len(sum_word_index)


# In[ ]:


sum_word_index_1500 = {}
counter = 0
for word in sum_word_index.keys():
    if sum_word_index[word] == 0:
        print("found 0!")
        break
    if sum_word_index[word] > VOCAB_SIZE:
        continue
    else:
        sum_word_index_1500[word] = sum_word_index[word]
        counter += 1


# In[ ]:



counter


# In[ ]:


#Padding: pad_sequences 1000 max_len
from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 400
pad_art_sequences = pad_sequences(article_sequences, maxlen=MAX_LEN, padding='post', truncating='post')


# In[ ]:


print(len(article_sequences[1]), len(pad_art_sequences[1]))


# In[ ]:


pad_sum_sequences = pad_sequences(summary_sequences, maxlen=MAX_LEN, padding='post', truncating='post')


# In[ ]:


print(len(summary_sequences[1]), len(pad_sum_sequences[1]))


# In[ ]:


pad_art_sequences.shape


# In[ ]:


pad_art_sequences


# In[ ]:


# Reshape: manual max_len * one-hot matrix
"""
encoder_inputs = np.zeros((2225, 400), dtype='float32')
encoder_inputs.shape

decoder_inputs = np.zeros((2225, 400), dtype='float32')
decoder_inputs.shape

for i, seqs in enumerate(pad_art_sequences):
    for j, seq in enumerate(seqs):
        encoder_inputs[i, j] = seq
        
for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        decoder_inputs[i, j] = seq
"""


# In[ ]:


decoder_outputs = np.zeros((2225,400, 2000), dtype='float32')
decoder_outputs.shape


# In[ ]:


for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        decoder_outputs[i, j, seq] = 1.


# In[ ]:


decoder_outputs.shape


# In[ ]:


embeddings_index = {}
with open('../input/glove6b50d/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[ ]:


art_embedding_matrix = embedding_matrix_creater(50, word_index=art_word_index_1500)
art_embedding_matrix.shape


# In[ ]:


sum_embedding_matrix = embedding_matrix_creater(50, word_index=sum_word_index_1500)
sum_embedding_matrix.shape


# In[ ]:


from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
encoder_embedding_layer = Embedding(input_dim = 2000, 
                                    output_dim = 50,
                                    input_length = MAX_LEN,
                                    weights = [art_embedding_matrix],
                                    trainable = False)


# In[ ]:


decoder_embedding_layer = Embedding(input_dim = 2000, 
                                    output_dim = 50,
                                    input_length = MAX_LEN,
                                    weights = [sum_embedding_matrix],
                                    trainable = False)


# In[ ]:


sum_embedding_matrix.shape


# In[ ]:





# In[ ]:


pip install chart-studio


# In[ ]:


# Building Encoder-Decoder Model
from numpy.random import seed
seed(1)


from sklearn.model_selection import train_test_split
import logging

import chart_studio.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
import pydot


import keras
from keras import backend as k
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,concatenate, Embedding, RepeatVector
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


# In[ ]:





# In[ ]:


# Hyperparams

MAX_LEN = 400
VOCAB_SIZE =1999
EMBEDDING_DIM = 50
HIDDEN_UNITS = 200
VOCAB_SIZE = VOCAB_SIZE + 1

LEARNING_RATE = 0.9
BATCH_SIZE = 128
EPOCHS =20


# In[ ]:


# encoder
encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_LSTM = LSTM(HIDDEN_UNITS)(encoder_embedding)
# decoder
decoder_inputs = Input(shape=(MAX_LEN, ))
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_LSTM = LSTM(200)(decoder_embedding)
# merge
merge_layer = concatenate([encoder_LSTM, decoder_LSTM])
decoder_outputs = Dense(units=VOCAB_SIZE+1, activation="softmax")(merge_layer) # SUM_VOCAB_SIZE, sum_embedding_matrix.shape[1]

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# In[ ]:


#Training your model and Validate it
import numpy as np
num_samples = len(pad_sum_sequences)
decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="int32")


# In[ ]:


# output
for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        if j > 0:
            decoder_output_data[i][j][seq] = 1


# In[ ]:


art_train, art_test, sum_train, sum_test = train_test_split(pad_art_sequences, pad_sum_sequences, test_size=0.2)


# In[ ]:


train_num = art_train.shape[0]
train_num


# In[ ]:


target_train = decoder_output_data[:train_num]
target_test = decoder_output_data[train_num:]


# In[ ]:


history = model.fit([art_train, sum_train], 
                     target_train, 
                     epochs=EPOCHS, 
                     batch_size=BATCH_SIZE,
                     validation_data=([art_test, sum_test], target_test))


# In[ ]:




