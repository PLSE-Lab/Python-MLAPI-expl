#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis on 2020 US presidential Election Candidates using Twitter Data.
# **Data Information**: Tweets from 1 Republican Party candidate(Donald Trump) and 3 Democrat Party candidates(Joe Biden, Elizabeth Warren, Bernie Sanders) <br>
# **Question**: Which Candidate is more negative on which topic? <br>
# **Method**: Word Embedding Model(FastText), LSTM-CNN model (RCNN) 

# ### Required Library

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import pandas as pd


# ### Data Preview
# There are 4 dataset: Trump, and 3 Democrat party Candidates, Joe Biden, Elizabeth Warren and Bernie Sanders <br>
# I collected their tweets from '2019-01-01' to '2019-11-12'. <br>
# I merged individual csv files into one df_tweet file (df_politics), just to make sure every dataset is perfectly arranged

# In[ ]:


tweets = "../input"
embedding_path = "../input/crawl-300d-2M.vec"  #where word2vector model is 

df_Trump = pd.read_csv(os.path.join(tweets,"DonaldTrump.csv"))
df_Bernie = pd.read_csv(os.path.join(tweets, "BernieSanders.csv"))
df_Joe = pd.read_csv(os.path.join(tweets, "JoeBiden.csv"))
df_Elizabeth = pd.read_csv(os.path.join(tweets, "ElizabethWarren.csv"))
df_politics = pd.concat((df_Trump, df_Bernie, df_Joe, df_Elizabeth), axis=0)
df_politics.head()


# ### Number of the data
# There are total of 11762 tweets. This might seem like a lot of data, but actually relatively small data compared to those which is used in other machine learning projects. <br>
# 
# 
# 
# 
# *   Donald Trump: 3929 tweets
# *   Elizabeth Warren: 3645 tweets 
# *   Bernie Sanders: 2681 tweets <br>
# * Joe Biden: 1507 tweets <br>
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


df_politics.shape


# ### Tokenizer
# 
# 
# 1.   Separe full text into list of words. For example,
# 
# > 'I love you' => ['I', 'love', 'you']
# 
# 2.   Transform list of words into list of numbers. For example,
# 
# >['I', 'love', 'you'] => [16, 322, 22]
# 
# 3. Make each list of numbers into vectorized value. For example,
# ```
# array([[    0,     0,     0, ...,    30, 10547,    14],
#        [    0,     0,     0, ...,    97,     2,   759],
#        [    0,     0,     0, ...,   329,   488,   120],
#        ...,
#        [    0,     0,     0, ...,    11,     7, 25414],
#        [    0,     0,     0, ...,    11,     7, 25415],
#        [    0,     0,     0, ...,    11,     7, 25419]], dtype=int32)
# ```
# 
# 

# In[ ]:


embed_size = 300
max_features = 130000
max_len = 220

df_politics["text"].fillna("no comment")   #if there is any "None" text in data, delete the entire row
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]   #Negative feeling lists
raw_text = df_politics["text"].str.lower()   

tk = Tokenizer(num_words = max_features, lower = True)  #Implement tokenizer model (it helps to divide the text into list of several words. For example, [I love apple] -> ['I', 'love', 'apple'])
tk.fit_on_texts(raw_text)  # Tokenize tweets 
df_politics["comment_seq"] = tk.texts_to_sequences(raw_text)   #Then make them into a number. For example, "I love apple " -> [[16, 322, 2271]]. Insert the printed number into the 'df_politics' table

tweets_pad_sequences = pad_sequences(df_politics.comment_seq, maxlen = max_len) #each row of the table has different length of "comment_seq". Make them into equal length of vector. 
print("The vectorized tweet's shape is: ", tweets_pad_sequences.shape) # Each text has transformed into list of numbers which has length of 84. 

df_politics[['text', 'comment_seq']]


# ### Embedding Words into Vector space
# Now I need to construct an embedding index, that puts each one of the words into a 300-dimensional vector space.<br>
# I used 'FastText' model which was developed from Facebook in 2017. <br>
# This is a very revolutionary model, far better than 'GLOVE' or 'Word2Vec'. It calculates the distance between each words in its semantics, and can be redistributed even though there is a new word.
# 

# In[ ]:


# def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
# embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path, encoding ="utf-8"))   #This code is to open embedding "Glove model"


# In[ ]:


# word_list = tk.word_index
# n_words = min(max_features, len(word_list))
# print("number of used words: {}".format(len(word_list)))  #Number of the whole words used in tweet data 
# embedding_matrix = np.zeros((n_words+1, embed_size)) 
# for word, i in word_list.items():
#     if i >= max_features: continue
#     embedding_vector = embedding_index.get(word)
#     if embedding_vector is not None: embedding_matrix[i] = embedding_vector

# embedding_matrix[0].shape


# ### Machine Learning Model
# I have pre-trained model, so I loaded actual trained model and make the predictions on our data.

# In[ ]:


model = load_model("../input/best_model.hdf5")


# It shows how the model actually looks like. This model is mixture of 'LSTM' and 'CNN' model. 

# In[ ]:


model.summary() 


# Let's train our web-crawled tweets using 'LSTM-CNN' model.

# In[ ]:


pred = model.predict(tweets_pad_sequences, batch_size = 1024, verbose = 1)


# In[ ]:


pred.max()


# WOW! This model has 99.75% accuracy! I really didn't expect that. <br>
# This means this model can almost 100% distinguish toxic words from the tweets. <br>
# Then let's see the summary of this distinguishment

# In[ ]:


toxic_predictions = pd.DataFrame(columns=list_classes, data=pred)
toxic_predictions['id'] = df_politics['date'].values
toxic_predictions['text'] = df_politics['text'].values
toxic_predictions.head()


# In[ ]:


toxic_predictions[list_classes].describe()


# In[ ]:


toxic_predictions['username'] = df_politics['username'].values
toxic_predictions['text'] = df_politics['text'].values
toxic_predictions.tail()


# ### Let's check which tweet showed the greatest negative feeling(toxic, insult, identity_hate)
# Interesting! It seems relatively correct 

# In[ ]:


for neg_word in ['toxic', 'insult', 'identity_hate']:
  pd.set_option('display.max_colwidth', -1)
  temp = toxic_predictions[toxic_predictions[neg_word] == max(toxic_predictions[neg_word])]
  print('The most {} text: {}'.format(neg_word, temp['text'].values))
  print(temp['username'])


# ### Compare Politicians' negative Feelings! 

# In[ ]:


Trump_predictions = toxic_predictions[toxic_predictions['username'] == 'realDonaldTrump']
Sanders_predictions = toxic_predictions[toxic_predictions['username'] == 'BernieSanders']
Warren_predictions = toxic_predictions[toxic_predictions['username'] == 'ewarren']
Biden_predictions = toxic_predictions[toxic_predictions['username'] == 'JoeBiden']


# In[ ]:


#Trump Summary
Trump_predictions[list_classes].describe()


# In[ ]:


#Sanders Summary
Sanders_predictions[list_classes].describe()


# In[ ]:


#Warren Summary
Warren_predictions[list_classes].describe()


# In[ ]:


#Biden Summary
Biden_predictions[list_classes].describe()


# ### Plotting

# First, I compared the occurence of each negative feelings. <br>
# Polticians showed the greatest feeling of toxic, followed by obscene, and insult.

# In[ ]:


x=toxic_predictions.iloc[:,0:6].sum()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("# per class")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# Not all tweets show only one type of feeling. <br>
# So, I tried to see what feeling comes frequent with which feeling. 

# In[ ]:


temp_df=toxic_predictions.iloc[:,0:6:]
corr=temp_df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True)


# ### Distribution of each candwidate
# very skewed distribution....Unlike what I expected, candidates show realtively low negative feelings in their tweets. <br>
# Hmm...Does this prove that candidates use clean words? Or....Is this Facebook's fault? 
# 

# In[ ]:


melt_df = pd.melt(toxic_predictions, value_vars=list_classes, id_vars='username')
sns.violinplot(x='variable', y='value', hue='username', data=melt_df)
plt.show()


# In[ ]:


print("Trump: " + Trump_predictions.loc[Trump_predictions['toxic'].idxmax()]['text'])


# In[ ]:


print(Trump_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)


# In[ ]:


print(Sanders_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)


# In[ ]:


print(Warren_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)


# In[ ]:


print(Biden_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)


# ### How many times did the candidates mention others?

# In[ ]:


import re
total = []
num = 0
candidates = ['Trump', 'Sanders','Biden','Warren']
for df_candidate in [df_Trump, df_Bernie, df_Joe, df_Elizabeth]:
    cd_list = [candidates[num]]
    for candidate in candidates:
        count = 0
        for i in df_candidate['text'].values:
            if re.search(candidate, i):
                temp = len(re.findall(candidate, i))
                count+=temp
        cd_list.append(count)
    total.append(cd_list)
    num+=1


# In[ ]:


header = ['','Trump', 'Sanders','Biden','Warren'] 
pd.DataFrame(total, columns=header, index= None)


# In[ ]:




