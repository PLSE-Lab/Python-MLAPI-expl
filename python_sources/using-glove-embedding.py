#!/usr/bin/env python
# coding: utf-8

# **Description**
# 
# Notebook to load the quora sincere/insincere questions data set and figure out some initial impressions.
# 
# The dataset contains the questions posted but no information about time of post or possibility to extract other posts made by the same user. So predictions will be made just based on the text content of each single question.
# 
# Looking through quora is a guilty pleasure of mine, and I definitely notice the inflamatory fake looking posts. Usually I'm impressed with the time and thought the responders put in to responding about the question, and it's a huge waste of their time. I have to admit, it can be hard to tell some of the sincere but perhaps ill-informed questions from fake ones, so I think this will be a fun thing to look at!
# 
# This is my first quora kernel and a work in progress! I'll go through some simple analysis try to choose one of the suggested word embedding tools and work on incorporating that into the kernel.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

## Adding in some more useful packages here
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))


# I'll start by loading the training data and taking a look at the first few entries

# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# This dataset contains just 3 values for each of the categories: 
# 
#     'qid' 
#     'question text' 
#     'target' value of 0 or 1
# 
# The first thing I see is that all of the entries we see so fat have a target value of 0 - so they are categorized as sincere. 
# 
# Lets see what fraction of the data is labeled insincere (has target value 1) and take a look at some example data that was categorized as insincere.

# In[ ]:


fig,ax = plt.subplots(1,1)
train.hist(column = 'target', ax = ax)
ax.set_title('Number of entries classified as sincere vs insincere')
ax.set_xticks([0,1])
print('Percent of insincere entries %.3f %%'%(100*(sum(train['target'])/len(train))))

train[train['target']==1].head()


# Only a little over 6% of the questions are insincere.
# 
# At first glance, in the 5 displayed insincere titles, there some words related to race and politics. Some sincere questions could also have these words but maybe some top words could be used as initial flags for insincere posts that warrant further examination.
# _______________________________________________________________________________________________________________________________
# * **Next step** - start parsing through the word data and find the initial trends in word usage amongst the insincere posts.
# I'll use a word cloud to visualize top words used in the questions. To speed up processing I'll choose just 1000 posts of each category
# I'm using the standard stop words from nltk.corpus (loaded above).

# In[ ]:


n_posts = 1000
q_S = ' '.join(train[train['target'] == 0]['question_text'].str.lower().values[:n_posts])
q_I = ' '.join(train[train['target'] == 1]['question_text'].str.lower().values[:n_posts])

wordcloud_S = WordCloud(max_font_size=None, stopwords=stop,scale = 2,colormap = 'Dark2').generate(q_S)
wordcloud_I = WordCloud(max_font_size=None, stopwords=stop,scale = 2,colormap = 'Dark2').generate(q_I)

fig, ax = plt.subplots(1,2, figsize=(20, 5))
ax[0].imshow(wordcloud_S)
ax[0].set_title('Top words sincere posts',fontsize = 20)
ax[0].axis("off")

ax[1].imshow(wordcloud_I)
ax[1].set_title('Top words INsincere posts',fontsize = 20)
ax[1].axis("off")

plt.show()


# There certainly looks to be a difference in the words used in 'insincere' posts, but again many of these words can appear in legitimate questions as well.
# 
# ____________________________________________________________________________________________________________________________________________________________
# 
# **Embeddings** I'll move on to loading and using the embeddings tools.
# 
# There are 4 options with links provided in the dataset description.
# I chose to start with 'glove' This is new stuff for me, so I used https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html and https://medium.com/@japneet121/word-vectorization-using-glove-76919685ee0b as references and modified their code to work with this dataset (other referenced to be added).
# 
# GloVe is feature description dataset built on a large corpus of words that represent words based on their co-occcurence with other words. In the file provided, each line lists one word that is followed by a vector of numbers that represents the word.
# 
# First we read in the embeddings file into a dictionary - each entry is a word, followed by the vector of numbers to represent its values
# 
# 
# 
# 

# In[ ]:


embeddings_index = {}
f = open('/kaggle/input/embeddings/glove.840B.300d/glove.840B.300d.txt')
for line in f:
    values = line.split(' ')
    word = values[0] ## The first entry is the word
    coefs = np.asarray(values[1:], dtype='float32') ## These are the vecotrs representing the embedding for the word
    embeddings_index[word] = coefs
f.close()

print('GloVe data loaded')


# Now preprocess the question text
# 
# As with the work cloud, I initially start by working on a subset of the posts.
# 6% of 10,000 posts will be 600, so I won't go lower than total of 10k posts.
# Current code includes all posts in analysis

# In[ ]:


import re

## Iterate over the data to preprocess by removing stopwords
lines_without_stopwords=[] 
for line in train['question_text'].values: 
    line = line.lower()
    line_by_words = re.findall(r'(?:\w+)', line, flags = re.UNICODE) # remove punctuation ans split
    new_line=[]
    for word in line_by_words:
        if word not in stop:
            new_line.append(word)
    lines_without_stopwords.append(new_line)
texts = lines_without_stopwords

print(texts[0:5])


# The following code uses some tools from keras to 'Tokenize' the questions text - ie assign numbers to every word
# 
# labels get converted to a 2-column array (1,0) for 0 and (0,1) for 1 in the target column.
# 
# This is the first preprocessing step to be able to use the embedding.

# In[ ]:


## Code adapted from (https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)
# Vectorize the text samples

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

MAX_NUM_WORDS = 1000
MAX_SEQUENCE_LENGTH = 100
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(train['target']))
print(data.shape)
print(labels.shape)


# In[ ]:


## More code adapted from the keras reference (https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py)
# prepare embedding matrix 
from keras.layers import Embedding
from keras.initializers import Constant

## EMBEDDING_DIM =  ## seems to need to match the embeddings_index dimension
EMBEDDING_DIM = embeddings_index.get('a').shape[0]
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# In[ ]:


## Peeking at the embedding matrix values
print(embedding_matrix.shape)
plt.plot(embedding_matrix[16])
plt.plot(embedding_matrix[37])
plt.plot(embedding_matrix[18])
plt.title('example vectors')


# Now try applying a keras sequential model - this isn't a great model yet - just copied over from the source
# 
# I'll update it as I make it better

# In[ ]:


## Code from: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa
## To create and visualize a model

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation

model = Sequential()
model.add(Embedding(num_words, 300, input_length=100, weights= [embedding_matrix], trainable=False))

model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


## Fit train data
print(labels.shape)
model.fit(data, np.array(labels), validation_split=0.1, epochs = 1)


# In[ ]:


## Model visualization code adapted from: https://medium.com/@sabber/classifying-yelp-review-comments-using-cnn-lstm-and-pre-trained-glove-word-embeddings-part-3-53fcea9a17fa

from sklearn.manifold import TSNE
## Get weights
embds = model.layers[0].get_weights()[0]
## Plotting function
## Visualize words in two dimensions 
tsne_embds = TSNE(n_components=2).fit_transform(embds)

plt.plot(tsne_embds[:,0],tsne_embds[:,1],'.')


# * results visualization to be improved..

# Apply the model to predict values

# In[ ]:


#test = pd.read_csv('../input/test.csv')
#test.head()
#pred = model.predict()
#pred = np.round(pred)


# In[ ]:


#df = pd.DataFrame({"qid": test_df["qid"], "prediction": pred})
#df.to_csv("submission.csv", index=False)

