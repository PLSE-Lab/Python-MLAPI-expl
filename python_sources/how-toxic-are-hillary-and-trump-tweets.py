#!/usr/bin/env python
# coding: utf-8

# 
# 
# I've been wanting to play with this dataset for a while. I've also been wanting to try to see how do models built on [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) perform on non-competition "real world" data. Here I will just use one model that was built inside of a [kernel](https://www.kaggle.com/tunguz/bi-gru-lstm-cnn-poolings-fasttext). The kernel scores in the 0.984x AUC range. It's a respectable score, but well below the top solutions that scored in the 0.988x range. 
# 
# ---- October 22 2018 Update - Added a few distribution plots and a more extensive lists of the most problematic tweets, 10 per topic. ----
# 
# ---- October 31 2018 Update - Added a few clarifications and explanations. ----

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Now, let's load the data and all the vector embeddings. 

# In[ ]:


tweets = pd.read_csv("../input/clinton-trump-tweets/tweets.csv")
embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"
tweets.head()


# In[ ]:


tweets.shape


# We see that there are a total of 6444 tweets, and as it will be shown below they are almost evenly distributed between Hillary and Trump. This may seem like a lot of tweets, but for text classification this would be a very small amount of data.

# We will embed words from these tweets into a word-vector space using one of the previously trained word embeddings. Here we use a 300-dimensional vector space that comes curtesy of FastText. We will also limit the length of text to 220 words. This is an overkill for tweets, but for general purpose it is rather small text length. The original was aimed at much longer text sizes, and this was a reasonable length for those purposes. The best embedding that we used in Toxic limited length to 900 words.

# In[ ]:


embed_size = 300
max_features = 130000
max_len = 220

tweets["text"].fillna("no comment")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
raw_text = tweets["text"].str.lower()


# In order for our pretrained models to work, we need to transform the text here into the appropriate vectorized format.

# In[ ]:


tk = Tokenizer(num_words = max_features, lower = True)
tk.fit_on_texts(raw_text)
tweets["comment_seq"] = tk.texts_to_sequences(raw_text)


# We also need to pad the tweets that are less than 220 words, which is essentially all of them.

# In[ ]:


tweets_pad_sequences = pad_sequences(tweets.comment_seq, maxlen = max_len)


# In[ ]:


tweets_pad_sequences.shape


# In[ ]:


tweets_pad_sequences


# This is one big 6444x220 array. We'll now need to construc an embedding index, that puts each one of the words into a 300-dimensional vector space:

# 

# In[ ]:


def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))


# In[ ]:


word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# Now we'll load the actual trained model and make the predictions on our data.

# In[ ]:


model = load_model("../input/bi-gru-lstm-cnn-poolings-fasttext/best_model.hdf5")


# In[ ]:


pred = model.predict(tweets_pad_sequences, batch_size = 1024, verbose = 1)


# Let's see what's the maximum probability for this model:

# In[ ]:


pred.max()


# In other words, at nearly 1.0 probability the model seems pretty confident about the "toxicity" of some of the tweets.
# 
# Now let's put the predictions into a dataframe, so we can have a better view of them and how they relate to the actual tweets.

# In[ ]:


toxic_predictions = pd.DataFrame(columns=list_classes, data=pred)


# In[ ]:


toxic_predictions.head()


# In[ ]:


toxic_predictions['id'] = tweets['id'].values
toxic_predictions['handle'] = tweets['handle'].values
toxic_predictions['text'] = tweets['text'].values


# In[ ]:


toxic_predictions.tail()


# In[ ]:


Hillary_predictions = toxic_predictions[toxic_predictions['handle'] == 'HillaryClinton']
Trump_predictions = toxic_predictions[toxic_predictions['handle'] == 'realDonaldTrump']


# In[ ]:


Hillary_predictions[list_classes].describe()


# In[ ]:


Trump_predictions[list_classes].describe()


# 

# 

# In[ ]:


melt_df = pd.melt(toxic_predictions, value_vars=list_classes, id_vars='handle')


# In[ ]:


melt_df.head()


# Let's plot the distributions of various values for both Hillary and Trump tweets. We'll use 'violing plots', as they seem very visually intutive and easy to understand.

# In[ ]:


sns.violinplot(x='variable', y='value', hue='handle', data=melt_df)
plt.show()


# We see that for each one of the categories the outliers really skew the distributions. Maybe if we clip them we can see how the "main" distributions behave:

# In[ ]:


melt_df['value'] = np.clip(melt_df['value'].values, 0, 0.2)


# In[ ]:


sns.violinplot(x='variable', y='value', hue='handle', data=melt_df)
plt.show()


# Still very skewed distributions. How about one more clipping?

# In[ ]:


melt_df['value'] = np.clip(melt_df['value'].values, 0, 0.05)
sns.violinplot(x='variable', y='value', hue='handle', data=melt_df)
plt.show()


# Nope, still no luck. Looks like the distributions are to some extent "scale independant". 

# Based on this summary statistics, it would seem that both of them score pretty low on average for all of the "Toxic" categories. However, there do seem to be a few notable "highly probable" problemeatic tweets in each one of the six categories, with notable exception of "threat". Which, I think, is a good thing. For what it's worth (not much at all, IMHO), Hillary's tweets seem to be, on the average, toxic, severaly toxic, and obscene, while Trump's tweets score higher on the average for threat, insult, and identity hate. 
# 
# Let's see what the "worst offenders" are in for both candidates. Let's start with the most toxic Hillary tweet.

# In[ ]:


Hillary_predictions.loc[Hillary_predictions['toxic'].idxmax()]['text']


# Meh, not really toxic. Seems like the word "mad", or the high frequency of special characters, have flagged this tweet as toxic. The same tweet was also marked as the top tweet in both "severe toxic" and "obscene" categories. 
# 
# Now let's look at "threats":

# In[ ]:


print(Hillary_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)


# In[ ]:


print(Hillary_predictions.sort_values(by=['severe_toxic'], ascending=False)['text'].head(10).values)


# In[ ]:


print(Hillary_predictions.sort_values(by=['obscene'], ascending=False)['text'].head(10).values)


# Not really toxic, severely toxic, or obscene IMHO. Not very inspiring or humanlike either. As if a chatbot was coming up with these ...

# How about threats?

# In[ ]:


Hillary_predictions.loc[Hillary_predictions['threat'].idxmax()]['text']


# Yeah, not much going on there. As predicted with very low probability of this actually being a threat.

# In[ ]:


print(Hillary_predictions.sort_values(by=['threat'], ascending=False)['text'].head(10).values)


# In[ ]:





# 

# What's Hillary's worst insult?

# In[ ]:


Hillary_predictions.loc[Hillary_predictions['insult'].idxmax()]['text']


# Ouch. That's definitely below the belt, but in a more indirect kind of way. And yeah, insluting. Good job, predictive modeling!
# 
# Here are the top 10 "insults"

# In[ ]:


print(Hillary_predictions.sort_values(by=['insult'], ascending=False)['text'].head(10).values)


# Let's look at identity hate:

# In[ ]:


Hillary_predictions.loc[Hillary_predictions['identity_hate'].idxmax()]['text']


# In[ ]:





# Hmm, that's interesting: seem the algorithm has marked Hillary's ReTweet of Trump's tweet. Seems like there is something deep going on here. Or the algorithm is just plain unreliable. 
# 
# Let's look at the top 10 "hateful" Hillary tweets:

# In[ ]:


print(Hillary_predictions.sort_values(by=['identity_hate'], ascending=False)['text'].head(10).values)


# OK, let's move onto Trump. First, his most toxic tweet:

# In[ ]:


Trump_predictions.loc[Trump_predictions['toxic'].idxmax()]['text']


# That's just weird: there is nothign toxic about it. The same tweet has been flagged as the most severly toxic and obscene tweet as well. Not very informative.
# 
# But let's take a closer look at the top 10 flagged tweets in each one of those categories.
# 
# First toxic:

# In[ ]:


print(Trump_predictions.sort_values(by=['toxic'], ascending=False)['text'].head(10).values)


# Severe toxic:

# In[ ]:


print(Trump_predictions.sort_values(by=['severe_toxic'], ascending=False)['text'].head(10).values)


# Obscene:

# In[ ]:


print(Trump_predictions.sort_values(by=['obscene'], ascending=False)['text'].head(10).values)


# Now how about threats?

# In[ ]:


Trump_predictions.loc[Trump_predictions['threat'].idxmax()]['text']


# Massive tax increases? yeah, I can see how this could be viewed as threatening.
# 
# Let's take a closer look at the top 10 worst 'threat' tweets:

# In[ ]:


print(Trump_predictions.sort_values(by=['threat'], ascending=False)['text'].head(10).values)


# 
# 
# How about the most insulting tweet?

# In[ ]:


Trump_predictions.loc[Trump_predictions['insult'].idxmax()]['text']


# Yeah, definitely insulting. On so many levels. I can't even ...

# Let's take a closer look at the top 10 worst 'insult' tweets:

# In[ ]:


print(Trump_predictions.sort_values(by=['insult'], ascending=False)['text'].head(10).values)


# Seems that back then Fox News was not exactly his favorite TV network. 
# 
# And what about identity hate?

# In[ ]:


Trump_predictions.loc[Trump_predictions['identity_hate'].idxmax()]['text']


# That one really made me LOL. And think. Is he mocking him for his "identity" of bing Jeb? Or mommy's boy? Or W's brother? Or a weakling? All of the above? So many choices  ...
# 
# Let's take a closer look at the top 10 worst 'identity hate' tweets:

# In[ ]:


print(Trump_predictions.sort_values(by=['identity_hate'], ascending=False)['text'].head(10).values)


# Seeems that Jeb really got under his skin!

# 
# In the end, this exercise shows both the strengths and limitations of algorithmic approach to toxic comment classification. Since the AUC score for the training sets is relatively high (almost 0.99 for  the top models), it is most likely that in the case human insight is even more relevant than for most other ML areas. Furthermore, even though we had a pretty large dataset to work with, it is very likely that in order to get even close to human level toxic text classification, we'd need several orders of magnitude larger training set, and/or deeper natural text understanding models. 

# In[ ]:





# In[ ]:




