#!/usr/bin/env python
# coding: utf-8

# ## Democrat vs. Republican tweets
# ### Simple neural network classification demo
# This notebook is basically my previous notebook that did classification with Naive Bayes, but using Jason Brownlee's Embedding-1D CNN-LSTM classifier from ["Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras"](https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/)

# In[1]:


# Load some libraries, mostly pandas & sklearn
# but also NLTK's TweetTokenizer since we're working with Twitter data
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from nltk.tokenize.casual import TweetTokenizer
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# initialize tokenizer
tokenizer = TweetTokenizer(reduce_len=True)

# fix random seed for reproducibility
np.random.seed(7)


# ### Load the dataset
# 
# First, we load everything into a Pandas DataFrame. (The dataset consists of the 200 most recent tweets, at the time of data collection, from each member of the US House of Representatives who has a Twitter account.)
# 
# Then, we get an 80/20 split of *Twitter handles* in the dataset, stratified according to the Party variable. This way, we'll have roughly similar proportions of Democrats & Republicans in the training & test sets separately as in the whole dataset. Finally, we split the full dataset into train & test sets based on those Twitter handles because we don't want any individual tweeter to appear in both training & test sets.

# In[2]:


all_tweets = pd.read_csv("../input/ExtractedTweets.csv")


# In[3]:


top_words = 10000  # We'll keep these many distinct tokens and drop less frequent ones

# Tokenize using the TweetTokenizer
all_tweets.Tweet = all_tweets.Tweet.apply(tokenizer.tokenize)
# Get the word counts
fdist = FreqDist(word for tweet in all_tweets.Tweet for word in tweet)
# Get top top_words terms, in order of frequency
terms = [term for term, count in fdist.most_common(top_words)]
# Replace all tokens with their rank (or 0 if not in the top top_words)
all_tweets.Tweet = all_tweets.Tweet.apply(lambda tweet: [terms.index(term) if term in terms else 0
                                                         for term in tweet])


# In[4]:


# Get an 80/20 train/test split of Twitter handles, stratified on Party
tweeters = all_tweets.iloc[:,:2].drop_duplicates()
handles_train, handles_test = train_test_split(tweeters.Handle, stratify=tweeters.Party, test_size=0.2, random_state=0)

# extract train & test sets from the all_tweets df
train = all_tweets[all_tweets.Handle.isin(handles_train)].reset_index().drop('index', axis=1)
test = all_tweets[all_tweets.Handle.isin(handles_test)].reset_index().drop('index', axis=1)


# In[5]:


# The purpose of this cell is just to format the data to match the format of keras.datasets.imdb
X_train = np.array(train.Tweet)
y_train = np.array((train.Party == 'Democrat').astype(int))
X_test = np.array(test.Tweet)
y_test = np.array((test.Party == 'Democrat').astype(int))


# In[6]:


# truncate and pad input sequences
max_review_length = 48 # The longest tweet is 56 tokens, and only a handfull are longer than 48
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# ### Train & evaluate
# 
# Training & evaluating the model results in ~72% accuracy on a per-tweet basis, which isn't too bad. There's probably a lot of hyperparameter tuning that could be done here.

# In[7]:


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[8]:


preds = np.round(model.predict(X_test).ravel())
print('Accuracy: {}'.format(str(round(accuracy_score(y_test, preds), 4))))


# ### Evaluate by Congressperson
# 
# It's not so unusual for a single tweet to be mis-categorized, so perhaps we should consider the overall categorization of each Congressperson's tweets. With this adjustment, we get ~96% accuracy.

# In[9]:


accuracy_df = test.drop(['Tweet'], axis=1)
accuracy_df['preds'] = preds
accuracy_df['Party'] = y_test
correct = 0
total = 0

print("Congresspersons whose tweets were mostly mis-classified:")
for name in accuracy_df.Handle.unique():
    sub_df = accuracy_df[accuracy_df.Handle == name].reset_index()
    sub_preds = sub_df.preds.value_counts()

    if sub_preds.index[0] == sub_df.Party[0]:
        correct += 1
    total += 1

    if sub_preds[sub_df.Party[0]]/len(sub_df) < 0.5:
        print("{} ({}) classified with accuracy: {}".format(name, sub_df.Party[0],
                                                                      str(round(sub_preds[sub_df.Party[0]]/len(sub_df), 4))))

print()
print("Accuracy of the model on a per-Congressperson-basis was: {}".format(str(round(correct/total, 4))))


# In[ ]:




