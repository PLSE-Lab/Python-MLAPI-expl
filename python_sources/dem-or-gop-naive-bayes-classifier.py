#!/usr/bin/env python
# coding: utf-8

# ## Democrat vs. Republican tweets
# ### Simple Naive Bayes classification demo

# In[8]:


# Load some libraries, mostly pandas & sklearn
# but also NLTK's TweetTokenizer since we're working with Twitter data

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.tokenize.casual import TweetTokenizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# initialize tokenizer
tokenizer = TweetTokenizer(reduce_len=True)


# ### Load the dataset
# 
# First, we load everything into a Pandas DataFrame. (The dataset consists of the 200 most recent tweets, at the time of data collection, from each member of the US House of Representatives who has a Twitter account.)
# 
# Then, we get an 80/20 split of *Twitter handles* in the dataset, stratified according to the Party variable. This way, we'll have roughly similar proportions of Democrats & Republicans in the training & test sets separately as in the whole dataset. Finally, we split the full dataset into train & test sets based on those Twitter handles because we don't want any individual tweeter to appear in both training & test sets.

# In[9]:


all_tweets = pd.read_csv("../input/ExtractedTweets.csv")


# In[10]:


# Get an 80/20 train/test split of Twitter handles, stratified on Party
tweeters = all_tweets.iloc[:,:2].drop_duplicates()
handles_train, handles_test = train_test_split(tweeters.Handle, stratify=tweeters.Party, test_size=0.2, random_state=0)

# extract train & test sets from the all_tweets df
train = all_tweets[all_tweets.Handle.isin(handles_train)].reset_index().drop('index', axis=1)
test = all_tweets[all_tweets.Handle.isin(handles_test)].reset_index().drop('index', axis=1)


# ### Construct a scikit-learn Pipeline
# 
# This is roughly the simples Pipeline possible: It's just a TF-IDF vectorizer and a classifier. The TF-IDF vectorizer tokenizes the text of each tweet, then calculates the TF-IDF weights for the terms in the tokenized tweet.

# In[11]:


nb_pipeline = Pipeline([
    ('vectorize', TfidfVectorizer(tokenizer=tokenizer.tokenize)),
    ('classifier', MultinomialNB())
])


# ### Train & evaluate
# 
# Training & evaluating the model results in 74.34% accuracy on a per-tweet basis, which isn't too bad.

# In[12]:


nb_pipeline.fit(train.Tweet, train.Party)


# In[13]:


preds = nb_pipeline.predict(test.Tweet)
print('Accuracy: {}'.format(str(round(accuracy_score(test.Party, preds), 4))))


# ### Evaluate by Congressperson
# 
# It's not so unusual for a single tweet to be mis-categorized, so perhaps we should consider the overall categorization of each Congressperson's tweets. With this adjustment, we get 94.25% accuracy, even with this very simple model.

# In[14]:


accuracy_df = test.drop(['Tweet'], axis=1)
accuracy_df['preds'] = preds
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




