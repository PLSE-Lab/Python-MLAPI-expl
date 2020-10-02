#!/usr/bin/env python
# coding: utf-8

# Code taken from https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup kernel in toxic classification challenge
# > 
# Adversarial validation is a mean to check if train and test datasets have significant differences. The idea is to use the dataset features to try and separate train and test samples.
# 
# So you would create a binary target that would be 1 for train samples and 0 for test samples and fit a classifier on the features to predict if a given sample is in train or test datasets!
# 
# Here we will use a LogisticRegression and a TF-IDF vectorizer to check if text features distributions are different and see if we can separate the samples. 
# 
# The best kernel on this is certainly [here](https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms) by [Konrad Banachewicz](https://www.kaggle.com/konradb)
# 
# Other resources can be found [on fastML](http://fastml.com/adversarial-validation-part-one/)
# 

# ### Objective: We want to find if distribution of train data and test data is similar or not?
# 
# If it is not we are gonna face huge leaderboard shakeup in second stage. So lets see

# In[ ]:


import numpy as np
import pandas as pd
trn = pd.read_csv("../input/train.csv", encoding="utf-8")
sub = pd.read_csv("../input/test.csv", encoding="utf-8")


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import regex
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    tokenizer=lambda x: regex.findall(r'[^\p{P}\W]+', x),
    analyzer='word',
    token_pattern=None,
    stop_words='english',
    ngram_range=(1, 1), 
    max_features=50000
)
trn_idf = vectorizer.fit_transform(trn.question_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.question_text)
sub_vocab = vectorizer.vocabulary_
all_idf = vectorizer.fit_transform(pd.concat([trn.question_text, sub.question_text], axis=0))
all_vocab = vectorizer.vocabulary_


# Convert vocab dictionnaries to list of words

# In[ ]:


trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
all_words = [word for word in all_vocab.keys()]


# Check a few figures on words not in train or test

# In[ ]:


common_words = set(trn_words).intersection(set(sub_words)) 
print("number of words in both train and test : %d "
      % len(common_words))
print("number of words in all_words not in train : %d "
      % (len(trn_words) - len(set(trn_words).intersection(set(all_words)))))
print("number of words in all_words not in test : %d "
      % (len(sub_words) - len(set(sub_words).intersection(set(all_words)))))


# This means there is not much substantial differences between train and test vocabularies or term frequencies
# 
# Let's check if a LinearRegression can make a difference between train and test using this.
# 
# We would take the output of the TF-IDF vectorizer fitted on train + test.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# Create target where all train samples are ones and all test samples are zeros
target = np.hstack((np.ones(trn.shape[0]), np.zeros(sub.shape[0])))
# Shuffle samples to mix zeros and ones
idx = np.arange(all_idf.shape[0])
np.random.seed(1)
np.random.shuffle(idx)
all_idf = all_idf[idx]
target = target[idx]
# Train a Logistic Regression
folds = StratifiedKFold(5, True, 1)
for trn_idx, val_idx in folds.split(all_idf, target):
    lr = LogisticRegression(solver = 'saga')
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))


# So Based on AUC score we cannot really find much difference between the train and test data.** Good for us. **
# 
# I did not see any wrong doing in the code so far but if you do please shout as loud as possible !
# 
# ** This to me means we might not have many surprises on the private LB. **
# 
# ** Although with the size of the test set so low we cannot be sure. **
# 

# You may want to push all this further and see the impact of a cleaner dataset.
