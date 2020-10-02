#!/usr/bin/env python
# coding: utf-8

# I've been rummaging through the Discussion and Kernels from the recently completed [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), and am finding all kinds of gems that we can use here in DonorsChoose.
# 
# First some background.  I am using 8-fold cross validation (CV) on all my models, saving off both the test predictions and the out-of-fold (OOF) train predictions.  The CV's on my non-text based models align very well with their corresponding leaderboard (LB) scores.  However, my models that also analyze text (using TF-IDF or neural networks) score ***much*** better on the LB than on their corresponding CV.   For example, I have a non-text LightGBM (LGB) model that has CV=0.7284 and LB=0.7279 for a reasonable delta of -0.0005, while another model with TF-IDF thrown in (as in [Oleg Panichev's excellent kernel](https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter)) has CV=0.7865 and LB=0.7947 for a massive +0.0082 difference. 
# 
# So is there a difference between the train and test text?  I have hijacked the code from Olivier's [adversarial_validation_and_lb_shakeup kernel](http://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup) here to check this out.  Olivier said:
# 
# *Adversarial validation is a means to check if train and test datasets have significant differences. The idea is to use the dataset features to try and separate train and test samples.  So you would create a binary target that would be 1 for train samples and 0 for test samples and fit a classifier on the features to predict if a given sample is in train or test datasets!*
# 
# *Here we will use a LogisticRegression and a TF-IDF vectorizer to check if text features distributions are different and see if we can separate the samples.*
# 
# *The [best kernel](https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms/notebook) on this is certainly here by [Konrad Banachewicz](https://www.kaggle.com/konradb).  Other resources can be found on [fastML](http://fastml.com/adversarial-validation-part-one/)*
# 
# If train and test do vary considerably we could then, as a remedy, find a subset of the train dataset that behaves similarly to the test dataset, and use this subset for cross validation.
# 
# First, let's read in the train and test datasets:
# 

# In[ ]:


import numpy as np
import pandas as pd
import os

data_path = os.path.join('..', 'input')
trn = pd.read_csv(os.path.join(data_path, 'train.csv'), low_memory=True)
sub = pd.read_csv(os.path.join(data_path, 'test.csv'), low_memory=True)

# Preprocess data
trn['comment_text'] = trn.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)
sub['comment_text'] = sub.apply(lambda row: ' '.join([
    str(row['project_essay_1']), 
    str(row['project_essay_2']), 
    str(row['project_essay_3']), 
    str(row['project_essay_4']),
    ]), axis=1)


# What follows has been taken ***verbatim*** from Olivier; adapted here to examine the project essay text ('comment_text'), train versus test.

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
    max_features=20000
)
trn_idf = vectorizer.fit_transform(trn.comment_text)
trn_vocab = vectorizer.vocabulary_
sub_idf = vectorizer.fit_transform(sub.comment_text)
sub_vocab = vectorizer.vocabulary_
all_idf = vectorizer.fit_transform(pd.concat([trn.comment_text, sub.comment_text], axis=0))
all_vocab = vectorizer.vocabulary_

trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
all_words = [word for word in all_vocab.keys()]

common_words = set(trn_words).intersection(set(sub_words)) 
print("number of words in both train and test : %d "
      % len(common_words))
print("number of words in all_words not in train : %d "
      % (len(trn_words) - len(set(trn_words).intersection(set(all_words)))))
print("number of words in all_words not in test : %d "
      % (len(sub_words) - len(set(sub_words).intersection(set(all_words)))))


# 90% of the words overlap between train and test, but there are quite a few words in train but not in test.  Note that train is a much bigger dataset (182080 versus 78035 rows).  
# 
# Per Olivier: *Let's check if a LinearRegression can make a difference between train and test using this.  We would take the output of the TF-IDF vectorizer fitted on train + test.*

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
    lr = LogisticRegression()
    lr.fit(all_idf[trn_idx], target[trn_idx])
    print(np.round(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]), 3))


# AUC scores for each fold are all almost exactly 0.500, completely neutral.  So the project essays from train and test exhibit the same TF-IDF characteristics.
# 
# But, wait a minute!  What about the big difference between CV and LB scores when text analysis is included in my models?  The [Leaderboard](https://www.kaggle.com/c/donorschoose-application-screening/leaderboard) says, "This leaderboard is calculated with approximately 30% of the test data.  The final results will be based on the other 70%, so the final standings may be different."
# 
# Is the text in the private leaderboard proposals radically different than in the training proposals ***and*** the public leaderboard proposals?  And, different in a way such that, when combined with the public leaderboard proposals, the combined corpus nets out to be very similar to the training proposals text?  Seems strange, but at this point that is my best guess.  My other guess is that I making some kind of dumb mistake calculating my CV scores, but at least in that case it caused me to learn about adversarial validation of text.
# 
# 

# 
