#!/usr/bin/env python
# coding: utf-8

# This kernel is based on the [olivier](https://www.kaggle.com/ogrellier) adversial validation [kernel](https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup). I have only added how to find a validation set using adversial validation because I didn't know how to make one.
# If my approach of making an adversarial validation set is wrong then please correct me.
# <br>Thanks

# Adversarial validation is a mean to check if train and test datasets have significant differences. The idea is to use the dataset features to try and separate train and test samples.
# 
# So you would create a binary target that would be 1 for train samples and 0 for test samples and fit a classifier on the features to predict if a given sample is in train or test datasets!
# 
# Here we will use a LogisticRegression and a TF-IDF vectorizer to check if text features distributions are different and see if we can separate the samples. 
# 
# The best kernel on this is certainly [here](https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms) by [Konrad Banachewicz](https://www.kaggle.com/konradb)
# 
# Other resources can be found [on fastML](http://fastml.com/adversarial-validation-part-one/)

# In[106]:


import numpy as np
import pandas as pd
trn = pd.read_csv("../input/train.csv", encoding="utf-8")
sub = pd.read_csv("../input/test.csv", encoding="utf-8")


# In[107]:


#assign target if set is test set or not
trn['is_test'] = 0
sub['is_test'] = 1


# In[108]:


orginal_train = trn.copy()


# In[109]:


train = pd.concat([trn, sub], axis=0)


# In[110]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import TfidfVectorizer\nimport regex\nvectorizer = TfidfVectorizer(\n    sublinear_tf=True,\n    strip_accents='unicode',\n    tokenizer=lambda x: regex.findall(r'[^\\p{P}\\W]+', x),\n    analyzer='word',\n    token_pattern=None,\n    stop_words='english',\n    ngram_range=(1, 1), \n    max_features=20000\n)\ntrn_idf = vectorizer.fit_transform(trn.comment_text)\ntrn_vocab = vectorizer.vocabulary_\nsub_idf = vectorizer.fit_transform(sub.comment_text)\nsub_vocab = vectorizer.vocabulary_\nall_idf = vectorizer.fit_transform(train.comment_text.values)\nall_vocab = vectorizer.vocabulary_")


# Convert vocab dictionnaries to list of words

# In[111]:


trn_words = [word for word in trn_vocab.keys()]
sub_words = [word for word in sub_vocab.keys()]
all_words = [word for word in all_vocab.keys()]


# Check a few figures on words not in train or test

# In[112]:


common_words = set(trn_words).intersection(set(sub_words)) 
print("number of words in both train and test : %d "
      % len(common_words))
print("number of words in all_words not in train : %d "
      % (len(trn_words) - len(set(trn_words).intersection(set(all_words)))))
print("number of words in all_words not in test : %d "
      % (len(sub_words) - len(set(sub_words).intersection(set(all_words)))))


# This means there are substantial differences between train and test vocabularies or term frequencies
# 
# Let's check if a LinearRegression can make a difference between train and test using this.
# 
# We would take the output of the TF-IDF vectorizer fitted on train + test.

# In[113]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import StratifiedKFold\nfrom sklearn.metrics import roc_auc_score\n\n#predictions to save each fold predictions results\npredictions = np.zeros(train.shape[0])\n\n# Create target where all train samples are ones and all test samples are zeros\ntarget = train.is_test.values\n# Shuffle samples to mix zeros and ones\nidx = np.arange(all_idf.shape[0])\nnp.random.seed(1)\nnp.random.shuffle(idx)\nall_idf = all_idf[idx]\ntarget = target[idx]\n# Train a Logistic Regression\nfolds = StratifiedKFold(5, True, 1)\nfor trn_idx, val_idx in folds.split(all_idf, target):\n    lr = LogisticRegression()\n    lr.fit(all_idf[trn_idx], target[trn_idx])\n    print(roc_auc_score(target[val_idx], lr.predict_proba(all_idf[val_idx])[:, 1]))\n    predictions[val_idx] = lr.predict_proba(all_idf[val_idx])[:, 1]')


# In[115]:



#seperate train rows which have been misclassified as test and use them as validation
train["predictions"] = predictions
predictions_argsort = predictions.argsort()
train_sorted = train.iloc[predictions_argsort]

#select only trains set because we need to find train rows which have been misclassified as test set and use them for validation
train_sorted = train_sorted.loc[train_sorted.is_test == 0]

#Why did I chose 0.7 as thereshold? just a hunch, but you should try different thresholds i.e 0.6, 0.8 and see the difference in validation score and please report back. :) 
train_as_test = train_sorted.loc[train_sorted.predictions > 0.7]
#save the indices of the misclassified train rows to use as validation set
adversarial_set_ids = train_as_test.index.values
adversarial_set = pd.DataFrame(adversarial_set_ids, columns=['adversial_set_ids'])
#save adversarial set index
adversarial_set.to_csv('adversarial_set_ids.csv', index=False)


# We can now use the ids to seperate an adversarial validation set from the train set and validate our models on adversarial set because traditional Kfold might not work in this competition as found by [olivier](https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup) and [Konrad Banachewicz](https://www.kaggle.com/konradb/adversarial-validation).
# 
# 

# In[ ]:




