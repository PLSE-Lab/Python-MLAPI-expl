#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing Project (NU MSAI)
# ### Authors: Grant Gasser, Sundar Thevar, Blaine Rothrock, Zhili Wang

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
DATA_PATH = '/kaggle/input/nlp-getting-started/'


# ## Exploratory Data Analysis
# 
# ### Notebook Sources: 
# * [NLP with Disaster Tweets - EDA, Cleaning and BERT](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert/notebook)

# ### Read and Inspect Data

# In[ ]:


train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
ds = ['train', 'test', 'sample submission']

print("Training set has {} rows and {} columns.".format(train.shape[0], train.shape[1]))
print("Test set has {} rows and {} columns.".format(test.shape[0], test.shape[1]))

print()

print(train.columns)
print(test.columns)


# Clearly not disasters..

# In[ ]:


train[train.target == 0].head()


# Disasters

# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Keywords
# Not including NaNs

# In[ ]:


print('Train Keyword Distribution:\n\n')
print(train.keyword.value_counts())
print('\n', '-' * 50, '\n')
print('Test Keyword Distribution:\n\n')
print(test.keyword.value_counts())


# ### Class Balance 
# * (4342 0's and 3271 1's)

# In[ ]:


temp = train['target'].value_counts(dropna = False).reset_index()
temp.columns = ['target', 'counts']
countplt = sns.countplot(x = 'target', data = train, hue = train['target'])
countplt.set_xticklabels(['0: Not Disaster (4342)', '1: Disaster (3271)'])


# ### Target Distribution in Keywords

# In[ ]:


train['target_mean'] = train.groupby('keyword')['target'].transform('mean')
fig = plt.figure(figsize=(8, 72), dpi=100)
sns.countplot(y=train.sort_values(by='target_mean', ascending=False)['keyword'],
              hue=train.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')
plt.show()
train.drop(columns=['target_mean'], inplace=True)


# ### Missing Values (NaNs)

# In[ ]:


print('Count NaN:')
print(train.isnull().sum(), '\n')
print('Percentage NaN:')
print(train.isnull().sum()/ len(train))


# In[ ]:


print('Count NaN:')
print(test.isnull().sum(), '\n')
print('Percentage NaN:')
print(test.isnull().sum()/ len(test))


# ### Number of Characters in `text` - Train Set

# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
train_len = train[train['target'] == 0]['text'].str.len()
ax1.hist(train_len,color='blue')
ax1.set_title('Not A Disaster')
train_len = train[train['target'] == 1]['text'].str.len()
ax2.hist(train_len,color='red')
ax2.set_title('Disaster')
fig.suptitle('Characters in Train Set\'s Text')
plt.show()


# ### Number of Words in `text` - Train Set

# In[ ]:


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
train_len = train[train['target'] == 0]['text'].str.split().map(lambda x: len(x))
ax1.hist(train_len,color='blue')
ax1.set_title('Not A Disaster')
train_len = train[train['target'] == 1]['text'].str.split().map(lambda x: len(x))
ax2.hist(train_len,color='red')
ax2.set_title('Disaster')
fig.suptitle('Words in Train Set\'s Text')
plt.show()


# ## Data Cleaning and Preparation

# ### Tokenizing and Removing punctutation
# Thoughts:
#    * use some regex code from [here](https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert/notebook)
#    * use the Regex Tokenizer

# #### Prepend keyword to text - Run only once!

# In[ ]:


train.loc[train['keyword'].notnull(), 'text'] = train['keyword'] + ' ' + train['text']
test.loc[test['keyword'].notnull(), 'text'] = test['keyword'] + ' ' + test['text']

# view
train[train['keyword'].notnull()].head()


# In[ ]:


train = train.drop(['id', 'keyword', 'location'], axis=1)
test = test.drop(['keyword', 'location'], axis=1) # keep id

train.head()


# In[ ]:


y_train = np.array(train['target'])


# In[ ]:


# NLTK Tweet Tokenizer for now
# from nltk.tokenize import TweetTokenizer
# tknzr = TweetTokenizer()

# x_train = []
# x_test = []

# # tokenize but put back into one string 
# for i, v in enumerate(train['text']):
#     x_train.append(' '.join(tknzr.tokenize(v)))

# for i, v in enumerate(test['text']):
#     x_test.append(' '.join(tknzr.tokenize(v)))

# x_train = np.array(x_train)
# x_test = np.array(x_test)

# x_train[:2]


# In[ ]:


# # if we want to clean up with regex ourselves
# def clean_text(text):
#     re.sub(r'[0-9]', '', str(text))
    
#     # add more here
    
#     return text


# train['text'] = train['text'].apply(lambda t: clean_text(t))
# test['text'] = test['text'].apply(lambda t: clean_text(t))


# ### Rectify Mislabeled Samples
# * 18 of them, need to manually fix

# ## Naive Bayes (Non-Neural Baseline Method)
# The goal of this step is to learn about naive bayes and establish a baseline non-neural network model
# 
# * Great Naive Bayes [Explanation and Tutorial](https://towardsdatascience.com/algorithms-for-text-classification-part-1-naive-bayes-3ff1d116fdd8)
# * Another [Explanation and Tutorial](https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/)

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import roc_auc_score

vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(train['text'])

# lots of garbage "words" here, hence the need for more pre-processing
print(vectorizer.get_feature_names()[2700:3000])


# ### Fit and predict

# In[ ]:


# alpha is smoothing param
model = BernoulliNB(alpha=1.0)
model.fit(x_train_vectorized, y_train)

# prepare test
x_test_NB = vectorizer.transform(test['text'])


# ## BERT + Neural Network
# The goal of this step is to replicate and compete with some of the best results in this competition (F1 score in low-mid 80s)

# 

# ## Generate own word embeddings with NNLM 
# The goal of this step is to generate word embeddings that are specific to the available Tweet data and see if we can improve performance
#    * Maybe replicate Transformer architecture (the more recent, successful architectures)
#    * Do we have enough data to do this? Most NNLMs are trained on millions of tokens.
#    * Do we have the compute resources to train such a model?

# In[ ]:





# ## Enter submission

# In[ ]:


sample_submission['target'] = model.predict(x_test_NB)
sample_submission.head()


# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# In[ ]:




