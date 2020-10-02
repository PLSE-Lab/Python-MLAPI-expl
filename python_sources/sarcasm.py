#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate


# In[ ]:


o_train = pd.read_csv("../input/train.csv")
o_valid = pd.read_csv("../input/valid.csv")

train = pd.read_csv("../input/train.csv")
valid = pd.read_csv("../input/valid.csv")

data = pd.concat([train, valid], sort=False)

example_sub = pd.read_csv("../input/sample_submission.csv")


# ### Visual Analysis

# In[ ]:


train.head(10)


# It seems that we have some weird URLs.

# In[ ]:


train[train.ID == 5022].article_link


# If we're gonna use the urls in the model, it may be a good idea to remove most of the trash from the URL.

# ---

# In[ ]:


plt.figure(figsize=(10, 6))

sb.countplot(y='is_sarcastic', data=train)


# In[ ]:


sarcastic = len(train[train.is_sarcastic == 1])
non_sarcastic = len(train[train.is_sarcastic == 0])

sarcastic / (non_sarcastic + sarcastic)


# The dataset seems balanced.

# ### Checking for empty values

# In[ ]:


print(np.where(pd.isnull(train)))
print(np.where(pd.isna(train)))
np.where(train.applymap(lambda x: x == ''))


# ### Baseline

# In[ ]:


sarcasm_classfication = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classify', LinearSVC(C=1))
])


# We made our baseline using SVM. Support Vector Machines are a type of ML algorithm that creates a hyperplane that divides our input space, while trying to keep this hyperplane at the maximum distance of the nearest data point of each class. The name "Suppor Vector" comes to the fact that the Xi inputs that are near the hyperplane are called support vectors.

# The CountVectorizer function is a kind of Tokenizer avaliable on the sklear lib. It takes every word of our input data and gives it an ID. Then, it literally counts the amount of times that each word appears on each document.
# 
# The problem with that is that we can have two different documents, that talk about the same thing, but if one is bigger than the other, it will have a greater count of some words. That's why we use Tfidf transformer. Tf stands for Term Frequency.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.headline, train.is_sarcastic)


# In[ ]:


sarcasm_classfication.fit(X_train, y_train)


# In[ ]:


print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))
print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))

cross_validate(sarcasm_classfication, train.headline, train.is_sarcastic, cv=5, scoring='roc_auc')


# ### Optimization

# #### Remove Punctuation

# In[ ]:


def remove_punctuation(dataframe):
    rgx = '(\'s|[!?,.:;\'$])'
    tmp = dataframe.copy()
    tmp['headline'] = tmp['headline'].str.replace(rgx, '')
    
    return tmp


# In[ ]:


tmp = train.copy()
tmp = remove_punctuation(tmp)

X_train, X_test, y_train, y_test = train_test_split(tmp.headline, tmp.is_sarcastic)

sarcasm_classfication.fit(X_train, y_train)

print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))
print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))

cross_validate(sarcasm_classfication, X_train, y_train, cv=5, scoring='roc_auc')


# The change doesn't seem relevant.

# ---

# #### Remove words

# In[ ]:


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx], idx) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    words = [x[0] for x in words_freq]
    count = [x[1] for x in words_freq]
    
    return pd.DataFrame({'Words': words[:n], 'Amount': count[:n]})


# In[ ]:


plt.figure(figsize=(10, 6))

tmp = get_top_n_words(train.headline, 9)

sb.barplot(x=tmp.Words, y=tmp.Amount)


# In[ ]:


t = train.copy()
for word in tmp.Words:
    t.headline.str.replace(word, '')
    
X_train, X_test, y_train, y_test = train_test_split(t.headline, t.is_sarcastic)

sarcasm_classfication.fit(X_train, y_train)

print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))
print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))

cross_validate(sarcasm_classfication, X_train, y_train, cv=5, scoring='roc_auc')


# The change doesn't seem relevant.

# ### Cleaning URLs

# In[ ]:


def remove_double_links(series):
    rgx = '(https?(?!.+https?).+)'
    
    tmp = series.copy()
    tmp['article_link'] = tmp['article_link'].str.extract(rgx)
    
    return tmp


# In[ ]:


def get_source(series):
    rgx = '((?!https?:)\/\/.+?\..+?\/)'
    
    tmp = series.copy()
    tmp['article_link'] = tmp['article_link'].str.extract(rgx)
    tmp['article_link'] = tmp['article_link'].str.strip(to_strip="/w")
    tmp['article_link'] = tmp['article_link'].str.strip(to_strip=r'^\.')
    return tmp


# In[ ]:


train = remove_double_links(train)
train = get_source(train)
train.groupby('article_link')['article_link'].describe()


# In[ ]:


def rename_article_link(dataframe):
    return dataframe.rename(columns={'article_link': 'source'})


# In[ ]:


train = rename_article_link(train)
train.head(10)


# In[ ]:


train['headline_and_source'] = train.source + ' ' + train.headline


# In[ ]:


train.head(10)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train['headline_and_source'], train.is_sarcastic)

sarcasm_classfication.fit(X_train, y_train)


# In[ ]:


print(roc_auc_score(y_train, sarcasm_classfication.decision_function(X_train)))
print(roc_auc_score(y_test, sarcasm_classfication.decision_function(X_test)))
cross_validate(sarcasm_classfication, train.headline_and_source, train.is_sarcastic, cv=25, scoring='roc_auc', return_train_score=True)


# I tried using cross-validation with 25 crossfolding and got a 1 in every single test. That could mean two things:
# 
# 1. The model is overfitting (somehow).
# 2. Since the change that I did was only including the source links in the model, maybe it is easy to verify which kind of article we have only using the source? Let's test.

# In[ ]:


tmp = train.copy()

X_tmp_train, X_tmp_test, y_tmp_train, y_tmp_test = train_test_split(tmp.source, tmp.is_sarcastic)

tmp_model = sarcasm_classfication

tmp_model.fit(X_tmp_train, y_tmp_train)

print(roc_auc_score(y_tmp_train, tmp_model.decision_function(X_tmp_train)))
print(roc_auc_score(y_tmp_test, tmp_model.decision_function(X_tmp_test)))

cross_validate(sarcasm_classfication, X_tmp_train, y_tmp_train, cv=5, scoring='roc_auc', return_train_score=True)


# Yeah. You can easily identify if the article is sarcastic or not only using the source. Maybe you don't even need to work with text. Let's try the good, old, RandomForest.

# The RandomForest is a type of algorithm uses multiple DecisionTrees. A Decision tree simply asks different questions about our training data and splits the tree based on these questions. The choice of what feature will be used to divide the input data is based on (in this case), Gini Impurity. Gini Impurity is the probability of incorrectly classifying an input data at that node. Decision Trees (and random forests) choose the feature that gives the higher amount of decrement in the gini impurity, for each node.

# A RandomForest expands the concept of the Decision Tree by:
# 
# 1. Training various trees with random samples of the data.
# 2. Chosing only a subset of the existing features to do each split.
# 
# Then, the final predictions are made by averaging the result of each tree.

# In[ ]:


tmp = train.copy()

tmp = tmp.drop(columns=['ID', 'headline', 'headline_and_source'])

tmp = pd.get_dummies(data=tmp, columns=['source'])

X_tmp_train, X_tmp_test, y_tmp_train, y_tmp_test = train_test_split(tmp.drop(columns=['is_sarcastic']), tmp.is_sarcastic)

tmp_model = RandomForestClassifier(n_jobs=-1, n_estimators=100)

tmp_model.fit(X_tmp_train, y_tmp_train)

prob = tmp_model.predict_proba(X_tmp_train)
prob = prob[:, 1]

print(roc_auc_score(y_tmp_train, prob))

prob = tmp_model.predict_proba(X_tmp_test)
prob = prob[:, 1]

print(roc_auc_score(y_tmp_test, prob))

cross_validate(tmp_model, X_tmp_train, y_tmp_train, cv=5, scoring='roc_auc', return_train_score=True)


# Yeah... No text needed.

# ## Metric

# As in the previous competition, we used ROC AUC as a measure of our model quality.
# 
# The ROC is a curve plotted using the true positive rate (tp / tp + tn) and false negative rate (fn / fn + tn), using different thresholds.

# ## Sending

# In[ ]:


valid = remove_double_links(valid)
valid = get_source(valid)
valid = valid.rename(columns={'article_link': 'source'})


# In[ ]:


predicted = sarcasm_classfication.predict(valid.source)

prediction_dataframe = pd.DataFrame({'ID': valid.ID, 'is_sarcastic': predicted})

prediction_dataframe.to_csv('output.csv', index=False)

