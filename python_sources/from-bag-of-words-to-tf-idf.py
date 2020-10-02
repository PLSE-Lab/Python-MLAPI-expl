#!/usr/bin/env python
# coding: utf-8

# # Chapter 4. The Effects of Feature Scaling: From Bag-of-Words to Tf-Idf

# This chapter is one of the chapters of the book, _Feature Engineering for Machine Learning_. Since I do not have enough resource on my local machine, I have ended up creating this kernel to practice the source code of the chapter 4 while reading the book. You can also examine all the main chapters' code over the original GitHub repository of the book: https://github.com/alicezheng/feature-engineering-book

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sklearn 

from sklearn.model_selection import train_test_split


# In[ ]:


# Load Yelp business data
biz_f = open('../input/yelp_academic_dataset_business.json')
biz_df = pd.DataFrame([json.loads(x) for x in biz_f.readlines()])
biz_f.close()


# In[ ]:


biz_df.shape


# In[ ]:


# Load Yelp reviews data
review_file = open('../input/yelp_academic_dataset_review.json')
review_df = pd.DataFrame([json.loads(next(review_file)) for x in range(biz_df.shape[0])])
review_file.close()


# In[ ]:


review_df.shape


# In[ ]:


biz_df.dropna(subset=['categories'], inplace=True)


# In[ ]:


# Pull out only Nightlife and Restaurants businesses
two_biz = biz_df[biz_df.apply(lambda x: 'Nightlife' in x['categories'] or 'Restaurants' in x['categories'], axis=1)]


# In[ ]:


two_biz.head()


# In[ ]:


# Join with the reviews to get all reviews on the two types of business
two_biz_reviews = two_biz.merge(review_df, on='business_id', how='inner')


# In[ ]:


two_biz_reviews.shape


# In[ ]:


# Trim away the features we won't use
two_biz_reviews = two_biz_reviews[['business_id', 'name', 'stars_y', 'text', 'categories']]


# In[ ]:


two_biz_reviews.shape


# In[ ]:


# Create the target column--True for Nightlife businesses, and False otherwise
two_biz_reviews['target'] = two_biz_reviews.apply(lambda x: 'Nightlife' in x['categories'], axis=1)


# In[ ]:


two_biz_reviews.head()


# ## Creating a Classification Dataset

#  Imbalanced datasets are problematic for modeling because the model will expend most of its effort fitting to the larger class. Since we have plenty of data in both classes, a good way to resolve the problem is to downsample the larger class (restaurants) to be roughly the same size as the smaller class (nightlife).

# In[ ]:


two_biz_reviews.target.value_counts(normalize=True)


# In[ ]:


# Create a class-balanced subsample to play with
nightlife = two_biz_reviews[two_biz_reviews.apply(lambda x: 'Nightlife' in x['categories'], axis=1)]
restaurants = two_biz_reviews[two_biz_reviews.apply(lambda x: 'Restaurants' in x['categories'], axis=1)]

nightlife_subset = nightlife.sample(frac=0.74, random_state=123)
restaurant_subset = restaurants.sample(frac=0.35, random_state=123)

combined = pd.concat([nightlife_subset, restaurant_subset])

# Split into training and test datasets
training_data, test_data = train_test_split(combined,  train_size=0.7, random_state=123)


# In[ ]:


training_data.shape


# In[ ]:


training_data.target.value_counts(normalize=True)


# In[ ]:


test_data.shape


# In[ ]:


test_data.target.value_counts(normalize=True)


# ## Scaling Bag-of-Words with Tf-Idf Transformation

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# Represent the review text as a bag-of-words 
bow_transform = CountVectorizer()
X_tr_bow = bow_transform.fit_transform(training_data['text'])

X_te_bow = bow_transform.transform(test_data['text'])
len(bow_transform.vocabulary_)


# In[ ]:


X_tr_bow[:1]


# In[ ]:


X_tr_bow.shape


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[ ]:


# Create the tf-idf representation using the bag-of-words matrix
tfidf_trfm = TfidfTransformer(norm=None)
X_tr_tfidf = tfidf_trfm.fit_transform(X_tr_bow)

X_te_tfidf = tfidf_trfm.transform(X_te_bow)


# In[ ]:


from sklearn.preprocessing import normalize


# In[ ]:


# Just for kicks, l2-normalize the bag-of-words representation
X_tr_l2 = normalize(X_tr_bow, axis=0)
X_te_l2 = normalize(X_te_bow, axis=0)


# In[ ]:


# test and train targets
y_tr = training_data['target']
y_te = test_data['target']


# ## Classification with Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


def simple_logistic_classify(X_tr, y_tr, X_test, y_test, description, _C=1.0):
    ### Helper function to train a logistic classifier and score on test data
    m = LogisticRegression(solver='lbfgs', C=_C).fit(X_tr, y_tr)
    s = m.score(X_test, y_test)
    print ('Test score with', description, 'features:', s)
    return m


# In[ ]:


m1 = simple_logistic_classify(X_tr_bow, y_tr, X_te_bow, y_te, 'bow')


# In[ ]:


m2 = simple_logistic_classify(X_tr_l2, y_tr, X_te_l2, y_te, 'l2-normalized')


# In[ ]:


m3 = simple_logistic_classify(X_tr_tfidf, y_tr, X_te_tfidf, y_te, 'tf-idf')


# ## Tuning Logistic Regression with Regularization

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


# Specify a search grid, then do a 5-fold grid search for each of the feature sets
param_grid_ = {'C': [1e-5, 1e-3, 1e-1, 1e0, 1e1, 1e2]}


# In[ ]:


# Tune classifier for bag-of-words representation
bow_search = GridSearchCV(LogisticRegression(), cv=5, param_grid=param_grid_)
bow_search.fit(X_tr_bow, y_tr)


# In[ ]:


bow_search.best_params_


# In[ ]:


# Tune classifier for L2-normalized word vector
l2_search = GridSearchCV(LogisticRegression(), cv=5,param_grid=param_grid_)
l2_search.fit(X_tr_l2, y_tr)


# In[ ]:


l2_search.best_params_


# In[ ]:


# Tune classifier for tf-idf
tfidf_search = GridSearchCV(LogisticRegression(), cv=5,param_grid=param_grid_)
tfidf_search.fit(X_tr_tfidf, y_tr)


# In[ ]:


tfidf_search.best_params_


# In[ ]:


import pandas as pd
# Let's check out one of the grid search outputs to see how it went
search_results = pd.DataFrame.from_dict({
    'bow': bow_search.cv_results_['mean_test_score'],
    'tfidf': tfidf_search.cv_results_['mean_test_score'],
    'l2': l2_search.cv_results_['mean_test_score']
})
search_results.index = param_grid_['C']
search_results
# Average cross validation classifier accuracy scores


# In[ ]:


# find which parameter is the best for each model
search_results.apply(lambda column: column.idxmax(), axis=0)
# higher values of C correspond to less regularization


# In[ ]:


# Our usual matplotlib incantations. Seaborn is used here to make the plot pretty.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

ax = sns.boxplot(data=search_results, width=0.4)
ax.set_ylabel('Accuracy', size=14)
ax.tick_params(labelsize=14)


# In[ ]:


# Train a final model on the entire training set, using the best hyperparameter 
# settings found previously. Measure accuracy on the test set.
m1 = simple_logistic_classify(X_tr_bow, y_tr, X_te_bow, y_te, 'bow', _C=bow_search.best_params_['C'])
m2 = simple_logistic_classify(X_tr_l2, y_tr, X_te_l2, y_te, 'l2-normalized', _C=l2_search.best_params_['C'])
m3 = simple_logistic_classify(X_tr_tfidf, y_tr, X_te_tfidf, y_te, 'tf-idf', _C=tfidf_search.best_params_['C'])


# If you have made it so far, thanks for examining! 

# In[ ]:




