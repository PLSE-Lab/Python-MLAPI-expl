#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings as fw
fw('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/drugsComTrain_raw.csv')
test = pd.read_csv('../input/drugsComTest_raw.csv')


# In[ ]:


train.head()


# In[ ]:


top_30_drugs = train.drugName.value_counts()[:30]
plt.figure(figsize = (15,7))
top_30_drugs.plot(kind = 'bar');
plt.title('Top 30 Drugs by Count',fontsize = 20);


# #### Here, we can see the problems faced by people by count.  Birth Control followed by Deprerssion , anxiety, pain and Bipolar disorder are the top problems faced by people
# 

# In[ ]:


top_30_problems = train.condition.value_counts()[:30]
plt.figure(figsize = (15,7))
top_30_problems.plot(kind = 'bar');
plt.title('Top 30 Problems',fontsize = 20);


# In[ ]:


import string
train['review_clean']=train['review'].str.replace('[{}]'.format(string.punctuation), '')
train.head()


# In[ ]:


train = train.fillna({'review':''})  # fill in N/A's in the review column


# In[ ]:


plt.figure(figsize = (15,7))
train.rating.value_counts().plot(kind = 'bar');
plt.xlabel('Ratings',fontsize = 15);
plt.title('Ratings by count',fontsize = 18);


# #### We'll consider ratings more than 5 as positive and less than or equal to 5 as negative.
# #### For the sentiment column, we use +1 for the positive class label and -1 for the negative class label. A good way is to create an anonymous function that converts a rating into a class label and then apply that function to every element in the rating column.

# In[ ]:


train['sentiment'] = train['rating'].apply(lambda rating : +1 if rating > 5 else -1)
train.head()


# ### Split into training and test sets
# #### Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. 
# ![](https://cdn-images-1.medium.com/max/1600/1*-8_kogvwmL1H6ooN1A1tsQ.png)

# In[ ]:


from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(train,test_size = 0.20)
print('Size of train_data is :', train_data.shape)
print('Size of test_data is :', test_data.shape)


# #### Build the word count vector for each review
# We will now compute the word count for each word that appears in the reviews. A vector consisting of word counts is often referred to as bag-of-word features. Since most words occur in only a few reviews, word count vectors are sparse. For this reason, scikit-learn and many other tools use sparse matrices to store a collection of word count vectors. Refer to appropriate manuals to produce sparse word count vectors. General steps for extracting word count vectors are as follows:
# 
# Learn a vocabulary (set of all words) from the training data. Only the words that show up in the training data will be considered for feature extraction. Compute the occurrences of the words in each review and collect them into a row vector. Build a sparse matrix where each row is the word count vector for the corresponding review. Call this matrix train_matrix. Using the same mapping between words and columns, convert the test data into a sparse matrix test_matrix. The following cell uses CountVectorizer in scikit-learn. Notice the token_pattern argument in the constructor.

# In[ ]:


import gc
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer()

train_matrix = vectorizer.transform(train_data['review_clean'].values.astype('U'))
test_matrix = vectorizer.transform(test_data['review_clean'].values.astype('U'))

gc.collect()


# ## Train a sentiment classifier with logistic regression.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
rf = clf.fit(train_matrix,train_data['sentiment'])


# Precision and recall In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# 
# The traditional F-measure or balanced F-score (F1 score) is the harmonic mean of precision and recall:
# ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/057ffc6b4fa80dc1c0e1f2f1f6b598c38cdd7c23)

# In[ ]:


y_pred = rf.predict(test_matrix)
from sklearn.metrics import f1_score
f1_score(y_pred,test_data.sentiment)


# In[ ]:


from sklearn import tree
import graphviz


clf = tree.DecisionTreeClassifier() # init the tree
clf = clf.fit(train_matrix, train_data.sentiment) # train the tree
# export the learned decision tree
dot_data = tree.export_graphviz(clf,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("sentiment") 


# In[ ]:




