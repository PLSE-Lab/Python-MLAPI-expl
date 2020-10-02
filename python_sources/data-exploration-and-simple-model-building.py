#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## This notebook is primarily for exploration purposes and training simple predictive models on the data. 
# ### PS: You wont find any high end DNNs like LSTMs or GRUs in this notebook
# 
# This notebook is for people who are just starting out in data sciences and text mining. Although Deep Learning models such as LSTMs are currently dominating the NLP landscape, it might be difficult for beginners to get a grasp over them without understanding the basics of text mining, also beginners wont totally appreciate the modern NN based architectures if they are not fully aware of the different classical machine learning methodologies and their pitfalls.
# 
# In this notebook, I am going to explain in simple steps how to go about exploring text data and how to make simple models like simple Logistic regression, SVM and Naiive Bayes Classifiers.

# In[ ]:


# First we need to load the data into a dataframe

df = pd.read_csv("../input/train.csv")


# In[32]:


df.head()


# In[ ]:


# Lets also have a look at the test data

tdf = pd.read_csv("../input/test.csv")
print(tdf.head())
del(tdf)


# So the test data is composed of only the comments, and our algorithm has to predict the toxicity score of the text between 0 and 1, o being the lowest and 1 being the highest.

# ### Lets have a look at some random comment texts and their labels to better understand the bias mentioned in the competition description

# ### Neutral/ slightly toxic comments

# In[ ]:


# Let's first have a look at some of the comments whose scores were above 0.0

random_indices = np.random.choice([i for i in range(len(df)) if df["target"][i] > 0.], 5)
for i in random_indices:
    print("Text: ", df["comment_text"][i])
    print("Score: ", df["target"][i])


# ### Non toxic comments

# In[ ]:


# Now let's have a look at completely non toxic comments

random_indices = np.random.choice([i for i in range(len(df)) if df["target"][i] == 0.], 5)
for i in random_indices:
    print("Text: ", df["comment_text"][i])
    print("Score: ", df["target"][i])


# ### Severely toxic comments

# In[ ]:


# Now lets have a look at very toxic comments: target > 0.75

random_indices = np.random.choice([i for i in range(len(df)) if df["target"][i] >= .75], 5)
for i in random_indices:
    print("Text: ", df["comment_text"][i])
    print("Score: ", df["target"][i])


# ### Now lets have a look at the distribution of the target

# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 5))
plt.hist(df['target'], bins = 100)
plt.show()


# It seems that the majority of the comments are non toxic (score == 0.0). Therefore, we sould have a look at the toxic ones separately

# In[ ]:


plt.figure(figsize=(20, 5))
plt.hist(df[df['target'] > 0.0]['target'], bins = 100)
plt.show()


# So there are very few comments with high scores, therefore, we should count all the comments with low scores (e.g., target > 0.10) as toxic

# ## Making the model
# Now we should start making a model to predict the target in unseen texts

# #### Training and Testing Data

# In[ ]:


tdf = df.loc[:30000, ["comment_text", "target"]]
tdf.head()


# #### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tdf["comment_text"], tdf["target"], test_size = .10)


# ## Feature Extraction
# 
# #### First I am going to import Count Vectorizer and TFIDF Vectorizers which are used to convert the texts into feature vector forms which can be used as by the machine learning algorithm. 
# For more information please visit https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# and https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# 
# These are used to convert the text data from strings which cannot be used directly by the machine learning algorithms into floats which can be used as features.
# In short these algorithms count the occurence of every word token in the texts and convert the word scores for each text into a float. For more details refer to the abovementioned links.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# ## Model
# Now I will build a simple model for binary text classification. The two classes being toxic and non toxic.
# I will explore different classic machine learning algorithms for this purpose. 
# However I wont go into RNNs in this notebook. Very soon I will release another notebook with LSTMs, word embeddings etc.

# ## Logistic Regression with Count Vectoriser

# In[34]:


cvect = CountVectorizer(min_df = 0.1, ngram_range=(1, 3), analyzer="word").fit(X_train)
X_trcv = cvect.transform(X_train)
X_tscv = cvect.transform(X_test)

# In order to convert the coninuous values of the target to binary, as logistic regression can accept only binary values (0 or 1) as the target values
# Here we choose 0.1 as a cutoff as we want to classify even slightly toxic comments as toxic
y_train_lg = np.array(y_train > 0.1, dtype=np.float)
y_test_lg = np.array(y_test > 0.1, dtype=np.float)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression().fit(X_trcv, y_train_lg)
print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))
print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))
predicted = clf.predict(X_tscv)

from sklearn.metrics import precision_score, recall_score
print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# ### Dummy Classifier
# We can see that the accuracy scores are not that bad. However, the precision and recall scores are just unacceptable. This is because of the imbalanced targets, as most of the targets have label 0 and few have label 1. So even a dumb classifier which always predicts the most common class would give a respectable accuracy score. So we need to compare our classifier's performance with once such most-common-class-classifier. Lets see how to do that...

# In[33]:


from sklearn.dummy import DummyClassifier

dclf = DummyClassifier(strategy="most_frequent").fit(X_trcv, y_train_lg)
print("Training Accuracy: {}".format(dclf.score(X_trcv, y_train_lg)))
print("Testing Accuracy: {}".format(dclf.score(X_tscv, y_test_lg)))
predicted = dclf.predict(X_tscv)

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# ### Performance:
# Therefore, we see that our classifier is not much better than a simple baseline model which just predicts all outputs to be the most frequent class.
# Therefore, we need better models. Hence we will explore other models better suited for text classification purposes viz Naive Bayes Classifier and Support Vector Machines

# #### Bernoulli Naive Bayes using Count Vectors as features

# In[35]:


from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB().fit(X_trcv, y_train_lg)
print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))
print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))
predicted = clf.predict(X_tscv)

from sklearn.metrics import precision_score, recall_score
print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# ### Improvement
# We can see that the precision and recall scores have gone up, but the total accuracy score has gone down.
# Next we will run:
# #### Multinomial NB with TFIDF and Count Vector features

# In[36]:


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_trcv, y_train_lg)
print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))
print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))
predicted = clf.predict(X_tscv)

from sklearn.metrics import precision_score, recall_score
print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# The precision and recall scores again went down. But the accuracy increased and became equal to the Dummy classifier
# Lets use TFIDF features now

# In[38]:


tfvect = TfidfVectorizer().fit(X_train)

X_trtf = tfvect.transform(X_train)
X_tstf = tfvect.transform(X_test)

clf = MultinomialNB().fit(X_trtf, y_train_lg)
print("Training Accuracy: {}".format(clf.score(X_trtf, y_train_lg)))
print("Testing Accuracy: {}".format(clf.score(X_tstf, y_test_lg)))
predicted = clf.predict(X_tstf)

from sklearn.metrics import precision_score, recall_score
print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# ### Improvement:
# In this model the precision and accuracy have improved, however, the recall is very low. Lets now play around with SVM Models, which are also used very often in text classification

# #### SVM Classifier with a linear kernel and TFIDF vectors

# In[39]:


from sklearn.svm import SVC

clf = SVC(kernel="linear").fit(X_trtf, y_train_lg)
print("Training Accuracy: {}".format(clf.score(X_trtf, y_train_lg)))
print("Testing Accuracy: {}".format(clf.score(X_tstf, y_test_lg)))
predicted = clf.predict(X_tstf)

from sklearn.metrics import precision_score, recall_score
print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# ### Major Improvement
# The above classifier has major improvements in terms of test accuracy, precision and recall.

# #### SVM Classifier with linear kernel and Count Vectors
# Now lets try the last classifier on our list. 

# In[40]:


clf = SVC(kernel="linear").fit(X_trcv, y_train_lg)
print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))
print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))
predicted = clf.predict(X_tscv)

from sklearn.metrics import precision_score, recall_score
print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))
print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))


# Again the above classifier underperformed. It is almost always impossible to say which classifier will give the best results. Its better to try different classifiers and go with the one with the best performance.

# ### End
# Unfortunately I have to end this notebook here. However, if this is helpful, please comment. I will do further analysis using simple machine learning algorithms and share with you.
# 
# Within a few weeks I will publish a notebook on how to train an LSTM Classifier for this task. Stay tuned.

# In[ ]:




