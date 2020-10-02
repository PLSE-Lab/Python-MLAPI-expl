#!/usr/bin/env python
# coding: utf-8

# Original notebook by Balaka Biswas
# https://www.kaggle.com/balaka18

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/email-spam-classification-dataset-csv/emails.csv")
df.head(20)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# # Creating the NB Model
# 
# In this project we are clasifying mails typed in by the user as either 'Spam' or 'Not Spam'. Our original dataset was a folder of 5172 text files containing the emails.
# 
# Now let us understand why we have separated the words from the mails. This is because, this is a text-classification problem. When a spam classifier looks at a mail, it searches for potential words that it has seen in the previous spam emails. If it finds a majority of those words, then it labels it as 'Spam'. **Why did I say majority ? -->**
# 
# *CASE 1* : suppose let's take a word 'Greetings'. Say, it is present in both 'Spam' and 'Not Spam' mails.
# 
# *CASE 2* : Let's consider a word 'lottery'.Say, it is present in only 'Spam' mails.
# 
# *CASE 3* : Let's consider a word 'cheap'. Say, it is present only in spam.
# 
# If now we get a test email, and it contains all the three words metioned above, there's high probability that it is a 'Spam' mail.
# 
# The most effective algorithm for text-classification problems is the Naive Bayes algorithm, that works on the classic Bayes' theorem. This theorem works on every individual word in the test data to make predictions(the conditional probability with higher probability is the predicted result). 
# 
# ________________________________________________________________________________________________________________________
# 
# Say, our test email(S)is,*"You have won a lottery"*
# 
# **HOW NAIVE BAYES WORKS ON THIS DATA -->**
# 
# P(S) = P('You') * P('have') * P('won') * P('a') * P('lottery') ____ 1
# 
# Therefore, P(S|Spam) = P('You'|Spam) * P('have'|Spam) * P('won'|Spam) * P('a'|Spam) * P('lottery'|Spam) ____ 2 
# 
# Same calculation for P(S|Not_Spam) ____ 3
# 
# If 2 > 3, then 'Spam' Else, 'Not_Spam'.
# 
# **WHAT IF THE PROBABILITY IS ZERO ?** Here comes the concept of Laplace Smoothing, where P(words) = (word_count + 1)/(total_no_of_words + no_of_unique_words)
# 
# ________________________________________________________________________________________________________________________
# 
# Here, we'll work on the existing Multinomial Naive Bayes classifier (under scikit-learn). To further understand how well Naive Bayes works for text-classification, we'll use another standard classifier, SVC, to see how the two models perform.
# 
# **WILL ENSEMBLE MODELS WORKS BETTER ?** Let us see. We will use Random Forests to compare.

# In[ ]:


X = df.iloc[:,1:3001]
X


# In[ ]:


Y = df.iloc[:,-1].values
Y


# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.25)


# # Naive Bayes

# In[ ]:


mnb = MultinomialNB(alpha=1.9)         # alpha by default is 1. alpha must always be > 0. 
# alpha is the '1' in the formula for Laplace Smoothing (P(words))
mnb.fit(train_x,train_y)
y_pred1 = mnb.predict(test_x)
print("Accuracy Score for Naive Bayes : ", accuracy_score(y_pred1,test_y))


# # Support Vector Machines
# 
# Support Vector Machine is the most sought after algorithm for classic classification problems. SVMs work on the algorithm of Maximal Margin, i.e, to find the maximum margin or threshold between the support vectors of the two classes (in binary classification). The most effective Support vector machines are the soft maximal margin classifier, that allows one misclassification, i.e, the model starts with low bias(slightly poor performance) to ensure low variance later.
# 
# ________________________________________________________________________________________________________________________
# 
# Let us see the model performance.

# In[ ]:


svc = SVC(C=1.0,kernel='rbf',gamma='auto')         
# C here is the regularization parameter. Here, L2 penalty is used(default). It is the inverse of the strength of regularization.
# As C increases, model overfits.
# Kernel here is the radial basis function kernel.
# gamma (only used for rbf kernel) : As gamma increases, model overfits.
svc.fit(train_x,train_y)
y_pred2 = svc.predict(test_x)
print("Accuracy Score for SVC : ", accuracy_score(y_pred2,test_y))


# As expected, SVM's performance is slightly poorer than Multinomia Naive Bayes

# # Random Forests (Bagging)
# 
# Ensemble methods turn any feeble model into a highly powerful one. Let us see if ensemble model can perform better than Naive Bayes

# In[ ]:


rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
# n_estimators = No. of trees in the forest
# criterion = basis of making the decision tree split, either on gini impurity('gini'), or on infromation gain('entropy')
rfc.fit(train_x,train_y)
y_pred3 = rfc.predict(test_x)
print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3,test_y))


# As expected, Random Forest Classifier performs the best among the three. Decision tree classifiers are excellent classifiers. Random forest is a popular ensemble model that uses a forest of decision trees. So, obviously, combibining the accuracy of 100 trees (as n_estimators=100 here), will create a powerful model.

# ## Ending notes:
# 
# This was a purely comparative study to check the workability of the dataset that I created, and to check how conventional models perform on my dataset. In my next kernel, I will show the code behind the extraction of this dataset from the raw text files.
