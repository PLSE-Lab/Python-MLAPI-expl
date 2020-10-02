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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#sklearn's train_test_split is a powerful package
#which can randomly split dataset into training and testing parts.
#And it is extremely easy to apply.
from sklearn.model_selection import train_test_split

#First, let's look at the iris dataset
iris = pd.read_csv('../input/Iris.csv')
iris.head()


# In[ ]:


iris.pop('Id')  #Id column will not to be used, so remove it.
target_values = iris.pop('Species') #or you can call it 'labels'
target_values.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace = True)

#Split iris dataset
train_data, test_data, train_target, test_target = train_test_split(iris, target_values, test_size=0.2)
#Let's check the content of train_data
train_data.head()


# In[ ]:


# Model 1: Gaussian Process
from sklearn.gaussian_process import GaussianProcessClassifier

gpc = GaussianProcessClassifier()
gpc.fit(train_data, train_target)


# In[ ]:


#Training score (correct rate)
gpc.score(train_data, train_target)


# In[ ]:


#Testing score (correct rate)
gpc.score(test_data, test_target)


# In[ ]:


#Model 2: Naive Bayes
#There are 3 types of Naive Bayes model: GaussianNB, MultinomialNB, BernolliNB
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
#Model 2.1 gnb
gnb = GaussianNB()
gnb.fit(train_data, train_target)


# In[ ]:


#Model 2.2 mnb
mnb = MultinomialNB()
mnb.fit(train_data, train_target)


# In[ ]:


#Model 2.3 bnb
bnb = BernoulliNB()
bnb.fit(train_data, train_target)


# In[ ]:


#Training Scores (correct rate)
print("gnb --", gnb.score(train_data, train_target))
print("mnb --", mnb.score(train_data, train_target))
print("bnb --", bnb.score(train_data, train_target))


# In[ ]:


#Testing Scores (correct rate)
print("gnb --", gnb.score(test_data, test_target))
print("mnb --", mnb.score(test_data, test_target))
print("bnb --", bnb.score(test_data, test_target))


# You may noticed the result of bnb is not as good as the others.
# The reason here is, in order to perform bnb, we need to transform the data to binary version, which can be done by specify 'binarize=True' inside bnb

# In[ ]:


bnb_binary = BernoulliNB(binarize=True)
bnb_binary.fit(train_data, train_target)


# In[ ]:


#Training Score (correct rate)
bnb_binary.score(train_data, train_target)


# In[ ]:


#Testing Score (correct rate)
bnb_binary.score(test_data, test_target)


# The result of bnb here is still not good.
# Can anyone tell me why the result is like this?
# I really do not know what should I do to improve the results.
# 
# Thank you
