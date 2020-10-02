#!/usr/bin/env python
# coding: utf-8

# # Why Stratification?
# 
#    For model evaluation we often use train_test_split from sklearn to split the data into train set and test set. This is fine if the sample size is large. If the sample size is small we cannot expect the same. 
#    
#    For iris dataset, the data is equally distributed between the three species of flowers. Have you ever given a thought that your train_test_split on the data gives equal proportion of these classes?
#    
#    If it is not so then we are training on one distribution and predicting from another distribution!
#    

# ![StratifiedRandomSampling.jpg](attachment:StratifiedRandomSampling.jpg)
# Image from Wikipedia

# 
#    This is where StratifiedShuffleSplit helps us. Let us built a logistic regressions with train_test_split and StrstifiedShuffleSplit and analyze the difference.
#    
#    This notebook is inspired from this article "https://blog.usejournal.com/creating-an-unbiased-test-set-for-your-model-using-stratified-sampling-technique-672b778022d5"

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris=pd.read_csv('../input/iris/Iris.csv',index_col='Id')
iris.head()


# In[ ]:


iris.info()


# **More on Iris:**
# Iris data set is collection of length and width of sepal and petal of iris flower and there species. The dataset has three species of Iris.
# 
# We can classify the flower species by using the four features. So it is a four feature three class classification problem.
# 
# Lets analyze how our target variable is distributed in the data.[](http://)

# In[ ]:


iris.Species.value_counts()


# In[ ]:


plt.pie(iris.Species.value_counts(),labels=iris.Species.unique(),autopct = '%1.2f%%')


# Here we note that the given dataset has equal distribution of target variables. If we use train_test_split will it get equal proportion from all species? i.e., will it be a strtified sample?
# 
# Lets look!

# In[ ]:


y = iris.pop('Species') #Target
X = iris  #DataFrame with features

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)
print(y_train.value_counts())
print(y_test.value_counts())


# In[ ]:


plt.figure(figsize=(10,15))
plt.subplot('121')
plt.pie(y_train.value_counts(),labels=y_train.unique(),autopct = '%1.2f%%',shadow = True)
plt.title('Training Dataset')

plt.subplot('122')
plt.pie(y_test.value_counts(),labels=y_test.unique(),autopct = '%1.2f%%', shadow =True)
plt.title('Test Dataset')

plt.tight_layout()


# We see a clear difference in the way the training data and test data are distributed!
# 
# Why this happens? What does train_test_split do?
# 
# The train_test_split function randomly splits the training set and testing set. But the problem here is the less amount of data(150 only). If we have only 6 data,two from each class and we split it into train and test we cannot expect a random split that contains equal proportions both in test and train. 
# 
# So lesser the data, more the chance that your test and train set are not stratified!  

# # StratifiedShuffleSplit
# How do we achieve stratified split on a small amount of data such as this?
# 
# Here stands tall the StratifiedShuffleSplit of sklearn. It actually forces the data to be stratified.  Lets split and see..

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit 

splitter=StratifiedShuffleSplit(n_splits=1,random_state=12) #we can make a number of combinations of split
#But we are interested in only one.

for train,test in splitter.split(X,y):     #this will splits the index
    X_train_SS = X.iloc[train]
    y_train_SS = y.iloc[train]
    X_test_SS = X.iloc[test]
    y_test_SS = y.iloc[test]
print(y_train_SS.value_counts())  
print(y_test_SS.value_counts())


# In[ ]:


plt.figure(figsize=(10,15))

plt.subplot('121')
plt.pie(y_train_SS.value_counts(),labels=y_train_SS.unique(),autopct = '%1.2f%%')
plt.title('Training Dataset')

plt.subplot('122')
plt.pie(y_test_SS.value_counts(),labels=y_test_SS.unique(),autopct = '%1.2f%%')
plt.title('Test Dataset')

plt.tight_layout()


# Thats Great! We have made a stratified sampling!
# 
# Now lets build two logistic regression models and compare them!

# # The Models

# In[ ]:


# Model 1 with stratified sample
model1=LogisticRegression()
model1.fit(X_train,y_train)
y_pred_m1=model1.predict(X_test)
acc_m1=accuracy_score(y_pred_m1,y_test)

print(acc_m1)


# In[ ]:


# Model 2 with stratified sample
model2=LogisticRegression()
model2.fit(X_train_SS,y_train_SS)
y_pred_m2=model1.predict(X_test_SS)
acc_m2=accuracy_score(y_pred_m2,y_test_SS)

print(acc_m2)


# In[ ]:


#visualizing result
plt.bar(['Random Split','Stratified split'],[acc_m1,acc_m2])
plt.title('Random vs Stratified split')


# Here we see a clear difference between the stratified and unstratified samples! The same Logistic regression has different accuracies! 
# 
# This is because the way they are tested are the way they are trained in case of stratified split. 

# 
# Now thats the end. We have made a stratified split! We found we could actually improve our accuracy!
# 
# Note : It is crucial as the data gets limited!
# 
# Feedbacks are welcomed!
# 
# If you like please **Upvote!**
