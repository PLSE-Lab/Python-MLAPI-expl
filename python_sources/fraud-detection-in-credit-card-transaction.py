#!/usr/bin/env python
# coding: utf-8

# > The objective of this model is to understand how the various factors determine the chances of a fraudulent transaction.
#     We also intend to make our model efficient to detect all fraudulent transactions

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


# In[ ]:


data = pd.read_csv("../input/creditcard.csv")

data.head(10)


# In[ ]:


data.describe()


# As we can see that the dataset is clean and the range of values is not too high, therefore we can directly work on the data.
# 

# In[ ]:


data.columns


# checking for null or missing values in the dataset

# In[ ]:


data.isnull().sum()


# In[ ]:


"We will find the correlation among the various features and the class"

corr = data.corr()
corr


# Plotting the correlation matrix using heatmap for better understanding and visualisation

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data=corr,vmax=1)
plt.show()


# As visualised from the heatmap the features are uncorrelated or weakly correlated. Thus all features independently effect the target variable.
# 
# We know that the feature time in this dataset have no effect on the target variable since it is only a reference with respect to the time of first transaction in the dataset. It can therefore be dropped.

# 

# In[ ]:


data = data.drop(['Time'],axis=1)


# Let us understand the data distribution of features 
# 

# In[ ]:


#execution of this section will require some time since there are a lot of computations.
#execute this code if you want to have a better insight into the distribution of data
"""data['Class'].unique()
fig, ax = plt.subplots(5,6,sharex=False, sharey=False, figsize=(20,24))
i=1
for column in data.columns:
    if(column != 'Class'):
        data0 = data.loc[data['Class']==0,column]
        data1 = data.loc[data['Class']==1,column]
        plt.subplot(5,6,i)
        i = i + 1
        data0.plot(kind='density',label='class 0')
        data1.plot(kind='density',label='class 1')
        plt.ylabel(column)
        plt.legend()"""


# Next we need to separate the features from the target variable and then divide the data for training and testing.
# We shall be using logistic regression for our model since it works well for binary classification and is based on the independence of features. However,  other models could be selected based on the choice of the developer keeping in view the requirements of the model. 

# In[ ]:


target = data['Class']
features = data.drop(['Class'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(features, target, test_size = 0.25, random_state = 43)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(train_X,train_Y)
pred_class = model.predict(test_X)
pred_class


# We create a confusion matrix to check the accuracy of our model .False Negative cases which implies that the transaction is fraudulent but the model predicts it to be not, is highly undesirable for our study.

# In[ ]:


from sklearn.metrics import confusion_matrix
con_matrix = confusion_matrix(test_Y, pred_class)
tot = test_Y.count()
con_matrix/tot


# As the probability of FP and FN are neglible (of the order of e-04),our model can give accurate results with high probability.
# 
# If willing, one can try other models and find the confusion matrix and take a model more accurate than this.
# Hope, that you understand the idea and like it. 
