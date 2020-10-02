#!/usr/bin/env python
# coding: utf-8

# In this notebook, I am looking at synthetically generated financial fraud data. I want to develop a technique which is able to detect fraud transactions. I plan to achieve this by testing various machine learning models on this data. 
# 
# Let's start!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Load data into a dataframe and have a look at the data.

# In[ ]:


filename = '/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv'
data = pd.read_csv(filename)
data.head()


# There are 11 columns. Column 'isFraud' is our target column. Let us check data types of each column.

# In[ ]:


data.dtypes


# There are three columns with data type 'object'. Let's convert them to 'string'.

# Let's check if there are any null values in the dataset.

# In[ ]:


data.isnull().sum()


# No null values found! A rare dataset! Let's check size of the dataset.

# In[ ]:


data.size


# Almost 70M rows! A big dataset! Let's check if it's balanced or not. Meaning, check what percentage of transactions are fraudulant.

# In[ ]:


perFraud = (data[data['isFraud']==1].size/data.size)*100
print(perFraud)


# Only 0.13% of transactions are fraudulant. It is a highly imbalanced dataset. Hence, we have to be careful in reporting our results and accuracy might not be a good parameter to report for this classification problem. We will look at precision and recall.
# 
# Also, to improve results we might have to apply pre-processing techniques of undersampling the non-fraudulant transactions or oversampling of fradulant transactions while train-test split.
# 
# Let us first try a straight forward Decision Tree Classifier.

# In[ ]:


data.columns


# In[ ]:


predictors = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
              'newbalanceDest', 'isFlaggedFraud']

XX = data[predictors]
X = pd.get_dummies(XX)  # one-hot-encoding
X.describe()


# In[ ]:


y = data.isFraud


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(X,y,random_state=0)


# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)


# In[ ]:


pred_y = clf.predict(test_X)


# In[ ]:


print(confusion_matrix(test_y, pred_y))
print(classification_report(test_y, pred_y))


# Let's try scaling the predictors with Random Forest Classifier.

# In[ ]:


sc = StandardScaler()
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)


# In[ ]:


clf = RandomForestClassifier(n_estimators=20, random_state=0)
clf.fit(train_X, train_y)


# In[ ]:


pred_y = clf.predict(test_X)


# In[ ]:


print(confusion_matrix(test_y,pred_y))
print(classification_report(test_y,pred_y))
print(accuracy_score(test_y,pred_y))


# Between the two implemented models, for random forest we see that our recall decreases by 9% from the decision tree results i.e. we missed out on more fraudulant transactions. This is not good! 
# 
# Let's try to increase our recall by using over-sampling or under-sampling.

# In[ ]:




