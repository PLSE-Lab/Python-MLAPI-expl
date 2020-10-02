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


# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns


# In[ ]:


loans = pd.read_csv('../input/loan_data.csv')


# In[ ]:


loans.head()


# In[ ]:


loans.info()


# In[ ]:


loans.describe()


# In[ ]:


#Exploratory data
plt.figure(figsize=(10,6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha = 0.5,color = 'blue',
     bins = 30,label = 'credit policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(alpha = 0.5,color = 'red',
     bins = 30,label = 'credit policy = 0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5,color = 'blue',
     bins = 30,label = 'not fully paid = 1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha = 0.5,color = 'red',
     bins = 30,label = 'not fully paid = 0')
plt.legend()
plt.xlabel('FICO')


# In[ ]:


#showing the counts of loans by purpose
plt.figure(figsize=(11,7))
sns.countplot(x = 'purpose',hue = 'not.fully.paid',data = loans,palette='Set1')


# In[ ]:


#the trend between FICO score and interest rate
sns.jointplot(x = 'fico',y = 'int.rate',data = loans,color = 'purple')


# In[ ]:


#the trend differed between not fully paid and credit policy
plt.figure(figsize = (11,7))
sns.lmplot(y = 'int.rate',x = 'fico',data = loans,hue = 'credit.policy',
           col = 'not.fully.paid',palette = 'Set2')


# In[ ]:


#Modelling
cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns = cat_feats,drop_first = True)


# In[ ]:


from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis = 1)
y = final_data['not.fully.paid']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 101)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


#Prediction and Evaluation

predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:




