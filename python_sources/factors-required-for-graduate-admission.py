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
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


df = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values


# In[ ]:


print('check sample \n',df.sample(2))


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


plt.hist(df['GRE Score'])
plt.hist(df['TOEFL Score'])
plt.hist(df['CGPA'])
plt.hist(df['SOP'])


# In[ ]:


#checking out coorealtion 
df.corr()


# In[ ]:


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# In[ ]:



df_sort = df.sort_values(by=df.columns[-1],ascending=False)
df_sort.head(5)

