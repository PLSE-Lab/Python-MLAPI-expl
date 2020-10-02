#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Importing the Dataset

# In[ ]:


insure = pd.read_csv('/kaggle/input/insurance/insurance.csv')


# Inspection of the Dataset

# In[ ]:


insure.head()


# In[ ]:


insure.mean()


# In[ ]:


insure.max()


# Exploratory Data Analysis

# In[ ]:


sns.catplot(x = 'smoker', kind='count',hue='sex',data=insure)


# In[ ]:


reg = sns.swarmplot(x='region', y='charges',hue='sex',data=insure)
reg.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# In[ ]:


smo = sns.swarmplot(x='smoker', y='charges',hue='sex',data=insure)
smo.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# In[ ]:


child = sns.swarmplot(x='children', y='charges',hue='sex',data=insure)
child.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)


# In[ ]:


gender = sns.swarmplot(x='sex', y='charges',data=insure)


# In[ ]:


sns.relplot(x='charges',y='bmi',data=insure,sizes=(40, 400))


# In[ ]:


sns.jointplot(x='age',y='charges',data = insure[(insure.smoker=='no')])


# In[ ]:


sns.jointplot(x='age',y='charges',data = insure[(insure.smoker=='yes')])


# In[ ]:


sns.lmplot(x="age", y="charges", hue="sex",
               height=5, data = insure[(insure.smoker=='yes')])


# In[ ]:


sns.lmplot(x="age", y="charges", hue="sex",
               height=5, data = insure[(insure.smoker=='no')])


# Data Preprocessing

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(insure.sex.drop_duplicates()) 
insure.sex = le.transform(insure.sex)

le.fit(insure.smoker.drop_duplicates()) 
insure.smoker = le.transform(insure.smoker)


# Linear Regression Model Training

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


x = insure.drop(['charges','region'], axis = 1)
y = insure.charges

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state = 0)
lr = LinearRegression()
lr.fit(x_train,y_train)


print(lr.score(x_test,y_test))


# In[ ]:


predictions = lr.predict(x_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:




