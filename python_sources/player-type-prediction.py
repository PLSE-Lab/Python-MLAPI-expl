#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:



dataset = pd.read_csv('/kaggle/input/fifa-20-latest-player-database/players.csv')


# In[ ]:


dataset.head()


# our target is to predict which position is best for an individual player based on his stats

# In[ ]:


dataset.shape


# FIRST OF ALL SOME DATA ENGINEERING AND ANALYSIS

# In[ ]:


dataset['Position'].value_counts()


# WE WOULD NEED TO DROP SOME DATA COLUMS WHICH ARE OF NO USE

# In[ ]:


dataset = dataset.drop(['Futhead Card Link','Name','Club','League'],axis=1)


# In[ ]:


dataset.head()


# now we would need to convert strings into numerical values

# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset['Position'] = labelencoder.fit_transform(dataset['Position'])
dataset['Work Rates'] = labelencoder.fit_transform(dataset['Work Rates'])
dataset['Strong Foot'] = labelencoder.fit_transform(dataset['Strong Foot'])


# In[ ]:


dataset.head()


# In[ ]:


x = dataset[['Pace','Shooting','Passing','Dribbling','Defense','Physical','Card Rating','Weak Foot','Skill Moves','Work Rates','Strong Foot']]
y = dataset['Position']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train,y_train)


# In[ ]:


log.score(x_test,y_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
grad = GradientBoostingClassifier()
grad.fit(x_train,y_train)


# In[ ]:


grad.score(x_test,y_test)


# In[ ]:


test = [[76,78,45,67,58,68,54,4,4,3,0]]
grad.predict(test)


# In[ ]:




