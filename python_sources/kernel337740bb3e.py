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


traindata = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
testdata = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")


# In[ ]:


traindata.head(15)


# In[ ]:


testdata.head()


# In[ ]:


traindata.shape


# In[ ]:


testdata.shape


# In[ ]:


traindata.info()


# In[ ]:


traindata.describe()


# In[ ]:


traindata.isnull().sum()


# In[ ]:


import seaborn as sns
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.countplot(x="SmokingStatus",data = traindata)


# In[ ]:


sns.countplot(x="SmokingStatus",data = testdata)


# In[ ]:


sns.countplot(x="SmokingStatus",hue='Sex',data = traindata)


# In[ ]:


sns.countplot(x="SmokingStatus",hue='Sex',data = testdata)


# In[ ]:


sns.countplot(x="Age",hue='Sex',data = traindata)
plt.xticks(Rotation = 90)


# In[ ]:


sns.countplot(x = 'Age',data=traindata)


# In[ ]:


labels = traindata['SmokingStatus'].value_counts().index
values = traindata['SmokingStatus'].value_counts().values


# In[ ]:


plt.pie(values,labels = labels,autopct = '%0.1f%%')
plt.show()


# In[ ]:


labels1 = testdata['SmokingStatus'].value_counts().index
values1 = testdata['SmokingStatus'].value_counts().values


# In[ ]:


plt.pie(values1,labels = labels1,autopct = '%0.1f%%')
plt.show()


# In[ ]:


traindata.head(2)


# In[ ]:


X = traindata.drop(['Patient','Percent','Sex','SmokingStatus'],axis=1)
y = traindata['Percent']


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


# In[ ]:


cls = RandomForestRegressor()


# In[ ]:


cls.fit(X,y)


# In[ ]:


xtest = testdata.drop(['Patient','Sex','SmokingStatus','Age'],axis=1)


# In[ ]:


pred = cls.predict(xtest)


# In[ ]:


cls.score(X,y)


# In[ ]:


pred


# In[ ]:




