#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

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


import matplotlib.pyplot as plt 
import seaborn as sns


# # Importing Data

# In[ ]:


df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# <font size=3 color='green'> Here there are 271 missing values in Year column. And we are not able to handle the missing year values. So, we simply drop it. </font>

# In[ ]:


df.drop(['Year'], axis=1, inplace=True)


# In[ ]:


df['Publisher'].replace(np.nan, df['Publisher'].mode()[0], inplace=True)


# In[ ]:


df.Publisher.isna().sum()


# ## Handling Categorical Features

# In[ ]:


cate_feat = [col for col in df.columns if df[col].dtypes == 'O']


# In[ ]:


cate_feat


# In[ ]:


cate_unique = list(map(lambda x: df[x].nunique(), cate_feat))


# In[ ]:


l = list(zip(cate_feat, cate_unique))


# In[ ]:


l


# <font size=3 color='green'> There are so many labels for each category so we cannot apply OneHotEncoding. But we can use LabelEncoding. </font>

# In[ ]:


df.head()


# # Label Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# In[ ]:


df1 = df.copy()


# In[ ]:


encoder = LabelEncoder()


# In[ ]:


df1.head()


# <font size=3 color='blue'> We are droping Name Column as its has very large number of labels. </font>

# In[ ]:


df1['Platform'] = encoder.fit_transform(df1[cate_feat[1]]) 
df1['Genre'] = encoder.fit_transform(df1[cate_feat[2]])


# In[ ]:


df1['Publisher'] = df1['Publisher'].replace('<','', inplace=True)


# In[ ]:


df1['Publisher'] = encoder.fit_transform(df1[cate_feat[3]])


# In[ ]:


X = df1.drop(['Global_Sales', 'Name'], axis=1)
y = df1['Global_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

tree = DecisionTreeRegressor(random_state=1)
tree.fit(X_train, y_train)


# In[ ]:


pred = tree.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print("\033[32mMean Absolute Error: {}\033[00m" .format(mae)) 


# In[ ]:




