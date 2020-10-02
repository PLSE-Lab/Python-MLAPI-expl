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


import pandas as pd
import numpy as np
data=pd.read_csv("../input/athlete_events.csv")


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.describe(include="object")


# In[ ]:


#sorting the dataframe according to year
data.sort_values('Year', ascending=True,inplace=True)
data.head()


# In[ ]:


#filling missing nan of (Age,Height and Weight) with 0
data.update(data[['Age','Height','Weight']].fillna(0))
data.head()


# In[ ]:


#converting float to int
cols = ['Age', 'Height','Weight']
data[cols] = data[cols].apply(np.int64)
data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.shape


# In[ ]:


data.dtypes


# In[ ]:


data.Sex.value_counts(normalize=True)


# In[ ]:


data['Medal'].value_counts(dropna=False)


# In[ ]:


data['Sport'].nunique()


# In[ ]:


data.NOC.nunique()


# In[ ]:


#US has won more medals than any other country, followed by france
data.Team.value_counts().head()


# In[ ]:


pd.crosstab(data.Year,data.Medal,margins=True)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
fig=plt.figure(figsize=(25,10))
sns.countplot(x="Year",data=data,hue="Medal")


# In[ ]:


import seaborn as sns
sns.countplot(x="Season",data=data)


# In[ ]:


#most of the people are in the age group of 20 to 25
sns.distplot(data.Age)


# In[ ]:


#most of the people are in the height group of 170cm to 185cm
sns.distplot(data.Height)


# In[ ]:


#most of the people are in the height group of 60kg to 70kg
sns.distplot(data.Weight)


# In[ ]:


#filling medal nan with unknown
#data.Medal=data.Medal.fillna('unknown')
#data.head()

