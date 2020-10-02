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
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")


# In[ ]:


pd.set_option("Display.max_columns",100)


# In[ ]:


df.head()


# In[ ]:


#percentage missing values columnwise
print(df.shape)
(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100


# In[ ]:


df.info()


# In[ ]:


state_accident_count=df.groupby(['State'],as_index=False)['ID'].count().sort_values(by = "ID",ascending=False)


# In[ ]:


state_accident_count.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


f, ax = plt.subplots(figsize=(8, 18))
ax = sns.barplot(x="ID", y="State",  data=state_accident_count)


# In[ ]:


f, ax = plt.subplots(figsize=(18, 8))
ax = sns.lineplot(x="State", y="ID", data=state_accident_count)
ax.set(ylabel='Total no of accidents')
plt.show()


# In[ ]:




