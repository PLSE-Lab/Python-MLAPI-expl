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
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df = pd.read_csv("/kaggle/input/lung-cancer-dataset/lung_cancer_examples.csv")


# In[ ]:


df.columns


# In[ ]:


df.describe()


# # **Relationship with Age and Result based on people with Smoke Habbit**

# In[ ]:


sns.lineplot(x='Age',y='Result',hue='Smokes',data=df)


# # **Relationship with Age and Result based on people with Alkhol Habbit**

# In[ ]:


sns.lineplot(x='Age',y='Result',hue='Alkhol',data=df)


# # **Relationship between 'Age', 'Smokes', 'AreaQ', 'Alkhol', 'Result'**

# In[ ]:


sns.pairplot(data=df)


# # **Relationship between Age and Alkhol**

# In[ ]:


plt.figure(figsize=(50, 10))
g=sns.barplot(x='Age',y='Alkhol',data=df)


# # **Relationship between Age and Smokes habbit**

# In[ ]:


plt.figure(figsize=(50, 10))
g=sns.barplot(x='Age',y='Smokes',data=df)


# # **Relationship between Age And Result**

# In[ ]:


plt.figure(figsize=(50, 10))
g=sns.barplot(x='Age',y='Result',data=df)


# In[ ]:


sns.distplot(df.Result)


# # **df1 is the filtered data with Result '1' who are with cancer**

# In[ ]:


df1 = df[df.Result==1]


# # **Distribution of Age with Cancer positive result**

# In[ ]:


sns.distplot(df1.Age)


# In[ ]:


df2 = df[df.Result==0]


# # **Distribution of Age with Cancer negative result**

# In[ ]:


sns.distplot(df2.Age)


# # **Distribution of Age with Cancer positive result who smokes**

# In[ ]:


sns.distplot(df1.Smokes)


# # **Distribution of Age with Cancer positive result who consumes Alkhol**

# In[ ]:


sns.distplot(df1.Alkhol)


# In[ ]:


sns.scatterplot(x="Age", y="Result", data=df)


# In[ ]:


sns.scatterplot(x="Age", y="Result",
                      hue="Smokes",
                      data=df)


# In[ ]:


sns.scatterplot(x="Age", y="Result",
                      hue="Alkhol",
                      data=df)


# In[ ]:


sns.heatmap(data=df[['Age', 'Smokes', 'AreaQ', 'Alkhol', 'Result']],cmap="YlGnBu")


# In[ ]:



plt.figure(figsize=(50, 20))
sns.violinplot(x='Age',y='Result',data=df)


# In[ ]:




