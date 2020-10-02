#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Download the dataset

dataset = pd.read_csv("../input/heart.csv")


# In[ ]:


dataset.head()


# In[ ]:


df = dataset.drop(['cp', 'trestbps', 'restecg', 'thalach', 'exang', 'ca', 'thal'], axis=1)
df.head()


# In[ ]:


df.info()


# In[ ]:


sns.jointplot(data=df, x='age', y='chol')


# In[ ]:


sns.lmplot(data=df, x='age', y='chol', hue='sex', palette='Set1')
plt.title('Analysis of Heart Disease Based on Age, Gender, and Cholesterol Level')


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr(), cmap="viridis")
plt.tight_layout()


# In[ ]:




