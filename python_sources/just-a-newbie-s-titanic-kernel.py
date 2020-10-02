#!/usr/bin/env python
# coding: utf-8

# # Just a Newbie's Titanic Kernel
# 
# This kernel is my first kaggle kernel in my life.
# 
# I referred [EDA To Prediction (DieTanic)](https://www.kaggle.com/ash316/eda-to-prediction-dietanic) to study technics of Data Processing.
# 
# Special Thanks for @ash316 for uploading valuable kernel.

# ## I. Importing Libraries and Data
# Let's import some libraries and Titanic dataset.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
print('I have: ' + str(os.listdir("../input")))


# ## II. Checking some contents of Dataset and Dataset Integrity
# We have to check the contents and Dataset Integrity. The dataset might be damaged or omitted some data.

# In[ ]:


data = pd.read_csv('../input/train.csv')

data.head()


# Contents look like OK. Then check the Dataset Integrity.

# In[ ]:


data.isnull().sum()


# Null values in **Age, Cabin, Embarked**. Maybe a data loss.

# ## III. Survived Count
# Then, Let us draw the graph of Survived Count using matplotlib.

# In[ ]:


f, ax = plt.subplots()

sns.countplot('Survived', data=data, ax=ax)

plt.show()


# Let us check in Pie Graph:

# In[ ]:


f, ax = plt.subplots(figsize=(5, 5.5))

data['Survived'].value_counts().plot.pie(explode=[0,0],autopct='%1.1f%%',ax=ax,shadow=True)
ax.set_title('Survived')
ax.set_ylabel('')

plt.show()


# Categorical Feature with Sex:

# In[ ]:


data.groupby(['Sex', 'Survived'])['Survived'].count()


# In[ ]:


f, ax = plt.subplots()
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax)
ax.set_title('The Relationship between Survived and Sex')
plt.show()


# In[ ]:


f, ax = plt.subplots()
sns.countplot('Sex', hue='Survived', data=data, ax=ax)
ax.set_title('Survived and Dead with Sex')
plt.show()

