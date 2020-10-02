#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


# load the data into dataframe
cereal_df = pd.read_csv("../input/cereal.csv")
cereal_df.head()


# In[3]:


# Day 2

# import matplotlib, a visualization library for pandas/python
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# checking the dataset to determine which column are numeric
cereal_df.describe()


# In[5]:


# pick a column with numeric variables
calories = cereal_df['calories']
protein = cereal_df['protein']
potass = cereal_df['potass']

# plot a histogram of the column
plt.hist(calories)
plt.title("Calories in Cereals")


# In[6]:


# Day 3
# import ttest_ind
from scipy.stats import ttest_ind


# In[7]:


# perform ttest on calories and protein
# perform ttest on column calories
ttest_ind(calories, protein, equal_var=False)


# In[8]:


# plot a histogram of the column
plt.hist(protein)
plt.title("Protein in Cereals")


# In[9]:


# perform ttest on calories and potassium
# perform ttest on column calories
ttest_ind(calories, potass, equal_var=False)


# In[10]:


#  plot a histogram of the column
plt.hist(potass)
plt.title("Potassium in Cereals")


# In[11]:


# day 4
# categorical variable
mfr = cereal_df['mfr']
cereal_type = cereal_df['type']


# In[16]:


# get the value counts of manufacturer
mfrFreqTable = mfr.value_counts()
# get the list of manufacturer
list(mfrFreqTable.index)


# In[17]:


# plot manufacturer with matplotlib
import matplotlib.pyplot as plt
plt.bar(mfrFreqTable.index, mfrFreqTable.values)
plt.show()


# In[18]:


# alternatively, use seaborn
sns.countplot(cereal_df['mfr'])


# In[20]:


# do the same for cereal type
typeFreqTable = cereal_type.value_counts()
typeFreqTable


# In[21]:


plt.bar(typeFreqTable.index, typeFreqTable.values)


# In[22]:


# using seaborn
sns.countplot(cereal_type)


# In[ ]:




