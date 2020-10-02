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


# This creates a pandas dataframe and assigns it to the titanic variable.
titanic_test = pd.read_csv("../input/titanic/test.csv")
titanic= pd.read_csv("../input/titanic/train.csv")
# Print the first 5 rows of the dataframe.
titanic.head()


# In[ ]:


titanic_test.head().T
#note their is no Survived column here which is our target varible we are trying to predict


# In[ ]:


#info method provides information about dataset like 
#total values in each column, null/not null, datatype, memory occupied etc
titanic.info()


# In[ ]:


titanic.describe()


# In[ ]:


#lets see if there are any more columns with missing values 
null_columns=titanic.columns[titanic.isnull().any()]
titanic.isnull().sum()


# In[ ]:


#how about test set??
titanic_test.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

labels = []
counts =[]
for col in null_columns:
    labels.append(col)
    counts.append(titanic[col].isnull().sum())

ind = np.arange(len(labels))

fig,ax = plt.subplots(figsize=(10,2))
ax.barh(ind,counts,color='purple')
ax.set_yticks(ind)
ax.set_yticklabels(labels)
ax.set_xlabel('Missing values')
ax.set_ylabel('Columns')
ax.set_title('Variables with missing values')


# In[ ]:


titanic.hist(bins=10,figsize=(9,7),grid=False);


# In[ ]:


g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple");


# In[ ]:


g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();


# In[ ]:




