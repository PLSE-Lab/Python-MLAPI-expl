#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import itertools
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('../input/uci-ionosphere/ionosphere_data_kaggle.csv')


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.countplot(x='feature1',data=df)
plt.show()


# In[ ]:


columns=df.columns[:8]
plt.subplots(figsize=(18,15))
length=len(columns)
for i,j in itertools.zip_longest(columns,range(length)):
    plt.subplot((length/2),3,j+1)
    plt.subplots_adjust(wspace=0.2,hspace=0.5)
    df[i].hist(bins=20,edgecolor='black')
    plt.title(i)
plt.show()


# In[ ]:


sns.distplot(df.feature3, kde=False, fit=stats.gamma);


# In[ ]:


with sns.axes_style("white"):
    sns.jointplot(x=df.feature2, y=df.feature3, kind="hex", color="k");


# In[ ]:


with sns.axes_style("white"):
    sns.jointplot(x=df.feature3, y=df.feature5, kind="hex", color="k");

