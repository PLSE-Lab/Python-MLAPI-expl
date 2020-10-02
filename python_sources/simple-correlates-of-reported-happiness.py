#!/usr/bin/env python
# coding: utf-8

# **National Correlates of Reported Feelings of Happiness**
# 
# In this notebook, I will do a simple exploration of the correlations between responses to question A008 (Self-reported feelings of happiness) and the other survey response items in order to prompt other research questions on the feeling of happiness.  Of course, these are nothing but correlations at the national level, and could never be interpreted so strongly as to say, e.g. "Importance of family causes happiness."  Nor, in fact, can we even assume that the same correlations observed at the national level will hold for individuals observed within or across nations.  For example, we could imagine a nation-level correlation between levels of poverty and sickness and volunteer work, possibly because the volunteer work is done to address these problems.  Within individuals, however, we may find that the individuals engaging in volunteer work may not be the same ones experiencing poverty and sickness.

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
import string
from scipy.stats import linregress
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/WVS_per_Country.csv')
codes = pd.read_csv('../input/Codebook.csv')


# In[7]:


df.head()
df.describe()


# In[6]:


codes.head()


# Walk over all the survey response items,  calculating regressions for each of the response items using A008 as our dependent variable and store all results.  Not all survey items have enough data points to bother looking at.  Output some progress information as we go.

# In[11]:


def regplot(x_code, y_code, df):
    # make reg plot and label according to the codes in question
    data = df[~(getattr(df, x_code).isnull()|getattr(df, y_code).isnull())]
    y_label = codes[codes['VARIABLE']==y_code]['LABEL'].to_string()
    x_label = codes[codes['VARIABLE']==x_code]['LABEL'].to_string()
    ax = sns.regplot(x=x_code, y=y_code, data=data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    plt.title(label)
    plt.show()
    

ylabel = str(codes[codes['VARIABLE']=='A008']['LABEL'].to_string())
results = []
for col in list(df.columns.values[5:]):
    if col[0] not in string.ascii_letters:
        continue
    if col[0] in ['S', 'X']:
        continue
    if col.endswith('Sd'):
        continue
    try:
        label = str(codes[codes['VARIABLE']==col]['LABEL'].to_string())
        data = df[~(df.A008.isnull()|getattr(df, col).isnull())]
        n = data.shape[0]
        print("{} rows with overlapping results for {}.".format(n, label))
        if n < 70:
            continue
        regr = linregress(data['A008'], data[col])
        if regr.rvalue in [0,1]: 
            print("R of {} returned for {}.".format(regr.rvalue, label))
            continue
        row = dict(
            code=col,
            regression=regr,
            label=label,
            n=n)
        # regplot('A008', col, df)
        results.append(row)
    except TypeError:
        continue


# Now find all "significant" results, and look at the sharpest slopes in each direction.

# In[12]:


significant = [row for row in results if row['regression'].pvalue<0.05]
by_slope = sorted(significant, key=lambda x: x['regression'].slope, reverse=True)
top = by_slope[0:10]
bottom = by_slope[-10:]


# In[13]:


for row in top:
    print(row)
    regplot('A008', row['code'], data)


# In[14]:


for row in bottom:
    print(row)
    regplot('A008', row['code'], data)


# What is odd is that one of the strongest, most visually striking correlations in the dataset is a negative correlation between reported "Feelings of happiness" and "Satisfaction with your life."  This is an intuitively unexpected result and seems to call for further investigation.

# In[ ]:




