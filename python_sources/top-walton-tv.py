#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Read data:

# In[ ]:


df = pd.read_csv('../input/Date and model wise sale.csv')
print ("data shape : ",df.shape)
df.head()


# In[ ]:


#total count of models
total_models = len(df.Model.unique())
print ("total tv models count : ",total_models)
#group by model 
gp_model = df.groupby('Model')['Count'].apply(lambda x: x.sum())
gp_model.sort_values(axis=0, ascending=False, inplace=True)
#top 5 models
print ('Top 5 model :')
print (gp_model.head(5))
top = gp_model.head(5)
#not successful 5 models
print ('Not successful 5 model :')
print (gp_model.tail(5))

#figure model vs. sel
f, ax = plt.subplots(figsize=(12, 48))
ax=sns.barplot(gp_model, gp_model.index,orient='h')
ax.set(title='Models VS. Sell (from 2014 to 2016)',xlabel='Count of sell', ylabel='Model name')
plt.show()


# In[ ]:


top_df.tail()


# In[ ]:


for model in top.index:
    top_df = df[df.Model == model]
    plt.figure(figsize=(16,8))
    plt.plot(pd.to_datetime(top_df['Date']),top_df['Count'])
    plt.show()

