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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


chocolate= pd.read_csv('../input/flavors_of_cacao.csv')
chocolate.head()


# In[ ]:


chocolate.isnull().sum()


# In[ ]:


chocolate.dtypes


# In[ ]:


# convert string to float
chocolate['Cocoa\nPercent']=chocolate['Cocoa\nPercent'].apply(lambda row: row[:-1]).astype('float')


# In[ ]:


sns.heatmap(chocolate.corr())


# In[ ]:


chocolate_null=chocolate[pd.isnull(chocolate['Broad Bean\nOrigin'])]


# In[ ]:


# fix column names
chocolate.columns =chocolate.columns.str.replace('\n', ' ').str.replace('\xa0', '')


# In[ ]:


chocolate.columns.tolist()


# In[ ]:


sns.countplot(x = 'Rating', data=chocolate)


# In[ ]:


rating_median_origin = chocolate.groupby(["Broad Bean Origin"])['Rating'].median()


# In[ ]:


# countries produce chocolate with best ratings
median_desc=rating_median_origin.sort_values(ascending=False)
top_10=median_desc[:10]
top_10.head(10)


# In[ ]:


top_10.plot('barh')


# In[ ]:


chocolate["Cocoa Percent"].describe()


# In[ ]:


rating_median_com= chocolate.groupby(["Company (Maker-if known)"])['Rating'].median()


# In[ ]:


# best 10 companies
median_desc_com=rating_median_com.sort_values(ascending=False)
best_10=median_desc_com[:10]
best_10.head(10)


# In[ ]:


best_10.plot('barh')


# In[ ]:


sns.kdeplot(chocolate["Rating"],color="coral", shade=True)
# mode is around 3.75, it's more valueable to look at the origins/companies with rating of 4.0


# In[ ]:


sns.kdeplot(chocolate["Cocoa Percent"],color="seagreen", shade=True)


# In[ ]:


from ggplot import*
ggplot(chocolate,aes(x ="Cocoa Percent", y ="Rating"))+    geom_point(size=5,color="blue",shape=3)+    facet_wrap("Company Location")+    xlab("Company Location")+ylab("Ratings")+ggtitle("Ratings over Company Location")


# In[ ]:


# split dataset
# split data use train_test_split
# X as row, y as column with test.data contains 30% (can change #) data from all dataset
from sklearn.model_selection import train_test_split
X, y =chocolate.iloc[:, 1:5].values,chocolate.iloc[:, 6].values
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=0)


# In[ ]:


print (X_train.shape)


# In[ ]:


print (X_test.shape)


# In[ ]:


print (y_train.shape)


# In[ ]:


print (y_test.shape)

