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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# First importing seaborn library
import seaborn as sns


# In[ ]:


#Creating a data frame
DF = pd.read_csv("../input/911.csv")


# In[ ]:


# Head of data frame
DF.head()


# In[ ]:


#Info of data frame
DF.info()


# In[ ]:


# no. of unique title names
DF["title"].nunique()


# In[ ]:


# splitting of title
DF["SR"]=DF["title"].apply(lambda title : title.split(":")[0])


# In[ ]:


DF.head()


# In[ ]:


DF["SR"].value_counts().head(10)


# In[ ]:


# plotting a graph using seaborn
sns.countplot(x = "SR", data = DF)


# In[ ]:


# converting the format of timestamp
DF["timeStamp"] = pd.to_datetime(DF["timeStamp"])


# In[ ]:


DF.head()


# In[ ]:


DF["hours"] = DF["timeStamp"].apply(lambda time : time.hour)
DF["month"] = DF["timeStamp"].apply(lambda time : time.month)
DF["day"] = DF["timeStamp"].apply(lambda time : time.dayofweek)


# In[ ]:


sns.countplot(x="month", data = DF, hue = "SR")


# In[ ]:


byMonth = DF.groupby("month").count()
byMonth.head()


# In[ ]:


byday = DF.groupby("day").count()
byday.head()
byday["twp"].plot()


# In[ ]:


byMonth["twp"].plot()


# In[ ]:




