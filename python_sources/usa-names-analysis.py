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


import os
import bq_helper
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")
query = """SELECT year, gender, name, sum(number) as count FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
data = usa_names.query_to_pandas_safe(query)
data.to_csv("usa_names_data.csv")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


sns.countplot(data['gender'])


# In[ ]:


sns.distplot(data['year'], bins = 10, kde = True)


# In[ ]:


yea = data.groupby('year')['year'].count()
yea


# In[ ]:


sns.lineplot(x= yea.index, y= yea.values)


# In[ ]:


data['name'].value_counts().head()


# In[ ]:


kag=data.groupby('year')['gender'].value_counts()
kag


# In[ ]:


sns.countplot(kag)

