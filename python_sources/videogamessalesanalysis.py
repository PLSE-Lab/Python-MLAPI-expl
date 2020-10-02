#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
print(data)


# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


print(data['EU_Sales'].value_counts(dropna =False))


# In[ ]:


data.describe()


# In[ ]:


data.boxplot(column='NA_Sales',by='Year')


# In[ ]:


data_new = data.head()
data_new


# In[ ]:


melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['NA_Sales','EU_Sales'])
melted


# In[ ]:


melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Other_Sales','Global_Sales'])
melted


# In[ ]:


melted.pivot(index = 'Name', columns = 'variable',values='value')


# In[ ]:


data1 = data.head(10)
data2= data.tail(10)
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)
conc_data_row


# In[ ]:


data1 = data['Name'].head()
data2 = data['Global_Sales'].head()
new_data_conc = pd.concat([data1,data2],axis =1)
new_data_conc


# In[ ]:


data.dtypes


# In[ ]:


data['Genre'] = data['Genre'].astype('category')


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data["Other_Sales"].value_counts(dropna =False)


# In[ ]:


data["Global_Sales"].value_counts(dropna =False)

