#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.******


# In[ ]:


data = pd.read_csv("../input/ArduinoSensorValues.csv")


# In[ ]:


data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1, data2],axis = 0, ignore_index = True)
conc_data_row


# In[ ]:


data1 = data["light_value"].head()
data2 = data["decibles"].head()
conc_data_col = pd.concat([data1, data2], axis = 1)
conc_data_col


# In[ ]:


data.dtypes


# In[ ]:


data["light_value"] = data["light_value"].astype("int")
data["decibles"] = data["decibles"].astype("int")


# In[ ]:


data.dtypes


# In[ ]:


data.info()


# In[ ]:


data["decibles"].value_counts(dropna = False)


# In[ ]:


data1 = data
data1["decibles"].dropna(inplace = True)


# In[ ]:


assert data["decibles"].notnull().all()


# In[ ]:


data["decibles"].fillna("empty",inplace=True)


# In[ ]:


assert data["decibles"].notnull().all()

