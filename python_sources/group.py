#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime,time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


consumer = pd.read_csv("../input/consumer/cacSurveys3May2016-withColumnNames.csv")


# In[ ]:


consumer.dropna()


# In[ ]:


# Drop a row by condition


# In[ ]:


new_data = consumer.sample(n= 10000)
new_data


# In[ ]:


new = new_data.drop(columns=['Column 13'])


# In[ ]:


final = new.rename(columns={"Column 1":"ID"})
final


# In[ ]:


date= pd.to_datetime(final['Date'],errors = 'coerce')
final['Date']=date 
#data1['Date'] = data1['Date'].astype('datetime64[ns]') 


# In[ ]:


from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
final['Parish'] = label_encoder.fit_transform(final['Parish']) 
final['Town'] = label_encoder.fit_transform(final['Town']) 
final['Shop Type'] = label_encoder.fit_transform(final['Shop Type']) 
final['Good or Service Name'] =label_encoder.fit_transform(final['Good or Service Name'])

  


# In[ ]:


final

