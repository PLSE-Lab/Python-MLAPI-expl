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


train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# We notice that "Province_state" has some missing values in the train dataset
# 

# In[ ]:


test.info()


# We notice that "Province_state" has some missing values in the test dataset too

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# # 1- Fatalties evolution :

# In[ ]:


Fatalities = train["Fatalities"].groupby(train['Date']).sum()
df= pd.DataFrame(Fatalities)
df.plot(kind='bar', color="blue")


# We can clearly see here that the fatalities are increasing exponentially

# # 2- Confirmed Cases evolution :

# In[ ]:


ConfirmedCases = train["ConfirmedCases"].groupby(train['Date']).sum()
df = pd.DataFrame(ConfirmedCases)
df.plot(kind='bar', color="red")
plt.show()


# The fatalities are increasing exponentially

# Since the Virus started in China, let's take a look at the evolution of confirmed cases and deaths in China 

# 

# # 3- Confirmed cases evolution in China

# In[ ]:


ConfirmedCases = train[train["Country_Region"]=='China']["ConfirmedCases"].groupby(train['Date']).sum()
df = pd.DataFrame(ConfirmedCases)
df.plot(kind='bar', color="red")
plt.show()


# # 4- Deaths evolution in China :

# In[ ]:


Fatalities = train[train["Country_Region"]=='China']["Fatalities"].groupby(train['Date']).sum()
df= pd.DataFrame(Fatalities)
df.plot(kind='bar', color="blue")


# 
# We can notice that the graph will eventually converge towards a limit. so there is a possibility that we can attack it with a regression equation that we have to find
