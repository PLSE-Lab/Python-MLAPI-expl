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


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data=pd.read_csv("../input/Market_Basket_Optimisation.csv")


# In[ ]:


get_ipython().system('pip install apyori')


# In[ ]:


from apyori import apriori 


# In[ ]:


data.head()


# In[ ]:


data.fillna(0,inplace=True)


# In[ ]:


data.head()


# In[ ]:


transactions = []
for i in range(0,len(data)):
    transactions.append([str(data.values[i,j]) for j in range(0,20) if str(data.values[i,j])!='0'])


# In[ ]:


rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)


# In[ ]:


result=list(rules)


# In[ ]:


result


# In[ ]:




