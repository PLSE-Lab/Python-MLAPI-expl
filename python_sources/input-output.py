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


#READ CSV
pd.read_csv('example.csv')
df.to_csv('My_Output')
pd.read_csv('My_Output',index=0)


# In[ ]:


READ EXCEL
pd.read_excel('Excel_Sample.xlsx', sheetname = 'sheet1')
df.to_excel('Excel_Sample.xlsx', sheet_name = 'NewSheet')


# In[ ]:


READ HTML
data = pd.read_html('....')


# In[ ]:


READ SQL
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table', engine)
sqldf = df.read_sql('my_table', con = engine)
sqldf


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




