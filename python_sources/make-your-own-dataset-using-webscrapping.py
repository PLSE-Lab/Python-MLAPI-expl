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


# **Web scraping is the process of collecting information from the web pages.
# When large amount of data is to be collected manually, huge amount of time and efforts is required. This is intense if the data to be collected is from regularly updated web pages. Hence, it is always better to have a script, which automatically extracts the data from web page and stores the it in required format.
# All you needed is a python package to parse HTML data from the website, arrange it in a tabular form and do some data cleaning. There are dozens of packages available to do this job, but my story continues with BeautifulSoup.**

# lets import libs

# In[ ]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup


# In[ ]:


#lets get the website
#we will make corona virus dataset
website='https://www.worldometers.info/coronavirus/#countries'
website_url=requests.get(website).text
soup = BeautifulSoup(website_url,'html.parser')


# In order to make a Pandas DataFrame, we need to transform all the text data in object soup in tabular format. In Picture 1 above, I also marked 3 green rectangles. 

# now we will get all the data from table in our website

# In[ ]:


my_table = soup.find('tbody')


# **variable my_table will contain the table on the web page but still in HTML format. It contains the other two terms <tr> and <td> which points to individual rows and cells respectively. So i will transform this my_table into actual table containing the columns and values.
# **

# In[ ]:


table_data = []

for row in my_table.findAll('tr'):
    #row_data = []
    for cell in row.findAll('td'):
        row_data.append(cell.text)
        if(len(row_data) >0):
                   data_item = {"Country": row_data[0],
                             "TotalCases": row_data[1],
                             "NewCases": row_data[2],
                             "TotalDeaths": row_data[3],
                             "NewDeaths": row_data[4],
                             "TotalRecovered": row_data[5],
                             "ActiveCases": row_data[6],
                             "CriticalCases": row_data[7],
                             "Totcase1M": row_data[8],
                             "Totdeath1M": row_data[9],
                             "TotalTests": row_data[10],
                             "Tot_1M": row_data[11]}
    table_data.append(data_item)


# In[ ]:


for cell in row.findAll('td'):
        row_data.append(cell.text)
len(row_data)


# Now you get the table_data which is actually a table with columns and values. This table will be now transformed into a pandas DataFrame df.

# In[ ]:


df=pd.DataFrame(table_data)
df


# export df as csv

# In[ ]:


df.to_csv('Covid19_data.csv', index=True)


# Note: This dataset needs to be cleaned before uses because this is raw data with unwanted keywords.

# In[ ]:




