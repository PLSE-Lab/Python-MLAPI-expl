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


#Read the File
chocolate = pd.read_csv("/kaggle/input/chocolate-bar-ratings/flavors_of_cacao.csv")


# In[ ]:


chocolate.head()


# In[ ]:


#Finding the null values 
chocolate.isnull().sum()


# In[ ]:


#Changing the column names
chocolate.columns=['Company','Bean_Orgin','REF','Review_Date','Cocoa_Percent','Company_Location','Rating','Bean_Type','Bean_Origin']


# In[ ]:


chocolate.info()


# Since it has very less amount of null values. So no need to worry about fill null values. Just move ahead further analysis.

# In[ ]:


#summery of the data
chocolate.describe(include='all')


# BY summerizing the data, Mostly the companies are using 70% cocoa powder to prepare chocolate bars. Average chocolate bar rating is 3.25. Venezuela is best Bean_Origin.The values are calculated from the given dataset and total data is 1795.

# In[ ]:


#Question 1: Where are the best cacao beans grown?
chocolate['Bean_Origin'].value_counts().head(10)


# In[ ]:


#Question:2 Which countries produces the high_rated bars?
chocolate1 = chocolate.groupby(['Bean_Origin'])['Rating'].max()
chocolate1.sort_values(ascending = False).head(20)


# In[ ]:


#Question:3 What is the realationship between cacao solids percentage and rating?
chocolate.groupby(['Cocoa_Percent'])['Rating'].max().sort_values(ascending = False).head(20)


# In[ ]:


#Question 4: Countries with highest Vendors
chocolate['Company_Location'].value_counts().head(20)


# According to analysis, This dataset answered 4 types of questions. I drawn the graph using google data studio. I didnt shown up along with file. My point of view Data analysis is not story telling it helps to find the answers. Thank you, I hope it helps to all those who are begineers. 
