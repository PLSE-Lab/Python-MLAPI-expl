#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import locale
from locale import atof
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# First, let us read the data from the file.

# In[ ]:


data= pd.read_csv("../input/list-of-countries-by-number-of-internet-users/List of Countries by number of Internet Users - Sheet1.csv")
data.describe()


# We do not learn much more than the fact that we have 215 rows in this table. Now let us inspect the composition of the data in itself, as well as see some part of the data frame.

# In[ ]:



data.info()


# In[ ]:


data.head(10)


# As we can see, there is a slight issue. The object data type of Population and Internet Users will not allow us to perform numeric operations on the dataframe. We need to make some small changes before we can continue.
# We first convert the object type to string, then replace the comma and then convert it into integer form.

# In[ ]:


data['Population']=data['Population'].str.replace(',', '').astype(int)
data['Internet Users']=data['Internet Users'].str.replace(',', '').astype(int)
data.info()


# Now we can simply find the population of each country with no access to find the countries with most potential to expand into.

# In[ ]:


data['Users Without Internet']= data['Population'] - data['Internet Users']
data.sort_values("Percentage", axis = 0, ascending = True, 
                 inplace = True, na_position ='last') 

data.head(10)


# As we can see these are the countries, whose large section of population is not logged onto the Internet. But the population size affects the total number of people that can be reached out to. So let us see in terms of Users Without Internet.

# In[ ]:



data.sort_values("Users Without Internet", axis = 0, ascending = False, 
                 inplace = True, na_position ='last') 

data.head(10)


# In[ ]:


bar_chart= data.head(10).plot.bar(x='Country or Area',y='Users Without Internet',subplots=True)


# As we can see, these are the countries where there is more Internet penetration to be reached and provide a large opportunity for ISPs of these countries to expand aggressively.
# India and China are expected to be in the running due to their large population size. It is fairly clear that the countries with the highest population overall tend to have more scope for expansion overall. 
# 
# DRC is the interesting outlier here with a population rank of 66 but with only 8 percent of their population connected to the Internet, they have more scope in terms of numbers as well as getting more population to the Internet. Other countries in the top ten have atleast 18 percentage of people using the Internet as well.

# In[ ]:




