#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataframe=pd.read_csv("../input/2017.csv")


# **-- Data preliminary review --**

# In[ ]:


dataframe.info()
#quick general information about data content


# In[ ]:


dataframe.dtypes
#Information data types in data


# In[ ]:


dataframe.columns
#Column in data contains header information


# ***-*- Top 10 country -*-***

# In[ ]:


dataframe.head(10)
#According to the survey, the 10 most happiest countries in the world


# **-- The last 10 countries --**

# In[ ]:


dataframe.tail(10)
#According to the survey, the world's 10 most unhappy countries


# **-- According to the survey, the world's most unhappy countries  2017--**

# In[ ]:


dataframe.loc[149 : 154 , "Country"]


# In[ ]:


dataframe.describe()


# **Countries with good financial status, non-corruption and high quality of life have higher levels of happiness.**

# In[ ]:


dataframe.loc[0:10,["Country","Happiness.Rank","Economy..GDP.per.Capita.","Health..Life.Expectancy.","Trust..Government.Corruption."]]


# **The level of happiness in countries with poor financial status, poor performance in corruption and low quality of life, or in countries that have lagged behind are very low.**

# In[ ]:


dataframe.loc[144:154,["Country","Happiness.Rank","Economy..GDP.per.Capita.","Health..Life.Expectancy.","Trust..Government.Corruption."]]


# In[ ]:


df=pd.read_csv("../input/2015.csv")


# According to the 2015 Yearbook, the happiest countries are ranked very likable.

# In[ ]:


df.head(10)


# As we shall see again, the most important thing in happiness is that we can observe economic condition and comfort in health. Another beauty is the level of freedom.

# In[ ]:


df.loc[0:9,["Country","Region","Economy (GDP per Capita)","Health (Life Expectancy)","Freedom"]]


# The unhappy region of the world, on the other hand, always appears to be the same region of Africa and its surroundings.
# The main reasons for this are wars, various epidemics, the weakness of the family structure and the oppressive management structure.

# In[ ]:


df.tail(10)


# In[ ]:


df=pd.read_csv("../input/2016.csv")


# It is seen that there is not much difference in the 2016 annual data and the countries that provide order continue this order in continuity.

# In[ ]:


df.head(10)


# In[ ]:


df["Region"].unique()


# In[ ]:


print(len(df["Region"].unique()),"unique Region")

