#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available pin the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


#  **Suicide has been a major concern across the world. This kernel will be analyzing the data from WHO to better understand the suicide problems by nation, time, sex and age group. In this way can we know the target while conducting suicide prevention in the future. **

# In[ ]:


# create the data frame called suicide with the csv file
# display the first few rows so we know what the data set looks like 
suicide = pd.read_csv('/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv')
suicide.head()


# Ar first, we want to organize the data to make it more practical for use by adding a column called suicide rate. Suicide rate can better reflect the severity of suicide problem for a particular demographic. 

# In[ ]:


# create a new column called suicide_rate
# calculated by the number of suicides divided by the population
suicide['suicide_rate'] = suicide.suicides_no/suicide.population 

# output a list of column names to check if we added the suicide_rate successfully 
suicide.columns.tolist()


# # Quick facts

# This section will look at the influence of sex, time and age on suicide rates.

# In[ ]:


# 1. group by sex
# 2. summarize the average for suicide rates
suicide.groupby('sex').suicide_rate.mean()


# In[ ]:


# 1. group by year
# 2. summarize the average
# 3. find the index of the year with the highest suicide rate
suicide.groupby('year').suicide_rate.mean().idxmax()


# In[ ]:


# 1. groupby age
# 2. summarize the average suicide rates
# 3. display the data in a descending order in the horizontal bar chart
suicide.groupby('age')    .suicide_rate.mean()    .sort_values().plot.barh()
    


# # Suicide among different nations

# In[ ]:


# check the data completedness for each nation 
suicide.groupby('country').size()


# Since numbers of rows are not the same among these countries, so some country has more available information than others. Plus, there are 141 countries in total, 

# We want to see which countries are facing a bigger issue of suicides.

# In[ ]:


# 1. group by country
# 2. summarize the mean of suicide rates
# 3. sort the values in a descending order 
# 4. display the top 10 countries with the higest suicide rates
suicide.groupby('country')    .suicide_rate.mean()    .sort_values(ascending = False)    .head(10)


# Geospatially, nine among the top ten nations are located in east Europe.

# Then, we will take a closer look at Hungary which has the highest suicide rate worldwide.

# In[ ]:


# create a new data frame called Hungary
# 1. set the index into country
# 2. select hungary
Hungary = suicide.set_index('country').loc['Hungary']
Hungary


# In order to figure out the reasons behind the high suicide rate, we want to see how age, year and sex have an impact here.

# In[ ]:


# 1. group by age, year and sex
# 2. summarize the mean of suicide rates
# 3. sort values
# 4. display the top 10 highest ones in a horizontal bar chart

Hungary.groupby(['age', 'year','sex'])    .suicide_rate.mean()    .sort_values()    .tail(10).plot.barh()


# This bar chart reveals an astonishing fact that the top 10 are all among the elder males over 75 years old.

# # **World trends of suicide across time.**

# In[ ]:


# 1. groupby by year to see the time trend
# 2. summarize the mean for suicide rate
# 3. plot it with a bar chart 
suicide.groupby('year')    .suicide_rate.mean()    .plot.bar()


# The bar chart illustrates that the world suicide rates increases until 1995 and then steadily decreases since then. Also, the world as a whole suffers high suicide rates from late 1980s to 1990s.

# # Conclusion

# 1. **Men** are three times more likely to commit suicide than women.
# 2. The world suicide rate hit the peak in **1995** and has been dropping ever since then.
# 3. The **elder generation** are the age group with the highest suicide rate, especially those who are 75+ years old.
# 4. Geographically speaking, **east Euporean countries** are at the greatest risk of suicide. 
# 5. **Hungary** has the world's highest suicide rate. And its top ten suicide rate comes from 75+ years old male during 1980s, which corresponds to the previous conclusions. 
# 6. Recommendations include to **pay more attention to the mental state of the elder males; the government especially in east Europe should work on public health care and pension to support the elder people. **

# In[ ]:




