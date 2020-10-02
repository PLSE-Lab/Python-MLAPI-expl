#!/usr/bin/env python
# coding: utf-8

# Let us analyze the data of Brazil Forests and see what insights we can get.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Loading the data
dataframe = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding= 'ISO-8859-1')
dataframe.head(30)


# In[ ]:


month_list = dataframe['month'].unique().tolist()
month_list


# Looks like the months are in different language. We can either use label encoder for changing to numbers or replace them to English manually

# In[ ]:


month_english_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
for i in range(0, len(month_list)):
    dataframe['month'][dataframe['month'] == month_list[i]] = month_english_list[i]
dataframe.head(30)


# Its better (I feel) to change into number using label encoder

# In[ ]:


# checking missing values
dataframe.isnull().sum()


# In[ ]:


dataframe.describe()
# Looks like there are just two numerical variables


# In[ ]:


dataframe.dtypes


# In[ ]:


print(dataframe.groupby(['state']).count())
state_plot = sns.countplot(x = 'state', data =  dataframe)
state_plot.set_xticklabels(state_plot.get_xticklabels(), rotation = 70)


# 1. RIO recorded the highest record of forest fires
# 2. Padaiba and Mato Grosso recorded the second and third highest records of forest fires.

# In[ ]:


essential_data = dataframe.groupby(by = ['year','state','month']).sum().reset_index()
essential_data


# In[ ]:


from matplotlib.pyplot import MaxNLocator, FuncFormatter
plt.figure(figsize = (10, 5))
plot = sns.lineplot(data = essential_data, x = 'year', y = 'number', markers = True)
plot.xaxis.set_major_locator(plt.MaxNLocator(19))
plot.set_xlim(1998, 2017)


# 1. 2003 showed the highest number of forest fires.
# 2. 2008 and 2001 showed similar number of forest fires.
# 3. The rate of forest fires got increased during the years 1998 to 2003, 2008-2009, 2011-2012, 2013-2018 
# 4. The rate of forest fires got decreased during the years 2003 to 2008, 2009-2011, 2012-2013

# Shall we split the dataset into increasing and decreasing number of forest fires and try to find some inner meanings  from them ?

# In[ ]:


year_number_data = essential_data[['year', 'number']]
year_number_data[year_number_data['year'] == 1998]


# In[ ]:


# The rate of forest fires got increased during the years 1998 to 2003, 2008-2009, 2011-2012, 2013-2017 
# The rate of forest fires got decreased during the years 2003 to 2008, 2009-2011, 2012-2013
increasing_list = [1998, 1999, 2000, 2001, 2002, 2008, 2011, 2013, 2014, 2015]
decreasing_list = [2003, 2004, 2005, 2006, 2007, 2009, 2010, 2012, 2016]


# In[ ]:


increasing_dataframe = pd.DataFrame()
for i in increasing_list:
    df = year_number_data[year_number_data['year'] == i]
    increasing_dataframe = increasing_dataframe.append([df])
increasing_dataframe.head()


# In[ ]:


decreasing_dataframe = pd.DataFrame()
for i in decreasing_list:
    df1 = year_number_data[year_number_data['year'] == i]
    decreasing_dataframe = decreasing_dataframe.append([df1])
decreasing_dataframe.head()


# In[ ]:


plt.figure(figsize = (10, 5))
plot = sns.lineplot(data = increasing_dataframe,
                    x = 'year',
                    y = 'number',
                    lw = 1,
                    err_style="bars",
                    ci=100)
plot = sns.lineplot(data = decreasing_dataframe,
                    x = 'year',
                    y = 'number',
                    lw = 1,
                    err_style="bars",
                    ci=100)
plot.xaxis.set_major_locator(plt.MaxNLocator(19))
plot.set_xlim(1998, 2017)


# We splitted the years into increasing number of fires and decreasing number of fires. We can conclude that the number fires are getting increasing over the past decades (even if we consider the drop in number of fires for few years, it doesnt win over the raise in number of fires for the remaining years). There is a serious need to consider saving the brazilian forest. 
