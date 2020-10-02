#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary liberaries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot') 
import os


# **Data Wrangling**

# In[ ]:


# load data to pandas dataframe
df = pd.read_csv('../input/who_suicide_statistics.csv')


# In[ ]:


# Review the data first 15 rows to understand it better
df.head(15)


# In[ ]:


# Drop rows with NaN values, this step is necessary as there are rows where suicide_no is empty 
# and same row has population number otherwise suicide/population will be biased.
df = df.dropna(axis =0)


# In[ ]:


df.head(15)


# **Analysis**

# In[ ]:


# First lets review the suicides with respect to age groups
df.groupby(by=['age'], as_index=False).sum().plot(x='age', y=['suicides_no', 'population'], kind='bar', secondary_y=['population'])


# Age group 35-55 years have highest suicides. However age group 75+ have highest suicides as compared to population

# In[ ]:


# Now lets see the suicide with respect to gender
df.groupby(by=['sex'], as_index=False).sum().plot(x='sex', y=['suicides_no', 'population'], kind='bar', secondary_y=['population'])


# Male have very high suicide rate as compared to female

# In[ ]:


# Plot the suicides and population with respect to years
df.groupby(by=['year'], as_index=False).sum().plot(x='year', y=['suicides_no', 'population'], kind='line', secondary_y=['population'])


# Suicides were highest during year 2000 (including few years before and after 2000) and afterwards suicides have lowered. Generally suicides are higher during recent years as compared to early years(1980-1990). Ignore the trend during last two years (2015/2016) as data is not complete during these years.
# Since population is also on rise, we must see the suicide per population. I will create a new column which will be suicides per million population. This will give us better picture.

# In[ ]:


dfyearsum = df.groupby(by=['year'], as_index=False).sum()
# calculating suicides per million population
dfyearsum['suicidesperpopulation'] = dfyearsum['suicides_no']*1000000/dfyearsum['population']

dfyearsum.plot(x='year', y=['suicides_no', 'suicidesperpopulation'], kind='line', secondary_y=['suicidesperpopulation'])


# Suicides per million population were highest during 1995 and afterwards there is decline is suicides.
# This suggests that suicides per population have reduced.   
# 

# In[ ]:


# Lets plot the suicides with respect to countaries
df_countrygroup = df.groupby(by=['country'], as_index=False).sum()
df_countrygroup['suicidespercapita'] = df_countrygroup['suicides_no']*1000000/ df_countrygroup['population']

plt.figure(figsize = (12,8))
plt.subplot(2,2,1)
df_countrygroup.sort_values(by=['suicides_no'], ascending=False).head(10).plot(x='country', y=['suicides_no'], kind='bar', title='TOP 10 country with suicides', ax=plt.gca())

plt.subplot(2,2,2)
df_countrygroup.sort_values(by=['suicidespercapita'], ascending=False).head(10).plot(x='country', y=['suicidespercapita'], kind='bar', title='TOP 10 country with suicides/Populattion', ax=plt.gca())

plt.subplot(2,2,3)
df_countrygroup.sort_values(by=['suicides_no'], ascending=True).head(10).plot(x='country', y=['suicides_no'], kind='bar', title='Bottom 10 country with suicides', ax=plt.gca())

plt.subplot(2,2,4)
df_countrygroup.sort_values(by=['suicidespercapita'], ascending=True).head(10).plot(x='country', y=['suicidespercapita'], kind='bar', title='Bottom 10 country with suicides/Populattion', ax=plt.gca())

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace = 0.2, hspace = 1.2)


# Above charts gives you conutry list with the highest/lowest suicides in total and also gives you country list with the highest and lowest suicides per million population.
# Russia and Ukrain appear in both lists of top countaries. Both of these countaries are geographically located nearby!! Something intresting!! [https://en.wikipedia.org/wiki/File:Russia_Ukraine_Locator.svg]
# 

# In[ ]:


# Lets see the yearly trends of Russia and Ukrain
df[df['country']=='Russian Federation'].groupby(by=['year'], as_index=False).sum().plot(x='year', y=['suicides_no', 'population'], kind='line', secondary_y=['population'], title= 'Russia')
df[df['country']=='Ukraine'].groupby(by=['year'], as_index=False).sum().plot(x='year', y=['suicides_no', 'population'], kind='line', secondary_y=['population'], title='Ukrain')


# Good to know that the suicide trends have lowered in both the countaries!!
# Both countaries had highest suicides during 1995-2000
# 
# I inquired aobut this fact and found a study which relate the high suicides in Russia and Ukraine to alcohal slaes. https://jacobspublishers.com/alcohol-consumption-and-suicide-trends-in-russia-belarus-and-ukraine/  However, alcohal cant fully explain this phenomena as the study.

# 
