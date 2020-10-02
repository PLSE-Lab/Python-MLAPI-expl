#!/usr/bin/env python
# coding: utf-8

# # Suicide Rates dataset from 1985-2016
# 
# Welcome to my notebook.
# 
# ## **CN Suicides:**
# This data analysis covers a dataset about Suicide Rates from 1985-2016. Because this is a delicate topic I advice you to be careful.
# If you feel depressed or experience suicidal thoughts please talk to someone professional about this. 
# Or call a help line in your country: [Get help here](http://ibpf.org/resource/list-international-suicide-hotlines) 
# Please take care of yourself!
# 
# This notebook is a hobby project a **WORK IN PROGRESS**

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


# # Extra packages I used:

# In[ ]:


import seaborn as sns # makes nicer plots (personal taste)
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from bokeh.plotting import figure
from bokeh.io import output_file, show


# # Loading the Dataset
# 
# **First we are going to load the Dataset: **

# In[ ]:


df = pd.read_csv(os.path.join(dirname, filename))
df.head()


# # Investigation of the Dataset

# In[ ]:


df.info()


# **We can see that there are a lot of NAN values for the 'HDI for year'- column. 
# But over all the data in all of the columns seems to be complete.**
# 
# ### We start with the EDA analysis of the Dataset:
# The first question was: ,,How does the worldwide suicide rates change over time."

# In[ ]:


plt.figure(figsize = (25,10))
trace1 = go.Bar(
                x = df.year,
                y = df.suicides_no,
                name = "year",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df.country)
data = [trace1]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# We can see throuout the years that suicides increase in the year 1989 and increase each untill 1996. But since 2009 we see an overall decline in total suicides.
# From 1989 on the largest block is from the Russian Federation with more than 22k recorded suicides in 1994 shortly after the breakdown of the Sowjet Union in 1990. In the recent years (2010 - 2015) the United States show the hightest number of suicides with ~20k in 2015 (11k+9k).
# 
# Note: you can see the different countries when you hove over a scpecific block.

# In[ ]:


plt.figure(figsize = (20,50))
sns.barplot(x = 'suicides/100k pop', y = 'country', data = df, ci = None, hue = 'year')
plt.title('Total number of suicides sorted by male & female')
plt.xlabel(' ')
plt.ylabel('number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# # Men or women?

# In[ ]:


df2 = df.groupby(['sex']).sum()
df2 = df2.reset_index()
df2.head()


# In[ ]:


sns.catplot(x = 'sex', y = 'suicides_no', data = df2, kind ='bar', height = 8)
plt.title('Total number of suicides sorted by male & female')
plt.xlabel(' ')
plt.ylabel('number')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Throughout the years there are 3-4 times more men comitting suicide than women. With an increase in suicides again in 2015 after a steady decline over the 90s. 
# But is this visible as well when we split up the data by countries?

# In[ ]:


plt.figure(figsize = (20,30))
sns.barplot(x = 'suicides_no' , y = 'country', data = df, ci = None, hue = 'sex')
plt.title('Total number of suicides sorted by male & female')
plt.xlabel('total number')
plt.ylabel('countries')
plt.show()


# In the plot above the gender distribution (only for female & male available) is shown. 
# Interestingly in **ALL** of the countries suicides amongst males is more commen than females. As we have seen before in the over all gender distribution. 
# Surprisingly the 3-4 times ratio of male to female suicides does not hold true for the Russian Federation where this ratio is more like 5-6 times. 

# # Inspecting the ,,age" column a little bit more deeply
# * as seen in the overview and in the description this column is not of the type int or float
#     + it's a range of ages

# In[ ]:


df3 = df.groupby(['age', 'year']).sum()
df3 = df3.reset_index()
df3.head()


# In[ ]:


plt.figure(figsize = (20,10))
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
sns.barplot(x = 'age',y = 'suicides_no', data = df3, hue = 'year', order = order)
plt.title('Total number of suicides sorted age')
plt.xlabel(' ')
plt.ylabel('number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# The majority of suicides is comitted from people with an age range from 35 - 54 years and the second highest rate is shown for the age 55 - 74 years. Because there is no ,,reason" given in this dataset it would be speculation why the age group from 35 - 74 is effected the most by suicides. 
# Sadly there is as well data available from child suicides (age 5 - 14). 

# In[ ]:


plt.figure(figsize = (15,55))
sns.barplot(x = 'suicides_no' , y= 'country', data = df, hue = 'age', ci = None) 
plt.title('Total number of suicides per Country & age group')
plt.xticks(rotation = 45)
plt.xlabel('total number')
plt.ylabel('County')
plt.show()


# If we take a closer look at which age group is comitting suicide the most in which country, we still see that mostly the age group 35 -74 years is represented (orange). In Mexico and Colombia however we see that the blue group (15 - 24 years) shows the highest numbers of suicides.  

# # Suicide rate for Germany
# 
# I am from Germany. So naturally I have an interest particular in German data.
# After extraction of the data related to Germany from the main dataset it is striking that the data collection starts with the year 1990. 
# Because Germany was reunited after in November 1989 there is no data on the suicide rates of "eastern" and "western" Germany respectively. 

# In[ ]:


df_ger = df.loc[df['country'] == 'Germany'] #slicing the DataFrame for all columns regarding ,,Germany"
df_ger.reset_index()


# First I dive in as usual and ask how did the number of suicides change over the years in Germany

# In[ ]:


plt.figure(figsize = (25,6))
sns.barplot(x = 'year', y = 'suicides_no', data = df_ger, ci = None)
plt.title('Total number of suicides in Germany between 1990 and 2015')
plt.xlabel('year')
plt.ylabel('total number')
plt.show()


plt.figure(figsize = (25,6))
sns.barplot(x = 'year', y = 'gdp_per_capita ($)', data = df_ger, ci = None)
plt.title('GDP per capita in Germany between 1990 and 2015 ')
plt.xlabel('year')
plt.ylabel('gdp per capita in $')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# The suicides in Germany decreased from 1990 until 2007 and increased a bit by 2010. Since then the number of suicides seems to be more or less stable. During the same time period however the GDP of Germany nearly doubled comparing the values from 2001 to 2014. 
# In general we can see hat during the increase in GDP from 1990 to 2007 the suicide rates are decreasing. In 2009 and 2010 the GDP was lower than in 2008 (following the financial crisis) and during the same time a slight increase in suicide numbers is visible as well. The suicie rate is reacting as well a litlle delayed to the GDP. An increase in GDP in 2011 did not result in lower suicide numbers in 2011 but in 2012 there is a decrease in suicides visible. 
# As conclusion the GDP and the suicide rates might be linked. 

# In[ ]:


plt.figure(figsize = (25,10))
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
sns.barplot(x = 'year', y = 'suicides_no', data = df_ger, ci = None, hue = 'age')
plt.title('Total number of suicides in Germany between 1990 and 2015 by age')
plt.xlabel('year')
plt.ylabel('total number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# When we look at the age distribution for suicied victims in Germany throughout the years we can see again that the age group from 35 - 54 and 55 - 74 is effected the most. The suicides amongst teenagers (15 - 24) does not seem to change much over time as the suicides amongst children (5 -14). 

# In[ ]:


df3_ger = df_ger.groupby(['age', 'year']).sum()
df3_ger = df3_ger.reset_index()
df3.head()


# In[ ]:


plt.figure(figsize = (25,10))
order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
sns.barplot(x = 'age', y = 'suicides_no', data = df3_ger, ci=None, hue = 'year', order = order) 
plt.title('Overview about suicide distribution in Germany between 1990 and 2015 grouped by age')
plt.xlabel('generation')
plt.ylabel('total number')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# In[ ]:


order = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '75+ years']
plt.figure(figsize = (25,6))
sns.barplot(x = 'year', y = 'suicides/100k pop', data = df_ger, ci=None, hue = 'age') 
plt.title('Suicides per 100.000 inhabitants in Germany between 1990 and 2015')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
plt.figure(figsize = (25,6))

sns.barplot(x = 'year', y = 'population', data = df_ger, ci=None, hue = 'age') 
plt.title('Population distribution in Germany between 1990 and 2015')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# If you look at the ratio of suicides per 100k people filtered by age group you see the peole +75 years old have the highest numbers of suicides. Here is to note that this depends strongly as well on the population during this time as we can se in the lower panel. Because a relatively low number of suicides shows more because the general there is only a low number of people with an age of 75+ in the overall population. Therefore suicide at an ,,old" age might be a serious problem in Germany.

# In[ ]:


sns.set(style="ticks")
sns.pairplot(df_ger)
plt.show()


# # Using Machine Learning to predict the suicide rate for 2016 and beyond
# 
# Naturally, we would like to ask a propper 

# In[ ]:


df_ger.head()


# In[ ]:


df_ger2 = df_ger


# In[ ]:


df_ger2.head()


# In[ ]:


df_ger2 = df.drop(['country', 'country-year','HDI for year'], axis = 1)
df_ger2[' gdp_for_year ($) '] = df_ger2[' gdp_for_year ($) '].str.replace(",","")
df_ger2 = pd.get_dummies(df_ger2, columns = ['sex', 'age', 'generation'])


# In[ ]:


df_ger2.head()


# In[ ]:


df_ger2.info()


# In[ ]:


from sklearn.model_selection import train_test_split
X = df_ger2.drop('suicides_no', axis = 1)
y = df_ger2['suicides_no']

X_train, X_test, y_train, y_test = train_test_split(X, y , random_state = 0, test_size = 0.25)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion = "entropy", n_estimators = 10)
model.fit(X_train, y_train)
print('RandomForrest: ' + str(model.score(X_test,y_test)))


# In[ ]:


from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_train, y_train)
print('Linear Regression: ' + str(model2.score(X_test,y_test)))


# In[ ]:




