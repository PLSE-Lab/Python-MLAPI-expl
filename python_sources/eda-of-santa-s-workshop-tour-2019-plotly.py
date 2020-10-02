#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ## Ho! Ho ! Ho
# Santa has exciting news! For 100 days before Christmas, he opened up tours to his workshop. Because demand was so strong, and because Santa wanted to make things as fair as possible, he let each of the 5,000 families that will visit the workshop choose a list of dates they'd like to attend the workshop.
# 
# 

# # About the data
# Each family has listed their top 10 preferences for the dates they'd like to attend Santa's workshop tour. Dates are integer values representing the days before Christmas, e.g., the value 1 represents Dec 24, the value 2 represents Dec 23, etc. Each family also has a number of people attending, n_people.
# 
# 

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


# # Loading the data

# In[ ]:


path = '/kaggle/input/santa-workshop-tour-2019/family_data.csv'


# In[ ]:


family_data=pd.read_csv(path)
family_data.head()


# # Exploratory Data Analysis

# ## Importing the plotting libraries

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go


# ## Count Plot of the 1st choice of families

# In[ ]:


#lets have a look at the families on basis on the choices
plt.figure(figsize=(10,10))
sns.countplot(x='choice_0',data=family_data)


# ## Observation:
# 
# * We can see that about 370 families prefer the christmas Eve for the Santa's workshop tour.
# * Another interesting observation is that the is some `trend` and some `seasonality` that can be seen in `choice_0`.
# 

# # Trend and Seasonality

# ## Trend in data
# In the above figure, we plotted a count plot of the 1st choice of each family to get an idea of what are the general preferences for attending the workshop. We observed an increasing trend (increasing as the X axis show the number of days from Christmas eve i.e. Dec 24th) in the number of families to prefer to dates close to Christmas. 
# 
# Lets construct a timeseries by converting the choices to dates. We will plot a line graph to get a better understanding of the trend.

# In[ ]:


import gc
def choiceToDate(x,col):
    for i in x.family_id:
        x.iloc[i,1]=x['choice'].iloc[i] - pd.DateOffset(family_data[col].iloc[i])

choice_plot= go.Figure()
date = '2019-12-25'

for col in family_data.columns[1:-1]:
    choice = pd.DataFrame({'family_id':family_data.family_id,'choice': [date]*len(family_data)})
    choice['choice']=pd.to_datetime(choice.choice)
    choiceToDate(choice,col)
    choice = choice.set_index('choice').resample('D').count()
    choice.columns = ['family_count']
    choice_plot.add_trace(go.Scatter(x=choice.index,
                    y=choice.family_count,
                    mode='lines+markers',                                     
                    name=col))
    del(choice)
    gc.collect()
choice_plot.update_layout(title='Choice of Dates to attend Santa\'s Workshop ({}st Preference)'.format(int(col[-1])+1),
                           xaxis_title='Date',
                           yaxis_title='No of families')
choice_plot.show()


# ## Observation:
# * We can see a clear trend in the number of families. There is an increasing trend.
# * We can also see that there is a seasonality i.e. periodically for 3 days the number of families go up significantly.
# 
# ** We can make use of the dynamic features of plotly to turn on and off each of the choice lineplots.**
# 
# 
# **The seasonality observed should be most probably be due the occurance of Weekends as they are Three consecutive days. Working parents may prefer to attend the workshop during the weekend off.**

# Let us check the last feature : `n_people`. 
# It gives the number of people registered per family.

# In[ ]:


n_people_plot = go.Figure(go.Pie(values=family_data.n_people,labels=family_data.n_people))
n_people_plot.update_layout(title='Pie Chart of number of persons in families')
n_people_plot.show()


# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot('n_people',data=family_data)
plt.show()


# ## Observation:
# * From the above visualization, we can see that maximum number of families have **4** persons.
# * Very few families have more that **6** persons

# # Conclusion
# * We have had a look at all the features of the data.
# * We can now go with understanding the loss function and building a model to predict the results.
# 
# # To be continued 

# In[ ]:




