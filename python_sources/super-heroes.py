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


# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
color = sns.color_palette()
import plotly.graph_objs as go
from plotly import tools
py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


info = pd.read_csv('../input/heroes_information.csv')
powers = pd.read_csv('../input/super_hero_powers.csv')


# In[3]:


info.head()


# In[4]:


#preprocessing
info.isnull().sum()


# In[5]:


info['Publisher'].value_counts()


# In[6]:


#treat missing values
info.replace(to_replace='-', value = 'Other',inplace=True)
info['Publisher'].fillna('Other',inplace=True)


# In[7]:


#drop column 'Unnamed'
info.drop('Unnamed: 0',axis=1,inplace=True)


# In[8]:


info.head()


# **  Which comic(Publisher) has the highest number of Superheroes? **

# In[9]:


info.groupby('Publisher')['name'].count().sort_values(ascending=False)


# In[10]:


sns.countplot(x='Gender', data = info)


# Most superheroes are Male

# In[11]:


sns.countplot(x='Alignment', data=info, hue='Gender')


# Most of the good superheros are male. However the number of bad superheroes and the number of good female superheroes are almost same. There are some superheroes that are neutral as well.

# In[12]:


#which publisher has more good v/s bad heroes
alignment_publisher = info[['Publisher', 'Alignment']]
alignment_publisher.head()


# In[13]:


fig, ax = plt.subplots()

fig.set_size_inches(18,10)
sns.countplot(x=alignment_publisher['Publisher'], data=alignment_publisher, hue='Alignment')


# as seen from figure, Marvel & DC comics have most number of superheroes so we will explore it (in the later section).

# In[28]:


powers.head()


# In[29]:


#convert True to 1 and false to 0
power = powers*1


# In[30]:


power.head()


# In[31]:


power.loc[:, 'no_of_powers'] = power.iloc[:, 1:].sum(axis=1)


# In[32]:


powerful_hero=power[['hero_names','no_of_powers']]


# In[33]:


powerful_hero = powerful_hero.sort_values(by='no_of_powers', ascending=False)
powerful_hero


# Spectre has highest number of powers whereas Amazo is on second place with 44 powers

# In[34]:


#plot top 10 superheroes
fig, ax = plt.subplots()

fig.set_size_inches(14,10.8)
sns.barplot(x=powerful_hero['hero_names'].head(10), y=powerful_hero['no_of_powers'].head(10), data=powerful_hero)


# None of superman/captain america are in top 10

# In[35]:


#height for top superheroes
newdata = info.merge(powerful_hero, how = 'inner', left_on='name', right_on='hero_names' )
newdata.drop('hero_names', axis=1, inplace=True)
newdata.head()


# There are some superheros who dont have any power. So to avoid those values, i have used inner join

# In[36]:


newdata['Height'].max()


# In[37]:


newdata[newdata['Height']==975]


# In[38]:


newdata['Height'].min()
newdata[newdata['Height']==-99]


# There are a lot of superheros that have minimum height . Thus, exploring the relationship between height and number of powers

# In[39]:


#height based on number of powers
height_power = newdata[['name','Height', 'no_of_powers']]
sorted_height = height_power.sort_values(by='no_of_powers', ascending=False)


# In[40]:


sorted_height.plot(x='no_of_powers', y='Height', kind='line')


# above graph proves that number of powers is not related to height. Superheroes with highest number of powers are smallest.

# In[41]:


newdata['Weight'].max()


# In[42]:


newdata[newdata['Weight']==900]


# In[43]:


#height based on number of powers
weight_power = newdata[['name','Weight', 'no_of_powers']]
sorted_weight = weight_power.sort_values(by='no_of_powers', ascending=False)
sorted_weight.head()


# In[44]:


sorted_weight.plot(x='no_of_powers', y='Weight', kind='line')


# Superhero with most powers has very low weight. But we can't relate weight directly to the number of powers, as superhero's in top 10 are a li'l on the heavier side as well.

# In[52]:


#explore marvel comics and dc comics data
marvel_data = newdata[newdata['Publisher']=='Marvel Comics']
marvel_data.head()


# In[88]:


dc_data = newdata[newdata['Publisher']=='DC Comics']
dc_data.head()


# In[73]:


#gender distribution withing Marvel Comics
gender_series = marvel_data['Gender'].value_counts()
gender = list(gender_series.index)
gender_percentage = list((gender_series/gender_series.sum())*100)

dc_gender_series = dc_data['Gender'].value_counts()
dc_gender = list(dc_gender_series.index)
dc_distribution = list((dc_gender_series/dc_gender_series.sum())*100)


# In[86]:


fig = {
    'data': [
        {
            'labels': gender,
            'values': gender_percentage,
            'type': 'pie',
            'name': 'marvel gender distribution',
            'domain': {'x': [0, .48],
                       'y': [0, .49]},
            'hoverinfo':'percent+name',
            'textinfo':'label'
            
        },
        {
            'labels': dc_gender,
            'values': dc_distribution,
            'type': 'pie',
            'name': 'DC gender distribution',
            'domain': {'x': [.52, 1],
                       'y': [0, .49]},
            'hoverinfo':'percent+name',
            'textinfo':'label'

        },
       
    ],
    'layout': {'title': 'Comics gender distribution',
               'showlegend': True}
}

py.iplot(fig, filename='pie_chart_subplots')


# Marvel comics has more number of superheroes overall.But (as seen from the charts), 
# DC comics has 71.9% male superheroes whereas Marvel comics only has 66.6%.
# whereas Marvel Comics has 29% females superheroes but DC comics only has 27.6% female superheroes. 

# In[53]:





# In[64]:


dc_data['Gender'].value_counts()


# In[87]:


gender_series


# In[ ]:




