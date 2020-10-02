#!/usr/bin/env python
# coding: utf-8

# DAY 1: 10 Plots and my inference from these plots

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


# In[ ]:


data_titanic=pd.read_csv("../input/titanic/train.csv")
data_titanic.head(10)


# In[ ]:


import plotly.express as px
fig = px.scatter(data_titanic, x='PassengerId', y='Survived', color='Pclass',size='Fare')
fig.show()


# 1-Inference:A big no of First class and who paid a lot survived....

# In[ ]:


fig = px.scatter(data_titanic, x='Age', y='Survived', color='Sex')
fig.show()


# 2-Inference:
# * A lot of old age men died and one particularly interesting most oldest man on the ship survived.
# * A lot of man died basically,most of them who could have been rich enough would have in there 30+
# * A lot of old women and childeren were saved due to priority for boat...

# In[ ]:


data_regression=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
data_regression.head()


# In[ ]:


fig = px.scatter(data_regression, x='OverallQual', y='SalePrice')
fig.show()


# 3-Inference:If you keep the quality high,It sells

# In[ ]:


fig = px.scatter(data_regression, x='Neighborhood', y='SalePrice',color='OverallQual')
fig.show()


# 4-Inference:
# * people pay higher for living in Northridge hights,Stony Brook,NorthRidge
# * Northridge heights,somerst,northridge,stony brook houses are highquality..

# In[ ]:


data_udemy=pd.read_csv('../input/udemy-courses/udemy_courses.csv')
data_udemy.head()


# In[ ]:


fig = px.scatter(data_udemy, x='subject', y='num_subscribers', size='num_reviews',color='level')
fig.show()


# 5-Inference
# * People Like to engage and learn more of web developement skills

# In[ ]:


data_campusrecruitment=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data_campusrecruitment.head()


# In[ ]:


fig = px.scatter(data_campusrecruitment, x='degree_p', y='salary',color='gender')
fig.show()


# 6-Inference
# * Highest scorer and Lowest scorer in their degree exams are women
# * Otherwise it is fairly distributed

# In[ ]:


fig = px.scatter(data_campusrecruitment, x='degree_t', y='salary',color='gender')
fig.show()


# 7-Inference
# * Men earn more than women in tech
# * For lower levels,salary same in management but salary gap increase....it would have been if I had the age data

# In[ ]:


data_whosuicide=pd.read_csv("")


# 

# In[ ]:


data_videogames=pd.read_csv("../input/videogamesales/vgsales.csv")
data_videogames.head()


# In[ ]:


fig = px.scatter(data_videogames, x='Rank', y='Global_Sales', color="Genre")
fig.show()


# 8-Inference
# * Winner Takes all.
# * There are way too many shooting game that are trash..

# In[ ]:


fig = px.scatter(data_videogames, x='Publisher', y='Rank', color="Genre",height=1700)
fig.show()


# 9-Inference
# * Nintendo,SOny,UEP Systems released a lot of games
# * certain produce certain types of game genre,
# * I don't understand why there are straight lines in the plot..

# In[ ]:


data_foot=pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")
data_foot.head()


# In[ ]:


fig = px.scatter(data_foot, x='home_team', y='away_team',color='tournament',height=1700)
fig.show()


# 10-Inference
# * IDK help me for that
