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
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objects as go
import plotly.offline as py
import plotly.express as px
import pycountry
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print('Successfully loaded')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')
data.head()


# In[ ]:


data = data.rename(columns={"CRIMINAL\nCASES": "Criminal", "GENERAL\nVOTES": "Genral_votes","POSTAL\nVOTES":"Postal_votes","TOTAL\nVOTES":"Total_votes"})


# In[ ]:


data = data.dropna()


# In[ ]:


Num_cons = data.groupby('STATE')['CONSTITUENCY'].nunique().sort_values(ascending = False).reset_index()

ax = px.bar(Num_cons,y='CONSTITUENCY',x='STATE',color = 'CONSTITUENCY')
ax.show()


# ### Uttar Pradesh has the highest number of CONSTITUENCY with 80 and the second is 48 from Maharashtra
# 

# In[ ]:


# Data Cleaning
data['Criminal'] = data['Criminal'].replace('Not Available','0').astype('int')


# In[ ]:


data['EDUCATION'] = data['EDUCATION'].replace('Post Graduate\n','Post Graduate')
data['EDUCATION'] = data['EDUCATION'].replace('Not Available','Others')
education = data['EDUCATION'].value_counts().reset_index()
education.columns = ['EDUCATION','COUNT']
ax = px.bar(education,x = 'EDUCATION', y = 'COUNT',color = 'EDUCATION')
ax.show()


# In[ ]:


winner = data[data['WINNER']==1]
ax = px.bar(winner,x = 'EDUCATION',y = 'WINNER').update_xaxes(categoryorder = "total descending")
ax.show()


# ### The highest number of Winners were Post Graduate

# In[ ]:


young_winner = data[data['WINNER']==1]
young_winner = young_winner.sort_values('AGE').head(10)
ax = px.bar(young_winner,x = 'NAME',y = 'AGE',color = 'AGE',hover_data = ['PARTY','STATE','CONSTITUENCY'])
ax.show()


# ### The youngest was from Andhra Pradesh - Age 26

# In[ ]:


old_winner = data[data['WINNER']==1]
old_winner = old_winner.sort_values('AGE',ascending = False).head(10)
ax = px.bar(old_winner,x = 'NAME',y = 'AGE',color = 'AGE',hover_data = ['PARTY','STATE','CONSTITUENCY'])
ax.show()


# ### The oldest winner was of the AGE -86 (Uttar Pradesh)

# In[ ]:


sns.distplot(data['AGE'],
             kde=False,
             hist_kws=dict(edgecolor="black", linewidth=2),
             color='#00BFC4')


# In[ ]:


criminal_cases = data.groupby('PARTY')['Criminal'].sum().reset_index().sort_values('Criminal',ascending=False).head(30)
ax = px.bar(criminal_cases, x = 'PARTY',y = 'Criminal',color = 'PARTY')
ax.show()


# ### The highest number of criminals were in BJP, Congress was quite closeby

# In[ ]:


crime = data[data['WINNER']==1]
criminal_cases = crime.groupby('PARTY')['Criminal'].sum().reset_index().sort_values('Criminal',ascending=False).head(30)
ax = px.bar(criminal_cases, x = 'PARTY',y = 'Criminal',color = 'PARTY')
ax.show()


# ### We can clearly see the number of WINNERS and those who had criminal records were high for BJP

# In[ ]:


## changing the datatype
data['GENDER'] = data['GENDER'].astype('category') 
data['WINNER'] = data['WINNER'].astype('category') 


# In[ ]:


Female_winners = data[(data['WINNER']==1) & (data['GENDER']=='FEMALE')]
ax = px.histogram(Female_winners, 'STATE')
ax.show()


# ### Highest number of Female winners were from West Bengal

# In[ ]:


male_winners = data[(data['WINNER']==1) & (data['GENDER']=='MALE')]
ax = px.histogram(male_winners, 'STATE')
ax.show()


# In[ ]:


votes = data.groupby('STATE')['Total_votes'].sum().sort_values(ascending = False).reset_index()
ax = px.bar(votes,x = 'STATE',y = 'Total_votes',color='STATE')
ax.show()


# ### Total votes casted is highest in UP were the highest.

# In[ ]:


category = data['CATEGORY'].value_counts().reset_index()
category.columns= ['CATEGORY','COUNT']
ax = px.bar(category,x = 'CATEGORY', y = 'COUNT', color = 'CATEGORY')
ax.show()


# - We can see that the number of candidates from GENERAL is the highest.

# In[ ]:


df = data[data['WINNER']==1]
category = df['CATEGORY'].value_counts().reset_index()
category.columns= ['CATEGORY','COUNT']
ax = px.bar(category,x = 'CATEGORY', y = 'COUNT', color = 'CATEGORY')
ax.show()


# In[ ]:




