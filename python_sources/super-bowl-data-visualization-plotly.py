#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import plotly.express as px
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df1 = pd.read_csv("/kaggle/input/superbowl-history-1967-2020/superbowl.csv")
df1.head()


# In[ ]:


df1.isna().sum() 


# In[ ]:


df1['Diff'] = df1['Winner Pts'] - df1['Loser Pts']


# In[ ]:


fig = px.scatter(df1, x="Loser", y="Winner"  ,size="Diff", hover_name="Stadium", color="Date",
  title=' a 1967-2020 games resume')
fig.show()


# In[ ]:


rb2=df1.State.value_counts()
rbg2=rb2.reset_index()
rbg2.rename(columns={'index':'State', 'State': 'num of played match'}, inplace=True)
rbg2


# In[ ]:


import plotly.graph_objects as go
rbg2.set_index('State', inplace=True)
r30=rbg2.rename( index={'Florida':"FL", 'California': "CA","Louisiana": "LA", "Texas":"TX","Arizona": "AZ" , "Georgia":"GA", "Michigan":"MI", "Minnesota":"MN","Indiana":"IN", "New Jersey":"NZ" })
final=r30.reset_index()
fig = go.Figure(data=go.Choropleth(
    locations=final['State'], # Spatial coordinates
    z = final['num of played match'].astype(float), 
    locationmode = 'USA-states',
    colorscale = 'Blues',
))

fig.update_layout(
    title_text = '1967-2020 number of played games per State (USA)',
    geo_scope='usa', 
)

fig.show()


# In[ ]:


a=pd.DataFrame(df1.MVP.value_counts()).reset_index().rename(columns = {'index' : 'Player'}).max()
print('Most Valuable Player 1967-2020 is',a[0] ,' with a total of: ', a[1] ) 


# In[ ]:


rb1=df1.Winner.value_counts()
rbg1=rb1.reset_index()
rbg1.rename(columns={'index':'Team' , 'Winner': 'num of won games'}, inplace=True)
import plotly.express as px
fig = px.line(rbg1, x="Team", y="num of won games"   ,
  title=' number of won games by team 1967-2020')
fig.show()


# In[ ]:


rb=df1.Loser.value_counts()
rbg=rb.reset_index()
rbg.rename(columns={'index':'Team' , 'Loser': 'num of lost games'}, inplace=True)
rr=rbg.head(5)
a=rr.sort_index(axis = 0, ascending = False)
fig = px.bar(a, y="Team", x="num of lost games" , orientation ='h', title=' TOP FIVE LOSING TEAMS 1967-2020')
fig.show()


# In[ ]:


nn=pd.merge(rbg, rbg1, on='Team', how='inner')
nn['total finals']=nn.sum(axis = 1, skipna = True) 
nn=nn.sort_values(by ='total finals', ascending = False ).head()
fig = px.pie(nn, names="Team", values="total finals" , hole=.6, title=' TOP FIVE TEAMS THAT MADE IT TO FINALS 1967-2020')
fig.show()

