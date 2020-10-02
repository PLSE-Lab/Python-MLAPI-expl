#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:




file = "../input/world-coronavirus-data-visualization-442020/corona_world_latest-04-04-2020.csv"


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


df=pd.read_csv(file)


# In[ ]:


df


# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.max_rows', None) 


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df['TotalDeaths']=df['TotalDeaths'].astype('int32')


# In[ ]:


df['NewDeaths']=df['NewDeaths'].astype('int32')


# In[ ]:



df['TotalRecovered']=df['TotalRecovered'].astype('int32')


# In[ ]:


df['Serious,Critical']=df['Serious,Critical'].astype('int32')


# In[ ]:


df['Deaths/1M pop']=df['Deaths/1M pop'].astype('int32')


# In[ ]:


df['TotalTests']=df['TotalTests'].astype('int32')


# In[ ]:


df['Tests/ 1M pop']=df['Tests/ 1M pop'].astype('int32')


# In[ ]:


df.head()


# In[ ]:


df


# In[ ]:


#checking  missing values

pd.isnull(df)


# In[ ]:


#There are no missing values


# In[ ]:


df.head()


# In[ ]:


df1=df.groupby(['Country,Other'])['TotalCases','TotalDeaths','TotalRecovered','ActiveCases'].sum().reset_index()


# df1

# In[ ]:


df1


# In[ ]:


df1.head()


# In[ ]:


world_TotalCases=df1['TotalCases'].sum()
world_TotalDeaths=df1['TotalDeaths'].sum()
world_TotalRecovered=df1['TotalRecovered'].sum()
world_ActiveCases=df1['ActiveCases'].sum()


# In[ ]:


print(world_TotalCases,world_TotalDeaths,world_TotalRecovered,world_ActiveCases)


# In[ ]:


#adding world data to df1 and df3 


data= [['World', world_TotalCases,world_TotalDeaths,world_TotalRecovered,world_ActiveCases]] 
df3=pd.DataFrame(data,columns=['Country,Other','TotalCases','TotalDeaths','TotalRecovered','ActiveCases'])
df3


# In[ ]:


df1


# In[ ]:


df2=pd.concat([df1,df3],ignore_index=True,sort=False)
print(df2.tail(5))
print(df3)


# In[ ]:


#bar plotiing 

fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
bars=['TotalCases','TotalDeaths','TotalRecovered','ActiveCases']
values=[1118304,59221,229245,829838]
ax.bar(bars,values)
plt.title('Data of coronavirus till 4 april 2020')





# In[ ]:


#bar ploting with color 

fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
bars=['TotalCases','TotalDeaths','TotalRecovered','ActiveCases']
values=[1118304,59221,229245,829838]
ax.bar(bars,values,color=('green','red','blue','yellow'))
plt.xlabel('coronavirus parameters')
plt.ylabel('counts')

plt.title('Data of coronavirus till 4 april 2020')


# In[ ]:


df2.tail()


# In[ ]:


#plotting pie chart 
values=[59221,229245,829838]
colors=['red','blue','yellow']
explode=[0.2,0,0]
labels=['TotalDeaths','TotalRecovered','ActiveCases']
plt.pie(values, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',shadow=False, startangle=360)
plt.title('corona virus total cases till 4th april 2020: '+str(df1['TotalCases'].sum()))
plt.tight_layout()

plt.show()


# In[ ]:


#plotting donut or ring

labels=['TotalDeaths','TotalRecovered','ActiveCases']
values = [59221,229245,829838]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.7)])

fig.show()


# In[ ]:


#Another way of pie chart
fig=px.pie(values=[59221,229245,829838],names=['TotalDeaths','TotalRecovered','ActiveCases'],title='corona virus total cases till 4th april 2020: '+str(df1['TotalCases'].sum())
            )
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[ ]:


df2.head()


# In[ ]:


Data_per_country=df2.groupby(['Country,Other'])['TotalCases'].sum().reset_index().sort_values('TotalCases',ascending=False).reset_index(drop=True)


# In[ ]:


Data_per_country


# In[ ]:


#plotting world graph
fig=px.choropleth(Data_per_country,locations=Data_per_country['Country,Other'],
                  locationmode='country names',hover_name=Data_per_country['Country,Other'],
                  color=Data_per_country['TotalCases'],
                  color_continuous_scale=px.colors.sequential.deep)
fig.update_layout(title='total cases of coronavirus across the world on 4th April 2020')
fig.show()


# In[ ]:


#top ten countries in comparison with world data in terms of total cases
fig=go.Figure(data=[go.Bar(
    x=Data_per_country['Country,Other'][0:11],
    y=Data_per_country['TotalCases'][0:11],
    textposition='auto',
    marker_color='black'
    )])
fig.update_layout(title='top ten countries with total cases of coronavirus across the world on 4th April 2020',
                  xaxis_title='Countries', yaxis_title='total cases')
fig.show()


# In[ ]:


df1.head()


# In[ ]:


df.head()


# In[ ]:


#bar plotting and comparison
fig=go.Figure()
fig.add_trace(go.Bar(
   x=df['Country,Other'][df.TotalCases >=10000],
   y=df['TotalCases'][df.TotalCases >=10000],
   marker_color='green',
   textposition='auto',
   name='TotalCases',
   text=df.TotalCases[df.TotalCases>=10000]
   ))


fig.add_trace(go.Bar(
   x=df['Country,Other'][df.TotalCases >=10000],
   y=df['TotalDeaths'][df.TotalCases >=10000],
   marker_color='red',
   textposition='auto',
   name='TotalDeaths',
   text=df.TotalDeaths[df.TotalCases>=10000]
   ))

fig.add_trace(go.Bar(
   x=df['Country,Other'][df.TotalCases >=10000],
   y=df['TotalRecovered'][df.TotalCases >=10000],
   marker_color='blue',
   textposition='auto',
   name='TotalRecovered',
   text=df.TotalRecovered[df.TotalCases>=10000]
   ))

fig.update_layout(title='Corona virus TotalCase greater than 10,000',
                  xaxis_title='Countries', yaxis_title='total cases')
fig.show()


# In[ ]:




