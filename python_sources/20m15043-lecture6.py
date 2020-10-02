#!/usr/bin/env python
# coding: utf-8

# # lecture 6
# we look at Corona cases for a specific country___Brazil

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

selected_country='Brazil'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))
df=df[df['Country/Region']==selected_country]
df=df.groupby('ObservationDate').sum()
print(df)


# this df is a cumulative information. 
# shows the total cases since the initial date. 
# how can we calculate day-to-day cases or daily increases?

# In[ ]:


df['daily_confirmed']=df['Confirmed'].diff()
df['daily_deaths']=df['Deaths'].diff()
df['daily_recovered']=df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovered'].plot()
plt.show()


# # how about to make an interactive chart

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index, y=df['daily_confirmed'].values, name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index, y=df['daily_deaths'].values, name='Daily deaths')
daily_recovered_object = go.Scatter(x=df.index, y=df['daily_recovered'].values, name='Daily recovered')

layout_object = go.Layout(title='Brazil daily cases 20M15043',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object, daily_deaths_object, daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('Brazil_daily_cases_20M15043.html')


# # how can we make an informative table?
# 
# maybe color the entries showing large values as some bright color, low value as dark color

# In[ ]:


df1=df#[['daily_confirmed']]
df1=df1.fillna(0.)
style_object=df1.style.background_gradient(cmap='jet').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(style_object)
f=open('table_20M15043.html','w')
f.write(style_object.render())


# # how can we calculate the global ranking
# 
# maybe using the latest data

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1=df.groupby(['ObservationDate','Country/Region']).sum()
df2=df[df['ObservationDate']=='06/12/2020'].sort_values(by=['Confirmed'], ascending=False).reset_index()
print(df2[df2['Country/Region']=='Brazil'])


# # ranking by accurate number

# In[ ]:


selected_country='Brazil'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df=df[df['ObservationDate']=='06/12/2020']
df=df.groupby(['ObservationDate','Country/Region']).sum()
df1=df.sort_values(by=['Confirmed'], ascending=False).reset_index()
df2=df.sort_values(by=['Deaths'], ascending=False).reset_index()
df3=df.sort_values(by=['Recovered'], ascending=False).reset_index()

print('CONFIRMED:',df1)
print('DEATHS:',df2)
print('RECOVERED:',df3)


# As shown in the table, the observations of Sao Paulo in Brazil in 6/12/2020 is the top 5 of global regions.
# from the perspective of a whole country, the observation cases and the deaths cases is increasing rapidly.
# However, the rate of increasing seems reducing recently.
# on the 6/8/2020, the recovered cases rapidly got large, about 94.3K.
# If we look at the accurate number of brazil.
# For now, the rank of total number of confirmed and deaths and recovered are all 2.
# 
# And now, although ministry of health stops publishing the total cases of Corona Virus, the government of Brazil is tring hard to avoid the larger spread of the virus.
# 
# information resource:
# https://www.chinanews.com/gj/2020/06-07/9205606.shtml
# 
# until now, 2020/06/20, 
# the total number of accumulate cases of Brazil is over 1 million
# 
# information resource:
# http://www.cneb.gov.cn/2020/06/20/ARTI1592620956767295.shtml
# 
# 

# In[ ]:




