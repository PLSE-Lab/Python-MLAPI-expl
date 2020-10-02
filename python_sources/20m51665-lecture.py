#!/usr/bin/env python
# coding: utf-8

# # This is exercise 6
# 
# We look at Corona cases for a specific country.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Italy'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)


# Note: This df is actually a cumulative information.
# 
# June 6, 2020 shows the total cases since initial date.
# 
# How can we calculate day-to-day cases?? Or daily increases??

# In[ ]:


df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()


# In[ ]:


print(df)


# # How about we make an interactive chart?
# 
# To do this, we'll need to load to the following modules.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

layout_object = go.Layout(title='Italy Daily cases 20M51665',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html('Italy_daily_cases_20M51665.html')


# # How can we make an informative table?
# 
# Maybe color the entries showing large values as some bright color, low values as some dark color.

# In[ ]:


df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M51665.html','w')
f.write(styled_object.render())


# # How can we calculate global ranking?
# 
# Maybe selecting only the latest date???

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1 = df.groupby(['ObservationDate','Country/Region']).sum()
df2 = df1[df1['ObservationDate']=='06/06/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()
print(df2[df2['Country/Region']=='Italy'])

