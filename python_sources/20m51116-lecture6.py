#!/usr/bin/env python
# coding: utf-8

# # This is exercise 6
# 
# I look at Corona cases for Italy.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Italy'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)


# In[ ]:


df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()


# In[ ]:


print(df)


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

layout_object = go.Layout(title='Italy daily cases 20M51116',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html('Italy_daily_cases_20M51116.html')


# In[ ]:


df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M51116.html','w')
f.write(styled_object.render())


# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1 = df[df['ObservationDate']=='06/12/2020']
df2 = df1.groupby(['Country/Region']).sum().sort_values(by=['Confirmed'],ascending=False).reset_index()
print(df2[df2['Country/Region']=='Italy'].index.values[0]+1)


# # Report
# 
# 1. I selected Italy as above.
# 2. We can see the diagrams in above analysis.
# 3. According to the analysis, global ranking of Italy is No.7. We can recognize that number of patient in Italy has been decreasing day by day from the time-series diagram. The most number of new patient is recorded on 21th, March as 6557 people.
# 4. In order to deal with corona spreading, Italian government decided to state the restriction of going out. In this situation, they counld not go out without people who had had permition from the police. Except grosary shops and drug stores, all shops and restaurants had cannot been opened for a while.
# Reference: https://www.businessinsider.com/coroanvirus-italy-lockdown-nationwide-rules-2020-3
