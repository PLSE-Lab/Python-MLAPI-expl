#!/usr/bin/env python
# coding: utf-8

# # Exercise 6 Covid-19 in Peru
# ### Wiranpat Simadhamnand, student number 17B00136

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf) # so that we can see the whole dataframe


# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
#print(df.columns)
#print(np.unique(df['Country/Region'].values)) # list names of country


# In[ ]:


selected_country = 'Peru'
df1 = df[df['Country/Region']==selected_country]
#print(df1)


# In[ ]:


df = df1.groupby('ObservationDate').sum()
#print(df) # sum the same date in all the states


# ## Plot

# Graph showing cumulated number of infected, recovered, active and dead individuals in Peru.

# In[ ]:


df['ActiveCase'] = df['Confirmed']-df['Recovered']-df['Deaths']
df['Confirmed'].plot(legend=True, color='#5ec691')
df['Recovered'].plot(legend=True, color='#fe7241')
df['Deaths'].plot(legend=True, color='#95389e')
df['ActiveCase'].plot(legend=True, color='#fe346e')


# Graph showing daily changes in number of infected and recovered individuals in Peru.

# In[ ]:


df['DailyConfirmed'] = df['Confirmed'].diff()
df['DailyDeaths'] = df['Deaths'].diff()
df['DailyRecovery'] = df['Recovered'].diff()
df['DailyConfirmed'].plot(legend=True, color='#40bad5')
df['DailyRecovery'].plot(legend=True, color='#fcbf1e')
df['DailyDeaths'].plot(legend=True, color='#f35588')


# ## Interactive chart

# Interactive graph showing daily changes in number of infected and recovered individuals in Peru.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['DailyConfirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['DailyDeaths'].values,name='Daily deaths')

layout_object = go.Layout(title='Peru daily case',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html('Peru_daily_cases_17B00136.html')


# ## Colored table

# In[ ]:


df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='YlGnBu').highlight_max('DailyConfirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_17B00136.html','w')
#f.write(styled_object.render())


# ## Global ranking of Peru

# In[ ]:


import pandas as pd
import numpy as np

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#df.index[df['ObservationDate']=='06/12/2020'].tolist()
df1 = df[40076:]  # data of the date 06/12/2020
df1 = df.groupby(['Country/Region']).sum()
df2 = df1.sort_values(by=['Confirmed'], ascending=False)
df2['Rank'] = np.arange(1,224)  # adding a column of worldwide rank
df2.head(20)


# ## Government's measurement and overall trend of Covid-19 in Peru
# Peru was one of the first country in Latin America that went into lockdown on March 16. Bans on traveling across provinces, entering and exiting the country were implemented. There were only 43 new cases on the day that the quarantine was declared. On March 18, the measures were tightened into a next step by setting a curfew from 8 pm to 5 am. However the situation did not get better even after the all the measures. On March 26, new daily cases reached 100 cases before which it started to increase rapidly and reached a peak on May 30 with 14000 daily new cases. It is said tobe because of people's behavior on not respecting the laws to make the spread worse.
# 
# In general, daily infected people in Peru increased over time with some fluctuation until now. The fluctuation is probably from the number of test that is conducted or the update system. For now (June 12), Peru ranks in 13th in the world considering from confrimed case on June 12. Even though right now the number of daily cases decreases from the highest point, it is still considered high at around 5000 cases per day and there is still no sign of it decreasing. In total, there are 214788 confirmed cases in which 107133 cases already recovered and 6088 deaths out of population of 32 millions people.

# ## References
# https://www.theguardian.com/global-development/2020/may/20/peru-coronavirus-lockdown-new-cases
