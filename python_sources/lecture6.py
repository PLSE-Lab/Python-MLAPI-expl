#!/usr/bin/env python
# coding: utf-8

# # Exercise 6  - Mako Mizuno (20M51808)
# # email: mizuno.m.af@m.titech.ac.jp
# I look at Corona Virus for Switzerland.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Switzerland'
df = pd. read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)


# In[ ]:


df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()


# In[ ]:


print(df)


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

layout_object = go.Layout(title='Switzerland daily cases 20M51808',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html('Switzerland_daily_cases_20M51808.html')


# In[ ]:


df1 = df
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M51808.html','w')
f.write(styled_object.render())


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.head()


# In[ ]:


data.index=data['ObservationDate']
data = data.drop(['SNo','ObservationDate'],axis=1)
data.head()


# In[ ]:


data_swiss = data[data['Country/Region']=='Switzerland']
data_swiss.tail()


# # Global Ranking of Switzerland

# In[ ]:


latest = data[data.index=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of Switzerland: ', latest[latest['Country/Region']=='Switzerland'].index.values[0]+1)


# # Analyzing the data
# 
# According to the first graph, it is obvious that the number of people who are confirmed as infected persons with COVID-19 is now decreasing and there is no sign of the secondary wave so far.
# The global ranking of Switzerland is 34th, and the day that has the largest value of "daily confirmed" is March 23, 2020.

# # Measures for COVID-19 in Switzerland
# 
# Judging from the first confirmation of the infected person in the end of this February, the national government made posters to announce the prevention of spreading the virus, and started to call for attention.  Furthermore, the government expanded regulation gradually, and made sponsors cancel large-scaled events.
# 
# On March 11th, 2020, Canton Ticino declared a state of emergency for the first time in Switzerland, and almost all school, movie cinemas and nightclubs closed.  Since then, each canton started to declare a state of emergency one by one.  However, the government did not impose a curfew after all.
# 
# On March 25th,2020, the government established the system that finances maximum 500,000franc without interest to support the small and medium-sized enterprises.  In addition, the Swiss National Bank founded the new system called "COVID-19 refinance facility" to provide logistical support for other banks.
# 

# # Reference
# https://www.swissinfo.ch/jpn/politics/covid-19_%E3%82%B9%E3%82%A4%E3%82%B9%E3%81%AE%E6%96%B0%E5%9E%8B%E3%82%B3%E3%83%AD%E3%83%8A1%E3%82%AB%E6%9C%88-%E6%94%BF%E5%BA%9C%E3%81%AF%E3%81%A9%E3%81%86%E5%AF%BE%E5%BF%9C%E3%81%97%E3%81%9F%E3%81%8B/45645374
