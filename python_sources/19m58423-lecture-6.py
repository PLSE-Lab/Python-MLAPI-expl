#!/usr/bin/env python
# coding: utf-8

#  This is exercise 6  19M58423
# 
# # 1. Select a country which you are not familiar with and has over 100 current cases of Corona. If you notice some error in the data, select a different country.
# 
# Selected country: South Korea

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold =np.inf)

Selected_country = 'South Korea'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header =0)
#print (np.unique(df['Country/Region'].values))
df = df[df['Country/Region']==Selected_country]
#df3 = df.groupby('ObservationDate').sum()
#print(df3)


# 
# # 2.1 Construct a time-series and colored tables of the following:
# Daily confirmed cases, Daily deaths, Daily recoveries

# In[ ]:


df['Daily_confirmed'] = (df['Confirmed'].diff())
df['Daily_deaths'] = (df['Deaths'].diff())
df['Daily_recovery'] = (df['Recovered'].diff())
#df['Daily_confirmed'] .plot()
#plt.show()


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

Daily_confirmed_object = go.Scatter(x=df.ObservationDate, y=df['Daily_confirmed'].values,name='Daily confirmed')
Daily_deaths_object = go.Scatter(x=df.ObservationDate, y=df['Daily_deaths'].values,name='Daily deaths')
Daily_recoveries_object = go.Scatter(x=df.ObservationDate, y=df['Daily_recovery'].values,name='Daily recoveries')

layout_object = go.Layout(title = 'South Korea daily cases 19M58423',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[Daily_confirmed_object,Daily_deaths_object,Daily_recoveries_object],layout=layout_object)
iplot(fig)


# 
# # 2.2 Construct **a colored tables** of the following:
# Daily confirmed cases, Daily deaths, Daily recoveries

# In[ ]:


df1 = df.iloc[:,[1,8,9,10]]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('Daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
#f = open('table_19M58423.html','w')
#f.write(styled_object.render())


# # 3. What is the global ranking of that country in terms of the number of cases? 
# 

# In[ ]:


dfx = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header =0)

rk = dfx[dfx['ObservationDate']=='06/06/2020']
rk = rk.groupby('Country/Region').sum()
rk = rk.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of South Korea: ', rk[rk['Country/Region']=='South Korea'].index.values[0]+1)


# # Analyze from the time-series whether the corona cases are decreasing with time. Which dates had the largest daily cases? Is it still increasing? Not clear? Visual inspection is okay.
# 
# From the time-series, it can be observed that 2020/03/03 had the largest daily confirmed, which were 851 persons. 2020/03/22 had the largest recoveries, which were 1369 persons. From the colored table, it can be seen that 2020/03/24 had the largest daily deaths, which were 9 persons.The number of deaths is very low if compared to other countries. The corona cases had the peak during the end of Feburary. After that period, the cases is decreasing drastically with time in March. After March, the number of  daily confirmed case are quite stable. From the end of May, the number is a little bit increased compared before, but it seems still to be under control. It is not incresting gernerally speaking, but the attention shoule be paid for the recent increase.
# 

# # Discuss how the national government of the country is addressing the issue by summarizing from the news.
# 
# 
# South Korea has shown there is another way to bring the disease under control. **Businesses across the country have largely carried on, and no city has been locked down**. With new cases declining, life in South Korea is returning to normal. More importantly, the country has one of the world's lowest fatality rates at just 1%. Korean officials attribute this to** a strategy called TRUST: transparency; robust screening and quarantine; unique but universally applicable testing and; strict control and treatment.** There is one approach called **drive-through test stations**, which allow health workers to take swabs to test for the virus without people leaving their cars. This approach is quite effective and helpful, which is now being copied by other countries, including Canada, Germany and the U.S.
# 
# **Reference: **
# Tiang, J., Ma , D., Huang, S., & Han, W. (2020, Apri 1). In Depth: Why South Korea is winning the coronavirus battle. Retrieved June 8, 2020, from Nikkei Asian Review: https://asia.nikkei.com/Spotlight/Caixin/In-Depth-Why-South-Korea-is-winning-the-coronavirus-battle
# 
