#!/usr/bin/env python
# coding: utf-8

# # Covid19 cases in Austria
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Austria'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
#print(np.unique(df['Country/Region'].values))
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)


# In[ ]:


#print(df['Confirmed'].diff())
#print(df['Deaths'].diff())
df['daily_confirmed']=df['Confirmed'].diff()
df['daily_deaths']=df['Deaths'].diff()
df['daily_recovery']=df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()


# In[ ]:


print(df)


# # make an interactive chart
# 

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index, y=df['daily_confirmed'].values, name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index, y=df['daily_deaths'].values, name='Daily deaths')

layout_object = go.Layout(title='Austria daily cases 20M51850', xaxis=dict(title='Date'), yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object, daily_deaths_object], layout=layout_object)
iplot(fig)
fig.write_html('Austria_daily_cases_20M51850.html')


# # make an informative table

# In[ ]:


df1=df#[['daily_confirmed']]
df1=df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_heat').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M51850.html','w')
f.write(styled_object.render())


# # calculate global ranking

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
df1 = df.groupby(['ObservationDate','Country/Region']).sum()
df2 = df[df['ObservationDate']=='06/16/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()
print(df2[df2['Country/Region']=='Austria'])


# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

df2 = df[df['Country/Region']=='Austria']
df2.tail()

df2 = df[df.index=='06/16/2020']
df2 = df2.groupby('Country/Region').sum()
df2 = df2.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of Austria: ', df[df['Country/Region']=='Austria'].index.values[0]+1)

#I am not sure how to find global ranking of Austria. The result seems to be mistaken. 


# # Discussion
# 
# From seeing the number of daily cases it was highest on 26th of March, after then it was decreasing until 20th of April and until now it's increasing and decreasing with only small number. Here is summary of prevention measures of Covid19 in Austria. 
# 
# On 15 March, a COVID-19 crisis management fund was announced, with EUR 4 billion in funding.On 18 March, a further EUR 38 billion support fund was announced. The measures include: 
# EUR 9 billion in guarantees and warranties; 
# EUR 15 billion in emergency aid; 
# EUR 10 billion in tax deferral[1].
# 
# On 17 March, Austria banned all arrivals from Italy, China's Hubei Province, Iran, and South Korea, excepting those who had a medical certificate no more than four days old that confirmed they were not affected by coronavirus.However, travel through Austria was possible[2].
# 
# On 30 March, the government laid out plans to introduce compulsory wearing of face in the supermarket from 6 April onwards, but will be extended to more public places in the near future[3].
# 
# The working hours of the employees are reduced between 10% and 90% and their salaries are adjusted accordingly.The employer receives a financial support from the Public Employment Service (AMS) on the basis of flat-rates determined by the AMS (short-time-work allowance).Two billion fund to cover living costs of self employed & gig workers and mini-companies[1].
# 
# Austria did a very good job starting with mandatory mask wearing and banning the arrivals from certain countries, since it's risky to spread the virus from Italy. Also after shutting down the shops and everything, during the lockdown they had a vaious measurements that easing the lockdown, such as financial supports, tax relief, loan guarantees etc. They managed to decrease the number from the peak. 
# 
# [1] https://home.kpmg/xx/en/home/insights/2020/04/austria-government-and-institution-measures-in-response-to-covid.html
# [2] https://edition.cnn.com/travel/article/coronavirus-travel-bans/index.html
# [3] https://orf.at/stories/3159909/
# 
# 
