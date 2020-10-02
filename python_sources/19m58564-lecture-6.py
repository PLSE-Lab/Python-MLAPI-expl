#!/usr/bin/env python
# coding: utf-8

# # Exercise 6
# SUMAYYA - 19M58564
# 
# For this exercise, I am going to see the trend of COVID-19 in Denmark

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Denmark'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
#print(np.unique[df['Country/Region'].values])
del df['SNo']
df_DE = df[df['Country/Region']==selected_country]
df_DE = df_DE.groupby('ObservationDate').sum()
print(df_DE)


# **Covid-19 cases trend in Denmark**

# In[ ]:


df1 = df_DE.diff()
print(df1.plot())


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df1.index, y=df1['Confirmed'].values, name='Confirmed')
daily_deaths_object = go.Scatter(x=df1.index, y=df1['Deaths'].values, name='Deaths')
daily_recovered_object = go.Scatter(x=df1.index, y=df1['Recovered'].values, name='Recovered')
layout_object = go.Layout(title='Denmark Daily Cases',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)


# **Informative Table of Covid-19 Cases in Denmark**

# In[ ]:


df1= df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('Confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_19M58564.html','w')
f.write(styled_object.render())


# **Denmark's Covid-19 Cases Rank**

# In[ ]:


latest = df[df['ObservationDate']=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()
print('Ranking of Denmark: ', latest[latest['Country/Region']==selected_country].index.values[0]+1)


# # Analysis and Discussion
# 
# Based on the Danish Health Authority, the first case in Denmark confirmed on February 27, 2020, and the first death in early March. Based on the graph, the outbreak reached its peak on April 7. After that, the cases are decreasing gradually until now.
# 
# Denmark's cases peaked at a pretty early stage. The decisiveness of the government leads to the current situation. Denmark was among the first countries in Europe to enforce lockdown and close its border. The government quickly respond to the outbreak as soon as March 11, two weeks after the first case. There is no legal restriction for the residents of Denmark to stay at home, but they were being obedient at that moment and supported the government directions. On April 15, a week after the peak, the Country allowed schools to open gradually, beginning with kindergarten and elementary schools with social distancing measures and strict hygiene procedures. On April 20, small businesses started to be allowed to open, and the government gave financial aid to them. The quarantine period was pretty much over early May as other sectors are allowed to go back in business. Since then, no hike in cases found as measures to limit the virus spread, such as limited mass gathering and social distancing rules, are applied. Recently, Denmark opened its borders with the neighbours except for Sweden because of the collective immunity method. Since the lockdown eased, Denmark ramp up testing, while before that the tests were limited to the only high-risk group because of limited kits, and the result is still showing good progress. 
# 
# The Danes appreciate the measures taken by government and health institutions. The decisiveness, combined with the resident's habit to obey rules is the factor of success and is one of the best models to control the outbreak in Europe. The Danes are not too physical in their social life so that it may be one of the factors of the falling cases. However, the economic impact is inevitable as a price to their decision to apply the lockdown early. Inflation and GDP drop is expected to reach its worst for the second time since the post-WWII period. Hopefully, with opening borders allowing countries to take on new normal, these consequences will be able to be reduced.
# 
# Articles as Reference:
# 
# https://www.sst.dk/en/English/Corona-eng/FAQ#uk-corona-faq-strategi
# https://www.institutmontaigne.org/en/blog/europe-versus-coronavirus-putting-danish-model-test
# https://www.thelocal.dk/20200610/denmark-says-easing-lockdown-has-not-increased-infections
# https://www.dailymail.co.uk/news/article-8406167/Coronavirus-infections-Denmark-FALLING-lockdown-eased-restaurants-shops-reopened.html
# 
