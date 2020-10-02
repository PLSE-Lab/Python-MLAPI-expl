#!/usr/bin/env python
# coding: utf-8

# # Exercise 6 : Covid 19 cases by country
# 
# ### Country of study : United Kingdom by 17B00076 Lamoonkit Jomphol

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
# print(df.columns)
# print(np.unique(df['Country/Region'].values))
country = 'UK'
# print(df[df['Country/Region']== country])
df1 = df[df['Country/Region']== country]
# print(df1)
print(df1.groupby('ObservationDate').sum())

df2 = df1.groupby('ObservationDate').sum()




# # To calculate daily increases
# 

# In[ ]:


# print(df2['Confirmed'].diff())
# diff = subtraction for daily basis
# df2 = cumulative data
df2['daily_confirmed'] = df2['Confirmed'].diff()
df2['daily_deaths'] = df2['Deaths'].diff()
df2['daily_recovered'] = df2['Recovered'].diff()
# print(df2)
df2['daily_confirmed'].plot()
df2['daily_recovered'].plot()
df2['daily_deaths'].plot()
plt.ylabel("Number of People")
plt.xlabel("Date")
plt.show()


# In[ ]:


#Cumulative plot
df2['Confirmed'].plot()
df2['Recovered'].plot()
df2['Deaths'].plot()
plt.ylabel("Number of People")
plt.xlabel("Date")
plt.show()


# # Interactive chart

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df2.index,y=df2['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df2.index,y=df2['daily_deaths'].values,name='Daily deaths')
daily_recovered_object = go.Scatter(x=df2.index,y=df2['daily_recovered'].values,name='Daily recovered')

layout_object = go.Layout(title='UK daily cases 17B00076',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('UK_DAILY_CASES_17B00076.html')


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df2.index,y=df2['Confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df2.index,y=df2['Deaths'].values,name='Daily deaths')
daily_recovered_object = go.Scatter(x=df2.index,y=df2['Recovered'].values,name='Daily recovered')

layout_object = go.Layout(title='UK daily cases 17B00076',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('UK_DAILY_CASES_17B00076.html')


# # Table 

# In[ ]:


# print(df2)
# df 3 = 
# df3 = df2[['daily_confirmed']]
df3 = df2
df3 = df3.fillna(0.)
# print(df3)
styled_object = df3.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('Table_17B00076.html','w')
f.write(styled_object.render())


# 
# # Global ranking
# Inspired by Ryza's method

# In[ ]:



# print df
date = '06/10/2020'
rank = df[df.index==date]
# df.index = row labels
rank = rank.groupby('Country/Region').sum()
rank = rank.sort_values(by='Confirmed',ascending=False).reset_index() 
print(rank)

print('Global rank of',country,'on',date,' = ', rank[rank['Country/Region']=='UK'].index.values [0]+1)


#read https://pandas.pydata.org/pandas-docs/stable/reference/frame.html


# # Discussion
# 
# 
# According to the cumulative graph, United Kingdom is clearly still experiecing more Covid-19 cases daily with slightly slower rate in late May. The Covid-19 cases in UK began to surge dramatically since late February and continued to rise until the present day.  On 10th June 2020, UK was ranked the 4th highest Covid-19 confirmed cases internationally with 291,588 confirmed new cases. From the daily cases graph,The highest daily confirmed cases was recorded on 10th April 2020 with 8,733 cases. It can also be inferred that the rate of infection has started to decrease around mid-April. Nevertheless, the situation of Covid-19 in United Kingdom is still critical due to the fact that the number of recovered cases is comparatively minuscule. 
# 
# In addition, there are several anomalies in the data. On 29th April, there was a record of 4421 deaths which is contrast to on-going trend with less than a thousand deaths daily. This was becuse the government started to include the death from care homes and wider community. Hence, the unorthodox surge. On 20th May, there was a negative number of daily confirmed cases which could be a result of the removal of mistakenly recorded cases. 
# 
# The british government started the lockdown policy on March 23th. Nevertheless, its first record of Covid-19 case dated back to 31th January. There have been several complains regarding how British government react poorly on the pandemic. Now, United Kingdom is one of the hardest hit country regarding the virus. The worst issue seems to be the shortages of medical equipment and PPE for the healthcare workers. 
# 
# Recently, the movement for the diverse ethinicity has also driven many british people to protest. People are gathered densely in big cities,bringing down statue. All of which indicates that a significant number of people do care more about the politic movement than the severity of the on-going pandamic.
# 
# 
# note : In my perception, one of the reason which made Covid-19 become more severe in western hemisphere could be the mask-wearing culture. It can be observed that in the early period of pandamic, the western countries rejected the use of face mask while the eastern country embraced face mask policy very seriously. In depth, people do percieve the use of face mask differently across the regions. In Japan, people do wear face mask as a fashion long before the pandemic hit while in western country such as UK, some people percieved face mask as a sign for serious illness and often being treated differently. Thus, it is intelligible on what might have gone wrong. 
# 
# 
# 
# 
# 
# References
# 
# Death tolls anomaly
# https://metro.co.uk/2020/04/29/uk-death-toll-rises-26097-care-homes-included-12628454/
# 
# Lockdown history
# https://www.independent.co.uk/life-style/health-and-families/coronavirus-lockdown-uk-remove-end-review-schools-when-government-a9453246.html
# 
# Criticism
# https://www.japantimes.co.jp/news/2020/06/11/world/uk-scientists-speak-out-covid-19/#.XuS8NkX7SUk
# https://www.theguardian.com/world/2020/may/10/100-days-later-how-did-britain-fail-so-badly-in-dealing-with-covid-19
# 
# PPE shortage
# https://www.theguardian.com/uk-news/2020/may/03/nearly-half-of-british-doctors-forced-to-find-their-own-ppe-new-data-shows
#   
# Protests
# https://www.bbc.com/news/uk-53023351
#   
#   
#   
#   
#   
