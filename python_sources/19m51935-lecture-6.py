#!/usr/bin/env python
# coding: utf-8

# # Exercise 6 
# 
# 19M51935 Tsamara Tsani
# 
# In this report I examine the corona virus cases in **Iran**.
# 
# First of all, we loaded the Iran cases from dataset.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

selected_country='Iran'
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df = data[data['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)


# # Daily Cases Plot
# 
# Then, we created the daily cases plot showing the daily confirmed, daily deaths and daily recovery in time series.

# In[ ]:


df['daily_confirmed']=df['Confirmed'].diff()
df['daily_deaths']=df['Deaths'].diff()
df['daily_recovery']=df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovery'].plot()
plt.legend()
plt.show()


# # Interactive Plot
# 
# Interactive plot is created to examine the data clearly in the day to day basis.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily Confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily Deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily Recovery')

layout_object = go.Layout(title='Iran daily cases 19M51935',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Iran_daily_cases.html')


# In the above graph we see that Iran has experienced **the first peak of COVID19 confirmed cases on March 30**. The cases has been slowdown after that. However, it begin to increase again and reach **the second peak on June 4**. The country currently experiencing **the second wave** of Coronavirus cases. The peak of recovery cases is shown on April 11. Appoximately 2 weeks after the peak of confirmed cases. This might be because the incubation period of coronavirus is said to be about 2 weeks. The peak of recovery cases maybe due to the peak of confirmed cases in the 2 weaks before.
# To grap the statistics of the data, we can conduct the following analysis.

# In[ ]:


print(df.describe())


# Here we can see the mean, median and percentile of the data. We found that **the average daily death in Iran is 75.93 people**. We will examine the severity of this by adding fatality rate parameter in the later section.

# # Informative Table
# 
# Here we highlighted the maximum number of each categories.
# 
# 

# In[ ]:


df1 = df
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='jet').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)


# In the above interactive plot we can not really see the maximum daily death number because it is relatively small compared to the daily confirmed cases and daily recovery cases. However, using this informative table, we can clearly see the maximum value highlighted in red color. We confirmed that **the highest death cases of 158 cases has occurred on April 4th, 2020.**

# # Iran Ranking
# Next, we observed Iran Ranking compared to the whole world

# In[ ]:


latest = data[data['ObservationDate']=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()
print('Ranking of Iran Confirmed Cases: ', latest[latest['Country/Region']=='Iran'].index.values[0]+1)

#Fatality Rate is defined as the ration of death per confirmed cases in certained period of time in the country
latest['Fatality_rate']=latest['Deaths']/latest['Confirmed']
print('Iran Fatality Rate:', latest[latest['Country/Region']=='Iran']['Fatality_rate'].values)

#Fatality Rate Ranking
latest_fat = latest.sort_values(by='Fatality_rate',ascending=False).reset_index()
print('Ranking of Iran Fatality Rate: ', latest_fat[latest_fat['Country/Region']=='Iran'].index.values[0]+1)


# From the above analysis we found that Iran is in the **11th Ranking** of Confirmed cases in the world. Fatality rate of the virus in the country is **0.04744**, which means that there are 4.7 deaths in every 100 confirmed cases. However, the fatality rate ranking of this country is **45**, which is relatively moderate compared to other countries.

# # Government measures
# 
# On late February, when the pandemic begin to spread in Iran, the country official has denied to conduct quarantine or travel ban measures. Even the country's leader has once mentioned a [conspiracy theory](https://www.independent.co.uk/news/world/middle-east/iran-coronavirus-us-target-country-special-version-covid19-a9417206.html) that COVID19 was created by US and specifically targetted Iran. The government also announce warning to its citizen that it will [punish anyone who spread false rumor](https://www.rferl.org/a/iran-says-3600-arrested-for-spreading-coronavirus-related-rumors/30583656.html) about the pandemic.
# 
# Without any travel restriction and social distancing measures, coronavirus cases in the country has continously growing. Annualy, the country celebrated the Persian New Year in the [Nowruz Festival](https://www.reuters.com/article/us-health-coronavirus-iran-nowruz/coronavirus-keeps-families-apart-on-eve-of-iranian-new-year-idUSKBN2162G2) from March 20 to April 3. The government set travel ban during this period of time, as many people may travel to tourists destinations or hometown. However, a lot of people disobyed the restriction, resulting on the first peak of confirmed cases on March 30, as shown in the interactive plot. The government then further restricted the social distancing measures. This contributed to the slowdown of the coronavirus cases in the following weeks.
# 
# The government began to open business, schools and mosques in the Mid May. However this has lead the country to face [the second wave](https://www.aljazeera.com/news/2020/06/iran-braces-coronavirus-wave-surge-infections-200605211903567.html) of coronavirus to the present time. 
# 
# Compared to other countries, the fatality rate of coronavirus in Iran stood in the 45th rank. The country claimed that this relatively moderate ranking is link to[ high portion of young population in the country and also mass testing](http://https://financialtribune.com/articles/national/103737/low-covid-19-death-toll-linked-to-young-population-mass-testing) which was conducted in the country.
# 
# However, many foreign media [doubted the credibility ](https://cpj.org/2020/03/amid-coronavirus-pandemic-iran-covers-up-crucial-i/)of confirmed cases and death cases announced by the Iranian government.
# 
# 
