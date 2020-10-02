#!/usr/bin/env python
# coding: utf-8

#  # This is excercise 6
#  I look at Corona cases for Netherlands.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)


selected_country='Netherlands'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(df['Country/Region'].values)
df=df[df['Country/Region']==selected_country]
df=df.groupby('ObservationDate').sum()
print(df)


# Note : This 'df' is  actually a cumlative information.
# 
# June 12,2020 shows the total cases since initial date.
# 
# How can we caluculate day-to-day cases

# In[ ]:


df['daily_confirmed']= df['Confirmed'].diff()
df['daily_deaths']= df['Deaths'].diff()
df['daily_recovery']= df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_deaths'].plot()
df['daily_recovery'].plot()
plt.show()


# # How about we make a interactive chart?
# 
# To do this, we will need to load two of the following modules.

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object=go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object=go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object=go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovery')

layout_object=go.Layout(title='Netherlands daily cases 19M58392',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig=go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Netherlands_cases_19M58392.html')


# # Condition
# 
# In the Netherlands, the spread of infection began around March 10. And the peak came a month later, and a month later the infection settled down.
# Currently, the number of infected people has not changed and has remained constant.
# The day with the most confirmed cases of infection is April 10. And the highest number of deaths is April 7.
# Also, the recoverer's data is obviously wrong.

# # How can wemake informative table?
# 
# Maybe color the entries showing large values as some bright color, low values as some dark color.

# In[ ]:


df1=df#[['daily_confirmed']]
df1=df1.fillna(0.)
styled_object=df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f=open('table_19M58392.html','w')
f.write(styled_object.render())


# # How can we caluculate global ranking?
# 
# Maybe selecting only latest date???

# In[ ]:


df=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df1=df.groupby(['ObservationDate','Country/Region']).sum()
df2=df[df['ObservationDate']=='06/06/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()
df=df2.groupby(['Country/Region']).sum()
print('Ranking of Netherlands:',df2[df2['Country/Region']=='Netherlands'].index.values[0]+1)


# # News Summary
# 
# In the Netherlands. The first death was confirmed on March 6th. A week later, the museum was closed and events for more than 100 people were canceled. In Japan, the school was closed first, but the school wasn't closed yet because the Netherlands decided that children weren't heavily affected by COVID-19.
# But a few days later, the closure of schools, restaurants and cafes was announced. We decided to do it until April 6th, including the various measures announced so far.
# On 16 March, it was decided to implement border measures across the EU, and restrictions on travel to the EU began.
# A week later, Intelligent Lockdown was decided. It has also been announced that the deadline for banning meetings that has been announced will be extended to June 1. In addition, there were various announcements. For example, it is forbidden for three or more people to act without taking a distance, or to close work that involves physical contact. There were also announcements that they would be fined.
# There was an announcement on 22nd April about the reopening of the elementary school. Specifically, the school started on May 11th, with half attendance and remote lessons combined in the classroom.
# And on May 6th, it was announced that junior high school will be reopened and jobs that involve contact will be resumed. Middle school was reopened on June 1st, and from May 11th, it was allowed to resume occupations involving contact. I think it was restarted as scheduled, but I could not find the news.
# In addition, restaurants, movie theaters, and museums will be reopened on June 1.
# In addition, it was mandatory to wear a mask when using public transportation. Violators can also be fined.

# # Reference
# 
# https://oranda.jp/info/corona-timeline/
# https://oranda.jp/info/lockdown/
# https://www.rivm.nl/en/novel-coronavirus-covid-19/dutch-response-to-coronavirus
# https://www.government.nl/topics/coronavirus-covid-19/tackling-new-coronavirus-in-the-netherlands/basic-rules-for-everyone

# 

# In[ ]:




