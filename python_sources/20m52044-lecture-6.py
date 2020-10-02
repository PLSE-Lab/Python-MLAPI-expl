#!/usr/bin/env python
# coding: utf-8

# # **COVID-19 cases in South Korea (Linux Farungsang 20M52044)**
# I choose to focusing on South Korea due to its impressive initial outbreak early prevention from the government. South Korea performed a mass testing system which played a very important role in terms of outbreak prevention. In this assignment, I will try to show Daily confirmed cases, Daily Deaths, Daily Recoveries,and global ranking to satisfy this week's task. As such, the explanation for every parts can be seen in the discussion section.

# # Table 1. Time-Series Table of Daily confirmed cases, Daily Deaths, and Daily Recoveries in South Korea (Until 2020/06/12) 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='South Korea'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
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


# # Figure 1. Interactive charts for Daily Confirmed Cases, and Daily Deaths in South Korea

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

layout_object = go.Layout(title='South_Korea_daily_cases_20M52044',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
fig.write_html('South_Korea_daily_cases_20M52044.html')


# # Table 2. Colored Table of Daily confirmed cases, Daily Deaths, and Daily Recoveries in South Korea (Until 2020/06/12)
# 
# In this Table, the bright color indicates large value and the dark color indicates low value of Daily confirmed cases, Daily Deaths, and Daily Recoveries in South Korea.

# In[ ]:


df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M52044.html','w')
f.write(styled_object.render())


# # South Korea global ranking

# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.index=data['ObservationDate']
data = data.drop(['SNo','ObservationDate'],axis=1)

latest = data[data.index=='06/02/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 


print('Ranking of South Korea: ', latest[latest['Country/Region']=='South Korea'].index.values[0]+1)


# # Discussion
# 
# **1. Construct a time-series and colored tables of the following: Daily confirmed cases, Daily deaths, Daily recoveries**
#   
#   The time-series and colored tables of Daily confirmed cases, Daily deaths, Daily recoveries can be seen in Table 1 and Table 2.
#     
# **2. What is the global ranking of that country in terms of the number of cases?**
#   
#   In the section South Korea global ranking, the result has been shown that as of 2020/06/12, the global ranking of South Korea in terms of the number of cases is 50. 
#     
# **3. Analyze from the time-series whether the corona cases are decreasing with time.**
#   
#   Figure 1 has shown that the corona cases are decreasing with time. However, time is not the only factor that makes the number of corona cases reduce. Detialed explanation will be in number 5. 
#     
# **4. Which dates had the largest daily cases? Is it still increasing? Not clear? Visual inspection is okay.**
#   
#   The largest Daily cases can be seen from Figure 1 on 2020/03/03 with the number of daily cases of 851. It is very clear from visual inspection in this figure that the number of daily cases in South Korea is not significantly increasing anymore.
#     
# **5. Discuss how the national government of the country is addressing the issue by summarizing from the news.**
#   
#   According to Japan Times, South Korea's main focus was aggressive testing and contact tracing. This resulting in more than 600 testing sites capable of testing 20,000 people per day from its 51 million people nationwide. Consequently, the daily confirmed cases can go down rapidly after the outbreak has been found. We can see a clear result of aggressive testing and contact tracing in figure 1. Moreover, this method of combating the outbreak is also recommended by WHO Strategic and Technical Advisory Group for Infectious Hazards (STAG-IH) (Bedford et al., 2020). In brief, countries must rapidly response, and should consider combination of contact tracing, promote personal hygiene protection, strong infection prevention in health facilities, and postpone/cancel large-scale public gatherings.  
# 

# # References
# 
# https://www.japantimes.co.jp/opinion/2020/05/05/commentary/world-commentary/south-korea-stopped-covid-19-early/#.XuYg1UUzZ3g
# 
# Bedford, J., Enria, D., Giesecke, J., Heymann, D. L., Ihekweazu, C., Kobinger, G., ... & Ungchusak, K. (2020). COVID-19: towards controlling of a pandemic. *The Lancet, 395*(10229), 1015-1018.
