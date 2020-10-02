#!/usr/bin/env python
# coding: utf-8

# **Dolgormaa Banzragch #20M51866**
# 
# **Exercise 6**
# 
# I chose china for this project
# 

# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

China = 'Mainland China'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
#print(np.unique(df['Country/Region'].values))
#print(df.columns)

#dfc means Data Frame China
dfc = df[df['Country/Region']==China]
dfc = dfc.groupby('ObservationDate').sum()
#print(dfc)

#creating new columns
dfc['Daily_Confirmed']=dfc['Confirmed'].diff()
dfc['Daily_Recovered']=dfc['Recovered'].diff()
dfc['Daily_Deaths']=dfc['Deaths'].diff()

#simple_visualization 
dfc['Daily_Confirmed'].plot()
dfc['Daily_Recovered'].plot()
dfc['Daily_Deaths'].plot()
plt.show()


# #The bizzare thing is that the daily recovered plot has a minus value in it (minus number of people ????) maybe the data was corrupted. 

#  **Creating an Interactive Graph**

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

layout_object = go.Layout(title='China daily cases #20M51866', xaxis=dict(title = 'date'), yaxis=dict(title='number of people'))
daily_confirmed_objets = go.Scatter(x=dfc.index, y=dfc['Daily_Confirmed'].values, name='Daily_Confirmed')
daily_deaths_objets = go.Scatter(x=dfc.index, y=dfc['Daily_Deaths'].values, name='Daily_Deathss')
daily_recovered_objets = go.Scatter(x=dfc.index, y=dfc['Daily_Recovered'].values, name='Daily_Recovered')
fig = go.Figure(data=[daily_confirmed_objets, daily_deaths_objets, daily_recovered_objets], layout=layout_object)
iplot(fig)
#fig.write_html('China_daily_cases_#20M51866.html')


# **Creating an Informative table**
# ****

# In[ ]:


df1 = dfc#[['Daily_Confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('Daily_Confirmed').set_caption('Daily Summaries')
display(styled_object)
#f = open('Table_20M51866.html','w')
#f.write(styled_object.render())


# **Calculating global ranking**

# In[ ]:


#Got the idea from Ryza's comment
latest = df[df['ObservationDate']=='06/12/2020']
#06/12/2020 was the latest possible date
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of China: ', latest[latest['Country/Region']=='Mainland China'].index.values[0]+1)


# **Analysis**
# Currently China is ranked at 18th in terms of the number of cases according to the latest possible date of the data-set that I am using. 
# Despite being th efirst country that had ccovid-19 outbreak the situation has been handled very fast(apprent from the plot).
# The death by covid-19 was surprisingly low in China compared to other countries and the highest date number recorded was in 04/17/2020 (1290 Deaths).
# Since 02/28.2020  (3393 cases), the covid-19 daily cases has been decreasing dramatically in China and starting from April the dily cases are nearly none relative to the previous months. I think the reason why they were able to controll the situation this well was because Chinese goverment has taken it seriously from the beggining and started serious quarantine early. 
# 
