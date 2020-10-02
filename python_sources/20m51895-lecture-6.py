#!/usr/bin/env python
# coding: utf-8

# # This is exercise 6
#  ## I would like to explore the situation in Germany.

# In[ ]:


import numpy as np
import matplotlib. pyplot as plt
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import plotly.graph_objs as go

np.set_printoptions(threshold=np.inf)
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header = 0)
df

selected_country = 'Germany'
filt1 = df['Country/Region'] == selected_country
df1 = df[filt1]
df2 = df1.groupby('ObservationDate').sum()
df2


# In[ ]:


df2['daily_confirmed'] = df2['Confirmed'].diff()
df2['daily_deaths'] = df2['Deaths'].diff()
df2['daily_Recovered'] = df2['Recovered'].diff()
df2


# In[ ]:


df2 = df2.fillna(0.)
df2.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Pastel1_r',subset=["Deaths"])                        .background_gradient(cmap='YlOrBr',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["daily_confirmed"])                        .background_gradient(cmap='Reds',subset=["daily_deaths"])                        .background_gradient(cmap='Greens',subset=["daily_Recovered"])


# # CURVE
# Acoording to the picture below, we could see that the peak of the COVID-19 outbreak in Germany came at the end of March. And there has been no significant increase in the last two months, which means the German goveronment seems has controled the spread of the virus.

# In[ ]:


daily_confirmed_object = go.Scatter(x = df2.index,y = df2["daily_confirmed"].values,name = "daily_confirmed")
daily_deaths_object = go.Scatter(x = df2.index,y = df2["daily_deaths"].values,name = "daily_deaths")
layout_object = go.Layout(title ='Germany daily cases 20M51895',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)
iplot(fig)
#fig.write_html('GERMANY_Daily_cases.html')


# # Global ranking

# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
data.index=data['ObservationDate']
latest = data[data.index=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 


print('Ranking of Germany: ', latest[latest['Country/Region']=='Germany'].index.values[0]+1)


# # DISCUSSION
# Although Germany ranks tenth in the world in the number of cases confirmed, the spread of the virus seems had stoped since the end of March. The curve is clearly falttening. And the proportion of fatalities is lower in Germany than in many other countries.
# I see three reasons why Germany is coming throuhg this crisis relatively well. 
# 
# First, the German health-care system was in good shape going into the crisis; everyone has had full access to medical care.
# 
# Second, Germany was not the first country to be hit by the virus, and thus had time to prepare. The government took the COVID-19 threat seriously from the beginning.
# 
# Third, Germany is home to many laboratories that can test for the test, and to many distinguished researchers in the field.
# 
# References:  
# 1.https://en.wikipedia.org/wiki/COVID-19_pandemic_in_Germany  
# 2.https://www.youtube.com/watch?v=z9eSeRgEIH4
