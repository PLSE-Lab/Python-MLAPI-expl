#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# insights about corona virus


# In[ ]:


import warnings 

warnings.filterwarnings("ignore") 


# Mainly now a days corona virus is dangerous virus to cultivate society. In this kernel you will get brief understaing of how exactly it happend and where is the most spread areas etc. hope you are enjoying my kernel.
#  ### Please do upvote and comment

# First we can check dataset format and how can we load the data from kaggle. Brief understanding of corona virus and for awareness of intuition this diease please go link: https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public 
# 
# #### Be safe and keep learning

# In[ ]:


get_ipython().system('ls ../input/novel-corona-virus-2019-dataset/')


# In[ ]:


#to read the data 2019_nCoV_data.csv
import numpy as np
import pandas as pd

data_corona = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
# display top five rows in dataset
data_corona.head()


# In[ ]:


#display last five rows in dataset
data_corona.tail()


# By observing top and bottom of rows conclude that shape of the size has to check. The sol number not gonna much important for analysis so removed.

# In[ ]:


data_corona['Date'] = data_corona['Date'].apply(pd.to_datetime).dt.normalize()
data_corona['Last Update'] = data_corona['Last Update'].apply(pd.to_datetime).dt.normalize()
data_corona.head()


# In[ ]:


#shape of the dataset
data_corona.shape


# In[ ]:


del data_corona['Sno']


# In[ ]:


#now lets see dataframe
data_corona.head()


# In[ ]:


data_corona.describe()


# first lets  understand the dataset consists of state, country,deaths,recovered,confirmed,date are fields in dataframe.So lets do analysis on country wise.

# ### Country wise analysis so that we can observe that which is most populate country to suffer corona.** 

# In[ ]:


# basic visualize to help counts in each country
data_corona['Country'].value_counts()


# As observation of above we can conclude that Mainland China has most number of cases across world and second US etc
#   ####*  - very rare in brazil and mexio,egypt so that not comes into picture.
#   ###*   - some how we can come up analysis on Mainland China and US

# In[ ]:


#list of all countries that present in country column
countries = data_corona['Country'].unique()
print(countries)

print("\nTotal countries affected by virus: ",len(countries))


# > > ## EDA on China and US**

# In[ ]:


data_corona.groupby('Country')['Confirmed'].sum()


# In[ ]:


data_corona.groupby(['Country','Province/State']).sum().head(50)


# In[ ]:


data_corona.groupby(['Country','Date']).sum().head(50)


# In[ ]:


data_corona.groupby(['Country','Last Update']).sum().head(50)


# bar plots
# pie plots 
# bar plots

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Import dependencies
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
from plotly import subplots
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from datetime import date
from fbprophet import Prophet
import math


# In[ ]:


# Exploring word cloud based on STATE value
from wordcloud import WordCloud
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(data_corona.Country))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('country.png')
plt.show()


# In[ ]:


import pandas_profiling
pandas_profiling.ProfileReport(data_corona)


# In[ ]:


data_corona.groupby('Date')['Confirmed','Recovered','Deaths'].sum()


# In[ ]:


#rate of confirmed cases and recovered cases

#(confirmed*100)/total no of cases


# In[ ]:


# Ploting daily updtes for 
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(x=data_corona.Date, y=data_corona.Confirmed, mode="lines+markers", name=f"MAX. OF {int(data_corona.Confirmed.max()):,d}" + ' ' + "CONFIRMED",line_color='red'))
fig_d.add_trace(go.Scatter(x=data_corona.Date, y=data_corona.Recovered, mode="lines+markers", name=f"MAX. OF {int(data_corona.Recovered.max()):,d}" + ' ' + "RECOVERED",line_color='deepskyblue'))
fig_d.add_trace(go.Scatter(x=data_corona.Date, y=data_corona.Deaths, mode="lines+markers", name=f"MAX. OF {int(data_corona.Deaths.max()):,d}" + ' ' + "DEATHS",line_color='Orange'))
fig_d.update_layout(template="ggplot2",title_text = '<b>Daily numbers for Confirmed, Death and Recovered </b>',
                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'), showlegend=True)
fig_d.update_layout(
    legend=dict(
        x=0.01,
        y=.98,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=7,
            color="Black"
        ),
        bgcolor="White",
        bordercolor="black",
        borderwidth=1
    ))
fig_d.show()


# In[ ]:


fig = px.bar(data_corona[['Country', 'Confirmed']].sort_values('Confirmed', ascending=True), 
             y="Confirmed", x="Country", color='Country', 
             log_y=True, template='ggplot2', title='Recovered Cases')
fig.show()


# In[ ]:


fig = px.bar(data_corona[['Country', 'Recovered']].sort_values('Recovered', ascending=True), 
             y="Recovered", x="Country", color='Country', 
             log_y=True, title='Confirmed Cases')
fig.show()


# In[ ]:


fig=px.bar(data_corona[['Country','Deaths']].sort_values('Deaths',ascending=True),
       x="Country",y="Deaths",color='Country',
       log_y=True,title='Death cases')

fig.show()


# In[ ]:


#looking fro lastest data world wide
get_ipython().system('ls ../input/novel-corona-virus-2019-dataset')


# In[ ]:


data_corona = data_corona[data_corona['Confirmed'] != 0]


# In[ ]:


plt.figure(figsize=(30,10))
sns.barplot(x='Country',y='Confirmed',data=data_corona,color='red')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(30,10))
sns.barplot(x='Country',y='Recovered',data=data_corona,color='blue')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(30,10))
sns.barplot(x='Country',y='Deaths',data=data_corona,color='pink')
plt.tight_layout()


# In[ ]:


data_corona.groupby("Country")["Deaths"].plot.bar()


# In[ ]:



g = sns.PairGrid(data_corona)
g.map(plt.scatter);


# In[ ]:


data_2 = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data_2.head(4)


# In[ ]:


data_2.describe()


# In[ ]:


import pandas_profiling
pandas_profiling.ProfileReport(data_2)


# In[ ]:


del data_2["SNo"]

data_2.head(3)


# In[ ]:


data_2.groupby('Province/State')['Deaths'].sum().reset_index().sort_values(by=['Deaths'],ascending=False).head()


# In[ ]:


data_corona.groupby('Province/State')['Deaths'].sum().reset_index().sort_values(by=['Deaths'],ascending=False).head()


# In[ ]:


# Reading Data

covid_open=pd.read_csv('../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
covid_confirmed=pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
covid_death= pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')


# In[ ]:


print(covid_open.shape)
covid_open.describe()


# In[ ]:


print("The shapeof the data death cases ",covid_death.shape)
covid_death.describe()


# In[ ]:


print("The shapeof the data confirmed cases ",covid_confirmed.shape)
covid_confirmed.describe()


# In[ ]:


no_cases = data_corona[data_corona['Confirmed']==0]
print(no_cases.count())
          


# In[ ]:


no_cases = data_corona[data_corona['Recovered']!=0]
no_cases
          


# In[ ]:


#expect china display remaining countries

no_china=data_corona=data_corona[data_corona['Country']!='China']

no_china.head(20)


# In[ ]:


no_china

