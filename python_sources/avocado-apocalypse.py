#!/usr/bin/env python
# coding: utf-8

# ![](http://www.organicfacts.net/wp-content/uploads/avocadofruit.jpg)

# 

# # Context

# 
# 
# It is a well known fact that Millenials LOVE Avocado Toast. It's also a well known fact that all Millenials live in their parents basements.
# 
# Clearly, they aren't buying home because they are buying too much Avocado Toast!
# 
# But maybe there's hope... if a Millenial could find a city with cheap avocados, they could live out the Millenial American Dream.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import calendar

from datetime import datetime

import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
from plotly import tools

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


# In[ ]:


avocado=pd.read_csv("../input/avocado.csv")
avocado=avocado.copy()


# # Basic details of the data

# In[ ]:


avocado.info()


# In[ ]:


print("shape of data",avocado.shape)


# In[ ]:


avocado.columns


# The dataset contains 14 columns.We can ignore the id column.So we have 13 features to consider.

# ## Is there any missing values?
# 

# In[ ]:


avocado.isnull().sum()


# The dataset is clean and neat,there is no missing values here.We will inspect for outliers later.

# We will split the date column into year,month and day for further analysis.

# In[ ]:


avocado['year']=avocado['Date'].apply(lambda x : x.split("-")[0])
avocado['month']=avocado['Date'].apply(lambda x : calendar.month_name[datetime.strptime(x,"%Y-%m-%d").month])
avocado['day']=avocado['Date'].apply(lambda x : calendar.day_name[datetime.strptime(x,"%Y-%m-%d").weekday()])


# In[ ]:


avocado.head(7)


# In[ ]:


avocado[avocado['day']!='Sunday']


# It is very interesting to note the all the data recorded here are on sundays.

# # Convensional or Organic ?

# In[ ]:


typeof=avocado.groupby('type')['Total Volume'].agg('sum')


# In[ ]:


values=[typeof['conventional'],typeof['organic']]
labels=['conventional','organic']

trace=go.Pie(labels=labels,values=values)
py.iplot([trace])


# Only 2% of the avocadros sold are organic.We will find out why.

# In[ ]:


conv=avocado[avocado['type']=='conventional'].groupby('year')['AveragePrice'].agg('mean')
org=avocado[avocado['type']=='organic'].groupby('year')['AveragePrice'].agg('mean')

trace1=go.Bar(x=conv.index,y=conv,name="conventional",
             marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

trace2=go.Bar(x=conv.index,y=org,name="organic",
             marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

data=[trace1,trace2]
layout=go.Layout(barmode="group",title="Comaparing organic and conventional avocadro prices over years",
                yaxis=dict(title="mean price"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# The prices of organic avocadros are always higher than conventional avocadros.                                    
# That can be the main reason why convensional avocadros are dominant in the market.
# 

# In[ ]:





# In[ ]:


conv=avocado[avocado['type']=='conventional'].groupby('year')['Total Volume'].agg('mean')
org=avocado[avocado['type']=='organic'].groupby('year')['Total Volume'].agg('mean')

trace1=go.Bar(x=conv.index,y=conv,name="conventional",
             marker=dict(
        color='rgb(58,200,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)

trace2=go.Bar(x=conv.index,y=org,name="organic",
             marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.7)



data=[trace1,trace2]

layout=go.Layout(barmode="group",title="Comaparing  mean Volume of organic and conventional avocadro  sold over years",
                yaxis=dict(title="Volume sold"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# In[ ]:


trace3=go.Scatter(x=conv.index,y=conv,name="conventional")
trace4=go.Scatter(x=org.index,y=org,name='organic')
fig=tools.make_subplots(rows=1,cols=2)
fig.append_trace(trace3,1,1)
fig.append_trace(trace4,1,2)

fig['layout'].update(height=500, title="", barmode="stack", showlegend=True,yaxis=dict(title="Mean Volume sold"))
py.iplot(fig)


# As you can see there is no downward slope in the  year vs mean volume of avocados sold

# # 2016 vs 2017

# In[ ]:


avocado['Date']=avocado['Date'].apply(lambda x : datetime.strptime(x,'%Y-%m-%d').date())


# In[ ]:



date_16=avocado[avocado['year']=='2016'].sort_values(by='Total Volume')
date_17=avocado[avocado['year']=='2017'].sort_values(by='Total Volume')
trace1=go.Bar(x=date_16['Date'],y=date_16['Total Volume'],name="2016")
trace2=go.Bar(x=date_17['Date'],y=date_17['Total Volume'],name='2017')
data=[trace1,trace2]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data)
py.iplot(fig)


# Total volume of avocados sold in different weeks of 2016 and 2017 is depicted in the above            
# figure.There seems to be no serious declines in the market.

# ## Which varient of avocado is more popular ?

# In[ ]:


types=date_17.groupby('month')[['4046','4225','4770']].agg('sum')
types=types.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]
trace1=go.Bar(x=types.index,y=types['4046'],name=' PLU 4046')
trace2=go.Bar(x=types.index,y=types['4225'],name=' PLU 4225')
trace3=go.Bar(x=types.index,y=types['4770'],name=' PLU 4770')

data=[trace1,trace2,trace3]
layout=go.Layout(barmode="group")
fig=go.Figure(data=data)
py.iplot(fig)


# ## Compairing popular varients 

# In[ ]:



types1=date_16.groupby('month')[['4046','4225','4770']].agg('sum')
types1=types1.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]


# In[ ]:


trace1=go.Scatter(x=types.index,y=types['4046'],name=' 2017PLU 4046',line=dict(color='green'))
trace2=go.Scatter(x=types.index,y=types['4225'],name=' 2017PLU 4225',line=dict(color='green'))
trace3=go.Scatter(x=types1.index,y=types1['4046'],name=' 2016PLU 4046',mode='markers+lines',line=dict(color='blue'))
trace4=go.Scatter(x=types1.index,y=types1['4225'],name=' 2016PLU 4225',mode='markers+lines',line=dict(color='blue'))

data=[trace1,trace2,trace3,trace4]

fig=go.Figure(data=data)
py.iplot(fig)


# We will inspect if there was any serious increase or deacrase in average prices of avocados

# ## Price distribution over years

# In[ ]:


price_16=date_16.groupby('month')['AveragePrice'].agg('mean')
price_16=price_16.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]

price_17=date_17.groupby('month')['AveragePrice'].agg('mean')
price_17=price_17.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]

price_15=avocado[avocado['year']=='2015'].groupby('month')['AveragePrice'].agg('mean')
price_15=price_15.loc[['January','February','March',"April",'May','June',"July","August",'September','October','November',"December"]]


# In[ ]:



trace2=go.Scatter(x=price_17.index,y=price_17,name='2017')
trace1=go.Scatter(x=price_16.index,y=price_16,name='2016')
trace3=go.Scatter(x=price_15.index,y=price_15,name='2015')
data=[trace2,trace1,trace3]

fig=go.Figure(data=data)
py.iplot(fig)


# In[ ]:





# In[ ]:


price_15=avocado[avocado['year']=='2015']['AveragePrice']
price_16=avocado[avocado['year']=='2016']['AveragePrice']
price_17=avocado[avocado['year']=='2017']['AveragePrice']
price_18=avocado[avocado['year']=='2018']['AveragePrice']

trace1=go.Box(y=price_15,name="2015")
trace2=go.Box(y=price_16,name='2016')
trace3=go.Box(y=price_17,name='2017')
trace4=go.Box(y=price_18,name='2018')
data=[trace1,trace2,trace3,trace4]
layout=go.Layout(title='Box plot of price per avocado ',yaxis=dict(title='price in dollars'))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# In 2017 there is a  rise in mean prices of avocadros.we will inspect that further.

# ## Box plot of Total sales

# In[ ]:


price_15=avocado[avocado['year']=='2015']['Total Volume']
price_16=avocado[avocado['year']=='2016']['Total Volume']
price_17=avocado[avocado['year']=='2017']['Total Volume']
price_18=avocado[avocado['year']=='2018']['Total Volume']

trace1=go.Box(y=price_15,name="2015")
trace2=go.Box(y=price_16,name='2016')
trace3=go.Box(y=price_17,name='2017')
trace4=go.Box(y=price_18,name='2018')
data=[trace1,trace2,trace3,trace4]
layout=go.Layout(title=" sales in diiferent years ",yaxis=dict(title="Volume sold"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)


# In[ ]:


avocado.groupby(['region','year'],as_index=False)['Total Volume'].agg('mean')


# In[ ]:



    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




