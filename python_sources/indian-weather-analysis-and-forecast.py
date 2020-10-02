#!/usr/bin/env python
# coding: utf-8

# [1. Peek Into The Dataset](#1)
# 
# 
# [2. Temprature throughout time](#2)
# 
# 
# [3. Warmest/Coldest/Average](#8)
# 
# 
# [4. Alarming Global Warming !!!](#4)
# 
# 
# [5. Monthly Tempratures throught history](#5)
# 
# 
# [6. Seasonal Analysys](#6)
# 
# 
# [7. Forecasing](#7)

# In[ ]:


import numpy as np ## For Linear Algebra
import pandas as pd ## To Work With Data
## For visualizations I'll be using plotly package, this creates interesting and interective visualizations.
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime ## Time Series analysis.


# In[ ]:


## Reading The data.
df = pd.read_csv("../input/weather-data-in-india-from-1901-to-2017/Weather Data in India from 1901 to 2017.csv")


# <a class='anchor' id='1'></a>
# ## Peek into the Dataset

# In[ ]:


df.head() ## This will show us top 5 rows of the dataset by default.


# #### Ahh, we've got an unexpected column named 'Unnamed: 0'. Well, this is a very common problem. We face this when our csv file has an index column which has no name. here is how we can get rid of it.

# In[ ]:


df = pd.read_csv("../input/weather-data-in-india-from-1901-to-2017/Weather Data in India from 1901 to 2017.csv", index_col=0)


# In[ ]:


df.head() ## This is how our dataset really looks:


# #### Now, we'll make an attribute that would contain date (month, year). So that we could get temprature values with the timeline.

# In[ ]:


df1 = pd.melt(df, id_vars='YEAR', value_vars=df.columns[1:]) ## This will melt the data
df1.head() ## This is how the new data looks now:


# In[ ]:


df1['Date'] = df1['variable'] + ' ' + df1['YEAR'].astype(str)  
df1.loc[:,'Date'] = df1['Date'].apply(lambda x : datetime.strptime(x, '%b %Y')) ## Converting String to datetime object
df1.head()


# ## Temprature throught time.
# 
# <a class="anchor" id="2"></a>

# In[ ]:


df1.columns=['Year', 'Month', 'Temprature', 'Date']
df1.sort_values(by='Date', inplace=True) ## To get the time series right.
fig = go.Figure(layout = go.Layout(yaxis=dict(range=[0, df1['Temprature'].max()+1])))
fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Temprature']), )
fig.update_layout(title='Temprature Throught Timeline:',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.update_layout(xaxis=go.layout.XAxis(
    rangeselector=dict(
        buttons=list([dict(label="Whole View", step="all"),
                      dict(count=1,label="One Year View",step="year",stepmode="todate")                      
                     ])),
        rangeslider=dict(visible=True),type="date")
)
fig.show()


# #### Woah... what is this now??? on the first look, this gives a feel that e've done something wrong. But, that's not true. On a closer look, by clicking on "One Year View", we can see that the graph seems distorted because this is how the values really are. The temprature varies every year with moths.
# ### Insights:
# - May 1921 has been the hottest month in india in the history. What could be the reason ?
# - Dec, Jan and Feb are the coldest months. One could group them together as "Winter".
# - Apr, May, Jun, July and Aug are the hottest months. One could group them together as "Summer".
# 
# #### But, since this is not how seasons work. We have four main seasons in India and this is how they are grouped:
# - Winter : December, January and February.
# - Summer(Also called, "Pre Monsoon Season") : March, April and May.
# - Monsoon : June, July, August and September.
# - Autumn(Also called "Post Monsoon Season) : October and November.
# 
# #### We also will stick to these seasons for our analysis.

# ## Warmest /Coldest/Average :
# 
# <a class='anchor' id='8'></a>

# In[ ]:


fig = px.box(df1, 'Month', 'Temprature')
fig.update_layout(title='Warmest, Coldest and Median Monthly Tempratue.')
fig.show()


# ### Insights:
# - January has  the coldest Days in an Year.
# - May has the hottest days in an Year.
# - July is the month with least Standard Daviation which means, temprature in july vary least. We can expect any day in july to be a warm day.

# In[ ]:


fig = px.scatter(df1, 'Date', 'Temprature', color='Month', 
                 color_discrete_sequence= ['blue','darkorchid','black','tomato','red','orange',
                                          'gold','crimson','hotpink','olive','darkgreen','dodgerblue'])
fig.update_layout(title='Temprature Clusturs of Months:',
                 xaxis_title = 'Time', yaxis_title='Temprature')
fig.show()


# ### Insights:
# - Despite having 4 seasons we can see 3 main clusturs based on tempratures.
# - Jan, Feb and Dec are the coldest months.
# - Apr, May, Jun, Jul, Aug and Sep; all have hotter tempratures.
# - Mar, Oct and Nov are the months that have tempratures neither too hot nor too cold.

# In[ ]:


fig = px.histogram(x=df1['Temprature'], nbins=200, histnorm='density')
fig.update_layout(title='Frequency chart of temprature readings:',
                 xaxis_title='Temprature', yaxis_title='Count')


# - There is a cluster from 26.2-27.5 and mean temprature for most months during history has been between 26.8-26.9 
# ### Let's see if we can get some insights from yearly mean temprature data. I am going to treat this as a time series as well.

# ## Yearly average temprature.
# 
# <a class="anchor" id="4"></a>

# In[ ]:


df['Yearly Mean'] = df.iloc[:,1:].mean(axis=1) ## Axis 1 for row wise and axis 0 for columns.
fig = go.Figure(data=[
    go.Scatter(name='Yearly Tempratures' , x=df['YEAR'], y=df['Yearly Mean'], mode='lines'),
    go.Scatter(name='Yearly Tempratures' , x=df['YEAR'], y=df['Yearly Mean'], mode='markers')
])
fig.update_layout(title='Yearly Mean Temprature :',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.show()


# ### We can see that the issue of global warning is true.
# - The yearly mean temprature was not incresing till 1980. It was only after 1979 that we can see the gradual increse in yearly mean temprature.
# - After 2015, yearly temprature has incresed drastically.
# - But, There are some problems in this figure.
# - We are seeing a monthly like up-down pattern in yearly tempratures as well.
# - This is not understandable. Because with months, we have a phenominan of seasons and the earth the revolving around sun in a eliptic path. But this pattern is not expected in yearly temprature.

# ## Monthly tempratues throught history.
# 
# <a class="anchor" id="5"></a>

# In[ ]:


fig = px.line(df1, 'Year', 'Temprature', facet_col='Month', facet_col_wrap=4)
fig.update_layout(title='Monthly temprature throught history:')
fig.show()


# #### We can see clear positive trendlines. Let's see if we could find any trend in seasonal mean tempratures.

# ## Seasonal Analysis
# 
# <a class="anchor" id="6"></a>

# In[ ]:


df['Winter'] = df[['DEC', 'JAN', 'FEB']].mean(axis=1)
df['Summer'] = df[['MAR', 'APR', 'MAY']].mean(axis=1)
df['Monsoon'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
df['Autumn'] = df[['OCT', 'NOV']].mean(axis=1)
seasonal_df = df[['YEAR', 'Winter', 'Summer', 'Monsoon', 'Autumn']]
seasonal_df = pd.melt(seasonal_df, id_vars='YEAR', value_vars=seasonal_df.columns[1:])
seasonal_df.columns=['Year', 'Season', 'Temprature']


# In[ ]:


fig = px.scatter(seasonal_df, 'Year', 'Temprature', facet_col='Season', facet_col_wrap=2, trendline='ols')
fig.update_layout(title='Seasonal mean tempratures throught years:')
fig.show()


# #### We can again see a positive trendline between temprature and time. The trendline does not have a very high positive correlation with years but still it is not negligable.
# #### Let's try to find out if we can get someyhing out of an animation?

# In[ ]:


px.scatter(df1, 'Month', 'Temprature', size='Temprature', animation_frame='Year')


# #### On first look, we can see som fluctuations but that doesn't give much of insights for us. However, if we again see by arranging bar below to early years and late years we can notice the change. But this is certainly not the best way to visualize it. Let's find some better way.

# ## Forecasting
# 
# <a class='ancor' id='7'></a>

# ### Let's try to forecast monthly mean temprature for year 2018.

# In[ ]:


## I am using decision tree regressor for prediction as the data does not actually have a linear trend.
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score 

df2 = df1[['Year', 'Month', 'Temprature']].copy()
df2 = pd.get_dummies(df2)
y = df2[['Temprature']]
x = df2.drop(columns='Temprature')

dtr = DecisionTreeRegressor()
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3)
dtr.fit(train_x, train_y)
pred = dtr.predict(test_x)
r2_score(test_y, pred)


# #### A high r2 value means that our predictive model is working good(For now, because there is a lot more than just the r_squared statistic, and we can't determine how good a model is based only on r2 statistic. But, that we'll discuss later. ). Now, Let's see the forecasted data for 2018.

# In[ ]:


next_Year = df1[df1['Year']==2017][['Year', 'Month']]
next_Year.Year.replace(2017,2018, inplace=True)
next_Year= pd.get_dummies(next_Year)
temp_2018 = dtr.predict(next_Year)

temp_2018 = {'Month':df1['Month'].unique(), 'Temprature':temp_2018}
temp_2018=pd.DataFrame(temp_2018)
temp_2018['Year'] = 2018
temp_2018


# In[ ]:


forecasted_temp = pd.concat([df1,temp_2018], sort=False).groupby(by='Year')['Temprature'].mean().reset_index()
fig = go.Figure(data=[
    go.Scatter(name='Yearly Mean Temprature', x=forecasted_temp['Year'], y=forecasted_temp['Temprature'], mode='lines'),
    go.Scatter(name='Yearly Mean Temprature', x=forecasted_temp ['Year'], y=forecasted_temp['Temprature'], mode='markers')
])
fig.update_layout(title='Forecasted Temprature:',
                 xaxis_title='Time', yaxis_title='Temprature in Degrees')
fig.show()


# ### yearly mean temprature of India in 2018 was 25.9 Source : https://www.statista.com/statistics/831763/india-annual-mean-temperature/

# In[ ]:




