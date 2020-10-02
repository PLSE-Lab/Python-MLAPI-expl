#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import cufflinks as cf
cf.go_offline()

from functools import reduce

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Read each file into seperate Dataframes

# In[2]:


df_income = pd.read_csv("..//input/MedianHouseholdIncome2015.csv", encoding="windows-1252")
df_schooling = pd.read_csv("..//input/PercentOver25CompletedHighSchool.csv", encoding="windows-1252")
df_race = pd.read_csv("..//input/ShareRaceByCity.csv", encoding="windows-1252")
df_poverty = pd.read_csv("..//input/PercentagePeopleBelowPovertyLevel.csv", encoding="windows-1252")
df_killing = pd.read_csv("..//input/PoliceKillingsUS.csv", encoding="windows-1252")


# **Median Income Dataset**

# In[3]:


df_income.head()


# In[4]:


df_income.info()


# There are some missing values, first we will observe the pattern of the missing values.

# To help us visualize the missing values, I have written a simple heatmap function. This function will plot the values for each column at all indices

# In[5]:


def heatMap(df):
    df_heat = pd.DataFrame()
    for each in df.columns:
         df_heat[each] = df[each].apply(lambda x: 1 if pd.notnull(x) else 0)
    df_heat.iplot(kind='heatmap')


# All missing values are from Wyoming. I will just fill them with 0s

# In[6]:


heatMap(df_income[df_income['Geographic Area'] == 'WY'])


# Before filling the NaN columns with 0s, we need to convert the column to numeric, we will also set the nonnumeric fields as NaN

# In[7]:


df_income['Median Income'] = df_income['Median Income'].apply(lambda x : pd.to_numeric(x,errors='coerce'))


# In[8]:


df_income.fillna(0,axis=1, inplace=True)


# Since Geographic Area and City are our index fields, it's advisable to convert them to categorical columns

# In[9]:


df_income['Geographic Area'] = df_income['Geographic Area'].astype('category')
df_income['City'] = df_income['City'].astype('category')


# Now we will rank the cities based on their median income, we will create two rank field, one for national and the  other for  state

# In[10]:


df_income['Area Rank- Income'] = df_income.groupby('Geographic Area')['Median Income'].rank(ascending=False,method='dense')
df_income['National Rank- Income'] = df_income['Median Income'].rank(ascending=False,method='dense')


# In[11]:


total = int(df_income['National Rank- Income'].max())


# Following text field will be used as the hoverinfo for the plotly graph we will create later

# In[12]:


df_income['Text1'] = df_income.apply(lambda x: "<b>City: {}</b><br><b>National Rank Income: {:,.0f} ({})</b>".format(x['City'], x['National Rank- Income'],total), axis=1)


# **Percent Over 25 Completed HighSchool Dataset**

# In[13]:


df_schooling.head()


# In[14]:


df_schooling.info()


# We will follow the same data cleaning and feature engineering procedure we have done for the previous datasets

# In[15]:


df_schooling['percent_completed_hs'] = df_schooling['percent_completed_hs'].apply(lambda x : pd.to_numeric(x,errors='coerce'))


# In[16]:


heatMap(df_schooling)


# In[17]:


df_schooling.fillna(0,axis=1, inplace=True)


# In[18]:


df_schooling['National Rank- Schooling'] = df_schooling['percent_completed_hs'].rank(ascending=False,method='dense')


# In[19]:


df_schooling['Geographic Area'] = df_schooling['Geographic Area'].astype('category')
df_schooling['City'] = df_schooling['City'].astype('category')


# In[24]:


#following lookup stores the total number of cities grouped by area
lkup = dict(df_schooling.groupby('Geographic Area').size())
    
#map the lookup to indivigual rows
df_schooling['Total Cities'] = df_schooling['Geographic Area'].map(lkup)

# create catogories
df_schooling['Geographic Area'] = df_schooling['Geographic Area'].astype('category')

# select all the cities with 100% pass rate
df_top_city_schooling = df_schooling[df_schooling['National Rank- Schooling'] == 1]

#select the least rank present, this would be used in the hoverinfo later
total = int(df_schooling['National Rank- Schooling'].max())

#count the number of cities in each area that have 100% pass rate 
lkup2 = dict(df_top_city_schooling.groupby('Geographic Area').size())
    
#map the lookup to indivigual rows    
df_schooling['No. of Top Cities'] = df_schooling['Geographic Area'].map(lkup2)

#create the text fields required for the hoverinfo.
df_schooling['Text2'] = df_schooling.apply(lambda x: "<b>Cities with pass % 100: {:,.0f}</b><br><b>Total Cities: {:,.0f}</b>".format(x['No. of Top Cities'], x['Total Cities']), axis=1)
df_schooling['Text2_1'] = df_schooling.apply(lambda x: "<br><b>National Rank Education: {:,.0f} ({})</b><br>".format(x['National Rank- Schooling'], total), axis=1)


# In[25]:


traces = (go.Bar(y=  (df_top_city_schooling.groupby('Geographic Area').size()*100/df_schooling.groupby('Geographic Area').size()).values,
                     x = [each[0] for each in lkup.items()],
                     text = df_schooling.groupby(['Geographic Area','Text2']).size().reset_index()['Text2'].values,
                     hoverinfo = "text",
                     marker = dict(color='rgb(111, 198, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                 opacity=0.6
              ))
layout = dict(title = "Number of Cities that have 100% High School Pass Rate for 25 and above year olds, Area Wise",
             xaxis = dict(dict(title = 'Area')),
             yaxis = dict(dict(title = "(# of Cities with Pass % 100/Total # of Cities in that Area) % ")))
fig = dict(data = [traces], layout=layout)
iplot(fig)


# Please note all the observations made below are based on the data present in the dataset and might not reflect the actual figures.
# 
# Observations
# *     California has the most number of cities with 100% high school pass rate for 25 and above - 117, it has a total of 1522 cities.
# *     Wyoming is doing considerably better than other states, among its 204 cities, 48 of them have 100% pass rate, followed by Nevada, which has 25 among 131. 
# *     DC has only one city and it doesn't have 100% pass rate.
# *     After DC, Tennessee is the worst performing state, among its 430 cities, only three has 100% pass rate.
# *     Best Performing 5 States - WY, NV, CO, NM and MT
# *     Worse Performing 5 States - DC, TN, AR, IL, and NA
# 
#     
#            
#     
#     

# **Share Race by City Dataset**

# In[26]:


df_race.info()


# In[27]:


#convert the numeric fields to numeric and coerce nonumeric to NaN
cols = df_race.columns[2:]
for each in cols:
    df_race[each] = df_race[each].apply(lambda x : pd.to_numeric(x,errors='coerce'))


# In[28]:


#fix the column name
df_race.rename(columns={'Geographic area':'Geographic Area'}, inplace=True)


# In[29]:


#set categories
df_race['Geographic Area'] = df_race['Geographic Area'].astype('category')
df_race['City'] = df_race['City'].astype('category')


# In[30]:


df_race.info()


# In[31]:


#create the text field for the hoverinfo
df_race['Text3'] = df_race.apply(lambda x: "<b>White: {}%</b><br><b>Black: {}%</b><br><b>Native: {}%</b><br><b>Asians: {}%</b><br><b>Hispanic: {}%</b><br>".
                                 format(x['share_white'], x['share_black'],
                                       x['share_native_american'], x['share_asian'],
                                       x['share_hispanic']), axis=1)


# **Percentage People Below Poverty Level Dataset**

# In[32]:


df_poverty.info()


# In[33]:


df_poverty['poverty_rate'] = df_poverty['poverty_rate'].apply(lambda x : pd.to_numeric(x,errors='coerce'))


# In[34]:


df_poverty['Geographic Area'] = df_poverty['Geographic Area'].astype('category')
df_poverty['City'] = df_poverty['City'].astype('category')


# In[35]:


heatMap(df_poverty)


# There a few missing values, we will fill them with 0s

# In[36]:


df_poverty['poverty_rate'].fillna(0,axis=0, inplace=True)


# In[37]:


df_poverty.info()


# In[38]:


#create Rank fields
df_poverty['Area Rank- Poverty'] = df_poverty.groupby('Geographic Area')['poverty_rate'].rank(ascending=False,method='dense')
df_poverty['National Rank- Poverty'] = df_poverty['poverty_rate'].rank(ascending=False,method='dense')


# In[39]:


total = int(df_poverty['National Rank- Poverty'].max())


# In[40]:


df_poverty['Text4'] = df_poverty.apply(lambda x: "<b>National Rank Poverty: {:,.0f} ({})</b><br>".format(x['National Rank- Poverty'],total), axis=1)


# Merge all the cleaned up datasets

# In[41]:


df_income_schooling_race_poverty = reduce(lambda left,right: pd.merge(left,right,on=['Geographic Area', 'City'], how='left'), [df_income, df_schooling, df_race, df_poverty])


# In[42]:


#select the top city with best median income from each region
df_top_city_income_final = df_income_schooling_race_poverty[df_income_schooling_race_poverty['Area Rank- Income'] == 1].set_index(['Geographic Area'])
df_top_city_income_final.fillna(' ', inplace=True)


# In[43]:


#create a trace for the top cities with best median income
traces1 = (go.Bar(y=  df_top_city_income_final['Median Income'],
                     x = df_top_city_income_final.index,
                     text = df_top_city_income_final['Text1'] + df_top_city_income_final['Text2_1'] + df_top_city_income_final['Text3'] + df_top_city_income_final['Text4'],
                     hoverinfo = "text",
                     marker = dict(color='rgb(111, 198, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                     name = 'Best',
                     opacity=0.6
                 )
              )


# In[65]:


#create a lookup, area wise, with worse performing Cities.
lkup3 = dict(df_income_schooling_race_poverty[df_income_schooling_race_poverty['Median Income'] != 0].groupby('Geographic Area')['Area Rank- Income'].max())


# In[66]:


#select the indices that match the criteria
loc = [df_income_schooling_race_poverty[(df_income_schooling_race_poverty['Geographic Area'] == Area) & (df_income_schooling_race_poverty['Area Rank- Income'] == Rank)].index[0] for Area, Rank in lkup3.items()]


# In[67]:


#pick the cities with worse median income.
df_worst_city_income_final = df_income_schooling_race_poverty.iloc[loc].set_index(['Geographic Area'])
df_worst_city_income_final.fillna(' ', inplace=True)


# In[68]:


#create a trace for cities with worst median income from each area
traces2 = (go.Bar(y=  df_worst_city_income_final['Median Income'],
                     x = df_worst_city_income_final.index,
                     text = df_worst_city_income_final['Text1'] + df_worst_city_income_final['Text2_1'] + df_worst_city_income_final['Text3'] + df_worst_city_income_final['Text4'],
                     hoverinfo = "text",
                     marker = dict(color='rgb(198, 91, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                     name= 'Worst',
                     opacity=0.6
                 )
              )


# In[69]:


#finally, plot the bar chart
layout = dict(title = "Median Income - Best and Worst Performing City from each State",
             xaxis = dict(dict(title = 'Area')),
             yaxis = dict(dict(title = "Median Income")))
fig = dict(data = [traces1,traces2], layout=layout)
iplot(fig)


# Please note all the observations made below are based on the data present in the dataset and might not reflect the actual figures.
# 
# Observations
# *  Crisman in CO has the best median income. It has 100% high school pass rate for people above 25 years, nobody is below poverty line
# *  Scarsdale in NY has the second best median income, it has a very good high school pass rate, ranked 13th among all cities, and a very low poverty rank of 747.
# *  Stanfield is the worse performing city, with a median income rank of 14588. Its high school pass rate rank is 255 and poverty rank 121 
# 

# **Police Killings US Dataset**

# In[70]:


#renmae the columns
df_killing.rename(columns={'state':'Geographic Area', 'city': 'City'}, inplace=True)


# In[71]:


df_killing.info()


# In[72]:


#convert Area to Category
df_killing['Geographic Area'] = df_killing['Geographic Area'].astype('category')


# Next we will create some state wise summary, which we will use in our next plotly plot

# In[73]:


# count number of cities that are in top 50 when it comes to median income and store them into a lookup
lkup4 = dict(df_income_schooling_race_poverty[df_income_schooling_race_poverty['National Rank- Income'] <= 50].groupby('Geographic Area').size())
    
#map the summary back to original dataset
df_income_schooling_race_poverty['National Rank50- Income'] = df_income_schooling_race_poverty['Geographic Area'].map(lkup4)

#create a text field to store the details for hoverinfo
df_income_schooling_race_poverty['Text5']  = df_income_schooling_race_poverty.apply(lambda x: "<br><b>Cities in top 50-Median Income: {:,.0f}</b><br>".
                                 format(x['National Rank50- Income']),
                                        axis=1)

# count the cities which have 75% of the population below poverty line
lkup5 = dict(df_income_schooling_race_poverty[df_income_schooling_race_poverty['poverty_rate'] >= 75].groupby('Geographic Area').size())
    
#map it back to original datset
df_income_schooling_race_poverty['75% below poverty # cities'] = df_income_schooling_race_poverty['Geographic Area'].map(lkup5)

#text field for hoverinfo
df_income_schooling_race_poverty['Text6']  = df_income_schooling_race_poverty.apply(lambda x: "<b>Cities with 75% or more below Poverty: {:,.0f}</b><br>".
                                 format(x['75% below poverty # cities']),
                                        axis=1)
# select the hoverinfo fields from the original dataset, only select one row for each Area.
df_state_text_lkup = df_income_schooling_race_poverty.drop_duplicates(subset=['Geographic Area','Total Cities','Text2','Text5','Text6'])[['Geographic Area','Total Cities','Text2','Text5','Text6']].dropna().set_index('Geographic Area')


# In[74]:


#first sum up the fatal shootings for each state and append the hoverinfo to it
df_state_inc = pd.merge(df_killing.groupby(['Geographic Area']).size().to_frame(), df_state_text_lkup, how='left', left_index=True, right_index=True)
df_state_inc.rename(columns={0:'Fatal Shootings'}, inplace=True)


# In[75]:


#create a new text field for hoverinfo, this will show the incidents per city for each state
df_state_inc['Text1'] = df_state_inc.apply(lambda x: "<b>Shooting Per City: {:.2f}</b><br>".format(x['Fatal Shootings']/x['Total Cities']),axis=1)


# In[76]:


traces  = (go.Bar(y=  df_state_inc['Fatal Shootings'],
                     x = df_state_inc.index,
                     text = df_state_inc['Text1'] + df_state_inc['Text5'] + df_state_inc['Text6'] + df_state_inc['Text2'] ,
                     hoverinfo = "text",
                     marker = dict(color='rgb(198, 91, 49)',
                                   line=dict(color='rgb(8,48,107)',
                                             width=2)
                                  ),
                     name= 'Fatal Shootings',
                     opacity=0.6
                 )
              )
layout = dict(title = "Fatal Shootings, State Wise",
             xaxis = dict(dict(title = 'State')),
             yaxis = dict(dict(title = "Number of Fatal Shoorings")))
fig = dict(data = [traces], layout=layout)
iplot(fig)


# Please note all the observations made below are based on the data present in the dataset so could be way off from the actual figures.
# 
# Observations
# 
# *   CA has highest number of shooting, but then it has more cities as well, the incidents per city is 0.28, second to only NV. 
# *   CA's 6 cities are in the top 50 list of cities with highest medium income, only 11 cities in that state have poverty % more than 75, 117 cities have a 100% high school pass rate for 25 and above
# *  IA and ND has least number of fatal police shootings per city,  0.01
# *  NV has got the highest number of fatal shootings per city, 0.32.
# *  15 of NY's cities are in the list of top 50 cities with highest medium income, top among all other states.

# In[77]:


df_killing['manner_of_death'].unique()


# Now will observe the details in the killings datset

# In[78]:


df_killing.groupby('armed').size().sort_values(ascending=False)[:10].iplot(kind='bar',title='Top 10 Most used Weapons')


# In[79]:


df_killing.groupby('age').size().iplot(kind='bar',title='Age of People Killed')


# In[80]:


df_killing.groupby('gender').size().iplot(kind='bar',title='Gender')


# In[81]:


df_killing.groupby('City').size().sort_values(ascending=False)[:11].iplot(kind='bar',title='Top 10 Cities with Most Number of Fatal Shootings')


# **Time series analysis**

# In[82]:


# Extract the year and month from the date and separate them as different columns. 
df_killing = pd.concat([pd.DataFrame([each for each in df_killing['date'].str.split('/').values.tolist()],
                             columns=['Day', 'Month', 'Year']),df_killing],axis=1)
df_killing['Year'] = df_killing['Year'].apply(lambda x: int(x) + 2000)

# Convert the Date column to datetime
df_killing.date = df_killing.date.apply(lambda x : pd.to_datetime(x,dayfirst=True))

#create day of the week column
df_killing['day_of_week'] = df_killing.date.apply(lambda x: pd.to_datetime(x).weekday())


# 2015 July had maximum number of Fatal shootings, followed by 2017 February

# In[83]:


df_killing.groupby([ 'Year', 'Month'])['date'].size().iplot(title = 'Fatal Shootings Monthly', yTitle = "Number", xTitle = "(Year, Month)" )


# 2016 had comparatively lesser number of fatal shootings than 2015. 2017's data is not complete enough to make any such inference

# In[84]:


df_killing[ 'Year'].value_counts().plot(kind='bar',title = 'Fatal Shootings Yearly',figsize=(8,5))


# in 2015, maximum cases recieved on any month were less than or equal to 104, while in 2016 it has reduced to 92, in 2017 it has gone up to 100.

# In[85]:


pd.crosstab(df_killing['Month'], df_killing['Year']).iplot(kind='box', title = 'Fatal Killings', xTitle ='Year',
                                                                    yTitle = 'Number')


# In[86]:


df_time = df_killing[['Year', 'Month', 'Day', 'day_of_week', 'date']]
df_time['Count'] = 1
df_time = df_time.groupby(['date','Year', 'Month', 'Day', 'day_of_week']).sum().reset_index()
df_time.set_index('date', inplace=True)


# In[87]:


df_time.info()


# In[88]:


df_time = df_time.reindex(pd.date_range(start="2015", end="2017-07-31", freq='D'))


# In[89]:


df_time.info()


# In[90]:


df_time.fillna(0, inplace=True)
df_time = df_time.reset_index()
df_time['Year'] = df_time['index'].apply(lambda x: x.strftime('%Y'))
df_time['Month'] = df_time['index'].apply(lambda x: x.strftime('%m'))
df_time['Day'] = df_time['index'].apply(lambda x: x.strftime('%d'))
df_time['day_of_week'] = df_time['index'].apply(lambda x: x.strftime('%w'))


# *  Maximum fatal shootings per day is 8 or less
# *  27th and 15th of every month always have alteast one fatal shooting per day. (for the data we have got)

# In[91]:


ax1 = pd.crosstab([df_time['Year'],df_time['Month']], df_time['Day'], values=df_time['Count'], aggfunc='sum').reset_index().drop(['Year','Month'],axis=1).plot(kind='box',figsize=(15,5))
ax1.set_ylabel("Count")
ax1.set_xlabel("Day of the Month")


# First day of the month has lesser number of fatal shootings and 23rd and 27th have the maximum

# In[92]:


df_killing.groupby([ 'Day'])['date'].size().iplot(title = 'Fatal Shootings by Day of the Month', yTitle = "Count", xTitle = "Day" )


# In[93]:


df = df_time.drop(['index','Year', 'Month', 'Day'], axis=1).reset_index()
ax1 = pd.crosstab(df.index,df.day_of_week,values=df.Count, aggfunc='sum').plot(kind='box',figsize=(10,5))
ax1.set_ylabel("Count")
ax1.set_xlabel("Day of the Week")
ax1.set_xticklabels(['Sun', 'Mon', 'Tue','Wed', 'Thu', 'Fri', 'Sat'])


# In[94]:


ax1=pd.crosstab(df_time.Year,df_time.Month,values=df_time.Count, aggfunc='sum').plot(kind='box',figsize=(15,5))
ax1.set_ylabel("Count")
ax1.set_xlabel("Month")


# In[95]:


df_killing.groupby([ 'Month'])['date'].size().iplot(title = 'Fatal Shootings by Month', yTitle = "Count", xTitle = "Month" )

