#!/usr/bin/env python
# coding: utf-8

# # Analysis of road accident patterns in India from 2001 to 2014 in different states

# In this notebook, we are analysing the pattern of road accidents in India. The data covers the period from year 2001 through 2014. We have worked on publicly available data. The workbook is open for further improvement with data from other sources.
# Credibility of data is not ascertained.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display as display
import plotly.graph_objs as go


# #### Getting the data from the files

# In[ ]:


# Read the data to the dataframe from the data files.

#state_year_month_df contains data for each state, segragated into year and month
state_year_month_df=pd.read_csv('../input/only_road_accidents_data_month2.csv')

#state_year_time_df contains data for each state, segragated into year and time of the day
state_year_time_df=pd.read_csv('../input/only_road_accidents_data3.csv')


# In[ ]:


state_year_month_df.head()


# In[ ]:


state_year_time_df.head()


# We can see that data is in a long format, ie the breakup of accidents in each state is given as separate rows, rathern than including separate columns for each year.

# In[ ]:


#Get all the state names in an array..
state_names=state_year_month_df['STATE/UT'].unique()
print(state_names)


# We can see from the above list that the states - Delhi, Dadra & Nagar Haveli has appeared with multiple names. So, we will make their names uniform.

# In[ ]:


#state_year_month_df=state_year_month_df['STATE/UT']
state_year_month_df['STATE/UT']=state_year_month_df['STATE/UT'].replace({'Delhi (Ut)': 'Delhi Ut', 'D & N Haveli':'D&N Haveli'})
print(state_year_month_df['STATE/UT'].unique())


# In[ ]:


# Reassiging state names to variable..
state_names=state_year_month_df['STATE/UT'].unique()


# ## Feature Engineering

# #### <i> Feature engineering 1: Clubbing month columns into seasons</I>
# Our data contains breakup of accident figures for every state and every month, since 2001 till 2014. We considered, clubbing monthly data to seasonal data, as monthly details is not required.

# In[ ]:


#display(state_year_month_df.head())

#Create season groups clubbing values from multiple month columns..
state_year_month_df['SUMMER']=state_year_month_df[['JUNE','JULY','AUGUST']].sum(axis=1)
state_year_month_df['AUTUMN']=state_year_month_df[['SEPTEMBER','OCTOBER','NOVEMBER']].sum(axis=1)
state_year_month_df['WINTER']=state_year_month_df[['DECEMBER','JANUARY','FEBRUARY']].sum(axis=1)
state_year_month_df['SPRING']=state_year_month_df[['MARCH','APRIL','MAY']].sum(axis=1)

#Delete month columns..
state_year_month_df=state_year_month_df.drop(['JANUARY','FEBRUARY','MARCH','APRIL','MAY','JUNE','JULY'
                                             ,'AUGUST','SEPTEMBER','OCTOBER','NOVEMBER','DECEMBER'], axis=1)
#Create groups of states, summing the values of accident number for each year..
state_grouped=state_year_month_df.groupby(['STATE/UT']).sum()

#Create % columns for noting the % of accidents happening in each state for each season..
state_grouped['%_SUMMER']=state_grouped['SUMMER']/state_grouped['TOTAL']
state_grouped['%_AUTUMN']=state_grouped['AUTUMN']/state_grouped['TOTAL']
state_grouped['%_WINTER']=state_grouped['WINTER']/state_grouped['TOTAL']
state_grouped['%_SPRING']=state_grouped['SPRING']/state_grouped['TOTAL']

display(state_grouped.iloc[:,1:].head())


# #### <I> Over the day accident breakup : Merge columns into categories - 'Night', 'Day', 'Afternoon', 'Evening'</I>

# In[ ]:


#Working on the over the day data...
state_year_time_df.rename(columns={'0-3 hrs. (Night)':'0-3',
                              '3-6 hrs. (Night)':'3-6',
                                '6-9 hrs (Day)':'6-9', '9-12 hrs (Day)':'9-12','12-15 hrs (Day)':'12-15','15-18 hrs (Day)':'15-18',
                                  '18-21 hrs (Night)':'18-21','21-24 hrs (Night)':'21-24'}, inplace=True)
state_time_grouped=state_year_time_df.groupby(['STATE/UT']).sum()

state_time_grouped['%_MORNING']=(state_time_grouped['6-9']+state_time_grouped['9-12'])/state_time_grouped['Total']
state_time_grouped['%_AFTERNOON']=(state_time_grouped['12-15']+state_time_grouped['15-18'])/state_time_grouped['Total']
state_time_grouped['%_EVENING']=(state_time_grouped['18-21']+state_time_grouped['21-24'])/state_time_grouped['Total']
state_time_grouped['%_NIGHT']=(state_time_grouped['0-3']+state_time_grouped['3-6'])/state_time_grouped['Total']

state_time_grouped=state_time_grouped.drop(state_time_grouped.columns[0:9], axis=1)
display(state_time_grouped.head())


# ## <U> Digging into the data to find patterns </U>

# ### First, lets see the seasonal distribution of accidents over the years, and for all the states taken together

# In[ ]:


plt.figure(figsize=(15,5))
ax=plt.subplot(1,2,1)
boxplot=state_grouped.boxplot(ax=ax,column=['%_SUMMER','%_WINTER','%_AUTUMN','%_SPRING'])

ax=plt.subplot(1,2,2)
state_grouped.loc[:,'SUMMER':'SPRING'].sum(axis=0).plot.pie(title='Seasonal distribution of all accidents in India(2001-14)',autopct='%1.1f%%')


# INTERPRETATION: 
# 
# From the boxplot it is evident that the median % of accidents is slightly high for winter and spring. Moreover, summer and spring are having more even distribution of accidents %. Outliers are high for summer.
# 
# Second plot(Pie plot) gives a general impression that accidents are nearly uniform over the seasons.

# #### Lets find out the states with highest percentage accidents in different seasons
# 
# Each plot depicts a season with the states with highest % of accidents in that season. 

# In[ ]:


plt.figure(figsize=(20,5))
plt.subplot(141)
summer_sorted=state_grouped.sort_values('%_SUMMER')
summer_sorted['%_SUMMER'].tail(5).plot.bar(title='Highest Summer Accidents')
plt.subplot(142)
winter_sorted=state_grouped.sort_values('%_WINTER')
winter_sorted['%_WINTER'].tail(5).plot.bar(title='Highest Winter Accidents')
plt.subplot(143)
autumn_sorted=state_grouped.sort_values('%_AUTUMN')
autumn_sorted['%_AUTUMN'].tail(5).plot.bar(title='Highest Autumn Accidents')
plt.subplot(144)
spring_sorted=state_grouped.sort_values('%_SPRING')
spring_sorted['%_SPRING'].tail(5).plot.bar(title='Highest Spring Accidents')


# What we can interpret:
#       For summer, the smaller states occupy the major positions. Same is for winter and autumn. For winter and autumn, Lakshadweep has high % of accidents. But , we can not make any interpretation as the overall number of accidents in Lakshwadeep is low. Some anomaly may be there. States with highesst spring time accidents share are almost at par.
#      
#     

# ## Yearly progress of states with highest accidents

# In[ ]:


highest_accident_states=state_grouped.sort_values('TOTAL', ascending=False)
high_states=list(highest_accident_states.head().index)
df4=state_year_month_df.loc[state_year_month_df['STATE/UT'].isin(high_states),['STATE/UT','YEAR','TOTAL']]

plt.figure(figsize=(10,5))
ax=plt.subplot(111)
for key, grp in df4.groupby(['STATE/UT']):
    ax = grp.plot(ax=ax, kind='line', x='YEAR', y='TOTAL', label=key)
  
plt.show()


# we can see something unusual for Tamilnadu in year 2005. Lets not dig deeper, but applaud other states, which have improved over years.

# ## Working on over the day accident data..

# #### Which are the states having highest number of accidents?

# In[ ]:


highest_accident_states=state_grouped.sort_values('TOTAL', ascending=False)
state_list=list(highest_accident_states.head().index)
print(state_list)


# #### States with highest accident numbers: How are accidents distributed over the day?

# In[ ]:



df=state_time_grouped.loc[state_time_grouped.index.isin(state_list)]

df_T=df.groupby('STATE/UT').sum().drop(['Total'], axis=1).T.plot.pie(subplots=True, figsize=(20, 5),autopct='%1.1f%%')


# One observation: Kerala has high afternoon accidents, but low night time accidents. Other than that, we dont see any other interesting patterns.

# ## Now, lets see the over the day distribution of all the accidents in India

# In[ ]:


## Break up accidents for all states over the time blocks:
#state_time_grouped.info()
df2=state_time_grouped.sum(axis=0)



df2.drop(['Total']).T.plot.pie(title='All accidents 2001-2014',subplots=True, figsize=(5,5),autopct='%1.1f%%')

df2=state_time_grouped.sum(axis=0)


# Interpretation: For all the states, and for all the years, we can see the afternoon accidents occupy major portion, followed by morning, and closely followed by evening.

# ## How has accidents in the country has grown over years 

# In[ ]:


df3=state_year_time_df.groupby(['YEAR']).sum()
df3.loc[:,'Total'].plot(title='Accidents growth in India')


# The accidents are always growing since 2001, but the rate of growth has declined in recent years. Good news!!

# ## States with highest % accidents in different timeblocks

# In[ ]:


#See the states with highest % accident in the every timeblock..
plt.figure(figsize=(10,5))
state_time_grouped.sort_values('%_MORNING',ascending=False).head().loc[:,['STATE/UT','%_MORNING']].plot(kind='bar', ax=plt.subplot(221), color='b')
state_time_grouped.sort_values('%_AFTERNOON',ascending=False).head().loc[:,['STATE/UT','%_AFTERNOON']].plot(kind='bar', ax=plt.subplot(222),color='g')
state_time_grouped.sort_values('%_EVENING',ascending=False).head().loc[:,['STATE/UT','%_EVENING']].plot(kind='bar', ax=plt.subplot(223),color='r')
state_time_grouped.sort_values('%_NIGHT',ascending=False).head().loc[:,['STATE/UT','%_NIGHT']].plot(kind='bar', ax=plt.subplot(224),color='y')


# ## Checking performance of states from 2001 to 2014

# In[ ]:


#Create a new dataframe - period_performance.
period_performance=pd.DataFrame(columns=['STATE/UT','%_CHANGE_2001_TO_2014'])

#Take one state name at a time,
for state in state_names:
    #print(state)
    total_2001=state_year_month_df.loc[(state_year_month_df['STATE/UT']==state) & (state_year_month_df['YEAR']==2001), 'TOTAL']
    total_2014=state_year_month_df.loc[(state_year_month_df['STATE/UT']==state) & (state_year_month_df['YEAR']==2014), 'TOTAL']
    value_2001=total_2001.iloc[0]
    value_2014=total_2014.iloc[0]
    change_in_percent= (value_2014-value_2001)*100/value_2001
   
    new_data=pd.Series({'STATE/UT':state, '%_CHANGE_2001_TO_2014':change_in_percent})
    period_performance=period_performance.append(new_data, ignore_index=True)


# In[ ]:


best_performing=period_performance.sort_values('%_CHANGE_2001_TO_2014')
#print(best_performing.head())
ax=best_performing.plot(kind='bar').set_xticklabels(best_performing['STATE/UT'])


# Interpretation: A few states/UTs have shown decrease in number of accidents. But, most of these states are smaller.
# Jharkhand, Assam and Punjab on the other hand are leading the states with high % increase of accidents from 2001 to 2014.

# ## Special Note: Please give your feedback and suggestions. Thank you.
