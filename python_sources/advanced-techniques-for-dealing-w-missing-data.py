#!/usr/bin/env python
# coding: utf-8

# # Handling Missing Data

# ## In this short tutorial, I will attemp to illustrate how to handle missing data using some "advanced" techniques. I am not going to implement pandas' build in methods (such ".interpolate" or ".fillna"). Hope you will find it useful!

# ### First, Let's import pandas library and check which columns need to be taken care of

# In[ ]:


# Import all of the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


events = pd.read_csv("../input/athlete_events.csv")
events.head(10)


# #### Does each column have an appropriate data type and which columns' missing data need to be fixed?

# In[ ]:


events.info()


# In[ ]:


events.isnull().sum()


# ### Part One - Explore and Understand by Visualize the missing Data

# In[ ]:


#How many different types of sports do Olympic Games have?
events['Sport'].nunique()


# In[ ]:


events['Medal'].value_counts(dropna=False)


# In[ ]:


events.groupby(['Year', 'Medal'])['Medal'].count().unstack().plot(kind='bar',figsize=(15,6))


# #### For Medal column, we need to convert all of the NaNs into "No Medal" or just string of zeros. The graph shows how the bronze, silver and gold medals were distributed in each Olympic Games. Also, starting from 1992, the winter and summer olympic games were held in different years 

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
events.groupby(['Year', 'Sex'])[['Height', 'Weight']].mean().unstack().plot(ax=ax1)
ax1.set_title('Average Height and Weight of Sportsmen/women')
events.groupby(['Year', 'Sex'])['Age'].mean().unstack().plot(ax=ax2,figsize=(15,10))
ax1.set_title('Average Age of Sportsmen/women')


# #### The weight and height of olympic sport-players didn't fluctuate much but their ages did (especially until 1950s, the age drastically changed from an olympic game to game). Probably, it is because of having very few sportsmen and women in early years of OG. Since 1910, females on average tend to be younger than males but the gap has been shrinking steadily since 80s  

# In[ ]:


#Here is the exact numbers of age difference in the last five OG
events.groupby(['Year', 'Sex'])['Age'].mean().unstack().tail()


# In[ ]:


grouped_df = events.groupby('Year')[['Height', 'Weight', 'Age']].count()


# In[ ]:


grouped_df.head(3)


# In[ ]:


attendees = events['Year'].value_counts().sort_index()
print('The Total Attendees from 1896 to 2016 in the List/Array:')
attendees.values


# In[ ]:


grouped_df['Total Attendees'] = attendees.values
grouped_df.head()


# #### There are also other ways of joining pandas series/df objects together - the common use cases are ".concat", ".join", ".merge" or ".append" methods

# In[ ]:


grouped_df[['Height', 'Weight', 'Total Attendees']].plot(figsize=(15,5), 
          title = 'Number of Existing Height and Weight vs Total Number of Olympic participants')


# In[ ]:


grouped_df[['Age', 'Total Attendees']].plot(figsize=(15,5))


# #### In early years of OG, there were way too many missing values of olympic players' height and weight. Let's analyze the proportion of it by ploting one more line-chart

# In[ ]:


grouped_df['Height non-NA'] = grouped_df['Height'] / grouped_df['Total Attendees']
grouped_df['Weight non-NA'] = grouped_df['Weight'] / grouped_df['Total Attendees']
grouped_df['Age non-NA'] = grouped_df['Age'] / grouped_df['Total Attendees']
grouped_df.head(3)


# In[ ]:


grouped_df[['Weight non-NA', 'Height non-NA', 'Age non-NA']].plot(figsize=(15,8), marker='o', alpha = .3, 
                                                 xticks = range(1896, 2018, 6), 
                                                 title='Proportion/Percentage of Non-missing Values for Weight, Height & Age')


# #### Until 1960, the players height and weight were unbearably low for data analysis (especially, for statistical analysis).

# ### Part Two - Take Actions on the missing Data

# #### If we pretend that height and weight are the most important part of our data or statistical analysis, then we can simply drop the "bad" data points. 

# In[ ]:


events_60up = events.loc[events['Year']>=1960, :]
events_60up.head(3)


# In[ ]:


events_60up.info()


# #### Change the NaN values to "No Medal" for the Medals column by running the following code:

# events_60up['Medal'].fillna(value='No Medal', inplace=True)

# #### For each gender with their playing sport type create a df of their average heights and plot it on a bar chart

# In[ ]:


height_grouped = events_60up.groupby(['Sport', 'Sex'])['Height'].mean().unstack()
height_grouped.sort_values(by='M', ascending=False).head()


# In[ ]:


height_grouped.sort_values(by='M').plot(kind='barh', figsize=(15,12))


# #### Olympic players' average height varies based on thier gender and playing sport type. Therefore, we are going to fill missing data points by mapping to their height averages of sport type and gender. 

# In[ ]:


#Practice how to grab a key for getting a value from python dictionary
#Assign height of sportswomen to a variable
f_height = height_grouped.to_dict()['F']
#what is the average value of female basketball player?
f_height['Basketball']


# In[ ]:


#set a condition which returns all NaN values for rows of female basketball and column of Height
events_60up.loc[(events_60up['Height'].isnull()) & (events_60up['Sex'] == 'F') & (events_60up['Sport'] == 'Basketball'), 'Height'] 


# #### After retrieving the selected rows and column, we can just put an "equal" sign to change the NaN values to 182 but doing it one by one (for each sport) is too slow; therefore, let's write a function which loops through each sport and then assigns the average values for each sport 

# In[ ]:


#How to iterate through to get pair of keys and values from python dictionary 
for k,v in f_height.items():
    print(k)


# In[ ]:


#Plug an average hight of sportswomen for each sport (type).
def f_height_fixer(df):
    for k,v in f_height.items():
        df.loc[(df['Height'].isnull()) & (df['Sex'] == 'F') & (df['Sport'] == str(k)), 'Height'] = v
    return df
#Do we have reasonable numbers of missing values for the height column? 
f_height_fixer(events_60up).info()        


# In[ ]:


# pandas' .info method tells us that we have more non-null values 
# but lets make sure that it has worked by taking a step further
events_60up.loc[(events_60up['Sex'] == 'F'), 'Height'].isnull().sum()


# In[ ]:


events_60up.loc[(events_60up['Sex'] == 'M'), 'Height'].isnull().sum()


# #### The function has fixed NaNs by replacing them with meaningful statistical means of sport types. Lets do the same thing for males

# In[ ]:


m_height = height_grouped.to_dict()['M']
def m_height_fixer(df):
    for k,v in m_height.items():
        df.loc[(df['Height'].isnull()) & (df['Sex'] == 'M') & (df['Sport'] == str(k)), 'Height'] = v
    return df
m_height_fixer(events_60up).info() 


# #### Now the Height column has floating numbers which needs to be rounded

# In[ ]:


events_60up['Height'].unique()


# In[ ]:


events_60up['Height'] = (round(events_60up['Height'])).astype(int)
events_60up['Height'].unique()


# In[ ]:


#Let's see the distribution of heights for males and females
events_60up['Height'].hist(by=events_60up['Sex'],figsize=(12,6),sharey=True, bins=25)


# ## I hope this "tutorial" was useful! If you are interested on doing further analysis, you have to answer the following bonus questions:
# ### - Do you think using sex and sport columns give us the most trustworthy results for missing values of height? Try to answer it by visualizing and using a mixture of other columns (i.e. sex and country; or sport, sex and year; or whatever combination you think suits based on plotting some graphs, grouping and applying some statistics) 
# ### - Age and Weight columns missing data points were not addressed. Try to solve the puzzle on your own by using various pandas and python methods 
# 
# ## If you liked my analysis, please upvote it. Also, please share your thoughts on the comment section below!
