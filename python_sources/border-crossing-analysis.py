#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# This dataset contains all the incidents of crossing the border into the US as provided by the Bureau of Transportation Statistics, Govt. of the US. This dataset tells about the incoming counts into the US.
# ---
# 
# This data can be useful to predict the daily or weekly or monthly or annual traffic that's going to accumulate on the borders so that the authorities can be aware of the number beforehand.
# ---

# Importing Libraries
# ---

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Creating Initial DF
# ---

# In[ ]:


df1 = pd.read_csv('../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
df1.isnull().sum()


# # Data Analysis

# Checking out what the data looks like
# ---

# In[ ]:


df1.head()


# Changing the name of the "Measure" and "Value" columns so that the data is a little easier to understand
# ---

# In[ ]:


df1['Mode_of_Transportation'] = df1['Measure']
df1['Crossings'] = df1['Value']
df1 = df1.drop(columns=["Measure", "Value"])
df1.head()


# Checking the data types of each column
# ---

# In[ ]:


df1.dtypes


# How many different ports/states/borders are we working with?
# ---

# In[ ]:


df1.nunique()


# How many total crossings are in the dataset?
# ---

# In[ ]:


df1['Crossings'].sum()


# Answer: a lot
# ---

# What's the breakdown of border crossings by state?
# ---

# In[ ]:


df1.groupby('State')['Crossings'].sum()


# Let's make a new column for the total border crossings by state
# --

# In[ ]:


df1['Total_Crossings_By_State'] = df1.groupby('State')['Crossings'].transform('sum')
df1[['State', 'Total_Crossings_By_State']]


# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=df1['State'], y=df1['Total_Crossings_By_State'], ax=ax)
plt.tight_layout()
plt.title('Breakdown of Border Crossings by State')
plt.show()


# Checking the breakdown of crossings by border
# ---

# In[ ]:


df1.groupby('Border')['Crossings'].sum()


# Making a new column for the total number of crossings in the dataset by border
# --

# In[ ]:


df1['Total_Crossings_By_Border'] = df1.groupby('Border')['Crossings'].transform('sum')
df1['Total_Crossings_By_Border']


# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=df1['Border'], y=df1['Total_Crossings_By_Border'], ax=ax)
plt.tight_layout()
plt.title('Breakdown of Border Crossings by Border')
plt.show()


# In[ ]:


df1['Total_Crossings_By_Border'] = df1.groupby('Border')['Crossings'].transform('sum')
df1[['Border', 'Total_Crossings_By_Border']]


# Now let's look at the border crossing trends over the years
# ---

# First, we're going to break down the "Date" column into 3 seperate columns for day/month/year
# ---

# In[ ]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1['Year'] = df1['Date'].dt.year
df1['Month'] = df1['Date'].dt.month
df1['Day'] = df1['Date'].dt.day
df1 = df1.drop(columns="Date")
df1.head()


# In[ ]:


df1['Total_Crossings_By_Year'] = df1.groupby('Year')['Crossings'].transform('sum')


# To help us do a time series analysis, let's a make a new colummn for the number of crossings in the year for that given row.
# ---

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.lineplot(x=df1['Year'], y=df1['Total_Crossings_By_Year'], ax=ax)
plt.title('Border Crossing Trends by Year')
plt.tight_layout()
plt.show()


# Let's plot a time series analysis of the crossings by year
# ---

# Now that we know what the trend seems to be for overall crossings, now let's find a trend for each individual border. To do this, I'll start with making a new DF for each border.
# --

# In[ ]:


canada_df = df1.loc[df1['Border'] == 'US-Canada Border']
canada_df.head()


# In[ ]:


mexico_df = df1.loc[df1['Border'] == 'US-Mexico Border']
mexico_df.head()


# Now I will add a new column to each new DF representing the total annual crossings for that specific border
# --

# In[ ]:


canada_df['Canada_Crossings_By_Year'] = canada_df.groupby('Year')['Crossings'].transform('sum')
canada_df[['Year', 'Canada_Crossings_By_Year']]


# In[ ]:


mexico_df['Mexico_Crossings_By_Year'] = mexico_df.groupby('Year')['Crossings'].transform('sum')
mexico_df[['Year', 'Mexico_Crossings_By_Year']]


# In[ ]:





# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.lineplot(x=df1['Year'], y=canada_df['Canada_Crossings_By_Year'], ax=ax)
plt.title('Border Crossing Trends by Year \n (US-Canada Border)')
plt.tight_layout()
plt.show()


# Making a new df2 that is a combination of the DF's I made for each border. By combining those two DF's I essentially have df1 + the new columns I just made to represent annual crossings by year for each specific border.
# --

# In[ ]:


df2 = pd.concat([canada_df, mexico_df], ignore_index=False)
df2.head()


# In[ ]:


df2['Mexico_Crossings_By_Year'].fillna('None', inplace=True)
df2['Canada_Crossings_By_Year'].fillna('None', inplace=True)
df2.isnull().sum()


# And now to plot the trends
# --

# In[ ]:


df2.groupby(['State', 'Year'])['Crossings'].sum()


# Here I am making a new column that represents the crossings for each year for each state.
# --

# In[ ]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.lineplot(x=df1['Year'], y=mexico_df['Mexico_Crossings_By_Year'], ax=ax)
plt.title('Border Crossing Trends by Year \n (US-Mexico Border)')
plt.tight_layout()
plt.show()


# Now let's plot the annual crossing trends for each state in the dataset
# --

# In[ ]:


for i in df2['State'].unique():
    if str(i) == 'OH':
        pass
    else:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_axes([0,0,1,1])
        sns.lineplot(
            x=df2['Year'],
            y=df2.loc[df2['State'] == str(i)]['Yearly_Crossings_By_State'],
            ax=ax
        )
        plt.title(f'Yearly Crossings for {i}')
        plt.tight_layout()
        plt.show()


# Note that Ohio has VERY few datapoints so that trend is essentially meaningless. Because of that I am not plotting the annual trend for border crossins in to / out of Ohio, even though it is my home state :(
# --

# Now I am going to check monthly trends of border crossings for each border as well as each individual state. I will do this using the total average for each month in the given border/state
# --

# Lets start with the borders. I will first make a new column in df2 representing the average crossings for each month for both borders.
# --

# In[ ]:


df2['Average_Monthly_Crossings_By_Month'] =     df2.groupby(['Border','Month'])['Crossings'].transform('mean')

df2[['Border', 'State', 'Month', 'Average_Monthly_Crossings_By_Month']]


# Not let's make a line plot for each border
# --

# In[ ]:


for i in df2['Border'].unique():
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_axes([0,0,1,1])
    sns.lineplot(
        x=df2['Month'],
        y=df2.loc[df2['Border'] == str(i)]['Average_Monthly_Crossings_By_Month'],
        ax=ax
    )
    plt.title(f'Average Monthly Crossings for the {i}')
    plt.tight_layout()
    plt.show()


# So as we can see, the US-Canada border definitely has some seasonality to it's border crossings. The US-Mexico border seems to fluctuate througout the year.
# --

# Now, let's follow the same process to check the monthly trends for each border state.
# --

# In[ ]:


df2['Average_Monthly_Crossings_By_State']  =     df2.groupby(['State', 'Month'])['Crossings'].transform('mean')

df2[['State', 'Month', 'Average_Monthly_Crossings_By_State']]


# In[ ]:


for i in df2['State'].unique():
    if str(i) == 'OH':
        pass
    else:
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_axes([0,0,1,1])
        sns.lineplot(
            x=df2['Month'],
            y=df2.loc[df2['State'] == str(i)]['Average_Monthly_Crossings_By_State'],
            ax=ax
        )
        plt.title(f'Average Monthly Crossings for {i}')
        plt.tight_layout()
        plt.show()


# As expected per the border trends. The states on the US-Canada border have a great deal of seasonality to their crossings (most crossings are in the summer) and the US-Mexico border crossings fluctuate throughout the year.
# --

# # That's enough for today, time to play video games.

# In[ ]:


df2['Yearly_Crossings_By_State'] = df2.groupby(['State', 'Year'])['Crossings'].transform('sum')

df2[['Year', 'State', 'Yearly_Crossings_By_State']]


# Now I'm going to find the trends for each state in the dataset.
# ---
