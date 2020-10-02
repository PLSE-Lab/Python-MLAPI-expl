#!/usr/bin/env python
# coding: utf-8

# # Data Story 6 and 7

# This notebook is used as a part of the day 6 and 7 of my data stories.
# 
# Check my other notebooks for more data stories

# Today I am going to analyze the data about US border crossings for various purposes.
# 
# Let's see what I can make out of the data

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# importing important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# For now these libraries are just fine. Let's check the data and then decide whether I will need more or not.

# In[ ]:


#summoning the data demon
data = pd.read_csv('../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
data.head(2)


# In[ ]:


# let's check the shape to make an estimate of how big the data is
data.shape


# Okay so the dataset is huge. 
# 
# It's not possible to analyze all at once. 
# 
# So it's better to break down the population into various samples and try to analyze them.

# If you are wondering what population and samples are then check the link below.
# 
# [https://towardsdatascience.com/what-is-the-difference-between-population-and-sample-e13d17746b16](http://)
# 
# So samples are basically chunks of data used for a partiular task.(correct me if I am wrong)

# In[ ]:


#let's check the decription of the data
data.describe(include='all')


# So it is confirmed that the dataset don't have any missing values at all.
# 
# Now let's decide on what qustions do I want to answer using the dataset.

# In[ ]:


# lets take a quick loot at the columns
data.columns


# Some of the questions that come to my mind after seeing the columns names are as follows:-
# 
# 1. Which is the busiest port?
# 2. Which state is having the highest border crossing through?
# 3. Which borders are the most used?
# 4. Which country is most associated in the border crossings?
# 5. Is there any pattern in the number of crossings with respect to any season of the year?
# 6. What type of vehicles are mostly used for crossing border?
# 7. Relation of the vehicles with value and comparing the value for different countries.

# So let's start from the first question and move down the list.

# # 1. Which is the busiest port?

# In[ ]:


#making a copy 
df = data.copy()

#lets look at the data
df.head()


# In[ ]:


#lets check unique number of ports in the data
print(df['Port Name'].nunique())


# In[ ]:


#let's check which port has highest frequency
df['Port Name'].value_counts()


# So the Eastport seems to be the most busy among other ports. Let's check what state is it in, it's port number and also how much value it is brining to US.
# 
# But before all that I need to check the number of unique values in the dates columns once.

# In[ ]:


#converting Date to datetime instance
df['Date'] = pd.to_datetime(df['Date'])

#checking the numbr of unique dates
df['Date'].nunique()


# In[ ]:


#checking the beginning and end of data
df['Date']


# So looking at the date column in partiular it is clear that the data contains info on every month from January 1996 to February 2020

# It is found out that EastPort appears in the dataset most frequently.
# 
# But one thing to remember is that checking the port on an yearly basis may give us more insights.

# In[ ]:


df['Date'].dt.year.value_counts()


# In[ ]:


# storing all unique years in a list
# ignoring 2020 as it have only 2 months of data
years = df['Date'].dt.year.unique().tolist()
years.remove(2020)
type(years)


# Now for each of the years, I will check out the busiest port and try to visualise them.

# In[ ]:


df2 = df['Port Name'].value_counts().reset_index()
df2.loc[0]


# In[ ]:


#creating two lists
counts = [] #counts of the busiest ports
busiest_ports = [] #busiest ports of the year
for year in years:
    df1 = df[df['Date'].dt.year==year]
    df1 = df1['Port Name'].value_counts().reset_index()
    busiest_ports.append(df1.loc[0]['index'])
    counts.append(df1.loc[0]['Port Name'])
    
fig = plt.figure(figsize=(15,8))
sns.barplot(years, counts)
plt.show()


# In[ ]:


busiest_ports[:20]


# So now some interesting data came up.
# 
# Eastport is proved to be the most busiest port since 2003 but one thing to notice is that the number of crossing border is mostly same for consecutive years.
# 
# So it will be right to assume that the Eastport was built mainly for this purpose on 2003 and hence it saw huge rise.

# Now let's plot the 5 busiest ports for each year

# In[ ]:


fig = plt.figure(figsize=(28,30))
for year,num in zip(years, range(1,25)):
    df1 = df[df['Date'].dt.year==year]
    df1 = df1['Port Name'].value_counts().reset_index()
    ax = fig.add_subplot(8,3,num)
    ax.bar(df1.loc[:4]['index'], df1.loc[:4]['Port Name'])
    ax.set_title(year)


# It can be seen that the port Otay Mesa remained in the top 5 till 2016 but then it got out of sight. 
# 
# It can be assumed that the port was closed for some reason.
# 
# Now that we have found out the busiest I think it's about time I move to the second question.

# # 2. Which state is having the highest border crossing through?

# In[ ]:


#getting a new copy of the data
df = data.copy()

#lets see the data
df.head(2)


# In[ ]:


#lets see the unique number of states in the dataset
df['State'].nunique()


# Okay so there are only 15 states involved with ports. This will be quick to analyse.
# 
# Let's see which state let most of the crossings in US in 2020

# In[ ]:


#converting date to datetime instance
df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


# creating a separate dataframe for 2020 data
df_2020 = df[df['Date'].dt.year==2020]


# In[ ]:


# checking unique states in 2020
df_2020['State'].unique()


# Now I have two options to choose the state bringing most crossings. Either I choose the count as I did in the case of ports.
# 
# Or I can choose the state based on the summation of all values that the state has brought in.
# 
# The second way looks a bit ambiguous since it is directly related to the measure(vehicle) used for crossing. 
# 
# The count looks more fair and hence I will stick to it.

# In[ ]:


df_2020['State'].value_counts()


# So the state with the most crossing is found to be North Dakota. Lets visualize the top 5

# In[ ]:


df1 = df_2020['State'].value_counts().reset_index()
fig = plt.figure(figsize=(10,5))
barlist = plt.bar(df1.loc[:4]['index'], df1.loc[:4]['State'])
barlist[0].set_color('m')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('c')
barlist[4].set_color('y')
plt.plot(df1.loc[:4]['index'], df1.loc[:4]['State'], c='red',linewidth=7.0)
plt.title('Top 5 states with highest cossings')
plt.show()


# I will end my today's analysis here. Let's continue with the next questions tomorrow.

# Okay so let's continue with the third question.

# # 3. Which borders are the most used?

# In[ ]:


data.head(3)


# Let's try and analyze which borders were the most busy in 2019

# In[ ]:


#change to datetimg
data['Date'] = pd.to_datetime(data['Date'])

#separating 2019 data
df_2019 = data[data['Date'].dt.year==2019]

#checking most used border
df_2019['Border'].value_counts()


# In[ ]:


borders = df_2019['Border'].value_counts().reset_index()

fig = plt.figure(figsize=(10,5))
sns.barplot(borders['index'], borders['Border'])
plt.xlabel('Borders')
plt.ylabel('Number of crossing')
plt.title('Border vs crossing in 2019')
plt.show()


# So it can be clearly seen that in 2019, US-Canada Border was the busiest but is that the same case for all of the previous years.
# 
# I think it's worth a try.

# In[ ]:


#getting unique years
years = data['Date'].dt.year.unique().tolist()

#getting border data for each year
canada = []
mexico = []
for year in years:
    df = data[data['Date'].dt.year==year]
    borders = df['Border'].value_counts().reset_index()
    canada.append(int(borders.loc[borders['index']=='US-Canada Border']['Border']))
    mexico.append(int(borders.loc[borders['index']=='US-Mexico Border']['Border']))


# In[ ]:


len(canada), len(mexico)


# In[ ]:


years


# In[ ]:


fig, ax = plt.subplots(figsize=(20,7))
x = np.arange(len(years))
width = 0.35
ax.bar(x-width/2, canada, width, label='US-Canada Border')
ax.bar(x+width/2, mexico, width, label='US-Mexico Border')

ax.set_ylabel('Crossing counts')
ax.set_title('Year wise border data')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()


# So we can see that the US-Canada Border maintains a higher border crossing count since 1996.

# Moving on to the next question.

# # 4. Which country is most associated in the border crossings?

# In[ ]:


data.head(3)


# From the previous question(busiest port) itself we came to know that Canada is the Country involved in highest number of US Border crossing.

# So the answer to the current question is also Canada.
# 
# This also provides us with some insights on the relation between the countries US and Canada.

# Moving to the next question.

# # 5. Is there any pattern in the number of crossings with respect to any season of the year?

# Now this is a very particular question.
# 
# For answering this first I need to decide which seasons to choose and what months will each season comprise of.
# 
# Rather than seasons lets divide the year into 4 quarters(each of 3 months).
# 
# Lets check this logic for 2019.

# In[ ]:


#separating 2019 data
df_2019 = data[data['Date'].dt.year==2019]

quars = []
for i in range(1,12,3):
    quars.append(df_2019[(df_2019['Date'].dt.month == i)|(df_2019['Date'].dt.month == i+1)|(df_2019['Date'].dt.month == i+2)].shape[0])
    
len(quars)


# In[ ]:


fig = plt.figure(figsize=(10,7))
plt.plot(['1st quar', '2nd quar','3rd quar', '4th quar'], quars)
plt.show()


# Now I can see that crossing are low in 1st and 4th quarter and increases in the 2nd quarter to highest.
# 
# Lets now visualize this for all years to see if same pattern is followed.

# In[ ]:


#getting unique years
years = data['Date'].dt.year.unique().tolist()

#getting border data for each year
first = []
second = []
third = []
fourth = []
for year in years:
    df = data[data['Date'].dt.year==year]
    borders = df['Border'].value_counts().reset_index()
    first.append(df[(df['Date'].dt.month == 1)|(df['Date'].dt.month == 2)|(df['Date'].dt.month == 3)].shape[0])
    second.append(df[(df['Date'].dt.month == 4)|(df['Date'].dt.month == 5)|(df['Date'].dt.month == 6)].shape[0])
    third.append(df[(df['Date'].dt.month == 7)|(df['Date'].dt.month == 8)|(df['Date'].dt.month == 9)].shape[0])
    fourth.append(df[(df['Date'].dt.month == 10)|(df['Date'].dt.month == 11)|(df['Date'].dt.month == 12)].shape[0])


# In[ ]:


fig, ax = plt.subplots(figsize=(20,7))
x = np.arange(len(years))
width = 0.2
ax.bar(x-(width*3)/2, first, width, label='First Quarter')
ax.bar(x-width/2, second, width, label='Second Quarter')
ax.bar(x+width/2, third, width, label='Third Quarter')
ax.bar(x+(width*3)/2, fourth, width, label='Fourth Quarter')

ax.set_ylabel('Crossing counts')
ax.set_xlabel('Years')
ax.set_title('Quarter wise crossing data')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()


# Okay so similar pattern is observed in most cases but 2016 shows some high variation.
# 
# The 1st and 2nd quarters are much higher compared to the others.
# 
# Not sure why.

# So the final conclusion is 2nd and 3rd quaretrs are when most crossing occur.

# # 6. What type of vehicles are mostly used for crossing border?

# For measuring this I will make use of the 'Measure' column because that's the column that contains data about what vehicles are used for the crossing.
# 
# Let's first see how many unique type of vehicles are there.

# In[ ]:


#checking all unique values in the 'Measure' column
data['Measure'].unique()


# Okay so there are a total of 12 different types of vehicles used for crossing the borders
# 
# Let's visualize their usage

# In[ ]:


#creating table for vehicle count
df = data['Measure'].value_counts().reset_index()

fig = plt.figure(figsize=(20,7))
sns.barplot(df['Measure'], df['index'])
plt.title('Vehicles used for crossing border')
plt.show()


# So it can clearly be seen that personal vehicles are the ones that are used the most.
# 
# One thing is surprising is that pedestrian count is more than all types of rail vehicles. This means there is low frequency of rail vehicles to cross the border.

# Moving to the next question

# # 7. Relation of the vehicles with value and comparing the value for different countries.

# Okay so the final question. This question is not actually clear in my own mind.
# 
# But will try and sort it out using the data.

# Let's break it up into two parts:-
# 
# 1. How is the vehicle related to the 'value' column?
# 2. How each country Canada and Mexico are related to the 'value' column?
# 
# But first let's see what the 'value' column actually signify.
# 
# According to the data dictionary
# Value -----> Counts of people crossing

# In[ ]:


# checking the sum of values for each vehicle
vehicles = data['Measure'].unique()
values = []

for vehicle in vehicles:
    df = data[data['Measure']==vehicle]
    values.append(df['Value'].mean()) #if you wonder why I took mean then check the below markdown


# So I took the mean in the above cell for the 'Value' column because sum is directly proportional the number fo such vehicles which we already saw is highly different.
# 
# So I took the mean to avoid any ambiguity.

# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.barplot(values, vehicles)
plt.title('Value by each vehicle')
plt.show()


# Okay so this is something interesting. While comparing the count of each vehicle the variation was a lot less.
# 
# But on seeing the number of people who crossed using those vehicles got much higher.

# Moving on to the second part.
# 
# For the second part we need to separate the values on the basis of the border and compare using value

# In[ ]:


data.head(2)


# In[ ]:


#separating on countries
countries = ['Canada', 'Mexico']
values = []

for country in countries:
    df = data[data['Border']=='US-{} Border'.format(country)]
    values.append(df['Value'].mean())


# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.barplot(values, countries)
plt.title('Country vs Value')
plt.xlabel('Number of People crossing US border from this country')
plt.show()


# Okay now this is something very interesting. While the count of crossing is very high frm the Canada border, as we have analysed in a previous question.
# 
# More number of people are actually from Mexico.

# So I will end my analysis on this dataset there. Hope you find it insightful.
# 
# If ou find any value then please upvote. It means a lot to me.

# # Good Bye. See you aain in a new dataset tomorrow

# In[ ]:




