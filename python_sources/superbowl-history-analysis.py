#!/usr/bin/env python
# coding: utf-8

# In this kernel I have performed Exploratory Data Analysis on the Superbowl History 1967-2020 dataset and tried to identify relationship between various features present in the dataset.

# I hope you find this kernel helpful and some **<font color='red'>UPVOTES</font>** would be very much appreciated

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Importing Required Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# setting plot style for all the plots
plt.style.use('fivethirtyeight')


# ## Describing the Dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')
df.head()


# The following data is given in the dataset
# 
# **1. Date:** Date on which the Superbowl Final was held.
# 
# **2. SB:** Superbowl Title
# 
# **3. Winner:** Name of the winning team.
# 
# **4. Winner Pts:** Points scored by the winning team.
# 
# **5. Loser:** Name of the losing team.
# 
# **6. Loser Pts:** Points scored by the losing team.
# 
# **7. MVP:** Most Valuable Player.
# 
# **8. Stadium:** Name of the stadium in which superbowl was held.
# 
# **9. City:** Name of the city where superbowl was held.
# 
# **10. State:** Name of the state where superbowl was held.

# ### Dimensions of dataset

# In[ ]:


print('Number of rows in the dataset: ', df.shape[0])
print('Number of columns in the dataset: ', df.shape[1])


# The dataset contains information about 54 superbowl finals.

# ### Features of the data set

# In[ ]:


df.info()


# There are **NO Null values** in the dataset. Also there are **3 numerical features** and **8 numerical features** in the dataset.

# Coverting the Date column in the dataframe to a datetime object

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.year


# ### Basic statistical details about the dataset

# In[ ]:


df.describe().round(decimals=3).drop('Date', axis=1)


# **The features described in the above data set are:**
# 
# **1. Count** tells us the number of NoN-empty rows in a feature.
# 
# **2. Mean** tells us the mean value of that feature.
# 
# **3. Std** tells us the Standard Deviation Value of that feature.
# 
# **4. Min** tells us the minimum value of that feature.
# 
# **5. 25%, 50%, and 75%** are the percentile/quartile of each features.
# 
# **6. Max** tells us the maximum value of that feature.

# ## Exploratory Data Analysis
# 

# ### 1. Points scored by winning and losing teams over the years

# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
ax.plot(df['Date'], df['Winner Pts'], marker='.', mew=5, color='dodgerblue', label='Winners')
ax.plot(df['Date'],df['Loser Pts'], marker='+', color='red', label='Losers', mew=3)
ax.set_xlabel('Years')
ax.set_ylabel('Points Scored')
ax.set_title('Points scored by Winning and Loosing Teams over the Years')
ax.legend()
plt.show()


# #### i. Highest points scored by a winning team till date

# In[ ]:


df[df['Winner Pts'] == df['Winner Pts'].max()][['Winner', 'Winner Pts', 'Date']]


# #### ii. Lowest points scored by a winning team till date

# In[ ]:


df[df['Winner Pts'] == df['Winner Pts'].min()][['Winner', 'Winner Pts', 'Date']]


# #### iii. Highest points scored by a losing team till date

# In[ ]:


df[df['Loser Pts'] == df['Loser Pts'].max()][['Loser', 'Loser Pts', 'Date']]


# #### iv. Lowest points scored by a losing team till date

# In[ ]:


df[df['Loser Pts'] == df['Loser Pts'].min()][['Loser', 'Loser Pts', 'Date']]


# ### 2. Number of times a team has won

# In[ ]:


# this dataframe contains the winning count of each team
winning_count = pd.DataFrame(df['Winner'].value_counts()).reset_index()
winning_count.index += 1
winning_count.rename(columns = {
    'index':'Team Name',
    'Winner':'Count'
}, inplace=True)

winning_count.sort_values(by='Count', ascending=False, inplace=True)


# In[ ]:


plt.figure(figsize=(20,7))
sns.barplot(y='Team Name', x='Count', data=winning_count,
           edgecolor='black',
           linewidth=2)
plt.title('Number of times each team has won throughout the years')
plt.xticks(rotation=90)
plt.show()


# **New England Patriots** and **Pittsburgh Steelers** have the most wins(11) till now.

# ### 3. Number of times a team has lost the match

# In[ ]:


losing_count = pd.DataFrame(df['Loser'].value_counts()).reset_index()
losing_count.index += 1
losing_count.rename(columns = {
    'index':'Team Name',
    'Loser':'Count'
}, inplace=True)

losing_count.sort_values(by='Count', ascending=False, inplace=True)


# In[ ]:


plt.figure(figsize=(20,7))
sns.barplot(x='Team Name', y='Count', data=losing_count,
           edgecolor='black',
           linewidth=2, palette='Blues_r')
plt.title('Number of times each team has won throughout the years')
plt.xticks(rotation=90)
plt.show()


# **Denver Broncos** and **New England Patriots** have the most number of loses till now.

# It's interesting to note that New **England Patriots** is the only team which has the most number of wins as well as loses in the SuperBowl.
# 
# The other top winner **Pittsburgh Steelers** has only lost 2 games till now.

# ### 4. Winning Margin over the years
# 
# Winning Margin is the difference between the scores of winning and losing teams[](http://)

# In[ ]:


df['Winning Margin'] = df['Winner Pts'] - df['Loser Pts']


# In[ ]:


plt.figure(figsize=(18,6))
plt.plot(df['Date'],df['Winning Margin'], marker='.', mew=3, linewidth=4,mec='black', color='dodgerblue')
plt.xlabel('Years')
plt.ylabel('Winning Margin')
plt.title('Winning Margin over the years')
plt.show()


# ### 5. Top 10 teams by winning points margin

# In[ ]:


df['Match'] = df['Winner'] + ' vs ' + df['Loser']
df2 = df.sort_values(by='Winning Margin', ascending=False)
df2 = df2.head(10)


# In[ ]:


plt.figure(figsize=(18,7))
sns.barplot(y='Match', x='Winning Margin', data=df2,
            edgecolor='black',linewidth=2)
plt.title('Top 10 teams by winning point margin')
plt.show()


# ### 6. Statewise Number of Matches

# In[ ]:


state_count = pd.DataFrame(df['State'].value_counts()).reset_index()
state_count.index += 1
state_count.rename(columns = {
    'index':'State',
    'State':'Count'
}, inplace=True)

state_count.sort_values(by='Count', ascending=False, inplace=True)


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(y='State', x='Count', data=state_count, edgecolor='black', linewidth=2)
g.set_title('Statewise Number of Matches', y=1.05)
g.set(xlabel='States', ylabel='Number of Matches held')
plt.show()


# ### 7. Top 5 cities where the most number of matches were held

# In[ ]:


city_count = pd.DataFrame(df['City'].value_counts()).reset_index()
city_count.index += 1
city_count.rename(columns = {
    'index':'City',
    'City':'Count'
}, inplace=True)

city_count.sort_values(by='Count', ascending=False, inplace=True)
city_count = city_count.head()


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(y='City', x='Count', data=city_count, edgecolor='black',linewidth=2)
g.set_title('Top 5 cities in terms of number of matches held', y=1.05)
g.set(xlabel='Cities', ylabel='Number of Matches held')
plt.show()


# ### 8. Top 5 MVP's(Most Valuable Players)

# In[ ]:


mvp_count = pd.DataFrame(df['MVP'].value_counts()).reset_index()
mvp_count.index += 1
mvp_count.rename(columns = {
    'index':'MVP',
    'MVP':'Count'
}, inplace=True)

mvp_count.sort_values(by='Count', ascending=False, inplace=True)
mvp_count = mvp_count.head()


# In[ ]:


plt.figure(figsize=(18,6))
g = sns.barplot(y='MVP', x='Count', data=mvp_count, edgecolor='black',linewidth=2)
g.set_title('Top 5 Most Valuable Players', y=1.05)
g.set(xlabel='MVP', ylabel='Count')
plt.show()


# ### 9. Stadiums where the most number of matches were held

# In[ ]:


stadium_count = pd.DataFrame(df['Stadium'].value_counts()).reset_index()
stadium_count.index += 1
stadium_count.rename(columns = {
    'index':'Stadium',
    'Stadium':'Count'
}, inplace=True)

stadium_count.sort_values(by='Count', ascending=False, inplace=True)
stadium_count = stadium_count.head(4)


# In[ ]:


plt.figure(figsize=(14,4))
g = sns.barplot(y='Stadium', x='Count', data=stadium_count, edgecolor='black',linewidth=2)
g.set_title('Stadiums where most number of matches were held', y=1.05)
g.set(xlabel='Stadium Names', ylabel='Count')
plt.show()


# Suggestions are welcome, **<font color='red'>UPVOTE</font>** if you found the notebook useful.
