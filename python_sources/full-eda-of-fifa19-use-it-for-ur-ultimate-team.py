#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/data.csv',index_col=0)


# In[ ]:


df.head()


# In[ ]:


df.columns # some of these are unnecessary, and we'll get rid of them


# In[ ]:


df.drop(columns=['ID','Photo','Flag','Club Logo','Work Rate','Weak Foot','Preferred Foot','Body Type','Real Face',
                 'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
                 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'], inplace=True)


# In[ ]:


df.dropna(subset=['Height','Weight'], inplace=True) # drop all rows where either of height or weight is missing


# **In the following few lines, we'll be converting the current string representations of heights, weights, values, wages and release clauses to their respective default formats, i.e. heights to inches, weights to  lbs, values, wages and release clauses to 'thousands of euros'**

# In[ ]:


def to_inches(row):
    lst = row.split("'")
    lst = pd.to_numeric(lst)
    new_height = (lst[0] * 12) + lst[1]
    return new_height


# In[ ]:


df['Height'] = df['Height'].apply(to_inches)


# In[ ]:


def weight_in_num(row):
    lst = row.split('lbs')
    lst = pd.to_numeric(lst)
    new_weight = lst[0]
    return new_weight


# In[ ]:


df['Weight'] = df['Weight'].apply(weight_in_num)


# In[ ]:


def value_change(row): #converting everything in terms of 'thousands of euros'
    if row[-1] == 'M':
        row = row[1:-1] # getting rid of euro and million sign
        num = pd.to_numeric(row)
        num = num * 1000
        return num
    else:
        return pd.to_numeric(row[1:-1])


# In[ ]:


df['Value'] = df['Value'].apply(value_change)


# In[ ]:


def release_clause_change(row):
    if row[-1] == 'M':
        row = row[1:-1]
        num = pd.to_numeric(row)
        num = num * 1000
        return num
    else:
        return pd.to_numeric(row[1:-1])


# In[ ]:


df['Release Clause'] = df[df['Release Clause'].notnull()]['Release Clause'].apply(release_clause_change)


# In[ ]:


def wage_change(row): # make wages in thousands of euros
    return pd.to_numeric(row[1:-1])


# In[ ]:


df['Wage'] = df['Wage'].apply(wage_change)


# In[ ]:


df.head(2)


# In[ ]:


sns.scatterplot(y='Special',x='SprintSpeed',data=df) # players with more 'special' rating tend to be 'speedier'


# In[ ]:


sns.scatterplot(x='Age',y='Value',data=df) # almost represents a normal distribution, which is good to see. Mr. Gauss would be a man.


# **Just in case you wanted to know the top players right now**

# In[ ]:


top_class_players = df[df['Value'] > 40000] # players with a market value of more than 40 million euros
top_class_players.head()


# And these are the teams for whom these players play... (Do you see your team here?!)

# In[ ]:


top_class_players.groupby('Club')['Name'].count().sort_values(ascending=False) # No wonder Madrid won 4 out of the last 6 Champions Leagues!


# In[ ]:


sns.boxplot(y='Vision',x='Height',data=df) # reaffirms the notion that shorter players are more creative, for eg. Cazorla, Messi, Mata, Silva etc.


# In[ ]:


sns.boxplot(y='Balance',x='Height',data=df) # and this is why shorter players are quicker, they can balance themseles better!


# In[ ]:


sns.boxplot(y='Strength',x='Height',data=df) # reaffirms the notion that shorter players are kinda weaker in duels


# **Now a bombshell is about to be dropeed!**

# In[ ]:


df.corr()['Age'] # no visible correlation anywhere! reaffirms the notion that age is just a number!


# **Now comes the main analysis, predicting the World Cup 2022 contenders!**

# We need a list of people who are raw today, but hold potential to be bright in the future... at least in the eyes of guys at FIFA!

# In[ ]:


df_raw_players_with_potential = df[(df['Overall'] <= 70) & (df['Potential'] > 80)]


# This will come in handy if you wanna get a feel of someone like Arsene Wenger or Guardiola or Klopp and develop raw players into superstars!

# In[ ]:


df_raw_players_with_potential.sort_values(by='Potential',ascending=False)  # watch out for these players!


# Wouldn't surprise me to find scouts of the giant clubs sitting in the stands of these teams' games: of AS Monaco, Hertha Berlin, Anderlecht, etc

# In[ ]:


df_raw_players_with_potential.groupby('Club')['Club'].count().sort_values(ascending=False)


# Now, for the World Cup probables, we'll set the benchmark at the age of the player being no more than 33 in 2022, i.e. 30 today, and a 'Potential' of no less than 75, if he is to be in contention for a place in the World Cup 2022 squad... pretty reasonable I guess

# In[ ]:


df_worldcup2022_probables = df[(df.Age<=30) & (df.Potential>75)]


# In[ ]:


df_worldcup2022_probables.sample()


# But this list also contains players from teams who have less than 20 players listed. And a world-cup preliminary squad is more than that, hence we'll happily drop those rows - anyway those teams wouldn't be fancied to win, if they don't have enough good players listed on FIFA.

# In[ ]:


df_worldcup2022_probables.groupby('Nationality')['Potential'].count()


# In[ ]:


df_worldcup2022_probables.groupby('Nationality').filter(lambda x: len(x) > 20)['Nationality'].value_counts()


# In[ ]:


df_worldcup2022_probables = df_worldcup2022_probables[df_worldcup2022_probables.groupby('Nationality')['Nationality'].transform('size') > 20]


# ***NOW COMES THE MAIN PREDICTIONS!***

# In[ ]:


df_worldcup2022_probables.groupby('Nationality')['Potential'].mean().sort_values(ascending=False)


# **No wonder Croatia and France feature at the top - both these teams had youthful squads in last year's World Cup, and reached the final, which means both of 'em will have more experienced squads in three years' time, and happier prospects for them both!**
