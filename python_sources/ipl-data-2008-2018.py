#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
from subprocess import check_output

import warnings                                            # Ignore warning related to pandas_profiling
warnings.filterwarnings('ignore') 


def annot_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/matches.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# ### Data cleaning: drop umpire3 column as you can see it has too much missing values.

# In[5]:


df.drop(['umpire3'], axis = 1, inplace = True)


# In[6]:


df.columns


# In[7]:


print(df['winner'].unique())
print(df['city'].unique())


# In[8]:


df.replace('Rising Pune Supergiant','Rising Pune Supergiants', inplace = True)
df.replace('Bangalore','Bengaluru', inplace = True)


# In[9]:


#fill missing values
df['city'].fillna(df['venue'], inplace = True)
df['winner'].fillna(df['result'], inplace = True)
df['player_of_match'].fillna(df['result'], inplace = True)
df['umpire1'].fillna('unknown', inplace = True)
df['umpire2'].fillna('unknown', inplace = True)


# ## Total number of matches b/w 2008-2018

# In[10]:


total_num_of_matches = df.groupby('season')['id'].count()
plt.figure(figsize=(10,5))
ax = sns.countplot("season", data = df, palette='viridis')
plt.title('Total number of matches b/w 2008-2018')
plt.ylabel('Number of Matches.')
annot_plot(ax,0.08,1)
plt.show()


# ## Total number of wins by each team b/w 2008-18:

# In[11]:


plt.figure(figsize=(12,7))
ax = sns.countplot("winner", data = df, order = df['winner'].value_counts().index,palette='viridis')
plt.title("Total number of wins by each team b/w 2008-2018")
plt.xticks(rotation=45, ha = 'right')
plt.ylabel('Number of matches')
annot_plot(ax,0.08,1)
plt.show()


# ## Number of matches won by each team b/w 2008-18:

# In[12]:


max_times_winner = df.groupby('season')['winner'].value_counts()
max_times_winner


# ## Graphs for all the season winners:

# **first way of plotting graph for all the season winners.**

# In[13]:


groups = max_times_winner.groupby('season')
fig = plt.figure()
count = 1

for year, group in groups:
    ax = fig.add_subplot(4,3,count)
    ax.set_title(year)
    ax = group[year].plot.bar(figsize = (10,15), width = 0.8)
    
    count+=1;
    
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('Matches Won')
    
    total_of_matches = []
    for i in ax.patches:
        total_of_matches.append(i.get_height())
    total = sum(total_of_matches)
    
    for i in ax.patches:
        ax.text(i.get_x()+0.2, i.get_height()-1.5,s= i.get_height(),color="black",fontweight='bold')
plt.tight_layout()
plt.show()


# **Second way of plotting graph for all the season winners.******

# In[14]:


plt.figure(figsize = (12,6))
ax = sns.catplot('winner', col='season',aspect=1, data = df, col_wrap = 2, kind = 'count',legend=True ,order = df['winner'].value_counts().index,palette='viridis')
ax.set_xticklabels(rotation=90, ha = 'right')
ax.set( ylabel = 'Number of matches')
plt.ylabel('Number of Matches')
plt.show()


# ## does toss winning affects the match winner? 

# In[15]:


plt.figure(figsize=(12,7))
ax = sns.countplot("winner", data = df, hue = 'toss_decision',order = df['winner'].value_counts().index,palette='viridis')
plt.title("Total number of wins for every team b/w 2008-2018")
plt.xticks(rotation=45, ha = 'right')
plt.ylabel('Number of Matches')
annot_plot(ax,0.08,1)
plt.show()


# In[16]:


Total_matches_played = df['team1'].value_counts() + df['team2'].value_counts()

toss_won = df['toss_winner'].value_counts()
toss_win_success_rate = (toss_won/Total_matches_played)*100
toss_win_success_rate_sort = toss_win_success_rate.sort_values(ascending = False)
plt.figure(figsize = (10,5))
ax = sns.barplot(x =toss_win_success_rate_sort.index, y = toss_win_success_rate_sort,palette='viridis')
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Toss Win success ratio.')
annot_plot(ax,0.08,1)
plt.show()


# ## Do you know who won the most player of the match?

# In[17]:


plt.figure(figsize=(12,6))

ax = sns.countplot("player_of_match", data = df,order = df['player_of_match'].value_counts()[:20].index,palette='viridis')
plt.title("Total number of Player of the match. ")
plt.xticks(rotation=90, ha = 'right')
plt.ylabel('Number of Player of the match')
plt.xlabel('Name of the top 20 Player of the match.')
annot_plot(ax,0.08,1)
plt.show()


# **Interpretation:** well, you have your answer :)

# ## Success rate of winning matches:

# In[18]:


matches_won = df.groupby('winner').count()
total_matches = df['team1'].value_counts()+ df['team2'].value_counts()

matches_won['Total matches'] = total_matches
win_df = matches_won[["Total matches","result"]]


# In[19]:


ax = win_df[['Total matches','result']].sort_values('Total matches',ascending=False).plot.bar(figsize=(12,6))
plt.ylabel('Total number of matches played')
annot_plot(ax,0.08,1)


# In[20]:


sucess_ratio = round((matches_won['id']/total_matches),4)*100
sucess_ratio_sort = sucess_ratio.sort_values(ascending = False)
plt.figure(figsize = (10,7))
ax = sns.barplot(x = sucess_ratio_sort.index, y = sucess_ratio_sort, palette='viridis' )
annot_plot(ax,0.08,1)
plt.xticks(rotation=45, ha = 'right')
plt.ylabel('Success rate of wining')
plt.show()


# ## Each Season Winner:

# In[21]:


each_season_winner = df.groupby('season')['season','winner'].tail(1)
each_season_winner_sort = each_season_winner.sort_values('season',ascending = True)
each_season_winner_sort


# In[22]:


sns.countplot('winner', data = each_season_winner_sort)
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Number of seasons won by any team.')
plt.show()


# ## Effect of home ground:

# In[23]:


plt.figure(figsize = (12,6))
venue = df[['city','winner','season']]
venue_season = venue[venue['season'] == 2018]
ax = sns.countplot('city', data = venue_season, hue = 'winner' )
plt.xticks(rotation=30, ha = 'right')
plt.ylabel('Number of matches.')
plt.show()


# **Interpretaion:** well, you can see in 2018 all the teams won most of the matches at their home ground.

# In[24]:


df['winner'].unique()


# In[25]:


df.replace('Sunrisers Hyderabad','SRH',inplace = True)
df.replace('Rising Pune Supergiants','RPS',inplace = True)
df.replace('Kolkata Knight Riders','KKR',inplace = True)
df.replace('Mumbai Indians','MI',inplace = True)
df.replace('Delhi Daredevils','DD',inplace = True)
df.replace('Gujarat Lions','GL',inplace = True)
df.replace('Chennai Super Kings','CSK',inplace = True)
df.replace('Rajasthan Royals','RR',inplace = True)
df.replace('Deccan Chargers','DC',inplace = True)
df.replace('Pune Warriors','PW',inplace = True)
df.replace('Kochi Tuskers Kerala','KTK',inplace = True)
df.replace('no result','Draw',inplace = True)
df.replace('Royal Challengers Bangalore','RCB',inplace = True)
df.replace('Kings XI Punjab','KXP',inplace = True)



# In[26]:


city_winner = df.groupby('city')['winner'].value_counts()

count=1
fig = plt.figure()

groups=city_winner.groupby('city')
for city,group in groups:
    ax = fig.add_subplot(8,4,count)
    ax.set_title(city)
    ax=group[city].plot(kind="bar",figsize=(13,20),width=0.8)
    
    count=count+1
    
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('Matches Won')
    
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
        #print(i.get_height())
    total = sum(totals)
    
    for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+0.2, i.get_height()-0.9,s= i.get_height(),color="black",fontweight='bold')
    
    
plt.tight_layout()
plt.show()


# In[27]:


df.replace('Dubai International Cricket Stadium','Dubai ICS', inplace = True)


# In[28]:


winner_city = df.groupby('winner')['city'].value_counts()

count=1
fig = plt.figure()

groups=winner_city.groupby('winner')
for winner,group in groups:
    ax = fig.add_subplot(7,2,count)
    ax.set_title(winner)
    ax=group[winner].plot(kind="bar",figsize=(12,20),width=0.8)
    
    count+=1
    
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('Matches Won')
    
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
        #print(i.get_height())
    total = sum(totals)
    
    for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x()+0.2, i.get_height()-0.9,s= i.get_height(),color="black",fontweight='bold')
    
    
plt.tight_layout()
plt.show()

