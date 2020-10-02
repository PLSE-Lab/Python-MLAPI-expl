#!/usr/bin/env python
# coding: utf-8

# # FA CUP FINAL : EDA AND VISUALIZATION

# ## Introduction
# 
# Welcome to my kernel for FA CUP Final Dataset. Here i'm going to give a short example and tutorial performing EDA to gather knowledge and information.
# 
# The first step here is we are going to import all the modules for this EDA, those modules are numpy, pandas, matplotlib and seaborn.

# ## Import Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(18,9), 'lines.linewidth': 5, 'lines.markersize': 5, "axes.labelsize":15}, style="whitegrid")


# ## Read File
# 
# Next we are going to read the csv file and open it on DataFrame.

# In[ ]:


df = pd.read_csv('../input/fa-cup-final-all-season-1872-2020/FA emirates Cup Final.csv')


# ## Quick look
# 
# let's take a look at our dataset by printing first 5 row.

# In[ ]:


df.head()


# let's take a look at our dataset info

# In[ ]:


df.info()


# Here we got four feature available, they are Year of the tournament held, the winner of the tournament, second place or runner up, and final score.

# ## Gather Information
# 
# It is time to gather some usefull information, lets start by doing something basic, like get the most winning team, most second place, and most appearance team.

# ### Most Winning Team
# 
# Leta start by sorting 8 best winning team.

# In[ ]:


win_team = df['Winners'].value_counts().nlargest(8)


# In[ ]:


sns.barplot(x=win_team, y=win_team.index, data=df)


# Here we have Arsenal as the best team with 13 time winning the FA Cup Final.

# ## Most Runner Up Team
# 
# The 8 best Runner Up

# In[ ]:


ru_team = df['Runners-up'].value_counts().nlargest(8)


# In[ ]:


sns.barplot(x=ru_team, y=ru_team.index, data=df)


# In the second place position, we have Manchester United and Everton at the top with their record 8 time runner up.

# ## Most Appearance Team
# 
# Lets get 8 most appearance team

# In[ ]:


app = pd.concat([df['Winners'],df['Runners-up']])
app_team = app.value_counts().nlargest(8)


# In[ ]:


sns.barplot(x=app_team, y=app_team.index, data=df)


# Now we learn that Manchester United and Arsenal have been participated in this FA Cup and goes to Final round 20 time each.

# ## Highest Win Ratio
# 
# Lets gather some more information by looking at highest win ratio for each team. This is calculated by how many time they win divide by total appearance.

# In[ ]:


mhwin = df['Winners'].value_counts() / app.value_counts()
mhwin_team = mhwin.fillna(0).sort_values(ascending=False).nlargest(25)


# In[ ]:


sns.barplot(x=mhwin_team, y=mhwin_team.index, data=df)


# Here we have quite a lot of team that has 1.0 win ratio, it is mean they have won the cup everytime they goes to final.

# ## Lowest Win Ratio
# 
# Get 25 worst win ratio on FA Cup History

# In[ ]:


mlwin_team = mhwin.fillna(0).sort_values().head(25)
sns.barplot(x=mlwin_team, y=mlwin_team.index, data=df)


# Here we have lowest win ratio, it is mean despite how many time they goes to final, they just never win

# ## Biggest Win
# 
# Lest take a look at the most brutal final on FA CUP History.

# In[ ]:


# Here we are going to perform feature engineering to create another feature that makes us easier to understand
# First we are going to create a new column call diff, this column contain only the difference of final score
# on every match

diff = []


for x in df.Score:
    result = int(x[0]) - int(x[2])
    diff.append(result)
    
df['diff'] = diff

# Now we want to make a column call match where its contain the year, winning team and the runner up.
# Remember what type Year columns is, it is an integer, we need to convert it to string before we can proceed.

date = []

for y in df.Year:
    result = str(y)
    date.append(result)
    
df['date'] = date

# Now we need all the component we want, lets create the match column.

df['match'] = df['date'] + ' ' + df['Winners'] + ' vs ' + df['Runners-up']

# lets take a quick look on our new dataset

big_win = df[['match','diff']].sort_values('diff', ascending=False).head(8)
big_win


# Looking Great, let's use seaborn to visualize this[](http://)

# In[ ]:


sns.barplot(x='diff', y='match', data=big_win)


# And the most brutal final goes to 2019 Manchester City vs Watford and 1903 Bury vs Derby County with 6 goals difference each.

# ## Best 5 Teams Performance Charts
# 
# Let's dig a little bit more deeper by visualizing performance record of 5 best team

# In[ ]:


# Here we are going to need a new dataframe called performance, which contain time (year in decade), 
# and best team performance (how many times they won every decade)

df_dict = {n: df.iloc[n:n+10, :] 
           for n in range(0, len(df), 10)}

dates = []

set = list(df_dict)
for x in set:
    ds = df_dict[x].Year[x]
    da = ds - 9
    dt = str(da) + '-' + str(ds)
    dates.append(dt)

performance = pd.DataFrame(dates, columns=['Time'])
performance.head()


# We got our time ready, lets head for each team performance

# ### Arsenal FC

# In[ ]:


arsFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Arsenal'].count()
    arsFC.append(result)

performance['Arsenal'] = arsFC


# In[ ]:


sns.lineplot(x='Time', y='Arsenal', data=performance)


# Arsenal FC performance, rarely won on few first decade, but they made a breakthrough by winning their first match in between 1924 - 1933. And got their best performance on early 2000 up until now, by winning 6 FA Cup title

# ### Manchester United FC

# In[ ]:


MUFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Manchester United'].count()
    MUFC.append(result)

performance['Manchester United'] = MUFC


# In[ ]:


sns.lineplot(x='Time', y='Manchester United', data=performance)


# Manchester United, on the second place best up until now had their best record at 1990 - 1999

# ### Chelsea FC

# In[ ]:


CheFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Chelsea'].count()
    CheFC.append(result)

performance['Chelsea'] = CheFC


# In[ ]:


sns.lineplot(x='Time', y='Chelsea', data=performance)


# Chelsea is making their late breakthrough at 1970 - 1979 by winning their very first FA Cup. And on par with Arsenal, they also get 6 FA Cup title on last 2 decade

# ## Tottenham Hotspur

# In[ ]:


totFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Tottenham Hotspur'].count()
    totFC.append(result)

performance['Tottenham Hotspur'] = totFC


# In[ ]:


sns.lineplot(x='Time', y='Tottenham Hotspur', data=performance)


# Tottenham hotspur had their best performance at 1960 - 1969 by winning 3 FA Cup Title

# ## Aston Villa

# In[ ]:


AVFC = []

for x in set:
    result = df_dict[x].Winners.loc[df_dict[x]['Winners'] == 'Aston Villa'].count()
    AVFC.append(result)

performance['Aston Villa'] = AVFC


# In[ ]:


sns.lineplot(x='Time', y='Aston Villa', data=performance)


# Aston Villa on 5th best FA CUP team had their best performance at 1890 - 1909 and 1914 - 1923 by winning 2 FA Cup title each.

# ### Best Team Performance Comparison
# 
# Finnaly lets perform comparison of performance between the 5 best team history

# In[ ]:


# lets take a look at our new dataset

performance.head()


# In[ ]:


# We are going to produce line chart contain all 5 performance

All = performance.melt('Time', var_name='cols',  value_name='vals')
sns.lineplot(x="Time", y="vals", hue='cols', data=All)


# # Summary
# 
# That is all the EDA i can show you now, my apologies for any miss spell english.
# it is great to explore something new, if you have any suggestion or question about this kernel,
# just ask on the comment, anything would be appreciated.
# 
# Thank you, have a great day
