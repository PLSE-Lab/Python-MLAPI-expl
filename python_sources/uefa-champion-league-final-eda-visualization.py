#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hai, in this kernel, i'm going perform EDA and Visualization to UEFA Champion league Final Dataset

# ## Import Modules

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# After import all the libraries we need, let's take a look at our dataset

# In[ ]:


df = pd.read_csv('../input/uefa-champion-league-final-all-season-19552019/UEFA Champion League All Season.csv')
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# This Dataset has 7 features, with no missing value, let's perform some EDA now

# ## Best Team
# 
# Best team is the team who won UEfA Champion League the most

# In[ ]:


sns.set(rc={'figure.figsize':(18,9), 'lines.linewidth': 5, 'lines.markersize': 5, "axes.labelsize":15}, style="whitegrid")

best_team = df.club[df['position'] == 'winner'].value_counts()


# In[ ]:


sns.barplot(x=best_team, y=best_team.index, data=df)


# Here we have real madrid CF with 13 times winning the UEFA Champion League, followed by AC Milan with 7 times, Liverpool FC 6 times, and FC Bayern Munchen with 5 times.

# ## Most Participate
# 
# The team who participate in UEFA Champion League the most

# In[ ]:


most_team = df['club'].value_counts()


# In[ ]:


sns.barplot(x=most_team, y=most_team.index, data=df)


# The most participate team in final of UEFA Champion League is real madrid with 16 times, followed by AC Milan 11 times and Bayern Munchen with 10 times

# ## Best Nation
# 
# Country that send their club to UEFA Champion League Final and win

# In[ ]:


best_nation = df.nation[df['position'] == 'winner'].value_counts()


# In[ ]:


sns.barplot(x=best_nation.index, y=best_nation, data=df)


# Spain has the highest value with 18 times win in UEFA Champion League

# ## Most Participate Nation
# 
# Country that send their team to UEFA Champion League Final

# In[ ]:


most_nation = df.nation.value_counts()


# In[ ]:


sns.barplot(x=most_nation.index, y=most_nation, data=df)


# Spain and Italy are the most country that almost has place in all UEFA Champion League Final, Spain has 18 times participate while italy has 27 times.

# ## Best Coach
# 
# Best Coach is the one who made their team win UEFA Champion League Final

# In[ ]:


best_coach = df.coach[df['position'] == 'winner'].value_counts().nlargest(15)


# In[ ]:


sns.barplot(x=best_coach, y=best_coach.index, data=df)


# Carlo Anceloti,Robert Parsley, and Zinedine Zidane has won UEFA Champion League Final 3 times each

# ## Most Participate Coach
# 
# Most participate coach in UEFA Champion League Final

# In[ ]:


most_coach = df.coach.value_counts().nlargest(15)


# In[ ]:


sns.barplot(x=most_coach, y=most_coach.index, data=df)


# There is 5 names who appear on UEFA Champion League Final 4 times each.

# ## Best Formation
# 
# Best formation used by winning team on UEFA Champion League Final, you may notice in thi dataset there is unknown value for formation and MVP because the lack of data and documentation, so you may search without including the unknown value.

# In[ ]:


best_formation = df.formation[(df['position'] == 'winner') & (df['formation'] != 'unknown') ].value_counts()


# In[ ]:


sns.barplot(x=best_formation.index, y=best_formation, data=df)


# 4-4-2 and 4-4-3 are the best formation used by the most winning team of UEFA Champion League Final

# ## Most MVP
# 
# Most MVP on UEFA Champion League Final FIFA Version

# In[ ]:


most_mvp = df.mvp[df['mvp'] != 'unknown'].value_counts()


# In[ ]:


sns.barplot(x=most_mvp, y=most_mvp.index, data=df)


# There is no one who ever be mvp on UEFA Champion League Final more than 1 times.

# ## Performance Charts
# 
# Let's take a look at performance charts by creating time series analysis, in this case i made a range for decade (10 Years)

# In[ ]:


dates = []

c = df['season'].unique()
ser = [c[x:x+10] for x in range(0, len(c), 10)]

for x in range(len(ser)):
    s = ' '.join(ser[x])
    f1 = s[1:5]
    f2 = s[-5:-1]
    f3 = str(f1) + '-' + str(f2)
    dates.append(f3)
    
performance = pd.DataFrame(dates, columns=['Time'])
performance


# ### Best team

# ### Real Madrid CF

# In[ ]:


RMCF = []

win = df[(df['position'] == 'winner') & (df['club'] == 'Real Madrid CF')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    RMCF.append(wins)
    
performance['Real Madrid CF'] = RMCF


# In[ ]:


sns.lineplot(x='Time', y='Real Madrid CF', data=performance)


# ### AC Milan

# In[ ]:


ACM = []

win = df[(df['position'] == 'winner') & (df['club'] == 'AC Milan')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    ACM.append(wins)
    
performance['AC Milan'] = ACM


# In[ ]:


sns.lineplot(x='Time', y='AC Milan', data=performance)


# ### Liverpool FC

# In[ ]:


LFC = []

win = df[(df['position'] == 'winner') & (df['club'] == 'Liverpool FC')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    LFC.append(wins)
    
performance['Liverpool FC'] = LFC


# In[ ]:


sns.lineplot(x='Time', y='Liverpool FC', data=performance)


# ### Bayern Munchen

# In[ ]:


BM = []

win = df[(df['position'] == 'winner') & (df['club'] == 'FC Bayern Munchen')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    BM.append(wins)
    
performance['FC Bayern Munchen'] = BM


# In[ ]:


sns.lineplot(x='Time', y='FC Bayern Munchen', data=performance)


# ### Barcelona FC

# In[ ]:


BFC = []

win = df[(df['position'] == 'winner') & (df['club'] == 'Barcelona FC')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    BFC.append(wins)
    
performance['Barcelona FC'] = BFC


# In[ ]:


sns.lineplot(x='Time', y='Barcelona FC', data=performance)


# ### AFC Ajax

# In[ ]:


AFCA = []

win = df[(df['position'] == 'winner') & (df['club'] == 'AFC Ajax')]

for y in range(len(ser)):
    wins = 0
    for s in ser[y]:
        for x in win.season:
            if x == s:
                wins += 1
    AFCA.append(wins)
    
performance['AFC Ajax'] = AFCA


# In[ ]:


sns.lineplot(x='Time', y='AFC Ajax', data=performance)


# # End
# 
# This conclude the EDA and Visualization of this Dataset, thank you.
