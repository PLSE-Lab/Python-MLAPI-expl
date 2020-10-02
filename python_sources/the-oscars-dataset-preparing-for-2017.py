#!/usr/bin/env python
# coding: utf-8

# With the 2017 Oscars fast approaching, we would like to explore the history of this prestigious award.
# Who were the most celebrated actors, actresses, and directors?  Will this year break any records

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/database.csv');

# Fill NaN in winners with 0, for a loss
df['Winner'] = df['Winner'].fillna(0);


# In[ ]:


dfwin = df.groupby('Name',as_index=False).sum()
dfwin = dfwin.sort_values('Winner', ascending=False)
dftop = dfwin.head(15)
sns.barplot(x='Winner', y='Name', data=dftop, color='goldenrod')
plt.xlabel('Number of Awards')
plt.ylabel('')
plt.title('Top 15 Oscar Winning Films')


# df.groupby('Name',as_index=False).mean()
# dfrate = dfrate.sort_values('Winner', ascending=False)
# 

# In[ ]:


dfact = df[(df.Award == 'Actor') | (df.Award == 'Actor in a Supporting Role')]
by_act = dfact.groupby('Name')
topact = pd.DataFrame()
topact['Wins'] = by_act['Winner'].sum()
topact['WinRate'] = by_act['Winner'].mean()
topact = topact.sort_values('Wins', ascending=False) 
topact['Thespian'] = topact.index
#print(topact)
topact = topact.head(20)
ax = sns.barplot(x='Wins', y='Thespian', color = 'royalblue',#palette='Blues_d', hue='WinRate',  
            data=topact.head(15))
plt.xlabel('Awards');
plt.ylabel('');
plt.title('Top 15 Oscar Winning Actors');
# ax.legend_.remove()


# In[ ]:





# In[ ]:


dfactress = df[(df.Award == 'Actress') | (df.Award == 'Actress in a Supporting Role')]
by_actress = dfactress.groupby('Name')
topactress = pd.DataFrame()
topactress['Wins'] = by_actress['Winner'].sum()
topactress['WinRate'] = by_actress['Winner'].mean()
topactress = topactress.sort_values('Wins', ascending=False) 
topactress['Thespian'] = topactress.index
topactress = topactress.head(20)
ax = sns.barplot(x='Wins', y='Thespian', color = 'orchid',#palette='Blues_d', hue='WinRate',  
            data=topactress.head(15))
plt.xlabel('Awards');
plt.ylabel('');
plt.title('Top 15 Oscar Winning Actresses');
# ax.legend_.remove()

