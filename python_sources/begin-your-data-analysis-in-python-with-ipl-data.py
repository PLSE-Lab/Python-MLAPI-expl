#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# # Welcome to Data Analysis in Python
# 
# In this tutorial, We will see how to get started with Data Analysis in Python. The **Python packages** that we use in this notebook are:
# 
# * `numpy`
# * `pandas`
# * `matplotlib`
# * `seaborn`
# 
# The **dataset** that we use in this notebook is **IPL (Indian Premier League) Dataset** posted on **Kaggle Datasets** sourced from **[cricsheet](http://cricsheet.org/)**.
# 
# ![MSD](https://s3.ap-southeast-1.amazonaws.com/images.deccanchronicle.com/dc-Cover-vk3o0lgt5njai0ql5hov9artq5-20170715144519.Medi.jpeg)

# **Data Science / Analytics** is all about finding valuable insights from the given dataset. Inshort, Finding answers that could help business. So, let us try to ask some questions reg. IPL.

# ### Questions:
# 
# * How many matches we've got in the dataset?
# * How many seasons we've got in the dataset?
# * Which Team had won by maximum runs?
# * Which Team had won by maximum wicket?
# * Which Team had won by closest Margin (minimum runs)?
# * Which Team had won by minimum wicket?
# * Which Season had most number of matches?
# * Which IPL Team is more successful? 
# * Has Toss-winning helped in winning matches?

# ### Loading required Python packages

# In[ ]:


import numpy as np # numerical computing 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visualization
import seaborn as sns #modern visualization
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


# In[ ]:


file_path = '../input/'


# In[ ]:


matches = pd.read_csv(file_path+'matches.csv')


# In[ ]:


matches.shape


# In[ ]:


matches.head()


# In[ ]:


matches.describe()


# In[ ]:


matches.info()


# ### How many matches we've got in the dataset?

# In[ ]:


#matches.shape[0]

matches['id'].max()


# ### How many seasons we've got in the dataset?

# In[ ]:


matches['season'].unique()


# In[ ]:


len(matches['season'].unique())


# ### Which Team had won by maximum runs?

# In[ ]:


matches.iloc[matches['win_by_runs'].idxmax()]


# ### Which Team had won by maximum wickets?

# In[ ]:


matches.iloc[matches['win_by_wickets'].idxmax()]


# ### Which Team had won by (closest margin) minimum runs?

# In[ ]:


matches.iloc[matches[matches['win_by_runs'].ge(1)].win_by_runs.idxmin()]


# The above code displays only one team because of the way we have solved the question with location the index, but that may not be appropriate because there could be more than one instance of such win. Hence to solve that, we have to tweak the approach a bit as below.

# In[ ]:


matches[matches[matches['win_by_runs'].ge(1)].win_by_runs.min() == matches['win_by_runs']]['winner']  #to handle the issue of only one team being shown 


# ### Which Team had won by minimum wickets?

# In[ ]:


matches.iloc[matches[matches['win_by_wickets'].ge(1)].win_by_wickets.idxmin()]


# ### Which season had most number of matches?

# In[ ]:


sns.countplot(x='season', data=matches)
plt.show()


# ### The most successful IPL Team

# In[ ]:


#sns.countplot(y='winner', data = matches)
#plt.show

data = matches.winner.value_counts()
sns.barplot(y = data.index, x = data, orient='h');


# ### Top player of the match Winners

# In[ ]:


top_players = matches.player_of_match.value_counts()[:10]
#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
ax.set_ylim([0,20])
ax.set_ylabel("Count")
ax.set_title("Top player of the match Winners")
#top_players.plot.bar()
sns.barplot(x = top_players.index, y = top_players, orient='v'); #palette="Blues");
plt.show()


# ### Has Toss-winning helped in Match-winning?

# How many Toss winning teams have won the matches? 

# In[ ]:


ss = matches['toss_winner'] == matches['winner']

ss.groupby(ss).size()


# What's the percentage of it?

# In[ ]:


#ss.groupby(ss).size() / ss.count()

#ss.groupby(ss).size() / ss.count() * 100

round(ss.groupby(ss).size() / ss.count() * 100,2)


# In[ ]:


#sns.countplot(matches['toss_winner'] == matches['winner'])
sns.countplot(ss);


# ### Team Performance

# In[ ]:


matches[matches['win_by_runs']>0].groupby(['winner'])['win_by_runs'].apply(np.median).sort_values(ascending = False)


# In[ ]:


#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Runs - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_runs', data=matches[matches['win_by_runs']>0], orient = 'h'); #palette="Blues");
plt.show()


# In[ ]:


matches[matches['win_by_wickets']>0].groupby(['winner'])['win_by_wickets'].apply(np.median).sort_values(ascending = False)


# In[ ]:


#sns.barplot(x="day", y="total_bill", data=tips)
fig, ax = plt.subplots()
#fig.figsize = [16,10]
#ax.set_ylim([0,20])
ax.set_title("Winning by Wickets - Team Performance")
#top_players.plot.bar()
sns.boxplot(y = 'winner', x = 'win_by_wickets', data=matches[matches['win_by_wickets']>0], orient = 'h'); #palette="Blues");
plt.show()

