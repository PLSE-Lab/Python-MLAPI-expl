#!/usr/bin/env python
# coding: utf-8

# **<font size="5">Data Visualization</font>**

# In[ ]:


import numpy as np 
# for scientific computation of multidimensional arrays
import pandas as pd 
# data structures dealing is done 
import matplotlib.pyplot as plt 
# For plotting graphs
import seaborn as sns
# Statistical Data Visualization
get_ipython().run_line_magic('matplotlib', 'inline')
# sets the backend of matplotlib to the 'inline' backend: With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook,


# **<font size="5">Reading DataFrames</font>**

# In[ ]:


'''score_df'''
d_df = pd.read_csv("../input/deliveries.csv")
# OR - pd.read_csv(data_path+"deliveries.csv"), where data_path = "../input/"
# reading deliveries dataset  
'''match_df'''
m_df = pd.read_csv("../input/matches.csv")
# OR - pd.read_csv(data_path+"matches.csv"), where data_path = "../input/"
# reading matches dataset
# csv- Comma seperated values


# In[ ]:


d_df.head(5)
# Printing the first five rows of the dataset to just look how the dataset is !


# In[ ]:


m_df.head(5)


# In[ ]:


print("Number of matches played: ",m_df.shape[0])
# https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.shape.html
print("Number of seasons: ",len(m_df.season.unique()))
# unique season values from dataset is printed.


# **<font size="3">Matches per Season</font>**

# In[ ]:


sns.countplot(x='season',data=m_df)
# counts the number of each season's value, i.e, counting the number of matches played per season
plt.plot()
# Plotting graph


# **<font size="3">Venue-wise Matches</font>**

# In[ ]:


plt.figure(figsize=(12,6))
# Sets the figure(plot) size.
sns.countplot(x='venue', data=m_df)
# same as above
plt.xticks(rotation='vertical')
# x-axis names vertically is shown
plt.plot()


# **<font size="3">Number of matches played by each team</font>**

# In[ ]:


temp_df = pd.melt(m_df, id_vars=['id','season'], value_vars=['team1', 'team2'])
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.melt.html
# https://youtu.be/qOkj5zOHwRE -> Video URL
plt.figure(figsize=(12,6))
sns.countplot(x='value', data=temp_df)
plt.xticks(rotation='vertical')
plt.show()


# **<font size="3">Number of wins/team</font>**

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='winner',data=m_df)
plt.xticks(rotation='vertical')
plt.show()


# **<font size="3">Champions each season</font>**

# In[ ]:


temp_df = m_df.drop_duplicates(subset=['season'],keep='last')[['season','winner']].reset_index(drop=True)
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop_duplicates.html
# https://stackoverflow.com/questions/44620465/python-reset-indexdrop-true-function-erroneously-removed-column
# https://stackoverflow.com/questions/33417991/pandas-why-are-double-brackets-needed-to-select-column-after-boolean-indexing
temp_df


# **<font size="3">Toss Decision</font>**

# In[ ]:


temp_series = m_df.toss_decision.value_counts()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.value_counts.html
labels = (np.array(temp_series.index))
# contains "bat" and "field"
# https://docs.scipy.org/doc/numpy-1.15.0/user/basics.creation.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.index.html
sizes = (np.array((temp_series/temp_series.sum())*100))
# calculating %ages
colors = ['Pink','SkyBlue']
plt.pie(sizes,labels = labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=90)
# https://www.commonlounge.com/discussion/9d6aac569e274dacbf90ed61534c076b#pie-chart
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html
plt.title("Toss decision Percentage")
plt.show()


# **<font size="3">Toss decision Per Season</font>**

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='season',hue='toss_decision',data=m_df)
# https://seaborn.pydata.org/generated/seaborn.countplot.html
plt.xticks(rotation='vertical')
plt.show()


# **<font size="3">Win %age of team BATTING second</font>**

# In[ ]:


no_of_wins = (m_df.win_by_wickets>0).sum()
no_of_loss = (m_df.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(no_of_wins + no_of_loss)
sizes = [(no_of_wins/total)*100, (no_of_loss/total)*100]
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage batting second")
plt.show()
# Similar to last pie chart


# **<font size="3">Win by Year, batting second</font>**

# In[ ]:


m_df["field_win"] = "win"
# made another column for win and loss only
m_df["field_win"].ix[m_df['win_by_wickets']==0] = "loss"
# 'ix' will select the location where 'win_by_wickets ==0' and stores "loss" in "field_win" column at that position, and rest will be stored with "win"
plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='field_win', data=m_df)
plt.xticks(rotation='vertical')
plt.show()


# **<font size="3">Top players of the match</font>**

# In[ ]:


temp_series = m_df.player_of_match.value_counts()[:10]
print(temp_series)
# value_counts() arrange in descending order, [:10] picks the first ten(top 10).
labels = np.array(temp_series.index)
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.index.html
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(temp_series), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top player of the match awardees")
plt.show()

