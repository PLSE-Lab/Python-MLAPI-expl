#!/usr/bin/env python
# coding: utf-8

# # Indian Premier League which is popularly known as IPL. Being a Cricket Fan and missing IPL due to Covid-19 ,I started analayzing ipl data. Here we go

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
    
        
        


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('/kaggle/input/ipl/matches.csv')
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop("umpire3",axis =1, inplace=True)


# *Replace the Team names to their short forms*

# In[ ]:


df.replace(['Sunrisers Hyderabad', 'Mumbai Indians', 'Gujarat Lions',
       'Rising Pune Supergiant', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Delhi Daredevils', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals', 'Deccan Chargers',
       'Kochi Tuskers Kerala', 'Pune Warriors', 'Rising Pune Supergiants'],['SRH','MI','GL','RPS','RCB','KKR','DD','KXIP','CSK','RR','DC','KTK','PW','RPS'],inplace = True)


# **Now lets look at the no. of matches played in the each season**

# In[ ]:


sns.countplot(x = 'season' , data = df)


# **Lets have a look at the team with the most number of wins**

# In[ ]:


sns.countplot(x = 'winner' , data = df)


# *MI is the team with most number of wins*

# **In which City most no. of matches have been played?**

# In[ ]:


fav_cities = df['city'].value_counts().reset_index()
fav_cities.columns = ['city','count']
sns.barplot(x = 'count',y = 'city', data = fav_cities[:10])


# *Again it's Mumbai XD*

# **Now Lets have a look at the venue where the most no. of matches have been played. I hope its not Mumbai this time :D**

# In[ ]:


fav_stadium = df['venue'].value_counts().reset_index()
fav_stadium.columns = ['venue','count']
sns.barplot(x = 'count',y = 'venue', data = fav_stadium[:10])


# **Yay it's Chinnaswamy Stadium***

# **Lets look at the toss decision percentage**

# In[ ]:


toss = df.toss_decision.value_counts()
labels = (np.array(toss.index))
sizes = (np.array((toss / toss.sum())*100))
colors = ['red', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()
          


# **It shows that most of the teams opted to field first after winning the toss**

# Now, lets look at the toss decision season wise

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='toss_decision', data=df)
plt.xticks(rotation='vertical')
plt.show()


# ****

# **Win percentage of team batting second**

# In[ ]:


num_of_wins = (df.win_by_wickets>0).sum()
num_of_loss = (df.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['green', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage of team batting second")
plt.show()


# **Season Wise win percentage of team batting second**

# In[ ]:


df["field_win"] = "win"
df["field_win"].loc[df['win_by_wickets']==0] = "loss"
plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='field_win', data=df)
plt.xticks(rotation='vertical')
plt.show()


# **Who won the most player of the match awards?**

# In[ ]:


MOM = df.player_of_match.value_counts()[:10]
labels = np.array(MOM.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(MOM), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top player of the match awardees")

plt.show()


# ****It's Universal Boss CHRIS GAYLE ****

# **Who is the top umpire?**

# In[ ]:


bestump_df = pd.melt(df, id_vars=['id'], value_vars=['umpire1', 'umpire2'])

best_ump = bestump_df.value.value_counts()[:10]
labels = np.array(best_ump.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(best_ump), width=width, color='r')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Umpires")

plt.show()


# **Its HDPK Dharmasena**

# **Toss Winner is a Match Winner?**

# In[ ]:


df['toss_winner_is_winner'] = 'no'
df['toss_winner_is_winner'].loc[df.toss_winner == df.winner] = 'yes'
temp_series = df.toss_winner_is_winner.value_counts()

labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss winner is match winner")
plt.show()


# *It Says 51% Yes.*

# **Lets look Season Wise**

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='toss_winner', hue='toss_winner_is_winner', data=df)
plt.xticks(rotation='vertical')
plt.show()


# # Thank You! Upvote if u find it good and useful! 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# 

# 

# In[ ]:




