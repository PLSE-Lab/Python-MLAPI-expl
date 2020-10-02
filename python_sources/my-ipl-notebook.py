#!/usr/bin/env python
# coding: utf-8

# In[231]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style ='whitegrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# Path of Importing

# In[185]:


matches_path = '../input/matches.csv'
deliveries_path = '../input/deliveries.csv'


# In[186]:


matches = pd.read_csv(matches_path)
deliveries = pd.read_csv(deliveries_path)


# Let's check out matches data, and then lets goto deliveries later

# In[187]:


matches.head(10).T


# **Lets check out what we have**
# 
# Lets check if there are any null vaues, also check how much data we have.

# In[188]:


print(matches.shape)
matches.isnull().sum()


# #All the umpire3 values are missing, so lets drop them 

# In[223]:


matches = matches.drop(labels='umpire3', axis=1)


# In[189]:


missing_city = matches.loc[matches.city.isnull()]
missing_city


# We see all the missing city values are played in Dubai International Cricket Stadium, so it is safe to say that the City is Dubai, 
# 
# Lets fill in the values

# In[190]:


matches.city = matches.city.fillna('Dubai')
matches.winner = matches.winner.fillna('Draw')


# Lets take a look at Missing winners,

# In[191]:


missing_winner = matches.loc[matches.winner.isnull()]
missing_winner


# Looking above, we can safely say that these are abandoned matches and theres no harm in keeping these records, we can drop them in future is needed, 
# **matches = matches.dropna(axis=0)** to drop all the indexes with NaN.
# 
# now lets fill in the umpires, http://www.espncricinfo.com/series/8048/scorecard/1082595/royal-challengers-bangalore-vs-delhi-daredevils-5th-match-ipl-2017/
# 
# umpire1 = Virender Sharma,  is VK Sharma
# umpire2 = Sundaram Ravi, is S Ravi

# In[192]:


mis_ump = matches.loc[matches.umpire1.isnull()]
mis_ump


# In[193]:


matches.umpire1 = matches.umpire1.fillna("VK Sharma")
matches.umpire2 = matches.umpire2.fillna("S Ravi")


# In[194]:


matches = matches.replace("Rising Pune Supergiant",'Rising Pune Supergiants')


# **Let the fun begin**
# 
# Let's see who's got the most Man-of-the matches awards?
# my guess, Suresh Raina.

# In[195]:


highest_mom = matches.player_of_match.value_counts()[:10]
#fig, ax = plt.subplots()
plt.figure(figsize=(12,6))
#ax.set_title("top match winners")
sns.barplot(x = highest_mom.index, y= highest_mom, orient ='v')
plt.title('top match winners')
plt.xticks(rotation = 'vertical')
plt.show()


# Gayle Storm!!! no wonder, but Yusuf Patan was a surprise second for me, hmm ...  Data
# 
# 
# Now, Let's check out which team has won the most number of matches, 
# My guess MI, after all they have three titles under their belt,

# In[196]:


highest_team_wins = matches.winner.value_counts()
plt.figure(figsize=(12,6))
sns.barplot(x = highest_team_wins.index, y = highest_team_wins, orient = 'v')
plt.title("most successful ipl team")
plt.xticks(rotation='vertical')
plt.show()


# Bingo. Mumbai Indians It is!!!
# 
# Lets check out, the most popular stadium i.e highest no of matches played, **bring out the graphs man!** 
# 

# In[197]:


plt.figure(figsize=(20,10))
sns.countplot(x='venue', data = matches)
plt.xticks(rotation='vertical')
plt.show()


# **what!!**
# Bangalore? and then Eden Gardens 
# 
# 
# Alright!! Alright!! Lets look at the champions

# In[198]:


matches.drop_duplicates(subset=['season'], keep='last')[['season','winner']].reset_index(drop=True)


# Lets check the umpiring in Ipl , Who has the most umpiring experience, **Kumara Dharmasena??**

# In[199]:


pool_of_umpires = matches.umpire1.value_counts()
pool_of_umpires2 = matches.umpire2.value_counts()
umpires_count = pd.concat([pool_of_umpires, pool_of_umpires2], axis=1)
umpires_count = umpires_count.fillna(0)
umpires_count = umpires_count.umpire1 + umpires_count.umpire2
#umpires_count.sum.sort_values(ascending=False)
plt.figure(figsize=(20,10))
sns.barplot(x = umpires_count.index, y=umpires_count, orient ='v')
plt.xticks(rotation='vertical')
plt.show()


# **Let's dive into the toss decisions**
# decisions taken so far,

# In[200]:


plt.figure(figsize=(12,6))
ax1 = sns.countplot(x='toss_decision',data = matches)


# Chasing is a popular because of the dew factor, but lets see if the wins support the toss decisions 
# 
# 
# Lets look at the results first! I don't remember how many ties.

# In[201]:


matches.result.value_counts()


# In[202]:


tie_matches = matches.loc[matches.result=='tie']
tie_matches


# In[203]:


Lets check out the toss decisions by ground


# In[204]:


plt.figure(figsize=(30,10))
sns.countplot(x='venue', hue = 'toss_decision', data = matches)
plt.xticks(rotation='vertical')
plt.show()


# Let's see which team won by high run margins, 100 runs margin in t20 means an  extremly one sided match, Lets check out those kind of matches 

# In[205]:


#matches.loc[matches.win_by_runs.idxmax()]
matches.loc[matches.win_by_runs > 100]


# What about highest wickets? I remeber DC won a match by 10 wickets vs MI back in 2008, Lets check out more of such occasions
# 
# have taken value 10 as it is the maximum number-of-wickets margin a team can win in cricket

# In[206]:


matches.loc[matches.win_by_wickets == 10]


# Lets check out the closest wins, i remember the 2017 finals, what a match that was, how many other such games are there?

# In[207]:


matches.loc[matches.win_by_runs == 1]


# Lets check out the relations b/w toss winner and match winner 

# In[208]:


toss_to_win = matches.loc[matches.toss_winner == matches.winner]
len(toss_to_win)/ matches.shape[0]


# As we can see there's only an advantage of 1% chances of winning a match with toss win, but combined wth deciding to chase first we can see a slight edge to win the toss and chase
# 
# Lets check if playing months make difference in toss decisions 

# In[209]:


matches.date = pd.to_datetime(matches['date'])


# In[210]:


matches['month_played'] = matches.date.dt.month


# Lets one hot encode a bunch of columns here, as we are going for predictions next

# In[211]:


matches.toss_decision = pd.get_dummies(matches.toss_decision)
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PW'],inplace=True)
encode = {'team1': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'team2': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'toss_winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13},
          'winner': {'MI':1,'KKR':2,'RCB':3,'DC':4,'CSK':5,'RR':6,'DD':7,'GL':8,'KXIP':9,'SRH':10,'RPS':11,'KTK':12,'PW':13,'Draw':14}}
matches.replace(encode,inplace =True)


# In[230]:


variable_mod = ['city','venue','toss_decision']
le =LabelEncoder()
for i in variable_mod:
    matches[i] = le.fit_transform(matches[i])


# In[232]:


def classification_model(model, data, predictor_labels, target_label):
    
    model.fit(data[predictor_label],data[target_label])
    predictions = model.predict(data[predictor_label])
    accuracy = metrics.accuracy_score(predictions,data[target_label])
    print('Accuracy : %s' % '{0:.3%}'.format(accuracy))
    kf = KFold(data.shape[0], n_folds=7)
    error = []
    for train, test in kf:
        train_predictors = (data[predictor_labels].iloc[train,:])
        train_target = data[target_labels].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictor_labels].iloc[test,:], data[target_labels].iloc[test]))
    
    print('Cross-Validation Score : %s' % '{0:.3%}'.format(np.mean(error)))

    model.fit(data[predictor_labels],data[target_labels]) 


# **to be continued**
