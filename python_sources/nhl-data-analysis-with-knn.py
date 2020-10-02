#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


team_df = pd.read_csv('../input/team_info.csv')


# In[ ]:


team_df.head()


# In[ ]:


#game_df.head()


# In[ ]:


"""
Add home and away team names.
"""
game_df = pd.read_csv('../input/game.csv')
game_df = game_df.merge(team_df[['team_id', 'teamName']],
              left_on='home_team_id', right_on='team_id') \
    .merge(team_df[['team_id', 'teamName']], left_on='away_team_id',
           right_on='team_id', suffixes=('home','away'))


# In[ ]:


game_df.head()


# # Distribution of Goals
# The distribution of home goals had a higher mean than away goals. "Home ice advantage?"

# In[ ]:


game_df[['away_goals','home_goals']].plot(kind='hist', figsize=(15,5), bins=10, alpha=0.5, title='Distribution of Home vs. Away Goals')


# # Average Goals per Team
# 

# In[ ]:


game_df.groupby('teamNamehome').mean()['home_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Scored in Home Games')
plt.show()
game_df.groupby('teamNameaway').mean()['away_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Scored in Away Games')
plt.show()


# # Average Goals Allowed per Team

# In[ ]:


game_df.groupby('teamNamehome').mean()['away_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Allowed in Home Games')
plt.show()
game_df.groupby('teamNameaway').mean()['home_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Allowed in Away Games')
plt.show()


# # Distribution of Point Differential
# 
# How much do teams usually win/lose by? Point differential is computed as:
# `point_diff` = `home team goals` - `away team goals`

# In[ ]:


game_df['point_diff'] = game_df['home_goals'] - game_df['away_goals']


# In[ ]:


game_df['point_diff'].plot(kind='hist',
                           bins=18,
                           title='NHL Point Differential (Negative Home team Loses, Positive Home team Wins)',
                           xlim=(-10,10))


# # Biggest Blowout

# In[ ]:


#Biggest Blowout was by 10 points
game_df['point_diff'].abs().max()


# In[ ]:


# Blowout game:
game_df.loc[game_df['point_diff'] == 10]


# Ouch that was a blowout. 
# Here are the video highlights in case you were wondering how it happened: https://www.nhl.com/bluejackets/video/recap-mtl-0-cbj-10/t-283041746/c-46026003

# Lets define game types as:
# - Blowout (abs point diff >= 3 points)
# - Normal Game (win > 2 points)
# - Tight Game (win by 1 point)

# In[ ]:


game_df['point_diff_type'] = game_df['point_diff'].abs().apply(lambda x: 'Blowout' if x>=3 else ('Normal' if x>=2 else 'Tight'))


# In[ ]:


# Create one dataframe with the point 
point_diff_team = pd.concat([game_df[['teamNamehome','point_diff_type','point_diff','date_time']].rename(columns={'teamNamehome':'team'}),
    game_df[['teamNameaway','point_diff_type','point_diff','date_time']].rename(columns={'teamNameaway':'team'})])


# In[ ]:


point_diff_team['date_time'] = pd.to_datetime(point_diff_team['date_time'])


# In[ ]:


for team, data in point_diff_team.groupby('team'):
    data.groupby(data['date_time'].dt.year).mean()['point_diff'].plot(kind='line', title='{} Average Point Diff By Year'.format(team), figsize=(15,2))
    plt.show()


# # TODO
# - Better metrics
# - Figure out how to make plot colors = team colors
# - More fun stuff
# 
# GO WINGS!

# # K Nearest Neighbors

# In[ ]:


team_stats_df = pd.read_csv('../input/game_teams_stats.csv')


# In[ ]:


team_stats_df.head()


# In[ ]:


team_stats_df.describe()


# In[ ]:


team_stats_df.columns


# In[ ]:


team_stats_df.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(team_stats_df.corr(), annot = True,linewidths=.2, fmt='.1f', ax=ax)


# In[ ]:


sns.distplot(team_stats_df['shots'])


# In[ ]:


sns.distplot(team_stats_df['goals'])


# In[ ]:


#make the 'won' column data binary
team_stats_df['won'] = team_stats_df['won']*1

team_stats_df = pd.concat([team_stats_df, pd.get_dummies(team_stats_df.HoA).rename(columns = '{}_binary'.format)],axis = 1)
team_stats_df = pd.concat([team_stats_df, pd.get_dummies(team_stats_df.settled_in).rename(columns = '{}_binary'.format)],axis = 1)
df_clean = team_stats_df.drop(['game_id','team_id','HoA','settled_in','head_coach','away_binary','OT_binary','SO_binary'], axis=1)
df_clean.head()


# In[ ]:


sns.pairplot(df_clean, hue='won')


# # Standardize the Variables

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()
scaler.fit(df_clean.drop('won',axis=1))
scaled_features = scaler.transform(df_clean.drop('won',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df_clean.columns[1:])
df_feat.head()


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df_feat
y = df_clean['won']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# # Using KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[ ]:


knn.fit(X_train,y_train)


# # Predictions and Evaluations

# In[ ]:


pred = knn.predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(confusion_matrix(y_test,pred))


# In[ ]:


print(classification_report(y_test,pred))


# # Choosing a K Value

# In[ ]:


error_rate = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# # K = 20

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[ ]:




