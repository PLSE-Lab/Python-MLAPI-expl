#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sqlite3 as sqlite

# Machine learning libraries
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


with sqlite.connect('../input/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * FROM Country", con)
    leagues = pd.read_sql_query("SELECT * FROM League", con)
    matches = pd.read_sql_query("SELECT * FROM Match", con)
    players = pd.read_sql_query("SELECT * FROM Player", con)
    player_atts = pd.read_sql_query("SELECT * FROM Player_Attributes", con)
    teams = pd.read_sql_query("SELECT * FROM Team", con)
    team_atts = pd.read_sql_query("SELECT * FROM Team_Attributes", con)
    


# In[ ]:


pd.options.display.max_columns = 999


# In[ ]:


matches.head()


# In[ ]:


columns1 = ['id', 'country_id', 'league_id', 'stage', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']


# In[ ]:


columns = ['id', 'country_id', 'league_id', 'stage', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']


# In[ ]:


X = matches[columns]


# In[ ]:


X.dropna(axis=0, how='any', inplace=True)


# In[ ]:


len(X)


# In[ ]:


def compare(h,a):
    if(h>a):
        return 1
    elif(h<a):
        return 2
    else:
        return 0


# In[ ]:


X['result'] = matches.apply(lambda r: compare(r['home_team_goal'], r['away_team_goal']), axis=1)


# In[ ]:


Y = X['result']


# In[ ]:


X = X.drop('result',1)
X = X.drop('home_team_goal',1)
X = X.drop('away_team_goal',1)


# In[ ]:


X


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


# In[ ]:


X_train.shape, Y_train.shape


# In[ ]:


model = ExtraTreesClassifier()
model.fit(X_train,Y_train)


# In[ ]:


randomForestModel = RandomForestClassifier()
randomForestModel.fit(X_train, Y_train)


# In[ ]:


randomForestModel.score(X_test, Y_test)


# In[ ]:


model.score(X_test,Y_test)


# In[ ]:


len(X)
X.shape


# In[ ]:


indices = np.argsort(model.feature_importances_)[::-1]
plt.figure(figsize=(10 * 2, 10))
index = np.arange(len(columns1))
bar_width = 0.35
plt.bar(index, model.feature_importances_*5, color='red', alpha=0.5)
plt.xlabel('features')
plt.ylabel('importance')
plt.title('Feature importance')
plt.xticks(index + bar_width,range(len(columns1)))
plt.tight_layout()
plt.show()
g=model.feature_importances_*5
h=pd.DataFrame(g)
l=pd.DataFrame(columns1)
l[1]=h.values
l.columns=['Feature','INFO_GAIN']
print(l)


# In[ ]:


match_team = pd.merge(matches, team_atts, left_on='home_team_api_id', right_on='team_api_id')


# In[ ]:


match_team


# In[ ]:


team_atts_date_sort = team_atts.groupby('team_api_id').apply(pd.DataFrame.sort, 'date')
team_atts_date_sort


# In[ ]:


team_atts_nodups = team_atts_date_sort.drop_duplicates(subset= 'team_api_id', keep = 'last', inplace=False)


# In[ ]:


team_atts_nodups.shape


# In[ ]:


matches


# In[ ]:


joinColumns = ['id', 'country_id', 'league_id', 'stage', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'SJH', 'SJD', 'SJA', 'VCH', 'VCD', 'VCA', 'GBH', 'GBD', 'GBA', 'BSH', 'BSD', 'BSA']


# In[ ]:


def matchresult(homeScore, awayScore):
    if(homeScore > awayScore):
        return 1
    elif(homeScore < awayScore):
        return 2
    else:
        return 0


# In[ ]:


X = matches[joinColumns]
X.dropna(axis=0, how='any', inplace=True)


# In[ ]:


X.shape


# In[ ]:


X.dtypes


# In[ ]:


X['result'] = X.apply(lambda r: matchresult(r['home_team_goal'], r['away_team_goal']), axis=1)
Y = X['result']
X = X.drop('result',1)
X = X.drop('home_team_goal',1)
X = X.drop('away_team_goal',1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


match_home = pd.merge(X, team_atts_nodups, left_on='home_team_api_id', right_on='team_api_id', how='inner')


# In[ ]:


match_home_away = pd.merge(match_home, team_atts_nodups, left_on='away_team_api_id', right_on='team_api_id', how='inner')


# In[ ]:


match_home_away


# In[ ]:


match_home_away['buildUpPlayDribblingClass_x'].value_counts()


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(match_home_away['buildUpPlayDribblingClass_x'])
list(le.classes_)
match_home_away['buildUpPlayDribblingClass_x'] = le.transform(match_home_away['buildUpPlayDribblingClass_x'])


# In[ ]:


match_home_away['buildUpPlayDribblingClass_x'].value_counts()


# In[ ]:


match_home_away


# In[ ]:


match_home_away_le = match_home_away.apply(le.fit_transform)


# In[ ]:


match_home_away_le


# In[ ]:


X_train.shape


# In[ ]:


randomForestModelNew = RandomForestClassifier()
randomForestModelNew.fit(X_train, Y_train)


# In[ ]:


randomForestModelNew.score(X_test, Y_test)


# In[ ]:


adaBoost = AdaBoostClassifier()
adaBoost.fit(X_train, Y_train)


# In[ ]:


adaBoost.score(X_test, Y_test)


# In[ ]:


matches.shape


# In[ ]:


newMatches = matches[ (-pd.isnull(matches.home_player_1)) & (-pd.isnull(matches.home_player_2)) & (-pd.isnull(matches.home_player_3)) & (-pd.isnull(matches.home_player_4)) & (-pd.isnull(matches.home_player_5)) & (-pd.isnull(matches.home_player_6)) & (-pd.isnull(matches.home_player_7)) & (-pd.isnull(matches.home_player_8)) & (-pd.isnull(matches.home_player_9)) & (-pd.isnull(matches.home_player_10)) & (-pd.isnull(matches.home_player_11)) & (-pd.isnull(matches.away_player_1)) & (-pd.isnull(matches.away_player_2)) & (-pd.isnull(matches.away_player_3)) & (-pd.isnull(matches.away_player_4)) & (-pd.isnull(matches.away_player_5)) & (-pd.isnull(matches.away_player_6)) & (-pd.isnull(matches.away_player_7)) & (-pd.isnull(matches.away_player_8)) & (-pd.isnull(matches.away_player_9)) & (-pd.isnull(matches.away_player_10)) & (-pd.isnull(matches.away_player_11)) ]


# In[ ]:


newMatches.shape


# In[ ]:


players[players.player_api_id == 26235]


# In[ ]:


player_atts


# In[ ]:


player_atts_group_sort = player_atts.groupby('player_api_id').apply(pd.DataFrame.sort, 'date')
player_atts_group_sort


# In[ ]:


player_atts_nodups = player_atts_group_sort.drop_duplicates(subset= 'player_api_id', keep = 'last', inplace=False)


# In[ ]:


player_atts_nodups.dtypes


# In[ ]:


player_atts_columns = list(player_atts.columns.values)


# In[ ]:


player_atts['overall_rating'].min()


# In[ ]:


def convertRating(rating):
    return int(rating/5);


# In[ ]:


print (convertRating(35));


# In[ ]:


player_atts_nodups.shape


# In[ ]:


player_atts_nodups.dropna(axis=0, how='any', inplace=True)


# In[ ]:


player_atts_nodups['overall_rating'] = player_atts_nodups.apply(lambda r: int(r['overall_rating']/5), axis=1)
player_atts_nodups['potential'] = player_atts_nodups.apply(lambda r: int(r['potential']/5), axis=1)
player_atts_nodups['crossing'] = player_atts_nodups.apply(lambda r: int(r['crossing']/5), axis=1)
player_atts_nodups['finishing'] = player_atts_nodups.apply(lambda r: int(r['finishing']/5), axis=1)
player_atts_nodups['heading_accuracy'] = player_atts_nodups.apply(lambda r: int(r['heading_accuracy']/5), axis=1)
player_atts_nodups['short_passing'] = player_atts_nodups.apply(lambda r: int(r['short_passing']/5), axis=1)
player_atts_nodups['volleys'] = player_atts_nodups.apply(lambda r: int(r['volleys']/5), axis=1)
player_atts_nodups['dribbling'] = player_atts_nodups.apply(lambda r: int(r['dribbling']/5), axis=1)
player_atts_nodups['curve'] = player_atts_nodups.apply(lambda r: int(r['curve']/5), axis=1)
player_atts_nodups['free_kick_accuracy'] = player_atts_nodups.apply(lambda r: int(r['free_kick_accuracy']/5), axis=1)
player_atts_nodups['long_passing'] = player_atts_nodups.apply(lambda r: int(r['long_passing']/5), axis=1)
player_atts_nodups['ball_control'] = player_atts_nodups.apply(lambda r: int(r['ball_control']/5), axis=1)
player_atts_nodups['acceleration'] = player_atts_nodups.apply(lambda r: int(r['acceleration']/5), axis=1)
player_atts_nodups['sprint_speed'] = player_atts_nodups.apply(lambda r: int(r['sprint_speed']/5), axis=1)
player_atts_nodups['agility'] = player_atts_nodups.apply(lambda r: int(r['agility']/5), axis=1)
player_atts_nodups['reactions'] = player_atts_nodups.apply(lambda r: int(r['reactions']/5), axis=1)
player_atts_nodups['balance'] = player_atts_nodups.apply(lambda r: int(r['balance']/5), axis=1)
player_atts_nodups['shot_power'] = player_atts_nodups.apply(lambda r: int(r['shot_power']/5), axis=1)
player_atts_nodups['jumping'] = player_atts_nodups.apply(lambda r: int(r['jumping']/5), axis=1)
player_atts_nodups['stamina'] = player_atts_nodups.apply(lambda r: int(r['stamina']/5), axis=1)
player_atts_nodups['strength'] = player_atts_nodups.apply(lambda r: int(r['strength']/5), axis=1)
player_atts_nodups['long_shots'] = player_atts_nodups.apply(lambda r: int(r['long_shots']/5), axis=1)
player_atts_nodups['aggression'] = player_atts_nodups.apply(lambda r: int(r['aggression']/5), axis=1)
player_atts_nodups['interceptions'] = player_atts_nodups.apply(lambda r: int(r['interceptions']/5), axis=1)
player_atts_nodups['positioning'] = player_atts_nodups.apply(lambda r: int(r['positioning']/5), axis=1)
player_atts_nodups['vision'] = player_atts_nodups.apply(lambda r: int(r['vision']/5), axis=1)
player_atts_nodups['penalties'] = player_atts_nodups.apply(lambda r: int(r['penalties']/5), axis=1)
player_atts_nodups['marking'] = player_atts_nodups.apply(lambda r: int(r['marking']/5), axis=1)
player_atts_nodups['standing_tackle'] = player_atts_nodups.apply(lambda r: int(r['standing_tackle']/5), axis=1)
player_atts_nodups['sliding_tackle'] = player_atts_nodups.apply(lambda r: int(r['sliding_tackle']/5), axis=1)
player_atts_nodups['gk_diving'] = player_atts_nodups.apply(lambda r: int(r['gk_diving']/5), axis=1)
player_atts_nodups['gk_handling'] = player_atts_nodups.apply(lambda r: int(r['gk_handling']/5), axis=1)
player_atts_nodups['gk_kicking'] = player_atts_nodups.apply(lambda r: int(r['gk_kicking']/5), axis=1)
player_atts_nodups['gk_positioning'] = player_atts_nodups.apply(lambda r: int(r['gk_positioning']/5), axis=1)
player_atts_nodups['gk_reflexes'] = player_atts_nodups.apply(lambda r: int(r['gk_reflexes']/5), axis=1)


# In[ ]:


player_atts_nodups['crossing'].min()


# In[ ]:


player_atts_nodups


# In[ ]:


player_atts_nodups['defensive_work_rate'].value_counts()


# In[ ]:


player_atts_nodups = player_atts_nodups[~player_atts_nodups['attacking_work_rate'].isin(['None', 'le', 'norm', 'stoc', 'y'])]


# In[ ]:


def convertRating(rating):
    return int(rating/5);


# In[ ]:


player_atts_nodups.dropna(axis=0, how='any', inplace=True)


# In[ ]:


player_atts_nodups['attacking_work_rate'].value_counts()


# In[ ]:


player_atts_nodups


# In[ ]:


y = player_atts_nodups['overall_rating']


# In[ ]:


player_atts_nodups_clean = player_atts_nodups.drop('id', axis=1, inplace=False)
player_atts_nodups_clean = player_atts_nodups_clean.drop('player_fifa_api_id', axis=1, inplace=False)
player_atts_nodups_clean = player_atts_nodups_clean.drop('player_api_id', axis=1, inplace=False)
player_atts_nodups_clean = player_atts_nodups_clean.drop('date', axis=1, inplace=False)
player_atts_nodups_clean = player_atts_nodups_clean.drop('overall_rating', axis=1, inplace=False)


# In[ ]:


player_atts_nodups_clean


# In[ ]:


player_atts_nodups_clean_le = player_atts_nodups_clean.apply(le.fit_transform)


# In[ ]:


player_atts_nodups_clean_le


# In[ ]:


player_atts_nodups_clean_le_train, player_atts_nodups_clean_le_test, y_train, y_test = train_test_split(player_atts_nodups_clean_le, y, test_size=0.4, random_state=0)


# In[ ]:


randomForestModel = RandomForestClassifier()
randomForestModel.fit(player_atts_nodups_clean_le_train, y_train)


# In[ ]:


randomForestModel.score(player_atts_nodups_clean_le_test, y_test)


# In[ ]:


player_atts_nodups_clean_le_columns = list(player_atts_nodups_clean_le.columns.values)
len(player_atts_nodups_clean_le_columns)


# In[ ]:


indices = np.argsort(randomForestModel.feature_importances_)[::-1]
plt.figure(figsize=(10 * 2, 10))
index = np.arange(len(player_atts_nodups_clean_le_columns))
bar_width = 0.35
plt.bar(index, randomForestModel.feature_importances_*5, color='red', alpha=0.5)
plt.xlabel('features')
plt.ylabel('importance')
plt.title('Feature importance')
plt.xticks(index + bar_width,range(len(player_atts_nodups_clean_le_columns)))
plt.tight_layout()
plt.show()
g=randomForestModel.feature_importances_*5
h=pd.DataFrame(g)
l=pd.DataFrame(player_atts_nodups_clean_le_columns)
l[1]=h.values
l.columns=['Feature','INFO_GAIN']
print(l)


# In[ ]:


player_atts_nodups_clean_le_new = SelectKBest(chi2, k=10).fit_transform(player_atts_nodups_clean_le, y)
player_atts_nodups_clean_le_new


# In[ ]:


player_atts_nodups_clean_le_new.dtype 


# In[ ]:


pl = player_atts_nodups_clean_le_new
pl.dtype


# In[ ]:


np.savetxt("Soccer.csv",pl,delimiter=',')


# In[ ]:


(player_atts_nodups_clean_le_new.max() - player_atts_nodups_clean_le_new.min()).idxmax()


# In[ ]:


player_atts_nodups_clean_le_new.range


# In[ ]:


player_atts_nodups_clean_le_new_train, player_atts_nodups_clean_le_new_test, y_train, y_test = train_test_split(player_atts_nodups_clean_le_new, y, test_size=0.4, random_state=0)


# In[ ]:


randomForestModel = RandomForestClassifier()
randomForestModel.fit(player_atts_nodups_clean_le_train_new, y_train)


# In[ ]:


randomForestModel.score(player_atts_nodups_clean_le_test, y_test)


# In[ ]:





# In[ ]:


randomForestModel = RandomForestClassifier()
randomForestModel.fit(player_atts_nodups_clean_le_new_train, y_train)


# In[ ]:


randomForestModel.score(player_atts_nodups_clean_le_new_test, y_test)


# In[ ]:




