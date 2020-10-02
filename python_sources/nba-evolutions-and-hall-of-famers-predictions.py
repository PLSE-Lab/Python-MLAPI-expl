#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


players = pd.read_csv('../input/Players.csv')
seasons_stats = pd.read_csv('../input/Seasons_Stats.csv')
player_data = pd.read_csv('../input/player_data.csv')


# In[ ]:


players.head()


# In[ ]:


len(players)


# In[ ]:


seasons_stats.head()


# In[ ]:


len(seasons_stats)


# In[ ]:


player_data.head()


# **Cleaning Data**

# In[ ]:


seasons_stats = seasons_stats[~seasons_stats.Player.isnull()]
players = players[~players.Player.isnull()]


# In[ ]:


players = players.rename(columns = {'Unnamed: 0':'id'})


# In[ ]:


num_players = player_data.groupby('name').count()
num_players =  num_players.iloc[:,:1]
num_players = num_players.reset_index()
num_players.columns = ['Player', 'count']
num_players[num_players['count'] > 1].head()


# There are some players with the same name, we have to be careful when using the data

# In[ ]:


seasons_stats = seasons_stats.iloc[:,1:]
seasons_stats = seasons_stats.drop(['blanl', 'blank2'], axis=1)


# In[ ]:


player_data['id'] = player_data.index


# In[ ]:


mj_stats = seasons_stats[seasons_stats.Player == 'Michael Jordan*']
mj_stats['Year'].iloc[0] - mj_stats['Age'].iloc[0] 


# We will substract one more year to match born column in players dataframe

# In[ ]:


seasons_stats['born'] = seasons_stats['Year'] - seasons_stats['Age'] - 1


# In[ ]:


players = players[~players.born.isnull()]


# We will concatenate players and player_data dataframes because none has all players

# In[ ]:


players_born = players[['Player', 'born']]


# In[ ]:


player_data = player_data[~player_data.birth_date.isnull()]


# In[ ]:


for i, row in player_data.iterrows():
    player_data.loc[i, 'born'] = float(row['birth_date'].split(',')[1])


# In[ ]:


player_data_born = player_data[['name', 'born']]
player_data_born.columns = ['Player', 'born']


# In[ ]:


born = pd.concat([players_born, player_data_born])


# In[ ]:


born = born.drop_duplicates()


# In[ ]:


born['id'] = born.index


# Changing these two Hall of Famers born year

# In[ ]:


born[born.Player == 'Magic Johnson*']


# In[ ]:


seasons_stats[seasons_stats.Player == 'Magic Johnson*'].head(1)


# In[ ]:


born[born.Player == 'Hakeem Olajuwon*']


# In[ ]:


seasons_stats[seasons_stats.Player == 'Hakeem Olajuwon*'].head(1)


# In[ ]:


id_magic = born[born.Player == 'Magic Johnson*'].id.values[0]
id_hakeem = born[born.Player == 'Hakeem Olajuwon*'].id.values[0]
born.loc[id_magic, 'born'] = 1959
born.loc[id_hakeem, 'born'] = 1962


# In[ ]:


data = seasons_stats.merge(born, on=['Player', 'born'])


# In[ ]:


data = data[data.Tm != 'TOT']


# Getting the number of seasons played each season for the players

# In[ ]:


data_season =  data[['id', 'Player', 'Year']].drop_duplicates()
data_season['season']  = data_season.groupby(['id', 'Player']).cumcount() + 1


# In[ ]:


data = data.merge(data_season, on=['id', 'Player', 'Year'])


# **All time scoring leaders**

# In[ ]:


all_time_scorers = data[['Player', 'PTS', 'id']].groupby(['id', 'Player']).sum()
all_time_scorers = all_time_scorers.reset_index()
all_time_scorers.columns = ['id', 'Player', 'PTS']
all_time_scorers = all_time_scorers.sort_values(by='PTS', ascending=False).head(20)
all_time_scorers


# In[ ]:


plt.figure(figsize=(12,6))
g = sns.barplot(all_time_scorers.Player, all_time_scorers.PTS)
g.set_xticklabels(labels = all_time_scorers.Player,  rotation=90)
plt.title('All Time Best Scorers')
plt.show()


# Let's have a look at the evolution of the first seven best scorer of all time

# In[ ]:


list_all_time_scorer_7 = all_time_scorers.head(7)['id'].tolist()
all_time_scorer_7 = data[data.id.isin(list_all_time_scorer_7)]


# In[ ]:


all_time_scorer_7['Total_PTS'] = all_time_scorer_7[['PTS','id']].groupby('id').cumsum()


# In[ ]:


plt.figure(figsize=(20,10))
sns.lineplot(x="season", y="Total_PTS", hue="Player", data=all_time_scorer_7)
plt.title('The 7 best All Time scorers evolutions')
plt.show()


# We can observe different informations :
# - Wilt Chamberlain had a great start with great points per game in his first ten seasons but then scores less
# - Same for Michael Jordan who was the closest to Chamberlain but ends at his 15th season
# - Kobe Bryant and Dirk Nowitzki score less but play longer 
# - Kareem Abdul-Jabbar and Karl Malone have a linear curve which means they kept scoring the same amount of points per game during their career
# - LeBron James has also a linear curve, he can pass Abdul-Jabbar in a few years if he keeps like this

# We can ask this question : when will LeBron James pass Kareem Abdul-Jabbar ?

# In[ ]:


total_kareem = all_time_scorers.iloc[0]['PTS']
lebron = all_time_scorer_7[all_time_scorer_7.Player == 'LeBron James']


# In[ ]:


from sklearn.linear_model import LinearRegression

X = lebron['season'].values.reshape(-1, 1)
y = lebron['Total_PTS'].values

reg = LinearRegression().fit(X, y)


# In[ ]:


lebron_seasons = lebron.loc[lebron['season'].idxmax()]['season']


# In[ ]:


X_test = np.array(range(lebron_seasons+1, lebron_seasons+8)).reshape(-1, 1)
y_pred = reg.predict(X_test)
print(y_pred)
print(y_pred > total_kareem)


# In[ ]:


print("LeBron wil pass Kareem Abdul-Jabbar in %sth season" % X_test[np.where(y_pred > total_kareem)][0][0])


# **Best players of all time **

# We will evaluate the players on their PER (Player Efficiency Rating)

# In[ ]:


# Filter players with at least 800 minutes played in the season (~ 10 min per game)
min_players = data[data.MP > 800]

per_players = min_players[['Player', 'PER', 'id']].groupby(['id', 'Player']).mean()
per_players = per_players.reset_index()
per_players = per_players.iloc[:,1:]
per_players.columns = ['Player', 'PER']
per_players = per_players.dropna()
per_players = per_players.sort_values(by='PER', ascending=False).head(20)
per_players


# In[ ]:


plt.figure(figsize=(12,6))
g = sns.barplot(per_players.Player, per_players.PER)
g.set_xticklabels(labels = per_players.Player,  rotation=90)
plt.title('All Time Best Players According to PER')
plt.show()


# **Points per Game in Career**

# In[ ]:


data['PPG'] = data['PTS'] / data['G']
min_players = data[data.MP > 800]


# In[ ]:


ppg_players = min_players[['Player', 'PPG', 'id']].groupby(['id', 'Player']).mean()
ppg_players = ppg_players.reset_index()
ppg_players = ppg_players.iloc[:,1:]
ppg_players.columns = ['Player', 'PPG']
ppg_players = ppg_players.dropna()
ppg_players = ppg_players.sort_values(by='PPG', ascending=False).head(20)
ppg_players


# In[ ]:


plt.figure(figsize=(12,6))
g = sns.barplot(ppg_players.Player, ppg_players.PPG)
g.set_xticklabels(labels = ppg_players.Player,  rotation=90)
plt.title('All Time Best Points Per Game in Career')
plt.show()


# In[ ]:


data.head()


# **Points through Years**

# In[ ]:


age_ppg = min_players.groupby('Age')['PPG'].mean()
age_ppg = age_ppg.reset_index()


# In[ ]:


plt.figure(figsize=(12,6))
g = sns.lineplot(x="Age", y='PPG', data=age_ppg)


# In[ ]:


print("Players tend to have their peak of points per game at", age_ppg.loc[age_ppg['PPG'].idxmax(), 'Age'])


# **Points Evolution**

# In[ ]:


years = min_players.groupby('Year').mean()
years = years.reset_index()


# In[ ]:


points = years[['Year', 'PTS']]


# In[ ]:


plt.figure(figsize=(15,8))
g = sns.lineplot(x="Year", y='PTS', data=points)
g.axvline(1954, 0,1, linestyle='dashed' ,color='red')
g.axvline(1972, 0,1, linestyle='dashed' ,color='red')
g.axvline(1992, 0,1, linestyle='dashed' ,color='red')
g.axvline(1999, 0,1, linestyle='dashed' ,color='red')
g.axvline(2007, 0,1, linestyle='dashed' ,color='red')
g.axvline(2012, 0,1, linestyle='dashed' ,color='red')
plt.title('Points Evolution')
plt.show()


# Differents points to observe :
#  - 1954 : The 24-second shot clock is introduced
#  - 1972 : No foul shots are attempted, except shooting fouls, until fifth team foul
#  - 1992 : Shot clock is reset only when ball hits the rim (previously it was reset if ball hit either the rim or backboard)
#  - 1999 : Lock out (50 matchs)
#  - 2011 : Lock out (66 matchs)

# **3 Points Evolution**

# In[ ]:


points_3 = years[['Year', '3PA']]


# In[ ]:


plt.figure(figsize=(15,8))
g = sns.lineplot(x="Year", y='3PA', data=points_3)
g.axvline(1994, 0,1, linestyle='dashed' ,color='red')
g.axvline(1997, 0,1, linestyle='dashed' ,color='red')
plt.title('3 Points Evolution')
plt.show()


# Differents points to observe :
#  - 1994 : Three shots awarded for foul on 3-point attempt
#  - 1997 : 3-point line lengthened to original 23 feet, 9 inches, except in the corners where the distance remained at 22 feet

# Let's look at the 3 points percentage taken 

# In[ ]:


years.head()


# In[ ]:


shots_taken = years[['Year', 'FGA', '2PA', '3PA']]


# In[ ]:


shots_taken = shots_taken[shots_taken.Year >= 1980]
shots_taken['2PA%'] = shots_taken['2PA'] / shots_taken['FGA']
shots_taken['3PA%'] = shots_taken['3PA'] / shots_taken['FGA']


# In[ ]:


plt.figure(figsize=(15,8))
g = sns.lineplot(x="Year", y='2PA%', data=shots_taken)
g = sns.lineplot(x="Year", y='3PA%', data=shots_taken)
g.axvline(1994, 0,1, linestyle='dashed' ,color='red')
g.axvline(1997, 0,1, linestyle='dashed' ,color='red')
g.set(xlabel='Year', ylabel='PA%')
plt.title('Points Taken Percentage Evolution')
plt.show()


# **Predicting Hall of Famers**

# In[ ]:


data.head()


# Adding features 

# In[ ]:


data['APG'] = data['AST'] / data['G']
data['RPG'] = data['TRB'] / data['G']
data['SPG'] = data['STL'] / data['G']
data['BPG'] = data['BLK'] / data['G']


# In[ ]:


# Adding mvps
mvp_players = {'Bob Pettit*': [1956, 1959],
                  'Bob Cousy*': [1957],
                  'Bill Russell*': [1958, 1961, 1962, 1963, 1965],
                  'Wilt Chamberlain*': [1960, 1966, 1967, 1968],
                  'Oscar Robertson*': [1964],
                  'Wes Unseld*': [1969],
                  'Willis Reed*': [1970],
                  'Kareem Abdul-Jabbar*': [1971, 1972, 1974, 1976, 1977, 1980],
                  'Dave Cowens*': [1973],
                  'Bob McAdoo*': [1975],
                  'Bill Walton*': [1978],
                  'Moses Malone*': [1979, 1982, 1983],
                  'Julius Erving*': [1981],
                  'Larry Bird*': [1984, 1985, 1986],
                  'Magic Johnson*': [1987, 1989, 1990],
                  'Michael Jordan*': [1988, 1991, 1992, 1996, 1998],
                  'Charles Barkley*': [1993],
                  'Hakeem Olajuwon*': [1994],
                  'David Robinson*': [1995],
                  'Karl Malone*': [1997, 1999],
                  'Shaquille O\'Neal*': [2000],
                  'Allen Iverson*': [2001],
                  'Tim Duncan': [2002, 2003],
                  'Kevin Garnett': [2004],
                  'Steve Nash': [2005, 2006],
                  'Dirk Nowitzki': [2007],
                  'Kobe Bryant': [2008],
                  'LeBron James': [2009, 2010, 2012, 2013],
                  'Derrick Rose': [2011],
                  'Kevin Durant': [2014],
                  'Stephen Curry': [2015, 2016],
                  'Russell Westbrook': [2017],
                  'James Harden': [2018]}


# In[ ]:


data['MVP'] = 0
for i, row in data.iterrows():  
    for k, v in mvp_players.items():
        for year in v:
            if row['Player'] != k:
                break
            elif(row['Year'] == year) & (row['Player'] == k):
                data.loc[i, 'MVP'] = 1
                break


# In[ ]:


# Adding nba champions
teams_champions = {'BOS': [1957, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1968, 1969, 1974, 1976, 1981, 1984, 1986, 2008],
                  'LAL': [1972, 1980, 1982, 1985, 1987, 1988, 2000, 2001, 2002, 2009, 2010],
                  'MNL': [1949, 1950, 1952, 1953, 1954],
                  'CHI': [1991, 1992, 1993, 1996, 1997, 1998],
                  'GSW': [1975, 2015, 2017, 2018],
                  'PHW': [1947, 1956],
                  'SAS': [1999, 2003, 2005, 2007, 2014],
                  'DET': [1989, 1990, 2004],
                  'MIA': [2006, 2012, 2013],
                  'PHI': [1967, 1983],
                  'SYR': [1955],
                  'HOU': [1994, 1995],
                  'NYK': [1970, 1973],
                  'STL': [1958],
                  'BLB': [1948],
                  'CLE': [2016],
                  'DAL': [2011],
                  'MIL': [1971],
                  'SEA': [1979],
                  'POR': [1977],
                  'ROC': [1951],
                  'WSB': [1978]}


# In[ ]:


data['Champion'] = 0
for i, row in data.iterrows():  
    for k, v in teams_champions.items():
        for year in v:
            if row['Tm'] != k:
                break
            elif(row['Year'] == year) & (row['Tm'] == k):
                data.loc[i, 'Champion'] = 1
                break


# In[ ]:


# Filter players with at least 800 min in a season
hall_of_fame = data[data.MP > 800]


# In[ ]:


list_famers = []
for i, row in hall_of_fame.iterrows():
    if '*' in row['Player']:
        list_famers.append(row['Player'])
list_famers = list(set(list_famers))


# In[ ]:


def is_hof(x):
    if '*' in x:
        return 1
    else:
        return 0


# In[ ]:


hall_of_fame['HOF'] = hall_of_fame['Player'].apply(is_hof)


# In[ ]:


# Season 1973 - 1974 : start to count Steals, Blocks and BPM
hall_of_fame = hall_of_fame[hall_of_fame.Year >= 1974]


# In[ ]:


mvps = data[['id', 'Player', 'MVP']]
mvps['Nb_MVP'] = mvps.groupby('id').cumsum()
mvps = mvps.groupby(['id', 'Player'], sort=False)['Nb_MVP'].max().reset_index()
mvps.sort_values(by='Nb_MVP', ascending=False).head(10)


# In[ ]:


champions = data[['id', 'Player', 'Champion']]
champions['Total_Champion'] = champions.groupby('id').cumsum()
champions = champions.groupby(['id', 'Player'], sort=False)['Total_Champion'].max().reset_index()
champions.sort_values(by='Total_Champion', ascending=False).head(10)


# In[ ]:


nb_seasons = data_season[['id', 'Player', 'season']]
nb_seasons = nb_seasons.groupby(['id', 'Player'], sort=False)['season'].max().reset_index()
nb_seasons.sort_values(by='season', ascending=False).head(10)


# In[ ]:


last_season = hall_of_fame[['id', 'Player', 'Year']]
last_season = last_season.groupby(['id', 'Player'], sort=False)['Year'].max().reset_index()
last_season.columns = ['id', 'Player', 'Last_Season']


# In[ ]:


total_points = data[['Player', 'PTS', 'id']].groupby(['id', 'Player']).sum()
total_points = total_points.reset_index()
total_points.columns = ['id', 'Player', 'Total_PTS']


# In[ ]:


career = hall_of_fame.groupby(['id', 'Player']).mean()
career = career.reset_index()
career.head()


# In[ ]:


career.columns


# In[ ]:


career_num = career[['id', 'Player', 'PER', 'TS%', 'OWS', 'DWS', 'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'FG%', '3P%', '2P%', 'eFG%', 'FT%', 'PPG', 'APG', 'RPG', 'SPG', 'BPG', 'HOF']]


# In[ ]:


career_num = career_num.merge(mvps, on=['id', 'Player'])
career_num = career_num.merge(champions, on=['id', 'Player'])
career_num = career_num.merge(nb_seasons, on=['id', 'Player'])
career_num = career_num.merge(last_season, on=['id', 'Player'])
career_num = career_num.merge(total_points, on=['id', 'Player'])


# In[ ]:


career_num.head()


# In[ ]:


career_num[career_num.HOF ==1]['season'].describe()


# In[ ]:


# Predict players with at least 8 seasons
career_num = career_num[career_num.season >= 8]


# In[ ]:


career_num = career_num.fillna(0)


# In[ ]:


career_num = career_num.drop(['Player'], axis=1)


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(career_num.corr(), cmap=sns.diverging_palette(220, 10, as_cmap=True))


# In[ ]:


# Looking at the features correlations with HOF
career_num.corr()['HOF'].sort_values(ascending=False)


# The Hall of Famers are mostly determined by Total Points, Win Share, Points per Game, Offensive Win Share and PER

# In[ ]:


len(career_num)


# In[ ]:


# train : older players who stop playing 20+ years ago (last data year is 2017 so 1997)
train = career_num[(career_num.Last_Season <= 1997) | (career_num.HOF == 1)]
test = career_num[(career_num.Last_Season > 1997) & (career_num.HOF == 0)]


# In[ ]:


print(len(train), len(test))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

X_train = train.drop('HOF', axis=1)
y_train = train['HOF']
X_test = test.drop('HOF', axis=1)


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest


# In[ ]:


pred_proba = random_forest.predict_proba(X_test)
pred_proba


# In[ ]:


y_pred_proba = []
for i in enumerate(pred_proba):
    y_pred_proba.append(i[1][1])
y_pred_proba = np.asarray(y_pred_proba)


# In[ ]:


results_hof = pd.DataFrame({
    "id": test["id"],
    "HOF": y_pred_proba
    })


# In[ ]:


career_player = career[['id', 'Player']]
results_hof = career_player.merge(results_hof, on='id')


# In[ ]:


results_hof = results_hof.sort_values(by='HOF', ascending=False)
results_hof = results_hof.head(20)
results_hof


# In[ ]:


plt.figure(figsize=(12,6))
g = sns.barplot(results_hof.Player, results_hof.HOF)
g.set_xticklabels(labels = results_hof.Player,  rotation=90)
plt.title('Players with Best Chances to be in the Hall of Fame')
plt.show()

