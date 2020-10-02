#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sqrt
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Cleaning the NFL Dataset
# 
# This notebook is my personal set of functions I'd like to use the clean up the NFL dataset. It mostly:
# 
# 1. Drops columns I feel would not be useful or involve too much work to prepare (for now)6.
# 2. Rolls play data into a single row
# 3. Resets play to move to the right for all entries
# 4. Makes player X,Y relative to the ball carrier
# 5. Standardizes team abbreviations
# 6. Other simple conversions
# 
# At the end, we try out some simple regression models.

# In[ ]:


from math import sin
from math import cos
from math import radians


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
raw_data.head()


# ## Missing Values
# 
# As we can see from the following, Temperature is missing from nearly 10% of the entries. I don't think it will be so valuable for prediction, so I'll drop the whole column. Humidity is also missing from lots of the entries, so I'll drop it, too. I can always revisit these later.
# 
# Offense formation is missing from some, and requires some parsing to make useful, so I'll drop it for now.
# 
# Defenders in the box is probably very useful for predicting run yardage, so I'll keep it and drop rows that are missing it.
# 
# Field position is essential for some of the data pipelines I want to put it, so I will drop plays missing that value for now.
# 
# Finally, we have orientation and direction data missing from some of the samples, so I will drop those, as well.
# 
# Note that if I want to drop a row, I must actually drop all 22 rows with the same playID.

# In[ ]:


numplays = len(raw_data.index)
print(numplays)
raw_data.isna().sum()/numplays*100


# In[ ]:


rowdrops = ['FieldPosition', 'DefendersInTheBox', 'Orientation', 'Dir', 'OffenseFormation']

badplays = raw_data[raw_data[rowdrops].isna().sum(axis=1) > 0]['PlayId']
badrows = raw_data[raw_data['PlayId'].isin(badplays)]
raw_data.drop(badrows.index, inplace=True)
raw_data.shape[0]/22


# ## Feature Drops
# 
# Displayname is probably not relevant, since NflId is already a unique identifier for each player. PlayerCollegeName is also probably not important for predicting performance compared to other features. 
# 
# The snap and handoff times are dropped for now. The delay between snap and handoff is almost always either 1 or 2 seconds, so I'm not sure how informative it is. I also would need to know the kickoff time to determine if length of game would be informative, but this data is also implicit in the quarter and gameclock information.
# 
# Temperature and humidity are dropped due to frequently missing values.
# 
# The other rows here need to be parsed before they can be useful, and I'll deal with that later if I want to.

# In[ ]:


raw_data.drop(['DisplayName', 'PlayerCollegeName'], axis=1, inplace=True)

raw_data.drop(['TimeHandoff', 'TimeSnap'], axis=1, inplace=True)

raw_data.drop(['Temperature', 'Humidity'], axis=1, inplace=True)

parse_later = ['WindSpeed',
               'WindDirection',
               'GameWeather',
               'Turf',
               'StadiumType',
               'Location',
               'Stadium',
              'OffensePersonnel']
raw_data.drop(parse_later, axis=1, inplace=True)


# In[ ]:


raw_data.isna().sum()


# ## Making Direction of Play Consistent
# 
# First, let's make the play always go to the right. 

# In[ ]:


dirSign = raw_data['PlayDirection'] == 'right'

raw_data['X'] = raw_data['X']*dirSign+(120-raw_data['X'])*~dirSign
raw_data['Dir'] = raw_data['Dir']*dirSign - raw_data['Dir']*~dirSign
raw_data['Orientation'] = raw_data['Orientation']*dirSign - raw_data['Orientation']*~dirSign

raw_data.drop(['PlayDirection'], axis=1, inplace=True)


# ## Standardize Team Abbreviations
# 
# Before doing some of the other preprocessing, we need the team abbreviations to all be consistent.

# In[ ]:


# standardize the abbreviations

def abbrConv(abbr):
    '''
    convert from the XTeamAbbr and fieldPosition to PossesionTeam
    '''
    if abbr == 'ARI':
        return 'ARZ'
    elif abbr == 'BAL':
        return 'BLT'
    elif abbr == 'CLE':
        return 'CLV'
    elif abbr == 'HOU':
        return 'HST'
    else:
        return abbr
    
raw_data['HomeTeamAbbr'] = raw_data['HomeTeamAbbr'].apply(abbrConv)
raw_data['VisitorTeamAbbr'] = raw_data['VisitorTeamAbbr'].apply(abbrConv)
raw_data['FieldPosition'] = raw_data['FieldPosition'].apply(abbrConv)


# ## Convert Yardline to X-Value
# 
# The yardline variable is from 0-50, so we should turn it into an X value. Here's how the logic goes:
# 
# If the possesion team is the same as the field position team the ball is on the left side and we just add 10 to the yardline.
# If the possesion team is not the same as the field position, the ball is on the right side of the field and X = 110-yardline

# In[ ]:


possmask = raw_data['FieldPosition'] == raw_data['PossessionTeam']
raw_data['YardLineX'] = possmask*(raw_data['YardLine']+10) + (1-possmask)*(110-raw_data['YardLine'])


# ## Convert Angles to X, Y components

# In[ ]:


raw_data['VX'] = raw_data['X'].apply(lambda a: sin(radians(a)))
raw_data['VY'] = raw_data['Y'].apply(lambda a: cos(radians(a)))


# In[ ]:





# ## Parse age

# In[ ]:


def getAge(birthday):
    # epxress birthday in years old
    return (pd.Timestamp.now() - pd.Timestamp(birthday)).days/365

raw_data['Age'] = raw_data['PlayerBirthDate'].apply(getAge)
raw_data.drop('PlayerBirthDate', axis=1, inplace=True)


# ## Parse Height

# In[ ]:


def parseHeight(height):
    # convert from ft-in to just inches
    feet, inches = map(int, height.split('-'))
    return 12*feet + inches
raw_data['PlayerHeight'] = raw_data['PlayerHeight'].apply(parseHeight)


# ## Add Feature for Distance to Runner

# In[ ]:


runner_table = raw_data[raw_data['NflId'] == raw_data['NflIdRusher']]
raw_data = raw_data.merge(runner_table[['PlayId','X','Y']], on='PlayId', suffixes=('','Runner'))
raw_data['SqDistToRunner'] = (raw_data['X']-raw_data['XRunner'])**2 + (raw_data['Y']-raw_data['YRunner'])**2


# ## Make X, Y Relative to the Runner

# In[ ]:


raw_data['X'] = raw_data['X'] - raw_data['XRunner']
raw_data['Y'] = raw_data['Y'] - raw_data['YRunner']
raw_data.drop(['XRunner','YRunner'], axis=1, inplace=True)


# ## Convert Gameclock to Seconds

# In[ ]:


def gameClock_to_seconds(clock):
    # convert mm:ss:xx to seconds
    minutes, seconds, _ = map(int, clock.split(':'))
    return 60*minutes+seconds

raw_data['GameClock'] = raw_data['GameClock'].apply(gameClock_to_seconds)


# ## Alter "XScoreBeforePlay"
# 
# Instead of Home and Visitor, make them Offense and Defense. In the process, add columsn that give the team name, whether or not they are on offense. Then drop possteam.

# In[ ]:


# first, add a column to indicate the team abbr
homemask = (raw_data['Team'] == 'home')
raw_data.loc[homemask,'TeamAbbr'] = raw_data[homemask]['HomeTeamAbbr']
raw_data.loc[~homemask,'TeamAbbr'] = raw_data[~homemask]['VisitorTeamAbbr']

raw_data['Offense'] = raw_data['TeamAbbr'] == raw_data['PossessionTeam']
raw_data.drop('PossessionTeam', axis=1, inplace=True)

raw_data['foomyscore'] = homemask*raw_data['HomeScoreBeforePlay'] + (1-homemask)*raw_data['VisitorScoreBeforePlay']
raw_data['footheirscore'] = ~homemask*raw_data['HomeScoreBeforePlay'] + (1-~homemask)*raw_data['VisitorScoreBeforePlay']
raw_data['OffenseScore'] = raw_data['Offense']*raw_data['foomyscore'] + ~raw_data['Offense']*raw_data['footheirscore']
raw_data['DefenseScore'] = ~raw_data['Offense']*raw_data['foomyscore'] + raw_data['Offense']*raw_data['footheirscore']
raw_data.drop(['foomyscore','footheirscore'], axis=1, inplace=True)


# In[ ]:


raw_data.columns


# ## Consolidating Rows with Merge
# 
# Each play has 22 rows, so we need to convert each of them to one row. The first way I tried to do this took forever, since it didn't rely on column operations, which are slow. Maybe this way will go faster?
# 
# First, separate out the columns that concern a single play.

# In[ ]:


play_data = raw_data[['GameId',
                      'PlayId',
                      'Season',
                      'YardLine',
                      'Quarter',
                      'GameClock',
                      'Down',
                      'Distance',
                      'HomeScoreBeforePlay',
                      'VisitorScoreBeforePlay',
                      'NflIdRusher',
                      'OffenseFormation',
                      'DefendersInTheBox',
                      'Yards',
                      'HomeTeamAbbr',
                      'VisitorTeamAbbr',
                      'Week',
                      'YardLineX',
                      'OffenseScore',
                      'DefenseScore']]
play_data = play_data.iloc[::22,:]


# In[ ]:


# now get the player data
player_data = raw_data[['PlayId',
                        'Team',
                        'TeamAbbr',
                        'X',
                        'Y',
                        'S',
                        'A',
                        'VX',
                        'VY',
                        'SqDistToRunner',
                        'Dis',
                        'Orientation',
                        'Dir',
                        'NflId',
                        'JerseyNumber',
                        'PlayerHeight',
                        'PlayerWeight',
                        'Position',
                        'Offense'
                       ]]

player_data.sort_values(['PlayId','Offense','SqDistToRunner'], inplace=True)
player_data


# In[ ]:


fat_table = play_data
for i in range(22):
    fat_table = fat_table.merge(player_data.iloc[i::22,], on='PlayId', suffixes=['',str(i)])


# In[ ]:


fat_table


# In[ ]:


fat_table.dtypes.value_counts()

objectcols = []
boolcols = []
for col in fat_table.columns:
    if fat_table.dtypes[col] == 'object':
        objectcols.append(col)
    elif fat_table.dtypes[col] == 'bool':
        boolcols.append(col)
        
from sklearn.preprocessing import LabelEncoder
objEncoders = []
for col in objectcols:
    encoder = LabelEncoder()
    fat_table[col] = encoder.fit_transform(fat_table[col])
    objEncoders.append( (col,encoder))


# # Regression Models
# 
# Let's start with trying to predict yardage with some orindary regression models. We can refit the output of the model to be appropriate for the official scoring later. Since I'm familiar with them, I want to focus on 
# 
# 1. Ordinary least squares
# 2. Ridge regression
# 3. Lasso regression
# 4. ElasticNet Regression
# 5. Logistc regression battery
# 
# Before doing that, let's standardize all the columns.
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

features = [col for col in fat_table.columns if col != 'Yards']
X = fat_table[features]
y = fat_table['Yards']

# linreg - no hyperparams
# ridge - alpha
# lasso - alpha
# elastic net - alpha, l1_ratio


# ## Ordinary Least Squares

# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(fat_table[features], fat_table['Yards'])
X_train = StandardScaler().fit_transform(X_train)
linreg = LinearRegression().fit(X_train, y_train)

X_test = StandardScaler().fit_transform(X_test)
yhat = linreg.predict(X_test)


print('RMSE: ', sqrt(MSE(yhat, y_test)))
print('R2: ', R2(yhat, y_test))


# ## Ridge

# In[ ]:


from sklearn.model_selection import GridSearchCV


alphas = [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]

ridge_search = GridSearchCV(Ridge(), param_grid={'alpha':alphas}, cv=5)

Xscaled = StandardScaler().fit_transform(X)
ridge_search.fit(Xscaled, y)


# In[ ]:


ridge_search.cv_results_


# In[ ]:


ax = plt.bar(range(1,1+len(alphas)), ridge_search.cv_results_['mean_test_score'], yerr=ridge_search.cv_results_['std_test_score'])
plt.gca().set_xticklabels(alphas, rotation=50)
best_ridge = ridge_search.best_estimator_


# ## Lasso

# In[ ]:


alphas = [0.01,0.05,0.1,0.5,1,2,4]

lasso_search = GridSearchCV(Lasso(), param_grid={'alpha':alphas}, cv=5)

Xscaled = StandardScaler().fit_transform(X)
lasso_search.fit(Xscaled, y)


# In[ ]:


lasso_search.cv_results_


# In[ ]:


ax = plt.bar(range(1,1+len(alphas)), lasso_search.cv_results_['mean_test_score'], yerr=lasso_search.cv_results_['std_test_score'])
plt.gca().set_xticklabels(alphas, rotation=50)
best_lasso = lasso_search.best_estimator_


# In[ ]:


lasso_coeffs = pd.Series(best_lasso.coef_, index=features)
lasso_coeffs.sort_values()


# In[ ]:


good_coeffs = lasso_coeffs[abs(lasso_coeffs) > 0.1].index
good_coeffs


# ## Summary
# 
# Just doing regression on the raw features is not good! However, its nice to see that the closest defender and runner's stats coming up as the most impotant features.

# # Alternative Features - Counting Nearby Players
# 
# The raw regression results are not very promising. Let's see if we can think of some better features. How about we count up the differences in how many players are 1, 2, 3 etc yards from the runner?

# In[ ]:


d_dist_cols = ['SqDistToRunner'] + ['SqDistToRunner' + str(i) for i in range(1,11)]
Ddists = fat_table[d_dist_cols]
o_dist_cols = ['SqDistToRunner' + str(i) for i in range(11,22)]
Odists = fat_table[o_dist_cols]
for i in range(1,10):
    fat_table['Owithin' + str(i)] = Odists[Odists < i**2].count(axis=1)
    fat_table['Dwithin' + str(i)] = Ddists[Ddists < i**2].count(axis=1)
    fat_table['diffwithin' + str(i)] = fat_table['Owithin'+ str(i)] - fat_table['Dwithin'+ str(i)]


# In[ ]:


j = 5
print(fat_table['diffwithin'+str(j)].unique())

for i in fat_table['diffwithin'+str(j)].unique():
    plt.hist(fat_table[fat_table['diffwithin'+str(j)] == i]['Yards'], alpha=.4, bins=20)
plt.legend()
plt.show()


# In[ ]:


features = [col for col in fat_table.columns if col != 'Yards']
X = fat_table[features]
y = fat_table['Yards']


# In[ ]:



X_train, X_test, y_train, y_test = train_test_split(fat_table[features], fat_table['Yards'])
X_train = StandardScaler().fit_transform(X_train)
linreg = LinearRegression().fit(X_train, y_train)

X_test = StandardScaler().fit_transform(X_test)
yhat = linreg.predict(X_test)


print('SMSE: ', sqrt(MSE(yhat, y_test)))
print('R2: ', R2(yhat, y_test))


# In[ ]:


alphas = [0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000,5000,10000,50000,100000]

ridge_search = GridSearchCV(Ridge(), param_grid={'alpha':alphas}, cv=5)

Xscaled = StandardScaler().fit_transform(X)
ridge_search.fit(Xscaled, y)


# In[ ]:


ridge_search.cv_results_


# In[ ]:


ax = plt.bar(range(1,1+len(alphas)), ridge_search.cv_results_['mean_test_score'], yerr=ridge_search.cv_results_['std_test_score'])
plt.gca().set_xticklabels(alphas, rotation=50)
best_ridge = ridge_search.best_estimator_


# In[ ]:


alphas = [0.0001,0.005,0.01,0.05,0.1,0.5,1,2,4]

lasso_search = GridSearchCV(Lasso(), param_grid={'alpha':alphas}, cv=5)

Xscaled = StandardScaler().fit_transform(X)
lasso_search.fit(Xscaled, y)


# In[ ]:


lasso_search.cv_results_


# In[ ]:


ax = plt.bar(range(0,len(alphas)), lasso_search.cv_results_['mean_test_score'], yerr=lasso_search.cv_results_['std_test_score'])
plt.gca().set_xticks(range(len(alphas)))
plt.gca().set_xticklabels(alphas, rotation=50, ha='right')
best_lasso = lasso_search.best_estimator_


# In[ ]:


lasso_coeffs = pd.Series(best_lasso.coef_, index=features)
lasso_coeffs.sort_values()


# In[ ]:


good_coeffs = lasso_coeffs[abs(lasso_coeffs) > 0.1].index
good_coeffs


# In[ ]:


F = good_coeffs
X2 = fat_table[F]
y2 = fat_table['Yards']
LinearRegression().fit(X2,y2).score(X2,y2)


# # 
