#!/usr/bin/env python
# coding: utf-8

# PUBG, the game where you fight with up to 99 others for a chicken dinner. I spent a couple hundred hours in this game while only winning the chicken dinner <50 times. Going to KFC would have been more efficient... Let's find out how we can predict the winner of the chicken dinner.

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read training data
df_train_raw = pd.read_csv('../input/train.csv')


# In[ ]:


df_train_raw.sample(5)


# # Exploring the data
# Let's see which features we have and what range of values they have.

# In[ ]:


df_train_raw.describe()


# The maxima for DBNO (63), heals (59), kills (60), revives (41), ride distance (48390m), road kills (42), swim distance (5286m), team kills (6), walking distance (17300m) and weapons acquired (76) are really odd.
# 
# The game as intended consists of 100 players playing solo, in duo's (2 players per team) or squads (4 players per team). Having 6 team kills is therefore impossible in the normal game. Achieving 60 kills is theoretically possible, but very unlikely. Same goes for the DBNO's, heals, revives and weapons acquired. It looks like the dataset contains some custom games or zombie mode games, where a few players fights against a horde of unarmed zombies.
# 
# The maximum ride, swim and walking distances are also huge. I thought this was a shooter, not a racing sim? 

# In[ ]:


print(('{m:d} unique matches').format(m=len(df_train_raw['matchId'].unique())))

print(('{tk:d} data entries in {m:d} different matches with exactly 3 team kills').format(
    tk=len(df_train_raw[df_train_raw['teamKills']==3]),
    m=len(df_train_raw[df_train_raw['teamKills']==3]['matchId'].unique())))

print(('{tk:d} data entries in {m:d} different matches with more than 3 team kills').format(
    tk=len(df_train_raw[df_train_raw['teamKills']>3]),
    m=len(df_train_raw[df_train_raw['teamKills']>3]['matchId'].unique())))
    
print(('{g:d} data entries in {m:d} different matches with exactly 1 group').format(
    g=len(df_train_raw[df_train_raw['numGroups']==1]),
    m=len(df_train_raw[df_train_raw['numGroups']==1]['matchId'].unique())))

n_group = 10
print(('{g:d} data entries in {m:d} different matches with less than {n:d} groups').format(
    g=len(df_train_raw[df_train_raw['numGroups']<n_group]),
    m=len(df_train_raw[df_train_raw['numGroups']<n_group]['matchId'].unique()),
    n=n_group))


# In 707 matches someone made exactly 3 team kills. He or she did not make any friends that match, but it is possible that it happened in regular games. Also, there are 92 games with more than 3 team kills, which must have been custom matches.
# 
# Only 3 matches had one group, with an average of ~30 players per group. Definitely exceeds the 4 player squad limit. Besides that, there were 593 matches with less than groups. My personal experience is that there were always 80+ players in a games, but I have heard that in the Oceania servers the player count could be much lower.
# 
# Based on these stats, up to 2% of the matches may be unreliable, but for now I am ignoring this and will continue with the complete dataset.
# 
# The line blow where df_train is defined is obsolete. For quick testing it was nice to use a smaller dataset. Now it is only used to rename the dataset.

# In[ ]:


df_train = df_train_raw#.iloc[0:100000]

del df_train_raw


# Let's see if we can get a first look of the important features and which features are correlated with each other.

# In[ ]:


f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(df_train.corr(), annot=True, fmt='.2f', linewidths=.5, ax=ax)


# Starting off with the most important:
# 
# **winPlacePerc**  is our to be predicted value. The features with the highest linear correlation are
# 1.  walkDistance (0.81)
# 2. killPlace (-0.71)
# 3. boosts (0.62)
# 4. weaponsAcquired (0.57)
# 5. damageDealt (0.44)
# 
# So you win by walking? Actually, my view point is that walking distance is a measure for the amount of time you have been alive. Survive longer and you have a higher chance on a higher ranking. Same explanation for the number of weapons acquired. Kill place is negatively correlated because a lower number means more kills. More kills probably means a higher win place, although the number of kills actually does not make the top 5! I think the kill place generalizes better than the actual number of kills. In 1 game the winner may have had 20 kills, while in another game the winner may only have 10 kills, which results in a large spread on a linear fit. For both the 20 and 10 kills the winners may have achieved killPlace = 1.
# 
# **Kills** correlates quite well with anything kill related:
# 1. damageDealt (0.89)
# 2. killStreak (0.80)
# 3. DBNOs (0.75)
# 4. killPlace (-0.73)
# 5. headshotKills (0.68)
# 6. longestKill (0.59)
# 
# We may be able to drop some of these features, but for now we continue with all features.

# # Feature engineering

# In[ ]:


# Reference: https://www.kaggle.com/anycode/simple-nn-baseline/code
def FeatureEngineering(df):
    df_size = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
        
    df_mean = df.groupby(['matchId','groupId']).mean().reset_index()
    
    df_sum = df.groupby(['matchId','groupId']).sum().reset_index()
    
    df_max = df.groupby(['matchId','groupId']).max().reset_index()
    
    df_min = df.groupby(['matchId','groupId']).min().reset_index()
    
    df_match_mean = df.groupby(['matchId']).mean().reset_index()
    
    df = pd.merge(df, df_size, how='left', on=['matchId', 'groupId'])
    del df_size
    df = pd.merge(df, df_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
    del df_mean
    df = pd.merge(df, df_sum, suffixes=["", "_sum"], how='left', on=['matchId', 'groupId'])
    del df_sum
    df = pd.merge(df, df_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
    del df_max
    df = pd.merge(df, df_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
    del df_min
    df = pd.merge(df, df_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    del df_match_mean
        
    columns = list(df.columns)
    columns.remove("Id")
    columns.remove("matchId")
    columns.remove("groupId")
    columns.remove("Id_mean")
    columns.remove("Id_sum")
    columns.remove("Id_max")
    columns.remove("Id_min")
    columns.remove("Id_match_mean")

    df = df[columns]
    return df


# # Scoring function
# The model performance will be judged by its mean absolute error which is defined in the function below.
# 
# [https://en.wikipedia.org/wiki/Mean_absolute_error](https://en.wikipedia.org/wiki/Mean_absolute_error)
# 
# The values can range between 0 (perfect) and 1 (worst possible), give that all predictions are in the interval [0-1].

# In[ ]:


def MAE(y_estimate, y_true):
    return sum(abs(y_estimate-y_true))/len(y_estimate)


# # Split the data in a train and test set
# Here y is defined as the output and X are all the features. As mentioned before, all features are included and none of the data entries are removed. The data is split in a training and test set.

# In[ ]:


from sklearn.model_selection import train_test_split

matchId = df_train['matchId'].unique()
matchIdTrain = np.random.choice(matchId, int(0.80*len(matchId)))

df_train2 = df_train[df_train['matchId'].isin(matchIdTrain)]
df_test = df_train[~df_train['matchId'].isin(matchIdTrain)]

y_train = df_train2['winPlacePerc']
X_train = df_train2.drop(columns=['winPlacePerc'])
y_test = df_test['winPlacePerc']
X_test = df_test.drop(columns=['winPlacePerc'])

X_train = FeatureEngineering(X_train)
X_test = FeatureEngineering(X_test)

# This commented out section is the train/test split without keeping the matches intact
#X = df_train.drop(columns=['winPlacePerc'])
#y = df_train['winPlacePerc']
#X = FeatureEngineering(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y)

print(('Training set size: {train:d}, test set size: {test:d}').format(train=len(X_train), test=len(X_test)))
print(X_train.describe())

del df_train2, df_test, df_train#, X, y


# # Model: Random Forest

# For the model we will use a Random Forest Regressor with 10 trees. 

# In[ ]:


def RandomForestModel():
    print('\nCreating and training random forest regressor')
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(n_jobs=4, n_estimators=10)
    rfr.fit(X_train, y_train)

    y_rfr = rfr.predict(X_test)
    score_rfr = MAE(y_rfr, y_test)
    print(('Random Forest training testset score: {s:.3f}').format(s=score_rfr))
    
    # Read the test set data and make predictions
    X_submit = pd.read_csv('../input/test.csv')
    df_submit = X_submit[['Id', 'matchId', 'groupId']]
    X_submit = FeatureEngineering(X_submit)
    y_submit = rfr.predict(X_submit)
    df_submit['prediction'] = y_submit
    
    # Return a dataframe with ID's and the prediction
    return df_submit    


# # Call model

# In[ ]:


df_submit = RandomForestModel()

df_submit.head()


# # Make corrections to the predictions
# Before submitting the predictions, we will correct them. What this correction means is best shown by an example:
# Say, we have 3 teams in 1 match with predicted scores of [0.83, 0.14, 0.56]. We know these scores should be [1.0, 0.0, 0.5] because of the way the scoring is defined. If there is only 1 team we will default the score to be 1.0.

# In[ ]:


print('Correcting predictions')
        
df_submit['prediction_mod'] = -1.0
matchId = df_submit['matchId'].unique()

for match in matchId:
    df_match = df_submit[df_submit['matchId']==match]

    df_max = df_match.groupby(['groupId']).max()
    pred_sort = sorted(df_max['prediction'])

    for i in df_max.index:
        groupPlace = pred_sort.index(df_max.loc[i]['prediction'])
        if len(pred_sort) > 1:
            df_max.at[i,'prediction_mod'] = groupPlace/(len(pred_sort)-1)
        else:
            df_max.at[i,'prediction_mod'] = 1.0

    for i in df_match.index:
        df_submit.at[i, 'prediction_mod'] = df_max['prediction_mod'].loc[df_match['groupId'].loc[i]]

y_submit_cor = df_submit['prediction_mod']
print('Submission scores corrected')

df_submit.head()


# # Make submission file

# In[ ]:


df_test = pd.read_csv('../input/sample_submission.csv')
df_test['winPlacePerc'] = df_submit['prediction_mod'].copy()

df_test.to_csv('submission_rfr.csv', index=False) 
print('Random Forest submission file made\n')

