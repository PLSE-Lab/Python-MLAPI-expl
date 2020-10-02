#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import joblib
import time
import re
import math

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, SGDRegressor, BayesianRidge, PassiveAggressiveRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import log_loss


# # Loading Data

# In[ ]:


teams = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv")
seasons = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSeasons.csv")
seeds = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv")
rankings = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MMasseyOrdinals.csv")
regular_results = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")


# # Data Exploration

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

regular_results.groupby(['Season']).mean().drop(["DayNum", "WTeamID", "LTeamID", "NumOT"], axis=1).plot(title="Average Point Scored (Regular Season)", ax = axes[0]).grid()
tourney_results.groupby(['Season']).mean().drop(["DayNum", "WTeamID", "LTeamID", "NumOT"], axis=1).plot(title="Average Point Scored (NCAA tourney)", ax = axes[1]).grid()


# In[ ]:


ax = regular_results.drop(["DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "NumOT"], axis=1).groupby(['Season', "WLoc"]).size().unstack(fill_value=0).plot.area(title="Location of winning team")

ax.legend(["AWAY", "HOME", "NEUTRAL"]);


# In[ ]:


tourney_results[tourney_results["DayNum"]==154].drop(["Season", "DayNum", "WScore", "LTeamID", "LScore", "NumOT", "WLoc"], axis=1).merge(teams, left_on="WTeamID", right_on="TeamID", how="left").groupby(["TeamName"]).size().sort_values()[-10:].plot(kind='barh', title="Top 10 NCAAM winners form 1985");


# In[ ]:


print("\nTop 10 NCAAM participants according to average ranking (from 2003)")
rankings.drop(["SystemName", "RankingDayNum", "Season"], axis=1).merge(teams[["TeamID", "TeamName"]], on="TeamID").groupby(["TeamName"]).mean().sort_values(by="OrdinalRank").drop(["TeamID"], axis=1)[:10]


# # Feature engineering

# In[ ]:


final_rankings = rankings[(rankings["RankingDayNum"]==133)]
final_rankings = final_rankings.pivot_table(index=["TeamID", "Season"], columns="SystemName", values="OrdinalRank", aggfunc="first")

mean_ranks = rankings.groupby(["TeamID", "Season", "RankingDayNum"]).mean()


# In[ ]:


def get_final_ranking(team_id, season, mean_ranks):
    
    if team_id not in mean_ranks.index.get_level_values(0):
        return 1000
    
    years = mean_ranks.index.get_level_values(1)
    
    if season in years:
        return mean_ranks.loc[team_id, season].unstack().tail(1)[0]
    else:
        years = years[years<season]
        if len(years)>0:
            return mean_ranks.loc[team_id, max(years)].unstack().tail(1)[0]
        else:
            return 1000


# In[ ]:


rounds = [[134, 135], [136, 137], [138, 139], [143, 144], [145, 146], [152], [154]]

def get_tourney_head_to_heads(results, teamA, teamB):
    team_A_wins = []
    team_B_wins = []

    for i in range(len(rounds)):
        team_A_wins.append(results[(results.WTeamID==teamA)                                   & (results.LTeamID==teamB) & (results.DayNum.isin(rounds[i]))].count()[0])
        team_B_wins.append(results[(results.WTeamID==teamB)                                   & (results.LTeamID==teamA) & (results.DayNum.isin(rounds[i]))].count()[0])
    
    return team_A_wins, team_B_wins

def get_round(day):
    for i in range(len(rounds)):
        if day in rounds[i]:
            return i
    return -1


# In[ ]:


def get_matches(seeds, season):
    matches = []
    teams = list(seeds[seeds.Season==season].TeamID.sort_values())
    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            matches.append((season, teams[i], teams[j]))
    return matches


# In[ ]:


def get_mean(values):
    numerical = []
    for value in values:
        numerical.append(int(re.sub("[A-Za-z]", "", value)))
    return np.mean(numerical)

overall_seeds = seeds.groupby("TeamID").agg({'Seed': get_mean})
overall_seeds = overall_seeds.astype('float64')


# # Features building
# 
# Before building the feature matrix we need to remove from data everything that is related to 2015 or later.<br>
# Furthermore we need to compute the features considering only the training set to avoid feature leakage.<br>
# I consider  all the matches before season 2015 as training set and all the matches of season 2015 and later as test set <br>
# For the regular season matches I can consider also season 2015 matches given that are played before Selection Sunday<br>

# In[ ]:


train_tourney_results = tourney_results[(tourney_results.Season>2002)&(tourney_results.Season<2015)]
train_regular_results = regular_results[(regular_results.Season>2002)&(regular_results.Season<=2015)]

test_tourney_results = tourney_results[(tourney_results.Season>=2015)]


# In[ ]:


regular_wins = train_regular_results.groupby(["WTeamID", "WLoc"]).count()        .unstack().drop(["DayNum", "WScore", "LTeamID", "LScore", "NumOT"], axis=1)
regular_wins.columns=['A', 'H', "N"]

regular_losses = train_regular_results.groupby(["LTeamID", "WLoc"]).count()        .unstack().drop(["DayNum", "WScore", "WTeamID", "LScore", "NumOT"], axis=1)
regular_losses.columns=['H', 'A', "N"]


# In[ ]:


winners = regular_results[['Season', "WTeamID", "WScore"]]
losers = regular_results[['Season', "LTeamID", "LScore"]]
winners.rename(columns={'WTeamID':'TeamID', 'WScore':'Score'}, inplace=True)
losers.rename(columns={'LTeamID':'TeamID', 'LScore':'Score'}, inplace=True)

overall_scores = pd.concat((winners, losers))
overall_scores = overall_scores.groupby(["Season", "TeamID"]).sum()
overall_scores


# In[ ]:


columns = ['Ranking A', 'Ranking B', 'Ranking diff', "H2H Tourney A0", "H2H Tourney A1", "H2H Tourney A2", "H2H Tourney A3", "H2H Tourney A4", "H2H Tourney A5", "H2H Tourney A6", "H2H Tourney B0", "H2H Tourney B1", "H2H Tourney B2", "H2H Tourney B3", "H2H Tourney B4", "H2H Tourney B5", "H2H Tourney B6", "Seed A", "Seed B", "Seed diff", "ScoreA", "ScoreB", "Score diff"]
def extract_match_features(teamA, teamB, tourney_results, regular_results, regular_wins, regular_losses, mean_ranks, seeds, season):
        features = []        
        
        #The overall final ranking for that year (final means the last ranking before NCAA)
        features.append(get_final_ranking(teamA, (season), mean_ranks))
        features.append(get_final_ranking(teamB, (season), mean_ranks))
        features.append(features[-2]-features[-1])


        #The number of wins for the previous head to heads of the two teams during the different phases of the tournament
        tourney_h2h = get_tourney_head_to_heads(tourney_results, teamA, teamB)
        features.extend(tourney_h2h[0])
        features.extend(tourney_h2h[1])        
        
        #The seed of the two teams
        seedA = int(seeds[(seeds.TeamID==teamA) & (seeds.Season==season)].Seed.values[0][1:3])
        seedB = int(seeds[(seeds.TeamID==teamB) & (seeds.Season==season)].Seed.values[0][1:3])

        features.append(seedA)
        features.append(seedB)
        features.append(seedA-seedB)
        
        #Overall scores during regular season
        scoreA = (overall_scores.loc[season, teamA].values[0])
        scoreB = (overall_scores.loc[season, teamB].values[0])
        
        features.append(scoreA)
        features.append(scoreB)
        features.append(scoreA-scoreB)
        
        return features

        
def extract_features(results, tourney_results, regular_results, final_rankings, seeds):
    labels = []
    train_features = []
    
    regular_wins = train_regular_results.groupby(["WTeamID", "WLoc"]).count()        .unstack().drop(["DayNum", "WScore", "LTeamID", "LScore", "NumOT"], axis=1)
    regular_wins.columns=['A', 'H', "N"]

    regular_losses = train_regular_results.groupby(["LTeamID", "WLoc"]).count()        .unstack().drop(["DayNum", "WScore", "WTeamID", "LScore", "NumOT"], axis=1)
    regular_losses.columns=['H', 'A', "N"]
    
    for index, match in (results).iterrows():
                
        teams = [match.WTeamID, match.LTeamID]
        teamA = min(teams)
        teamB = max(teams)
        
        # print(f"{index}: {teamA}, {teamB}")
        
        if teamA == match.WTeamID: 
            labels.append(1.0) 
        else: 
            labels.append(0.0)
        features = extract_match_features(teamA, teamB, tourney_results, regular_results, regular_wins, regular_losses, mean_ranks, seeds, match.Season)
        
        train_features.append(features)
        if index%1000 == 0:
            print(f"{index}")
        
    train_features_df = pd.DataFrame(train_features)
    train_features_df.columns = columns
    return train_features_df, labels


# In[ ]:


#features extraction
X_train, y_train = extract_features(train_tourney_results, train_tourney_results, train_regular_results, mean_ranks, seeds)
    
joblib.dump(X_train, "features.joblib")
joblib.dump(y_train, "labels.joblib")


# In[ ]:


X_test, y_test = extract_features(test_tourney_results, train_tourney_results, train_regular_results, mean_ranks, seeds)

joblib.dump(X_test, "test_features.joblib")
joblib.dump(y_test, "test_labels.joblib")


# In[ ]:


pd.set_option('display.max_columns', 100)
X_train


# In[ ]:


print("Features correlation matrix")
fig = plt.figure(figsize =(15, 15))

ax = fig.add_subplot(111)
cax = ax.matshow(X_train.tail(100).corr())

ax.set_xticks(np.arange(len(columns)))
ax.set_yticks(np.arange(len(columns)))

ax.set_xticklabels(columns, rotation=90, fontsize=12)
ax.set_yticklabels(columns, fontsize=12)

fig.colorbar(cax);


# # Training

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adadelta, SGD, Adagrad, Nadam, Adamax, RMSprop


# In[ ]:


# build model

model = Sequential()
model.add(Dense(128, input_shape=X_train.loc[0].shape, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300)


# In[ ]:


def get_results(matches, model):
    X = []
    for i, match in enumerate(matches):
        if i%1000 == 0:
            print(i)
            
        X.append(extract_match_features(match[1], match[2], train_tourney_results, train_regular_results, regular_wins, regular_losses, mean_ranks, seeds, match[0]))
    
    X = pd.DataFrame(X)
    labels = model.predict(X)
    return X, labels


# In[ ]:


def dump_results(matches, labels, filename):
     with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['ID', 'Pred'])

        for i, match in enumerate(matches):
            writer.writerow([f"{match[0]}_{match[1]}_{match[2]}", labels[i]])
    
        


# In[ ]:


matches = []
for season in range(2015, 2020):
    matches.extend(get_matches(seeds, season))
    
X, predictions = get_results(matches, model)
predictions = [a[0] for a in predictions]
predictions -= np.min(predictions)
predictions /= np.max(predictions) #normalization

dump_results(matches, predictions, "predictions.csv")


# In[ ]:


dump_results(matches, predictions, "predictions.csv")

