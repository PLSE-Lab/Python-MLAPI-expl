#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 1: Load libraries
import pandas as pd
import numpy
import math
import csv
import random
from sklearn import linear_model, model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# Step 2: Load data
folder = '../input'
season_data = pd.read_csv(folder + '/stage2datafiles/RegularSeasonDetailedResults.csv')
tourney_data = pd.read_csv(folder + '/stage2datafiles/NCAATourneyDetailedResults.csv')
seeds = pd.read_csv(folder + '/stage2datafiles/NCAATourneySeeds.csv')
frames = [season_data, tourney_data]
all_data = pd.concat(frames)
stat_fields = [
    # offensive statistics
    'fgm',              # field goal made
    'fga',              # field goal attempted
    'fgp',              # field goal percentage
    'fgm3',             # 3pt field goal made
    'fga3',             # 3 points field goal attempted
    '3pp',              # 3 points field goal percentage
    'ftm',              # free throw made
    'fta',              # free throw attempted
    'ftp',              # free throw percentage
    'ef_fg_perc',       # effective field goal percentage
    'f_throw_factor',   # free throw factor
    # defensive statistics
    'totalreb_perc',    # total rebound percentage
    'or',               # offensive rebound
    'offreb_perc',      # offensive rebound percentage
    'dr',               # defensive rebound
    'defreb_perc',      # defensive rebound percentage
    'to',               # turnover
    'ast',              # assist
    'ast_ratio',        # assist ratio 
    'to_ratio',         # turnover ratio
    'to_factor',        # turnover factor both defense & offense
    'stl',              # steal
    'blk',              # block
    'pf',               # personal foul
]

prediction_year = 2019
base_elo = 1600
team_elos = {}
team_stats = {}
X = []
y = []
submission_data = []
def initialize_data():
    for i in range(1985, prediction_year+1):
        team_elos[i] = {}
        team_stats[i] = {}
initialize_data()


# In[ ]:


# Step 3: Explore the data
all_data.head(10)


# In[ ]:


# Step 4: Define Helper functions
def get_elo(season, team):
    try:
        return team_elos[season][team]
    except:
        try:
            # Get the previous season's ending value.
            team_elos[season][team] = team_elos[season-1][team]
            return team_elos[season][team]
        except:
            # Get the starter elo.
            team_elos[season][team] = base_elo
            return team_elos[season][team]

def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)
    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff
    return new_winner_rank, new_loser_rank

def get_stat(season, team, field):
    try:
        l = team_stats[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0
    
def update_stats(season, team, fields):
    if team not in team_stats[season]:
        team_stats[season][team] = {}
    for key, value in fields.items():
        # Make sure we have the field.
        if key not in team_stats[season][team]:
            team_stats[season][team][key] = []
        if len(team_stats[season][team][key]) >= 9:
            team_stats[season][team][key].pop()
        team_stats[season][team][key].append(value)
        
def predict_winner(team_1, team_2, model, season, stat_fields):
    features = []
    # Team 1
    features.append(get_elo(season, team_1))
    for stat in stat_fields:
        features.append(get_stat(season, team_1, stat))
    # Team 2
    features.append(get_elo(season, team_2))
    for stat in stat_fields:
        features.append(get_stat(season, team_2, stat))
    #return model.predict_proba([features]).clip(0.025, 0.975)
    return model.predict_proba([features])


# In[ ]:


# Step 5: Feature Selection and Feature Engineering
## Our classifier will make its decision based off of the values for 25 features. One important feature is 
## a ranking metric called ELO while the remaining 24 features are traditional basketball metrics as described below:

### Features
"""
	wfgm: field goals made
	wfga: field goals attempted
	wfgm3: three pointers made
	wfga3: three pointers attempted
	wftm: free throws made
	wfta: free throws attempted
	wor: offensive rebounds
	wdr: defensive rebounds
	wast: assists
	wto: turnovers
	wstl: steals
	wblk: blocks
	wpf: personal fouls
"""
### Engineered Features
"""
    fgp: field goal percentage
    3pp: three point percentage
    ftp: free throw percentage
    ef_fg_perc: Effective Field Goal Percentage
    f_throw_factor: Free throw Factor
    totalreb_perc: Total Rebound Percentage	
    offreb_perc: Offensive Rebound Percentage    
    defreb_perc: Defensive Rebound Percentage
    ast_ratio: Assist Ratio
    to_ratio: Turnover Ratio    
    to_factor: Turnover Factor
"""
def build_season_data(all_data):
    # Calculate the elo for every game for every team, each season.
    # Store the elo per season so we can retrieve their end elo
    # later in order to predict the tournaments without having to
    # inject the prediction into this loop.
    for index, row in all_data.iterrows():
        # Used to skip matchups where we don't have usable stats yet.
        skip = 0
        # Get starter or previous elos.
        team_1_elo = get_elo(row['Season'], row['WTeamID'])
        team_2_elo = get_elo(row['Season'], row['LTeamID'])
        # Add 100 to the home team (# taken from Nate Silver analysis.)
        if row['WLoc'] == 'H':
            team_1_elo += 100
        elif row['WLoc'] == 'A':
            team_2_elo += 100         
        # We'll create some arrays to use later.
        team_1_features = [team_1_elo]
        team_2_features = [team_2_elo]
        # Build arrays out of the stats we're tracking..
        for field in stat_fields:
            team_1_stat = get_stat(row['Season'], row['WTeamID'], field)
            team_2_stat = get_stat(row['Season'], row['LTeamID'], field)
            if team_1_stat is not 0 and team_2_stat is not 0:
                team_1_features.append(team_1_stat)
                team_2_features.append(team_2_stat)
            else:
                skip = 1
        if skip == 0:  # Make sure we have stats.
            # Randomly select left and right and 0 or 1 so we can train
            # for multiple classes.
            if random.random() > 0.5:
                X.append(team_1_features + team_2_features)
                y.append(0)
            else:
                X.append(team_2_features + team_1_features)
                y.append(1)
        # AFTER we add the current stuff to the prediction, update for
        # next time. Order here is key so we don't fit on data from the
        # same game we're trying to predict.
        if row['WFTA'] != 0 and row['LFTA'] != 0:
            stat_1_fields = {
                # offense statistics
                'fgm': row['WFGM'],
                'fga': row['WFGA'],
                'fgp': (row['WFGM'] / row['WFGA']) * 100,
                'fgm3': row['WFGM3'],
                'fga3': row['WFGA3'],
                '3pp': (row['WFGM3'] / row['WFGA3']) * 100,
                'ftm': row['WFTM'],
                'fta': row['WFTA'],
                'ftp': (row['WFTM'] / row['WFTA']) * 100,
                'ef_fg_perc': 100 * (row['WFGM'] + (0.5 * row['WFGM3'])) / row['WFGA'],
                'f_throw_factor': (row['WFTM'] / row['WFGM']) / (row['WFTA'] / row['WFGA']),
                # defense statistics
                'totalreb_perc': (row['WDR'] + row['WOR']) / (row['WDR'] + row['WOR'] + row['LDR'] + row['LOR']),
                'or': row['WOR'],
                'offreb_perc': 100 * (row['WOR'] / (row['WOR'] + row['LDR'])),
                'dr': row['WDR'],
                'defreb_perc': 100 * (row['WDR'] / (row['WDR'] + row['LOR'])),
                'to': row['WTO'],
                'ast': row['WAst'],
                'ast_ratio': 100 * (row['WAst'] / row['WFGA'] + (0.475 * row['WFTA']) + row['WAst'] + row['WTO']),
                'to_ratio': 100 * (row['WTO'] / row['WFGA'] + (0.475 * row['WFTA']) + row['WAst'] + row['WTO']),
                'to_factor': row['WTO'] / (row['WFGA'] + (0.475 * row['WFTA']) + row['WTO']),
                'stl': row['WStl'],
                'blk': row['WBlk'],
                'pf': row['WPF'],
            }         
            stat_2_fields = {
                # offense statistics
                'fgm': row['LFGM'],
                'fga': row['LFGA'],
                'fgp': (row['LFGM'] / row['LFGA']) * 100,
                'fgm3': row['LFGM3'],
                'fga3': row['LFGA3'],
                '3pp': (row['LFGM3'] / row['LFGA3']) * 100,
                'ftm': row['LFTM'],
                'fta': row['LFTA'],
                'ftp': row['LFTM'] / row['LFTA'] * 100,
                'ef_fg_perc': 100 * (row['LFGM'] + (0.5 * row['LFGM3'])) / row['LFGA'],
                'f_throw_factor': (row['LFTM'] / row['LFGM']) / (row['LFTA'] / row['LFGA']),
                # defense statistics
                'totalreb_perc': (row['LDR'] + row['LOR']) / (row['WDR'] + row['WOR'] + row['LDR'] + row['LOR']),
                'or': row['LOR'],
                'offreb_perc': 100 * (row['LOR'] / (row['LOR'] + row['WDR'])),
                'dr': row['LDR'],
                'defreb_perc': 100 * (row['LDR'] / (row['LDR'] + row['WOR'])),
                'to': row['LTO'],
                'ast': row['LAst'],
                'ast_ratio': 100 * (row['LAst'] / row['LFGA'] + (0.475 * row['LFTA']) + row['LAst'] + row['LTO']),
                'to_ratio': 100 * (row['LTO'] / row['LFGA'] + (0.475 * row['LFTA']) + row['LAst'] + row['LTO']),
                'to_factor': row['LTO'] / (row['LFGA'] + (0.475 * row['LFTA']) + row['LTO']),
                'stl': row['LStl'],
                'blk': row['LBlk'],
                'pf': row['LPF'],
            }
            update_stats(row['Season'], row['WTeamID'], stat_1_fields)
            update_stats(row['Season'], row['LTeamID'], stat_2_fields)
        # Now that we've added them, calc the new elo.
        new_winner_rank, new_loser_rank = calc_elo(
            row['WTeamID'], row['LTeamID'], row['Season'])
        team_elos[row['Season']][row['WTeamID']] = new_winner_rank
        team_elos[row['Season']][row['LTeamID']] = new_loser_rank
    return X, y
X, y = build_season_data(all_data)


# In[ ]:


# Step 6: Use Logistic Regression to Predict Game Outcomes
#model = linear_model.LogisticRegressionCV()
print("Let's hope to be correct 75% of the time")
#print(cross_validation.cross_val_score(model, numpy.array(X), numpy.array(y), cv=10, scoring='accuracy', n_jobs=-1).mean())
clf1 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=80, max_features=2, max_leaf_nodes=None, min_samples_leaf=4,
            min_samples_split=12, min_weight_fraction_leaf=0.0,
            n_estimators=100, n_jobs=1, oob_score=False, random_state=0,
            verbose=0, warm_start=False)


clf2 = linear_model.LogisticRegressionCV(cv = 5)

clf3 = BaggingClassifier(base_estimator=None, n_estimators = 1000, max_samples=20, random_state=0)
clf4 = ExtraTreesClassifier(n_estimators = 100, random_state=0)
clf5 = SVC(probability=True)
clf6 = KNeighborsClassifier(n_neighbors=5)
eclf = VotingClassifier(estimators = [('RFC', clf1),
                                      ('logit',clf2),
                                      ('Bag',clf3),
                                      ('ETC', clf4),
                                      ('KNN',clf6)],
                       voting = 'soft')
clfs = [clf1,clf2,clf3,clf4,clf6]
print('here')
for clf in clfs:
    clf.fit(X, y)
    print(clf)
eclf = eclf.fit(X, y)

#print(model_selection.cross_val_score(eclf, numpy.array(X), numpy.array(y), cv=10, scoring='accuracy', n_jobs=-1).mean())
print("eclf fit")
tourney_teams = []
for index, row in seeds.iterrows():
    if row['Season'] == prediction_year:
        tourney_teams.append(row['TeamID'])
tourney_teams.sort()


# In[ ]:


for team_1 in tourney_teams:
    for team_2 in tourney_teams:
        if team_1 < team_2:
            prediction = predict_winner(
                team_1, team_2, eclf, prediction_year, stat_fields)
            label = str(prediction_year) + '_' + str(team_1) + '_' +                 str(team_2)
            submission_data.append([label, prediction[0][0]])
            print('\r',team_1, team_2, sep='\t')


# In[ ]:


# Step 7: Submit Results
print("Writing %d results." % len(submission_data))
d = {'ID': [row[0] for row in submission_data], 'Pred': [row[1] for row in submission_data]}
#submission_data2=pd.DataFrame(submission_data)
submission_data2=pd.DataFrame(data=d)
submission_data2.to_csv("stage2-submission_1.csv", index=False)
def build_team_dict():
    team_ids = pd.read_csv(folder + '/stage2datafiles/Teams.csv')
    team_id_map = {}
    for index, row in team_ids.iterrows():
        team_id_map[row['TeamID']] = row['TeamName']
    return team_id_map
team_id_map = build_team_dict()
readable = []
less_readable = []  # A version that's easy to look up.
readable_pd = pd.DataFrame(columns=['Winner','Loser','Probability'])
print("to csv")
for pred in submission_data:
    parts = pred[0].split('_')
    less_readable.append(
        [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]])
    # Order them properly.
    if pred[1] > 0.5:
        winning = int(parts[1])
        losing = int(parts[2])
        proba = pred[1]
    else:
        winning = int(parts[2])
        losing = int(parts[1])
        proba = 1 - pred[1]
    readable_pd = readable_pd.append({'Winner':team_id_map[winning], 'Loser':team_id_map[losing], 'Probability':proba}, ignore_index=True)
readable_pd.to_csv('readable.csv', index=False)


# In[ ]:


readable_pd.append({'Winner':team_id_map[winning], 'Loser':team_id_map[losing], 'Probability':proba}, ignore_index=True)


# In[ ]:


readable_pd


# In[ ]:


# # Step 8: Get Relative Feature Importances
# def get_feature_importances(estimator, norm_order=1):
#     """Retrieve or aggregate feature importances from estimator"""
#     importances = getattr(estimator, "feature_importances_", None)

#     if importances is None and hasattr(estimator, "coef_"):
#         if estimator.coef_.ndim == 1:
#             importances = np.abs(estimator.coef_)

#         else:
#             importances = np.linalg.norm(estimator.coef_, axis=0,
#                                          ord=norm_order)

#     elif importances is None:
#         raise ValueError(
#             "The underlying estimator %s has no `coef_` or "
#             "`feature_importances_` attribute. Either pass a fitted estimator"
#             " to SelectFromModel or call fit before calling transform."
#             % estimator.__class__.__name__)

#     return importances
    
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# importances = np.array(get_feature_importances(eclf))[:]
# statdict = np.array([
#             'elo',
#             # offense
#             'field goals made',
#             'field goals attempted',
#             'field goal percentage',
#             'three point field goal made',
#             'three point field goal attempted',
#             'three point field goal percentage',
#             'free throw made',
#             'free throw attempted',
#             'free throw percentage',
#             'effective field goal percentage',
#             'free throw factor',
#             # defense
#             'total rebound percentage',
#             'offensive rebounds',
#             'offensive rebound percentage',
#             'defensive rebounds',
#             'defensive rebound percentage',
#             'turnovers',
#             'assist',
#             'assist ratio',
#             'turnover ratio',
#             'turnover factor',
#             'steals',
#             'blocks',
#             'personal fouls',
#            ],dtype='str')
    
#  feature_importance = abs(model.coef_[0][:25])
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5


# featfig = plt.figure(figsize=(16,12))
# featax = featfig.add_subplot(1, 1, 1)
# featax.barh(pos, feature_importance[sorted_idx], align='center')
# featax.set_yticks(pos)
# featax.set_yticklabels(statdict[sorted_idx], fontsize=28)
# featax.set_xlabel('Relative Feature Importance',fontsize=24)

# plt.tight_layout()   
# plt.show()
# print(feature_importance)

