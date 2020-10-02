#!/usr/bin/env python
# coding: utf-8

# **Introduction** 
# 
# This notebook is intended for people who are looking for ways to generate meaningful advanced features for both the primary team and opponent. 
# We will also start with a basic RandomForest model, and tune some hyper parameters to hopefully improve the prediction score. 
# 
# **Contents** 
# 
# > **Part 1 - Features and Advanced Stats**
# 	1. Import data
# 	2. Rearrange the data structure to enable feature generation for the primary and opposition teams
# 	3. Aggregate raw stats per season-team
#     4. Derive Advanced Stats
# 	5. Get NCAA Tourney data and pre process it
# 
# > **Part 2 - Model**
# 	1. Get a train, valid test split
# 	2. Basic RandomForest Model
# 	3. Tune for number of trees
# 	4. Number of samples per leaf/node
# 	5. Tune for max features
#     6. Run model on test data
#    
# > **References** - 
# > 1. fast.ai
# > 2. https://www.basketball-reference.com/about/glossary.html
# > 3. https://stats.nba.com

# **Section 1 - Imports and import data**
# 
# In addition to the basics such as numpy and pandas, we will also import the fastai library

# In[ ]:


import numpy as np 
import pandas as pd
import os
#fast ai
from fastai.imports import *
from pandas_summary import DataFrameSummary
#Model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#import data
in_path = '../input/datafiles/'
RegularSeasonDetailedResults = pd.read_csv(in_path + 'RegularSeasonDetailedResults.csv')
RegularSeasonDetailedResults.shape


# Lets look at a quick summary of the variables in the dataframe

# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


# In[ ]:


display_all(RegularSeasonDetailedResults.describe(include='all').T)


# In[ ]:


RegularSeasonDetailedResults.head(5)


# As we can see in the first few examples, this dataframe is structured to have data for both winning (WTeamID) and losing(LTeamID) team features in the same row. In the next section, lets split that into one row per team per game and its opponent. 

# **Section 2 - Re arrange dataframe**
# 
# In this section, lets rearrange to show stats for each team against its opponent.  This should double the row count from the original dataframe.
# 

# In[ ]:


def setWinAndLoseTeamsRecords(RegularSeasonDetailedResults):
    #Convert the data frame from one record per game to one record per team-game
    regSesW = RegularSeasonDetailedResults.rename(columns = {'WTeamID': 'TEAMID',
                                                             'WScore': 'SCORE',
                                                             'WFGM': 'FGM',
                                                             'WFGA': 'FGA',
                                                             'WFGM3': 'FGM3',
                                                             'WFGA3': 'FGA3',
                                                             'WFTM': 'FTM',
                                                             'WFTA': 'FTA',
                                                             'WOR': 'OR',
                                                             'WDR': 'DR',
                                                             'WAst': 'AST',
                                                             'WTO': 'TO',
                                                             'WStl': 'STL',
                                                             'WBlk': 'BLK',
                                                             'WPF': 'PF',
                                                             'LTeamID': 'O_TEAMID',
                                                             'LScore': 'O_SCORE',
                                                             'LFGM': 'O_FGM',
                                                             'LFGA': 'O_FGA',
                                                             'LFGM3': 'O_FGM3',
                                                             'LFGA3': 'O_FGA3',
                                                             'LFTM': 'O_FTM',
                                                             'LFTA': 'O_FTA',
                                                             'LOR': 'O_OR',
                                                             'LDR': 'O_DR',
                                                             'LAst': 'O_AST',
                                                             'LTO': 'O_TO',
                                                             'LStl': 'O_STL',
                                                             'LBlk': 'O_BLK',
                                                             'LPF': 'O_PF'
                                                            })

    regSesL = RegularSeasonDetailedResults.rename(columns = {'LTeamID': 'TEAMID',
                                                             'LScore': 'SCORE',
                                                             'LFGM': 'FGM',
                                                             'LFGA': 'FGA',
                                                             'LFGM3': 'FGM3',
                                                             'LFGA3': 'FGA3',
                                                             'LFTM': 'FTM',
                                                             'LFTA': 'FTA',
                                                             'LOR': 'OR',
                                                             'LDR': 'DR',
                                                             'LAst': 'AST',
                                                             'LTO': 'TO',
                                                             'LStl': 'STL',
                                                             'LBlk': 'BLK',
                                                             'LPF': 'PF',

                                                             'WTeamID': 'O_TEAMID',
                                                             'WScore': 'O_SCORE',
                                                             'WFGM': 'O_FGM',
                                                             'WFGA': 'O_FGA',
                                                             'WFGM3': 'O_FGM3',
                                                             'WFGA3': 'O_FGA3',
                                                             'WFTM': 'O_FTM',
                                                             'WFTA': 'O_FTA',
                                                             'WOR': 'O_OR',
                                                             'WDR': 'O_DR',
                                                             'WAst': 'O_AST',
                                                             'WTO': 'O_TO',
                                                             'WStl': 'O_STL',
                                                             'WBlk': 'O_BLK',
                                                             'WPF': 'O_PF',
                                                             })

    regSes = (regSesW, regSesL)
    regSes = pd.concat(regSes, ignore_index = True, sort = False)
    regSes = regSes[['Season','TEAMID', 'DayNum', 'SCORE', 'O_TEAMID', 'O_SCORE',
                 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR',
                 'AST', 'TO', 'STL', 'BLK', 'PF', 'NumOT',
                 'O_FGM', 'O_FGA', 'O_FGM3', 'O_FGA3', 'O_FTM', 'O_FTA', 'O_OR', 'O_DR',
                 'O_AST', 'O_TO', 'O_STL', 'O_BLK', 'O_PF'
                 ]]
    return regSes


# In[ ]:


regSes = setWinAndLoseTeamsRecords(RegularSeasonDetailedResults)
print ('RegularSeasonDetailedResults shape: ', RegularSeasonDetailedResults.shape)
print ('regSes shape after rearranging: ', regSes.shape)


# **Section 3 - Aggregate raw stats per season-team**
# 
# We'll be performing a sum per season and team for all raw features we can then use in the next section to derive advanced features.

# In[ ]:


#Add GameNum so it can later help derive the game mins within Section 4- Derive Advanced Stats
regSes['GameNum'] = 1


# In[ ]:


def aggregateRawData(regSes):    
    regSes_Avg = regSes.groupby(['Season', 'TEAMID'])['SCORE','O_SCORE',  'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 
                 'AST', 'TO', 'STL', 'BLK', 'PF',
                 'O_FGM', 'O_FGA', 'O_FGM3', 'O_FGA3', 'O_FTM', 'O_FTA', 'O_OR', 'O_DR', 
                 'O_AST', 'O_TO', 'O_STL', 'O_BLK', 'O_PF', 'NumOT', 'GameNum'
                               ].agg('sum').reset_index()
    return regSes_Avg


# In[ ]:


regSes_aggregate = aggregateRawData(regSes)
regSes_aggregate.shape


# **Section 4 - Derive Advanced Stats**
# 
# Derive NCAA advanced stats - Reference - https://www.basketball-reference.com/about/glossary.html

# In[ ]:


def GetAdvancedStats(NCAA_features):
    NCAA_features ['EFG']       = (NCAA_features ['FGM'] + (NCAA_features ['FGM3']*0.5))/NCAA_features ['FGA']
    NCAA_features ['TOV']       = NCAA_features ['TO']/((NCAA_features ['FGA'] + 0.44) + (NCAA_features ['FTA']+NCAA_features ['TO']))
    NCAA_features ['ORB']       = NCAA_features ['OR']/(NCAA_features ['OR'] + NCAA_features ['O_DR'])
    NCAA_features ['DRB']       = NCAA_features ['DR']/(NCAA_features ['DR'] + NCAA_features ['O_OR'])
    NCAA_features ['FTAR']      = NCAA_features ['FTA']/(NCAA_features ['FGA'])
    NCAA_features ['TS']        = NCAA_features ['SCORE']/((NCAA_features ['FGA']*2) + (0.88 * NCAA_features ['FTA']))
    NCAA_features ['ASTTO']     = (NCAA_features ['AST']/(NCAA_features ['TO']))
    NCAA_features ['ASTR']      = (NCAA_features ['AST'] * 100) / ( (NCAA_features ['FGA'] + (NCAA_features ['FTA']*0.44)) + NCAA_features ['AST'] + NCAA_features ['TO'] )
    NCAA_features ['TR']        = NCAA_features ['OR'] + NCAA_features ['DR']
    NCAA_features ['O_TR']      = NCAA_features ['O_OR'] + NCAA_features ['O_DR']
    NCAA_features ['REBP']      = 100 * (NCAA_features ['TR']) / (NCAA_features ['TR'] + NCAA_features ['O_TR'])
    NCAA_features ['POSS']      = 0.5 * ((NCAA_features ['FGA'] + 0.4 * NCAA_features ['FTA'] - 1.07 * (NCAA_features ['OR'] / (NCAA_features ['OR'] + NCAA_features ['O_DR'])) * (NCAA_features ['FGA'] - NCAA_features ['FGM']) + NCAA_features ['TO']) + (NCAA_features ['O_FGA'] + 0.4 * NCAA_features ['O_FTA'] - 1.07 * (NCAA_features ['O_OR'] / (NCAA_features ['O_OR'] + NCAA_features ['DR'])) * (NCAA_features ['O_FGA'] - NCAA_features ['O_FGM']) + NCAA_features ['O_TO']))
    NCAA_features ['O_POSS']    = 0.5 * ((NCAA_features ['O_FGA'] + 0.4 * NCAA_features ['O_FTA'] - 1.07 * (NCAA_features ['O_OR'] / (NCAA_features ['O_OR'] + NCAA_features ['DR'])) * (NCAA_features ['O_FGA'] - NCAA_features ['O_FGM']) + NCAA_features ['O_TO']) + (NCAA_features ['FGA'] + 0.4 * NCAA_features ['FTA'] - 1.07 * (NCAA_features ['OR'] / (NCAA_features ['OR'] + NCAA_features ['O_DR'])) * (NCAA_features ['FGA'] - NCAA_features ['FGM']) + NCAA_features ['TO']))
    NCAA_features ['GM']        = (40*NCAA_features ['GameNum']) + (5*NCAA_features ['NumOT'])
    NCAA_features ['PACE']      = 40 * ((NCAA_features ['POSS'] ) / (2 * (NCAA_features ['GM'] / 5)))
    NCAA_features ['DRTG']      = 100* (NCAA_features ['O_SCORE']/NCAA_features ['POSS'])
    NCAA_features ['ORTG']      = 100* (NCAA_features ['SCORE']/(NCAA_features ['O_POSS']))
    NCAA_features ['OFF3']      = (NCAA_features ['FGM3']/(NCAA_features ['FGA3']))
    NCAA_features ['DEF3']      = (NCAA_features ['O_FGM3']/(NCAA_features ['O_FGA3']))
    NCAA_features ['O_EFG']     = (NCAA_features ['O_FGM'] + (NCAA_features ['O_FGM3']*0.5))/NCAA_features ['O_FGA']
    NCAA_features ['O_TOV']     = NCAA_features ['O_TO']/((NCAA_features ['O_FGA'] + 0.44) + (NCAA_features ['O_FTA']+NCAA_features ['O_TO']))
    NCAA_features ['DEFRTG']    = 100*NCAA_features ['O_SCORE']/(NCAA_features ['O_FGA'] + NCAA_features ['O_TO'] + (0.44* NCAA_features ['O_FTA']) - NCAA_features ['O_OR'])
    NCAA_features ['OFFRTG']    = 100*NCAA_features ['SCORE']/(NCAA_features ['FGA'] + NCAA_features ['TO'] + (0.44*NCAA_features ['FTA']) - NCAA_features ['OR'])
    NCAA_features ['TOR']       = (NCAA_features ['TO'] * 100) / (NCAA_features ['FGA'] + (NCAA_features ['FTA'] * 0.44) + NCAA_features ['AST'] + NCAA_features ['TO'])
    NCAA_features ['STLTO']     = NCAA_features ['STL']/NCAA_features ['TO']
    NCAA_features ['PIE']       = (NCAA_features ['SCORE'] + NCAA_features ['FGM'] + NCAA_features ['FTM'] - NCAA_features ['FGA']  - NCAA_features ['FTA']  + NCAA_features ['DR'] + (.5 * NCAA_features ['OR']) + NCAA_features ['AST'] + NCAA_features ['STL'] + (.5 * NCAA_features ['BLK']) - NCAA_features ['PF'] - NCAA_features ['TO']) / ((NCAA_features ['SCORE'] + NCAA_features ['FGM'] + NCAA_features ['FTM'] - NCAA_features ['FGA']  - NCAA_features ['FTA']  + NCAA_features ['DR'] + (.5 * NCAA_features ['OR']) + NCAA_features ['AST'] + NCAA_features ['STL'] + (.5 * NCAA_features ['BLK']) - NCAA_features ['PF'] - NCAA_features ['TO'])  + (NCAA_features ['O_SCORE'] + NCAA_features ['O_FGM'] + NCAA_features ['O_FTM'] - NCAA_features ['O_FGA'] - NCAA_features ['O_FTA'] + NCAA_features ['O_DR'] + (.5 * NCAA_features ['O_OR']) + NCAA_features ['O_AST'] + NCAA_features ['O_STL'] + (.5 * NCAA_features ['O_BLK']) - NCAA_features ['O_PF'] - NCAA_features ['O_TO']))
    NCAA_features ['O_STLTO']   = NCAA_features ['O_STL']/NCAA_features ['O_TO']
    NCAA_features ['O_TOR']     = (NCAA_features ['O_TO'] * 100) / (NCAA_features ['O_FGA'] + (NCAA_features ['O_FTA'] * 0.44) + NCAA_features ['O_AST'] + NCAA_features ['O_TO'])
    NCAA_features ['O_FTAR']    = NCAA_features ['O_FTA']/(NCAA_features ['O_FGA'])
    NCAA_features ['O_TS']      =  NCAA_features ['O_SCORE']/((NCAA_features ['O_FGA']*2) + (0.88 * NCAA_features ['O_FTA']))
    NCAA_features ['O_ASTTO']   = (NCAA_features ['O_AST']/(NCAA_features ['O_TO']))
    NCAA_features ['O_ASTR']    = (NCAA_features ['O_AST'] * 100) / ( (NCAA_features ['O_FGA'] + (NCAA_features ['O_FTA']*0.44)) + NCAA_features ['O_AST'] + NCAA_features ['O_TO'] )
    return NCAA_features


# In[ ]:


regSes_adStats = GetAdvancedStats(regSes_aggregate)
regSes_adStats.shape


# We will drop all of the raw aggregated features as teams will play a different number of games per season as the advanced features are all inherently normalized.  Eg Defensive Rating is the number of points given up per 100 possessions.  

# In[ ]:


#Remove raw features
regSes_adStats = regSes_adStats.drop(['SCORE', 'O_SCORE', 
                 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 
                 'AST', 'TO', 'STL', 'BLK', 'PF', 'NumOT',  
                 'O_FGM', 'O_FGA', 'O_FGM3', 'O_FGA3', 'O_FTM', 'O_FTA', 'O_OR', 'O_DR', 
                 'O_AST', 'O_TO', 'O_STL', 'O_BLK', 'O_PF','GameNum','POSS','GM','O_POSS' ], axis=1)
regSes_adStats.shape


# **Section 5 - Get NCAA Tourney data and pre process it**
# 
# The plan is to join all the features we just generated from the reg season games with the NCAA tourney games, along with the NCAA tourney outcome. All available tournament games will be used within our model.

# In[ ]:


def NCAASetWinAndLoseTeamsRecords(NCAATourneyCompactResults):
    #Convert the data frame from one record per game to one record per team-game
    NCAA_res_w = NCAATourneyCompactResults.rename(columns = {'WTeamID': 'NCAA_TEAMID',
                                                           'LTeamID': 'NCAA_O_TEAMID',
                                                           'WScore':'NCAA_SCORE',
                                                           'LScore':'NCAA_O_SCORE'
                                                             })
    NCAA_res_l = NCAATourneyCompactResults.rename(columns = {'LTeamID': 'NCAA_TEAMID',
                                                           'WTeamID': 'NCAA_O_TEAMID',
                                                           'LScore':'NCAA_SCORE',
                                                           'WScore':'NCAA_O_SCORE'
                                                             })

    NCAA_Ses = (NCAA_res_w, NCAA_res_l)
    NCAA_Ses = pd.concat(NCAA_Ses, ignore_index = True, sort = False)
    #Derive the outcome of who won[1] or loss[0]
    NCAA_Ses ['OUTCOME'] = np.where(NCAA_Ses['NCAA_SCORE']>NCAA_Ses['NCAA_O_SCORE'], 1, 0)
    NCAA_Ses = NCAA_Ses[['Season','NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME']]
    return NCAA_Ses


# In[ ]:


NCAATourneyCompactResults = pd.read_csv(in_path + 'NCAATourneyCompactResults.csv')
NCAA_Ses = NCAASetWinAndLoseTeamsRecords(NCAATourneyCompactResults)
NCAA_Ses.shape


# In[ ]:


#join ncaa and regSes_adStats ses data for primary team
NCAA_reg = pd.merge(NCAA_Ses, regSes_adStats, how='inner', 
                   left_on=['Season', 'NCAA_TEAMID'], 
                   right_on=['Season', 'TEAMID'])
#join to add your opponent's regSes_adStats 
NCAA_reg = pd.merge(NCAA_reg, regSes_adStats, how='inner',
                    left_on=['Season', 'NCAA_O_TEAMID'],
                    right_on=['Season', 'TEAMID'], suffixes =['', '_op'] )
NCAA_reg.shape


# **Add Seed data**

# In[ ]:


NCAATourneySeeds = pd.read_csv(in_path + 'NCAATourneySeeds.csv')
Seeds = NCAATourneySeeds.copy()
Seeds['Seed'] = Seeds.Seed.str.replace('[a-zA-Z]', '')
Seeds['Seed']=Seeds['Seed'].astype('int64')
Seeds.head(5)


# In[ ]:


NCAA_reg = pd.merge(NCAA_reg, Seeds, how='inner',
                    left_on=['Season', 'NCAA_TEAMID'],
                    right_on=['Season', 'TeamID'])
NCAA_reg = pd.merge(NCAA_reg, Seeds, how='inner',
                    left_on=['Season', 'NCAA_O_TEAMID'],
                    right_on=['Season', 'TeamID'], suffixes =['', '_op'] )
NCAA_reg.shape


# Lets look at our generated features dataframe

# Lets drop some of the demographic features we dont need for the model

# In[ ]:


cols_to_drop = ['TEAMID', 'TeamID', 'TeamID_op']
NCAA_features = NCAA_reg.drop(cols_to_drop, axis=1)
NCAA_features.shape


# In[ ]:


NCAA_features.head(5)


# Lets save this file for now using a 'feather' format as recommended per fastai

# In[ ]:


NCAA_features.to_feather('NCAA_features')


# **Part 2 - Model**
# 
# Now that we have our feature set, lets use that to train a RandomForest Model

# **Section 1 - Get a train test split**
# 
# Let's split our data based on a time series. Let's use data between 2003-2016 to train our model; 2017 for validation; and use 2018 to test our model.  As a rationale, it is good practice to do a train, validation, and test split.  We will use the validation dataset to incrementally train our dataset while tuning. We will use test dataset as an unbiased evaluation of a final model fit on the training dataset.

# In[ ]:


NCAA_features = pd.read_feather('NCAA_features')
NCAA_features.shape


# In[ ]:


train = NCAA_features[NCAA_features.Season <= 2016]
valid = NCAA_features[NCAA_features.Season == 2017]
test = NCAA_features[NCAA_features.Season == 2018]
print("train shape = ", train.shape)
print ("valid shape = ", test.shape)
print ("test shape = ", test.shape)


# In[ ]:


X_train = train.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)
y_train = train[['OUTCOME']]
X_valid = valid.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)
y_valid = valid[['OUTCOME']]
X_test = test.drop(['Season', 'NCAA_TEAMID', 'NCAA_O_TEAMID', 'OUTCOME'], axis=1)
y_test = test[['OUTCOME']]


# **Section 2 - Basic RandomForest Model**
# 
# Let's start with a tree based model.

# *Method to print model scores. We will use this to determine how our model is performing as we make changes.  Out-of-bag (OOB) score is used to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was not included in training that tree.  This allows us to see whether the model is over-fitting, without needing a separate validation set.*

# In[ ]:


def print_score(m):
    print ("train score :", m.score(X_train, y_train))
    print ("valid score :", m.score(X_valid, y_valid))
    if hasattr(m, 'oob_score_'): print ("oob_score : ", m.oob_score_)


# In[ ]:


m = RandomForestClassifier(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# As expected we get a very good training score, but not so good validation score on a basic model. 

# **Draw Tree**
# 
# Lets look at single tree (n_estimators=1) to get an idea of where the features are split.
# 
# FastAI V1.0 doesn't have the draw tree method, so I'm using it from its previous version here. 

# In[ ]:


m = RandomForestClassifier(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


from sklearn.tree import export_graphviz
import IPython
import graphviz
def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))


# In[ ]:


draw_tree(m.estimators_[0], X_train, size=7, precision=3)


# The first split is based on Offensive rating. Which makes sense, as the primary team's performance will depend on its offensive rating. However, this is just a single tree. 
# 
# Lets keep building the tree, and eventually we will look at this using feature importances. 

# **Section 3 - Tune for number of trees**

# In[ ]:


m = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, random_state=0)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=500, n_jobs=-1, oob_score=True, random_state=0)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=700, n_jobs=-1, oob_score=True, random_state=0)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


m = RandomForestClassifier(n_estimators=800, n_jobs=-1, oob_score=True, random_state=0)
m.fit(X_train, y_train)
print_score(m)


# Im going to keep the number of trees at 700. 
# 
# *Note - you could use a library like GridSearchCV and/or tune for the exact number of trees between 500 and 700 to see how high the score can get, but Im going to leave this at 700 for now.*

# **Section 4 - Number of samples per leaf/node**
# 
# One way to reduce over-fitting is to grow our trees less deeply. We do this by specifying (with min_samples_leaf) that we require some minimum number of rows in every leaf node. This has two benefits:
# 
# * There are less decision rules for each leaf node; simpler models should generalize better
# * The predictions are made by averaging more rows in the lead node, resulting in less volatility
# 
# Lets set that we require a minimum of 3 samples per leaf/node

# In[ ]:


m = RandomForestClassifier(n_estimators=700, min_samples_leaf=3, n_jobs=-1, oob_score=True, 
                           random_state=0)
m.fit(X_train, y_train)
print_score(m)


# **Section 5 - Tune for max features**
# 
# We can also increase the amount of variation amongst the trees by not only using a sample of rows for each tree, but to also using a sample of columns for each split. We do this by specying max_features, which is the proportion of features to randomly select from at each split. 
# 
# Lets try the following options - 
# 
# * 0.5
# * sqrt
# * log2
# * 25, 50, 80

# In[ ]:


for m_f in (0.5, 'sqrt', 'log2', 25, 50):
    print ("max_features = ", m_f)
    m = RandomForestClassifier(n_estimators=700, min_samples_leaf=3, n_jobs=-1, 
                              oob_score=True, random_state=0, 
                              max_features=m_f)
    m.fit(X_train, y_train)
    print_score(m)
    print (" ")


# We observe that 25 gives us the best score. 

# **Summary of model parameters **
# 
# * n_estimators = 700
# * min_samples_leaf = 3
# * max_features = 25
# 
# *Note - as we add or edit additional features, we need to go through this exercise again to pick the best parameters*

# **Section 6 - Run model on test data**
# 
# Lets see how our model performs on the test data

# In[ ]:


def print_score_test(m):
    print ("train score :", m.score(X_train, y_train))
    print ("test score :", m.score(X_test, y_test))
    if hasattr(m, 'oob_score_'): print ("oob_score : ", m.oob_score_)


# In[ ]:


m = RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features=25, 
                           n_jobs=-1, oob_score=True, random_state=0)
m.fit(X_train, y_train)
print_score_test(m)


# Lets actually combine the train and valid data into one train dataset, and use that to train our model. And get a test prediction score with that. 

# In[ ]:


print ("X_train shape before append: ", X_train.shape)
print ("X_valid shape: ", X_valid.shape)
print ("y_train shape before append: ", y_train.shape)
print ("y_valid shape: ", y_valid.shape)
X_train = X_train.append(X_valid)
y_train = y_train.append(y_valid)
print ("X_train shape after append: ", X_train.shape)
print ("y_train shape after append: ", y_train.shape)


# In[ ]:


m = RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features=25, 
                           n_jobs=-1, oob_score=True, random_state=0)
m.fit(X_train, y_train)
print_score_test(m)


# **Feature Importance**
# 
# We want to know how the model is making predictions. The best way to see this is with feature importance.

# Below is the function from fastai, which makes getting feature importances easier

# In[ ]:


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[ ]:


fi = rf_feat_importance(m, X_valid)
fi[:10]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:10]);


# This makes a lot more sense than the features we saw in the single tree earlier in this notebook. 
# 
# Seed and Seed_op are inherently a model in itself, to show which teams are better ranked. Therefore it makes sense they are the top important features. 
# 
# PIE and PIE_op are also comprehensive stats, therefore make sense they are in the top list. Per NBA stats website (https://stats.nba.com), a team's PIE rating and a team's winning percentage correlate at a R2 of 0.908. 

# **Summary**
# 
# This is our take on a starter set of advanced features, and basic model tuning. The scores arent as high yet, but the plan is to continue to work on building more/better features, and better models. 

# In[ ]:




