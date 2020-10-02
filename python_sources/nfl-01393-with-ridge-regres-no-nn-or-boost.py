#!/usr/bin/env python
# coding: utf-8

# Just FYI - This is completely original work.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import math, copy, time, os

from kaggle.competitions import nflrush

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


def grid_plot_histograms(df,grid_columns = 3,sample_size = 5000,figsize = (16,8)):
    """I created this setup to flexibly create subplots of histograms."""
    features = list(df.columns)
    grid_count = len(features)

    fig, axarr = plt.subplots(math.ceil(grid_count/grid_columns),grid_columns,figsize = figsize)
    
    # this uses the iterator "i" to determine which subplot to fill
    for i, feature in enumerate(features):
        axarr[math.floor(i/grid_columns)][i%grid_columns].hist(df[feature].sample(sample_size),bins = 30);
        axarr[math.floor(i/grid_columns)][i%grid_columns].set_title('{} \n {}'.format(feature,feature_dict[feature]));
    
    # This uses i again to delete the remaining subplots
    for i in range(grid_count,math.ceil(grid_count/grid_columns)*grid_columns):
        fig.delaxes(axarr[math.floor(i/grid_columns)][i%grid_columns]);
    
    plt.tight_layout();


# In[ ]:


train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory = False)


# In[ ]:


dist = train["Yards"].hist(density = True, cumulative = True, bins = 200)
plt.title('Cumulative Distribution - Yards');


# In[ ]:


#Datatypes

dtypes = train.dtypes
int_feats = list(dtypes[dtypes =='int64'].index)
float_feats = list(dtypes[dtypes =='float64'].index)
obj_feats = list(dtypes[dtypes =='object'].index)
dtypes.value_counts()


# In[ ]:


feature_dict = {
    'GameId':'a unique game identifier',
    'PlayId':'a unique play identifier',
    'Team':'home or away',
    'X' : 'player position along the long axis of the field.',
    'Y' : 'player position along the short axis of the field.',
    'S' :'speed in yards/second',
    'A' : 'acceleration in yards/second^2',
    'Dis' : 'distance traveled from prior time point, in yards',
    'Orientation' : 'orientation of player (deg)',
    'Dir' : 'angle of player motion (deg)',
    'NflId': 'a unique identifier of the player',
    'DisplayName':' player\'s name',
    'JerseyNumber':'jersey number',
    'Season' : 'year of the season',
    'YardLine' : 'the yard line of the line of scrimmage',
    'Quarter' : 'game quarter (1-5, 5 == overtime)',
    'GameClock' : 'time on the game clock',
    'PossessionTeam' : 'team with possession',
    'Down' : 'the down (1-4)',
    'Distance' : 'yards needed for a first down',
    'FieldPosition' : 'which side of the field the play is happening on',
    'HomeScoreBeforePlay' : 'home team score before play started',
    'VisitorScoreBeforePlay' :'visitor team score before play started',
    'NflIdRusher' : 'the NflId of the rushing player',
    'OffenseFormation' : 'offense formation',
    'OffensePersonnel' : 'offensive team positional grouping',
    'DefendersInTheBox' : 'number of defenders lined up near the line',
    'DefensePersonnel' : 'defensive team positional grouping',
    'PlayDirection' : 'direction the play is headed',
    'TimeHandoff' :'UTC time of the handoff',
    'TimeSnap' : 'UTC time of the snap',
    'Yards' : 'the yardage gained on the play',
    'PlayerHeight' :'player height (ft-in)',
    'PlayerWeight' : 'player weight (lbs)',
    'PlayerBirthDate' : 'birth date (mm/dd/yyyy)',
    'PlayerCollegeName' : 'where the player attended college',
    'HomeTeamAbbr' : 'home team abbreviation',
    'VisitorTeamAbbr' : 'visitor team abbreviation',
    'Week' : 'week into the season',
    'Stadium' : 'stadium where the game is being played',
    'Location' : 'city where the game is being player',
    'StadiumType' : 'description of the stadium environment',
    'Turf' : 'description of the field surface',
    'GameWeather' : 'description of the game weather',
    'Temperature' : 'temperature (deg F)',
    'Humidity' : 'humidity',
    'WindSpeed' : 'wind speed in miles/hour',
    'WindDirection' : 'wind direction',
    'SecondsToHandoff' : 'Custom Feature: TimeHandoff minus TimeSnap in seconds'
    }


# In[ ]:


grid_plot_histograms(train[float_feats].dropna(),3,5000,(16,10))


# In[ ]:


# A few examples to get familiar
train[float_feats].head(8).transpose()


# In[ ]:


grid_plot_histograms(train[int_feats],4,5000,(16,10))


# In[ ]:


# A few examples to get familiar
train[int_feats].head(8).transpose()


# # Functions

# It might be a bit harder to read this way, but I created all these functions so I could test different models and make small feature changes for quick experimentation.  Basically, all the functions allowed me to compile stats on different features... keep reading and you'll see what I mean.

# In[ ]:


def time_seconds(x):
    x = time.strptime(x[:-5], "%Y-%m-%dT%H:%M:%S")
    x = time.mktime(x)
    return x

def create_cdf(y):
    y_cdf = copy.deepcopy(y.to_frame())
    y_cdf.columns = ['Yards']
    y_cdf.head()
    for i in list(range(-99,100)):
    #for i in list(range(0,10)):    
        y_cdf['Yards' + str(i)] = y_cdf['Yards'].apply(lambda x: 1 if i >= x else 0)
    y_cdf.drop('Yards',1,inplace = True)
    #print('y_cdf.shape {}'.format(y_cdf.shape))
    y_cdf.head(3)
    return y_cdf

def add_custom_feats(train):
    train = train.loc[train['NflId']==train['NflIdRusher']]
    train['SecondsToHandoff'] = train['TimeHandoff'].apply(time_seconds) - train['TimeSnap'].apply(time_seconds)
    train['OffenseTeam'] = train['PossessionTeam']
    train['adj_height'] = train['PlayerHeight'].apply(lambda x: int(x[0])*12 + int(x[2:4]))
    train['OffenseHome'] = train[['OffenseTeam','HomeTeamAbbr']].apply(lambda x: 1 if x[0] == x[1] else 0, axis = 1)
    train['DefenseTeam'] = train[['OffenseHome','HomeTeamAbbr','VisitorTeamAbbr']].apply(lambda x: x[2] if x[0] == 1 else x[1], axis = 1)
    train['OffenseLead'] = train[['OffenseHome','HomeScoreBeforePlay','VisitorScoreBeforePlay']].apply(lambda x: x[1]-x[2] if x[0] == 1 else x[2]-x[1], axis = 1)
    train['YardsToGo'] = train[['FieldPosition','OffenseTeam','YardLine']].apply(         lambda x: (50-x['YardLine'])+50 if x['OffenseTeam']==x['FieldPosition'] else x['YardLine'],1)
    train['SecondsToHandoff'] = train['TimeHandoff'].apply(time_seconds) - train['TimeSnap'].apply(time_seconds)
    train['turf'] = train['Turf'].apply(lambda x: int('turf' in x.lower() or 'artific' in x.lower()))
    train['quarter_seconds_left'] = train['GameClock'].apply(lambda x: int(x[0:2])*60 + int(x[3:5]))
    train['game_seconds_left'] = train['Quarter'].map({1:2700, 2:1800, 3:900, 4:0}) + train['quarter_seconds_left']
    train['game_seconds_passed'] = 3600 - train['game_seconds_left']
    train['OffensePoints'] = train[['OffenseHome','HomeScoreBeforePlay','VisitorScoreBeforePlay']].apply(lambda x: x[1] if x[0] == 1 else x[2], axis = 1)
    train['OffensePointsPerMinute'] = train['OffensePoints'] / train['game_seconds_passed']*60
    return train

def add_yard_bins(test_df, test_data, bins = [0,2,5,8,10,15,20,30,40,50,60,70,80,100]):
    # Add yard_bins
    test_df['YardsToGo'] = test_df[['FieldPosition','OffenseTeam','YardLine']].apply(         lambda x: (50-x['YardLine'])+50 if x['OffenseTeam']==x['FieldPosition'] else x['YardLine'],1)
    yard_bins = pd.get_dummies(pd.cut(test_df['YardsToGo'],bins = bins))
    yard_bins.columns = [str(i) for i in yard_bins.columns.tolist()]
    test_data = pd.merge(test_data,yard_bins,left_index = True, right_index = True)
    test_data.columns = [str(x).replace(']','').replace('(','') for x in test_data.columns]
    return test_data

def add_categories(train_data, cat_feats, top_n_categories = 120, count_min = 50):
    #print('train_data shape: {}'.format(train_data.shape))
    categorical_stats = pd.DataFrame()
    for i in cat_feats:
        stats = train[[i,'Yards']].groupby(i).agg(['mean','count','std','max','min'])
        stats.columns = stats.columns.droplevel(0)
        stats['feature'] = i
        stats['feature_type'] = stats.index.values
        stats.reset_index(inplace = True)
        stats.drop(i,1,inplace = True)
        stats = stats[['feature','feature_type','mean','count','std','max','min']]
        categorical_stats = categorical_stats.append(stats)

    # I basically just made up this calculation to select the most significant categorical features-values, and it seems to work well
    categorical_stats['mean_difference'] = (categorical_stats['mean'] - 4.21).abs()
    categorical_stats['significance'] = categorical_stats['mean_difference'] * categorical_stats['count']**0.2

    cat_feats_keep = categorical_stats[categorical_stats['count']>count_min]             .sort_values('significance',ascending = False).head(top_n_categories)
    
    #print('train_data shape: {}'.format(train_data.shape))
    for feature in set(cat_feats_keep['feature']):
        feature_types = cat_feats_keep[cat_feats_keep['feature']==feature]['feature_type'].tolist()
        feature_dummies = pd.get_dummies(train[feature].apply(lambda x: 0 if x not in feature_types else x),prefix = feature)
        feature_dummies.drop(feature + '_0',1,inplace = True)
        train_data = pd.merge(train_data,feature_dummies,left_index = True, right_index = True)
#        print('{} Added: new train_data shape: {}'.format(feature,train_data.shape))    
    return train_data, cat_feats_keep

def add_player_feats_df(train_data, train_original, top_n_player_counts = 500, top_n_players_significance = 100):
    players = train_original[['NflId','Yards']].groupby('NflId').agg(['mean','count'])
    players.columns = ['mean','count']
    players['weight'] = players['count']**.2
    players['difference'] = players['mean'] - 4.21
    players['significance'] = (players['difference'] * players['weight']).abs()
    #players['significance'] = players['count']
    players_keep = players
    players_keep = players_keep.sort_values('count',ascending = False).head(top_n_player_counts)
    players_keep = players_keep.sort_values('significance',ascending = False).head(top_n_players_significance)
    #### Creating player_feats (get_dummies) for Top Players
    player_list = list(players_keep.index)
    player_feats = train_original[['PlayId','NflId']][train_original['NflId'].isin(player_list)]
    player_feats.set_index('PlayId',inplace = True)
    player_feats = pd.get_dummies(player_feats['NflId'], prefix = 'NflId')
    player_feats = player_feats.groupby(player_feats.index).sum()
    train_data = pd.merge(train_data, player_feats, left_index = True, right_index = True, how = 'left')
    train_data.fillna(0,inplace = True)
    return train_data, player_list

def array_in_range(pred):
    ones = np.ones(pred.shape)
    zeros = np.zeros(pred.shape)
    pred = np.maximum(pred,zeros)
    pred = np.minimum(pred,ones)
    return pred

def array_increasing(pred):
    for i in range(1,pred.size):
        pred[0][i] = max(pred[0][i],pred[0][i-1])
    return pred

def model_scores(models, X_train, y_train, X_test, y_test, params, description):
    model_results = pd.DataFrame()
    model_data = {}
    for model in models:    
        model.fit(X_train, y_train)
        model_name = model.__class__.__name__
        test_predictions = model.predict(X_test)
        train_predictions = model.predict(X_train)
        test_score = round(mean_squared_error(y_test, test_predictions),4)
        train_score = round(mean_squared_error(y_train, train_predictions),4)
        test_mae = round(mean_absolute_error(y_test, test_predictions),4)
        train_mae = round(mean_absolute_error(y_train, train_predictions),4)
        model_data['model_name'] = [model_name]
        model_data['test_score'] = [test_score]
        model_data['MAE'] = [test_mae]
        model_data['params'] = [str(params)]
        model_data['feature_count'] = [X_train.shape[1]]
        model_data['description'] = [description]
        model_data['model'] = [model]
        model_data['train_score'] = [train_score]
        print('{} MSE: {}'.format(model_name,test_score))
        print('{} MAE: {}'.format(model_name,test_mae))
        model_results = model_results.append(pd.DataFrame.from_dict(model_data, orient = 'columns'))
    model_results.sort_values('test_score', ascending = True, inplace = True)
    return model_results

def transform_pred_cdf(prediction,sample_prediction_df):
    prediction = array_increasing(array_in_range(prediction))
    pred_target = pd.DataFrame(index = sample_prediction_df.index,                                columns = sample_prediction_df.columns,                                data = prediction)
    return pred_target

def dummy_all(train_data, train, features):
    for i in features:
        train_data = pd.merge(train_data, pd.get_dummies(train[i], prefix = i), left_index = True, right_index = True)
    return train_data

def fit_transform_num_feats(train, num_feats, scaler, imp):
    train_data = train[num_feats]
    train_data = pd.DataFrame(index = train_data.index, columns = train_data.columns, data = scaler.fit_transform(train_data))
    train_data = pd.DataFrame(index = train_data.index, columns = train_data.columns, data = imp.fit_transform(train_data))
    return train_data

def print_parameters(params, num_feats, bins, cat_feats):
    print('params:{} - num_feats:{} - bins:{} - cat_feats:{}'.format(params,num_feats,bins,cat_feats))

def add_team_stats(train,train_original):
    trainXY = copy.deepcopy(train[['X','Y','S','A','Position','Team']])
    origXY = copy.deepcopy(train_original[['X','Y','PlayId','S','A','Position','Team']])
    origXY.set_index('PlayId', inplace = True)
    trainXY.columns = ['train_' + x for x in trainXY.columns]
    #print(set(train_original['Position']))
    locations = pd.merge(trainXY,origXY, left_index = True, right_index = True, how = 'left')
    locations['PlayerDistance'] = locations.apply         (lambda x: np.sqrt(np.square(x['train_X'] - x['X']) +  np.square(x['train_Y'] - x['Y'])), axis = 1)
    locations.reset_index(inplace = True)
    distances = locations.groupby(['PlayId','Team'])['PlayerDistance','S','A'].mean().reset_index()
    team_stats = distances.pivot_table(index = ['PlayId'], columns = ['Team'], values = ['PlayerDistance','S','A'])
    team_stats.columns = ['%s%s' % (a, '|%s' % b if b else '') for a, b in team_stats.columns]
    team_stats['speed_difference'] = team_stats['S|home'] - team_stats['S|away']
    train = pd.merge(train, team_stats,left_index = True, right_index = True)
    return train, team_stats.columns

def add_position_stats(train,train_original):
    trainXY = copy.deepcopy(train[['X','Y','S','A','Position','Team','OffenseDefense']])
    origXY = copy.deepcopy(train_original[['X','Y','PlayId','S','A','Position','Team','OffenseDefense']])
    origXY.set_index('PlayId', inplace = True)
    trainXY.columns = ['train_' + x for x in trainXY.columns]
    #print(set(train_original['Position']))
    locations = pd.merge(trainXY,origXY, left_index = True, right_index = True, how = 'left')
    locations['PlayerDistance'] = locations.apply         (lambda x: np.sqrt(np.square(x['train_X'] - x['X']) +  np.square(x['train_Y'] - x['Y'])), axis = 1)
    locations.reset_index(inplace = True)

    distances = locations.groupby(['PlayId','OffenseDefense','Position'])['PlayerDistance','S','A'].mean().reset_index()
    team_stats = distances.pivot_table(index = ['PlayId'], columns = ['OffenseDefense','Position'], values = ['S'])
    team_stats.columns = ['{}|{}|{}'.format(a,b,c) for a, b, c in team_stats.columns]
    team_stats.fillna(0,inplace = True)
    train = pd.merge(train, team_stats,left_index = True, right_index = True)
    return train, team_stats.columns


# # Data Pulls

# In[ ]:


train_original = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory = False)
train = copy.deepcopy(train_original[train_original['NflId']==train_original['NflIdRusher']])
train.index = train['PlayId']


# # Experiment Tester

# You can run tons of experiments here and keep track of performance of each model and the the parameters or other statistics in case you need to go back and use it again.

# In[ ]:


description_suffix = 0
experiments = pd.DataFrame()


# In[ ]:


### Train Feature Engineering
scaler = StandardScaler()
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

### Setup
description_suffix += 1

description = 'old team stats' + str(description_suffix)
top_n_player_counts = 900
top_n_players_significance = 120
top_n_categories = 120
count_min = 100
bins = [0,2,4,6,8,10,15,20,30,40,50,60,70,80,90,100]
params = str((top_n_player_counts, top_n_players_significance, top_n_categories, count_min))

##Custom Feats
train = add_custom_feats(train)

##Number Feats
train, team_stats_columns = add_team_stats(train, train_original)
#train, position_stats_columns = add_position_stats(train, train_original)

# New stuff since last commit
offense = ['RB','TE','WR','OL']
defense = ['DL','LB','DB']

for i in offense:
    train[i] = train['OffensePersonnel'].apply(lambda x: int(x[x.find(i) - 2]) if x.find(i)>0 else 0)
for i in defense:
    train[i] = train['DefensePersonnel'].apply(lambda x: int(x[x.find(i) - 2]) if x.find(i)>0 else 0)


num_feats = ['S', 'A', 'Dis', 'adj_height', 'Temperature', 'Humidity', 'SecondsToHandoff', 'Distance', 'PlayerWeight', 'OffenseLead', 'quarter_seconds_left', 'game_seconds_left', 'YardsToGo']
#num_feats.extend(['S|defense', 'S|offense', 'speed_difference'])
num_feats.extend(team_stats_columns)
num_feats.extend(offense)
num_feats.extend(defense)

train_data = fit_transform_num_feats(train, num_feats, scaler, imp)

#Player Feats
train_data, player_list = add_player_feats_df(train_data, train_original, top_n_player_counts, top_n_players_significance)

#Yard Bins
train_data = add_yard_bins(train, train_data, bins) #clean up columns

# Categories
cat_feats = ['DisplayName','PlayerCollegeName','OffensePersonnel','DefensePersonnel','Position',             'OffenseFormation','Down','OffenseTeam','NflIdRusher']
train_data, cat_feats_keep = add_categories(train_data, cat_feats, top_n_categories, count_min)

# Dummies:
#train_data = dummy_all(train_data, train, ['DefendersInTheBox'])

# Train/Test Split
X, y = train_data, train['Yards']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Exhaustive List
test_models = [
    Ridge(),
    Lasso(),
    ElasticNet(),
    Ridge(**{'alpha': 220, 'solver': 'lsqr'})
    #LogisticRegression(),
    #Ridge(),
    #MLPRegressor(),
    #inearSVR(),
    #BaggingRegressor(),
    #LGBMRegressor(),
    #VotingRegressor([('r',Ridge()),('f',RandomForestRegressor())]),
    #GradientBoostingRegressor(),
    #XGBRegressor()
    ]

print_parameters(params, num_feats, bins, cat_feats)

model_results = model_scores(test_models, X_train, y_train, X_test, y_test, params, description)
experiments = experiments.append(model_results)
experiments.sort_values('test_score', ascending = True)


# In[ ]:


y_cdf = create_cdf(y)
final_model = Ridge(**{'alpha': 220, 'solver': 'lsqr'}).fit(X, y_cdf)


# In[ ]:


# Setup
env = nflrush.make_env()
iter_test = env.iter_test()


# In[ ]:


def prep_prediction_submission(test_df, sample_prediction_df, final_model):
    #(test_df, sample_prediction_df) = next(iter_test)
    test_df_original = copy.deepcopy(test_df)
    test_df = test_df.loc[test_df['NflId']==test_df['NflIdRusher']]
    test_df = add_custom_feats(test_df)
    test_df.set_index('PlayId',inplace = True) #only on test
    test_df, test_df_columns = add_team_stats(test_df, test_df_original)
    
    offense = ['RB','TE','WR','OL']
    defense = ['DL','LB','DB']
    for i in offense:
        test_df[i] = test_df['OffensePersonnel'].apply(lambda x: int(x[x.find(i) - 2]) if x.find(i)>0 else 0)
    for i in defense:
        test_df[i] = test_df['DefensePersonnel'].apply(lambda x: int(x[x.find(i) - 2]) if x.find(i)>0 else 0)
        
    test_data = test_df[num_feats]
    test_data = pd.DataFrame(index = test_data.index, columns = test_data.columns, data = scaler.transform(test_data))
    test_data = pd.DataFrame(index = test_data.index, columns = test_data.columns, data = imp.transform(test_data))
    # Add categories
    for feature in set(cat_feats_keep['feature']):
        feature_types = cat_feats_keep[cat_feats_keep['feature']==feature]['feature_type'].tolist()
        feature_dummies = pd.get_dummies(test_df[feature].apply(lambda x: 0 if x not in feature_types else x),prefix = feature)
        if feature + '_0' in list(feature_dummies.columns):
            feature_dummies.drop(feature + '_0',1,inplace = True)
        if feature_dummies.shape[1] > 0:
            test_data = pd.merge(test_data,feature_dummies,left_index = True, right_index = True)
    # Add player feats
    add_feats = set(test_df_original['NflId']).intersection(player_list)
    for i in add_feats:
        test_data[i] = 1
    test_data = add_yard_bins(test_df, test_data,bins) #clean up columns
    # Columns (missing and order)
    new_columns = np.setdiff1d(list(train_data.columns),list(test_data.columns))
    for i in new_columns:
        test_data[i] = 0
    test_data = test_data[train_data.columns]
    prediction = final_model.predict(test_data)
    pred_target = transform_pred_cdf(prediction,sample_prediction_df)
    return pred_target


# In[ ]:


progress_counter = 0

for (test_df, sample_prediction_df) in iter_test:
    progress_counter+=1
    if progress_counter%250 == 0:
        print(progress_counter)
    pred_target = prep_prediction_submission(test_df, sample_prediction_df, final_model)
    env.predict(pred_target)


# In[ ]:


env.write_submission_file()

