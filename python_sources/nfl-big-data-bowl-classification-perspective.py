#!/usr/bin/env python
# coding: utf-8

# # NFL Big Data Bowl
# 
# In the competition we are asked to predict how many yards a NFL player will gain after receiving a handoff.
# 
# There is no real EDA for this kernel. I wanted to make a kernel that demonstrates this modeling process from the classification perspective. If you are interested in good EDA kernels and pairing it with this kernel there is a great kernel by Rob Mulla here https://www.kaggle.com/robikscube/big-data-bowl-comprehensive-eda-with-pandas and also another good kernel by SRK here https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-nfl.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from multiprocessing import Pool
from kaggle.competitions import nflrush

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
pd.set_option('display.max_rows', 500)


# In[ ]:


# Features get distance relative to the rushing player for each player on the field starting with home -> away

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# # Functions to generate features
# 
# Like a lot of other kernels already I will do some simple feature generation. In the data for each `PlayId` there is 22 rows. Each row represents a single player. Overall, there are 11 players per team on the field during a play. I pivot the data so there is one row for each `PlayId`. I only generated one new features `Force` which is the players weight in ibs. and the acceleration in $\dfrac{yards}{second ^ 2}$. Not the "correct" definition of force but I'm using that here.
# 
# I also just convert categorical variables to an integer label. From what I have seen so far it will be important to have a nice function or class that can handle all of the feature building because you can only get access to the testing data in slices.

# In[ ]:


def generate_category_labels(x):
    unique_categories = x.unique().tolist() + ['unknown']  # For this competition allow for this in test data
    label = range(len(unique_categories))
    mapping = dict(list(zip(unique_categories, label)))
    
    return x.map(mapping)

def build_features(df, college_names_map=None, display_names_map=None, position_names_map=None):
    
    # Add new features
    df['Force'] = df['PlayerWeight'] * df['A']  # F = ma (mass is players weight in ibs and acceleration is measure in yards/sec ** 2)
    
    # Have relative position so do not need the cumcount
    df = df.sort_values(['GameId', 'PlayId']).reset_index(drop=True)
    df['row_number'] = df.groupby(['GameId', 'PlayId']).cumcount() + 1
    
    # Split player height
    height = df['PlayerHeight'].str.split('-', n = 1, expand = True)
    df['PlayerHeightFeet'] = height[0].astype(int)
    df['PlayerHeightInches'] = height[1].astype(int)

    # Split player birthdate
    birth_date = df['PlayerBirthDate'].str.split('/', n = 2, expand = True)
    df['PlayerDOBMonth'] = birth_date[0].astype(int)
    df['PlayerDOBDay'] = birth_date[1].astype(int)
    df['PlayerDOBYear'] = birth_date[2].astype(int)

    if college_names_map is None:
        college_names = df['PlayerCollegeName'].unique().tolist() + ['unknown']
        college_names_map = dict(list(zip(college_names, range(len(college_names)))))
    df['PlayerCollegeName'] = df['PlayerCollegeName'].map(college_names_map)

    if display_names_map is None:
        display_names = df['DisplayName'].unique().tolist() + ['unknown']
        display_names_map = dict(list(zip(display_names, range(len(display_names)))))
    df['DisplayName'] = df['DisplayName'].map(display_names_map)

    if position_names_map is None:
        position_names = df['Position'].unique().tolist() + ['unknown']
        position_names_map = dict(list(zip(position_names, range(len(position_names)))))
    df['Position'] = df['Position'].map(position_names_map)
    
    # Unstack individual player features
    values = ['X', 'Y', 'S', 'A', 'Dis', 'Orientation', 'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'PlayerHeightFeet', 'PlayerHeightInches',
              'PlayerWeight', 'PlayerDOBMonth', 'PlayerDOBDay', 'PlayerDOBYear', 'PlayerCollegeName', 'Position', 'Force']

    group_df = df.pivot_table(index=['GameId', 'PlayId'], columns=['row_number'], values=values).reset_index()

    columns = []
    for c in group_df.columns:
        col_name = str(c[0]) + str(c[1])
        columns.append(col_name)

    group_df.columns = columns
    
    # Get play id specific features
    if 'Yards' in df.columns:
        play_features = ['Team', 'Season', 'YardLine', 'Quarter', 'GameClock', 'Down', 'Distance', 'PossessionTeam', 'FieldPosition',
                         'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
                         'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff', 'TimeSnap', 'HomeTeamAbbr',
                         'VisitorTeamAbbr', 'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather', 'Temperature',
                         'Humidity', 'WindSpeed', 'WindDirection', 'Yards']
    else:
        play_features = ['Team', 'Season', 'YardLine', 'Quarter', 'GameClock', 'Down', 'Distance', 'PossessionTeam', 'FieldPosition',
                         'HomeScoreBeforePlay', 'VisitorScoreBeforePlay', 'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
                         'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff', 'TimeSnap', 'HomeTeamAbbr',
                         'VisitorTeamAbbr', 'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather', 'Temperature',
                         'Humidity', 'WindSpeed', 'WindDirection']

    play_id_feature_df = df.groupby(['PlayId'])[play_features].first()
    simple_df = pd.merge(group_df, play_id_feature_df, on=['PlayId'])
    
    # Convert categorical variables
    categorical_variables = ['Team', 'PossessionTeam', 'FieldPosition', 'OffenseFormation', 'OffensePersonnel',
                             'DefensePersonnel', 'PlayDirection', 'HomeTeamAbbr', 'VisitorTeamAbbr', 'Stadium', 'Location',
                             'StadiumType', 'Turf', 'GameWeather', 'WindDirection']
    
    for column in categorical_variables:
        simple_df[column] = generate_category_labels(simple_df[column])
        
    return simple_df, college_names_map, display_names_map, position_names_map


# In[ ]:


# Build features and label maps for training
# Label maps well be applied to test data
simple_df, college_names_map, display_names_map, position_names_map = build_features(train_df)


# # Building Classes
# 
# By examining the data we know two things:
# 
# 1. There is not a yardage gain for all of the -99 to 99 yards that the metric needs to be computed
# 2. There will be some class imbalance
# 
# This means that we will need to be creative in the way that we model this problem as a multiclass classification problem. Below is a plot of the `Yards` which is what we wnat to predict. In the NFL positive yardage between 1 and 10 is very common and even negative yardage between -5 and -1 is also fairly common. For the purposes of this kernel I will build classes for -3 to 11 yards and a class for all yards less than -3 and a class for all yards greater than 11. 
# 
# This is somewhat aribitrary and based off my knowledge of the game. You can experiment with this to see what works.
# 
# Below the plot is a function to create the classes and targets. It is by no means optimized and can certainly use some work but it works all the way through to the end of the kernel.

# In[ ]:


simple_df['Yards'].value_counts().plot(kind='bar', figsize=(20, 10))


# In[ ]:


# Building the targets
# Want to do as a multiclass classification problem

target = simple_df['Yards']
actuals = simple_df['Yards']

start = -3
end = 11

def create_target_bins(x, start=-3, end=11):
    if x in set(range(start, end + 1)):
        return f'Yards{x}'
    elif x < start:
        return f'Yards_less_than_{start}'
    elif x > end:
        return f'Yards_greater_than_{end}'

def create_class_names(target, start=-3, end=11):

    bin0 = [f'Yards_less_than_{start}']
    bins = [f'Yards{i}' for i in range(start, end + 1)]
    bin_last = [f'Yards_greater_than_{end}']

    classes = bin0 + bins + bin_last
    target_classes = target.apply(create_target_bins)
    return classes, target_classes

classes, target_classes = create_class_names(target, start=start, end=end)

map_ = dict(list(zip(classes, range(len(classes)))))
inverse_map_ = dict(list(zip(range(len(classes)), classes)))

target = target_classes.map(map_)
target.unique()


# In[ ]:


objectives = pd.DataFrame(list(zip(target, actuals)), columns=['target', 'Yards'])


# # Generate Train/Test set
# 
# Like any other machine learning problem we want to be able to evaluate our model performance. Here I use a simple train/test split by calling one iteration of `KFold`. Theoretically, what happend in the past will should not have any impact on the next play in the NFL so currently I choose `KFold`. In the future I will be trying a time series validation strategy as well.
# 
# Also for now I am dropping a few columns that I just did not have the time to deal with properly yet.

# In[ ]:


# We can do two types of validation KFold (because a previous play should not affect the current play or time series where we hold out each teams most recent game)
# This is what we will be predicting for the LB.

group_id = simple_df['PlayId']
train_inds, test_inds = next(KFold(n_splits=10, random_state = 2019).split(simple_df))

X_train = simple_df.iloc[train_inds]
X_test = simple_df.iloc[test_inds]

y_train = objectives.iloc[train_inds]['target']
y_test = objectives.iloc[test_inds]['target']

y_test_actual = objectives.iloc[test_inds]['Yards'].values

test_ids = X_test['PlayId'].values
X_train.drop(['PlayId', 'GameId', 'Yards', 'GameClock', 'TimeSnap', 'TimeHandoff', 'WindSpeed'], axis=1, inplace=True)
X_test.drop(['PlayId', 'GameId', 'Yards', 'GameClock', 'TimeSnap', 'TimeHandoff', 'WindSpeed'], axis=1, inplace=True)


# In[ ]:


# For this competition I want to add focal loss as the loss metric with the imbalanced multiclass classification I think this well help
# If we want to mimic counts then from the test counts should be computed for each fold as well as aggregate metrics

params = {
          "objective" : "multiclass",
          "num_class" : len(classes),
          "num_leaves" : 2 ** 8,
          "max_depth": 5,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 2,        # subsample_freq
          "verbosity" : -1
}

lgb_train, lgb_valid = lgb.Dataset(X_train, y_train.values), lgb.Dataset(X_test, y_test.values)
model = lgb.train(params, lgb_train, 1000, valid_sets=[lgb_train, lgb_valid], verbose_eval=5, early_stopping_rounds=10)


# In[ ]:


lgb.plot_importance(model, max_num_features=25)


# # Building the predictions
# 
# This is the part that took me some time to think about. What the metric is asking for is a cumulative probability distribution $P(Y \leq n)$. If we model this as a multiclass classification we have the probabilities we just have to take the cumulative sum of each row. The interesting thing here for me is what to do with the binned classes?
# 
# My idea was to take the probability for the binned classes and divide that probability by the number of steps between -99 and -3 for the class where yards $\le -3$ and the number of steps between 11 and 99 for the class where yards $\ge 11$.
# 
# In the function I also clip the probabilities because I got an error that the probabilities were not in $(0, 1]$ even though they were. I think it is just float precision computations.

# In[ ]:


def build_cdf_predictions(X, model, classes, start=-3, end=11):
    predictions = model.predict(X, num_iteration=model.best_iteration)
    predictions = pd.DataFrame(predictions)
    
    cdf_predictions = np.zeros((predictions.shape[0], 199))
    actual_classes = [f'Yards{i}' for i in range(-99, 100)]
    cdf_predictions = pd.DataFrame(cdf_predictions, columns=actual_classes)
    predictions.columns = classes

    begin_columns = [f'Yards{i}' for i in range(-99, start)]
    cdf_predictions.loc[:, begin_columns] = np.tile(predictions.iloc[:, 0].values, (len(begin_columns), 1)).T / len(begin_columns)

    columns = [f'Yards{i}' for i in range(start, end + 1)]
    cdf_predictions.loc[:, columns] = predictions.loc[:, columns].values

    end_columns = [f'Yards{i}' for i in range(end + 1, 100)]
    cdf_predictions.loc[:, end_columns] = np.tile(predictions.iloc[:, -1].values, (len(end_columns), 1)).T / len(end_columns)
    
    predictions = pd.DataFrame(cdf_predictions).cumsum(axis=1)
    return predictions.clip(0.0, 1.0)


# In[ ]:


predictions = build_cdf_predictions(X_test, model, classes, start, end)


# # Get the test score
# 
# Finally, we want to get the test score from our model. Here it is `0.1704`.

# In[ ]:


def crps_score(y_true, y_pred):
    
    N = len(y_true)
    n = np.asarray([i for i in range(-99, 100)])
    n = np.tile(n, (len(y_true), 1))
    y_true = np.tile(y_test.values, (n.shape[1], 1)).T
    
    heavyside_calculation = ((n - y_true) >= 0).astype(int)
    score = ((y_pred - heavyside_calculation) ** 2).sum(axis=1)
    return score.sum() / (199 * N)

crps_score(y_test_actual, predictions)


# # Making predictions
# 
# In order to make predictions we have to call the special python module `nflrush` for this competition. There seem to be a lot of rules around this API. For example, you can only call it once in a notebook session. If you want to be able to call it again you have to restart your notebook session.
# 
# Also, you can only get the next iteration of the test data AFTER you have made a prediction on the current `PlayId`. We have to iterate though and build our features on each `PlayId` frame and then make the prediction. Once, we iterate through each test_df frame we can submit the predictions from `env.predict` and when it is completely finished we make an output submission with `env.write_submission`.
# 
# Again, this kernel is still in testing phase and the speed of the prediction process is not optimzed in any way.

# In[ ]:


env = nflrush.make_env()
iter_test = env.iter_test()


# In[ ]:


# Build test data frame and examine
test_df = []
for test_data, _ in tqdm(iter_test):
    test_data, _, _, _ = build_features(test_data, college_names_map, display_names_map, position_names_map)
    test_data.drop(['PlayId', 'GameId', 'GameClock', 'TimeSnap', 'TimeHandoff', 'WindSpeed'], axis=1, inplace=True)
    predictions = build_cdf_predictions(test_data, model, classes, start, end)
    print(predictions)
    env.predict(predictions)
env.write_submission_file()


# In[ ]:




