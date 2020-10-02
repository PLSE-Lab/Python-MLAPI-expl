#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import sqlite3
import math
import matplotlib.pyplot as plt
import datetime as dt # to handle dates
import time # for measuring training time
import torch
from copy import deepcopy
import random

np.set_printoptions(threshold=np.inf)
pd.set_option('max_columns',1000) 
pd.set_option('max_row',300)

conn = sqlite3.connect('../input/database.sqlite') # the database

TEST_DATA_SIZE = 200
VALIDATION_DATA_SIZE = 200
training_cut_off = TEST_DATA_SIZE + VALIDATION_DATA_SIZE


def sql_query_2_dataframe(query):
        return pd.read_sql_query(query, conn)

def get_team_attributes(matches_df, team_attribute_df):
    # get all home team attributes
    team_attributes_home = pd.merge(matches_df, team_attribute_df, how='inner', left_on="home_team_api_id", right_on="team_api_id",
                            suffixes=('', '_home_team'))
    team_attributes_home.rename(columns={'date': 'date_match'}, inplace=True)
    team_attributes_home.rename(columns={'id': 'id_match'}, inplace=True)

    # get most recent team attributes
    team_attributes_home = team_attributes_home[(team_attributes_home['date_home_team'] <= team_attributes_home['date_match'])]
    idx = team_attributes_home.groupby(['id_match'])['date_home_team'].transform(max) == team_attributes_home['date_home_team']
    team_attributes_home = team_attributes_home[idx]

    columns_to_drop = list(filter(lambda cln: False if (cln == 'id') else True, matches_df.columns.values))
    columns_to_drop = list(map(lambda cln: cln + '_match' if (cln == 'date') else cln, columns_to_drop))
    team_attributes_home = team_attributes_home.drop(columns_to_drop + ['date_home_team'], 1)

    # get all away team attributes
    team_attributes_away = pd.merge(matches_df, team_attribute_df, how='inner', left_on="away_team_api_id",
                                 right_on="team_api_id", suffixes=('', '_away_team'))
    team_attributes_away.rename(columns={'date': 'date_match'}, inplace=True)
    team_attributes_away.rename(columns={'id': 'id_match'}, inplace=True)

    # get most recent team attributes
    team_attributes_away = team_attributes_away[(team_attributes_away['date_away_team'] <= team_attributes_away['date_match'])]
    idx2 = team_attributes_away.groupby(['id_match'])['date_away_team'].transform(max)            == team_attributes_away['date_away_team']
    team_attributes_away = team_attributes_away[idx2]
    team_attributes_away = team_attributes_away.drop(columns_to_drop + ['date_away_team'], 1)

    return (team_attributes_home, team_attributes_away)


def get_player_attributes(matches_df, player_and_attribute_df, is_home, player_id):
    matches_df.rename(columns={'date': 'date_match'}, inplace=True)
    matches_df.rename(columns={'id': 'id_match'}, inplace=True)

    right_cond = 'home_player_' + str(player_id) if is_home else 'away_player_' + str(player_id)
    matches_with_player = pd.merge(matches_df, player_and_attribute_df, how='inner', left_on=right_cond, right_on="player_api_id")

    # Filter out all ratings with dates greater than match date; sfxs[0] is appended because the identifier depends on it
    matches_with_past_ratings = matches_with_player[(matches_with_player['date'] <= matches_with_player['date_match'])]
    result = matches_with_past_ratings.sort_values('date', ascending=False).drop_duplicates(['id_match'])

    columns_to_drop = list(filter(lambda cln: False if (cln == 'id_match') else True, matches_df.columns.values))
    columns_to_drop = list(map(lambda cln: cln + '_match' if (cln == 'date') else cln, columns_to_drop))
    result = result.drop(columns_to_drop + ['player_api_id'], 1)

    return result


def preprocess_player_data(matches):
    # for each row caculate match's result and store it
    matches['resultsOfMatch'] = matches.apply(lambda row : determine_result(row['home_team_goal'], row['away_team_goal']),         axis = 1)
    matches = onehot('resultsOfMatch', matches)
        
    # Maps for ordinary features
    high_med_low_map = {'high':0.9, 'medium':0.5, 'low':0.1}
    lots_normal_little_map = {'Lots':0.9, 'Normal':0.5, 'Little':0.1}
    short_mixed_long_map = {'Short':0.9, 'Mixed':0.5, 'Long':0.1}
    safe_normal_risky_map = {'Safe':0.9, 'Normal':0.5, 'Risky':0.1}

    # drop Players' name and caculate age of players
    for i in range(1,12):
        hpb =  'birthday_hp_' + str(i)
        apb =  'birthday_ap_' + str(i)
       
        # create columns for home- and away player i
        matches['age_hp_' + str(i)] = matches.apply(lambda row : calculate_age(row['date_match'], row[hpb]), axis = 1)
        matches['age_ap_' + str(i)] = matches.apply(lambda row : calculate_age(row['date_match'], row[apb]), axis = 1)
        
        # one hot encoding and mapping of players' data
        pf_ap = 'preferred_foot_ap_' + str(i) # one-hot
        pf_hp = 'preferred_foot_hp_' + str(i) # one-hot
        
        awr_ap = 'attacking_work_rate_ap_' + str(i) # ordinary
        dwr_ap = 'defensive_work_rate_ap_' + str(i) # ordinary
        awr_hp = 'attacking_work_rate_hp_' + str(i) # ordinary
        dwr_hp = 'defensive_work_rate_hp_' + str(i) # ordinary
        
        matches = onehot(pf_ap, matches)
        matches = onehot(pf_hp, matches)
        
        matches[awr_ap] = mapping(matches[awr_ap], high_med_low_map)
        matches[awr_hp] = mapping(matches[awr_hp], high_med_low_map)
        matches[dwr_ap] = mapping(matches[dwr_ap], high_med_low_map)
        matches[dwr_hp] = mapping(matches[dwr_hp], high_med_low_map)
        
        columns_to_drop = [hpb, apb, 'date_ap_' + str(i), 'date_hp_' + str(i), 'player_fifa_api_id_ap_' + str(i), 'player_fifa_api_id_hp_' + str(i), 'id_ap_' + str(i), 'id_hp_' + str(i)]

        matches = matches.drop(columns_to_drop, 1)
    
    columns_to_drop = ['date_match','id_match',
                     'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal']
    matches = matches.drop(columns_to_drop, 1)
        
    return matches

# Normalize a whole dataframe with min-max-normalization
def normalization(df):
    for column in df:
        max = df[column].max()
        min = df[column].min()
        if (max == min):
            df[column] = 1
        else:
            df[column] = df[column].map(lambda cell: (cell - min)/(max - min))

# Mapping for ordinary attributes
def mapping(df_column,mapping):
    df_column = df_column.map(mapping)
    return df_column

# One hot encode a column of a dataframe
def onehot(column, dataframe):
    onehot_df = pd.get_dummies(dataframe[column], prefix = column)
    dropped_df = dataframe.drop(column, 1)
    return pd.concat([dropped_df, onehot_df], axis = 1)

# Determine the result of a match from the goals
def determine_result(home_goals, away_goals):
    if (home_goals > away_goals):
        return 'hw' # home win
    elif (home_goals < away_goals):
        return 'aw' # away win
    else:
        return 'draw'

# Calculate age for a player in years. Both input dates are string
def calculate_age(match_date, birthday):
    date_format = '%Y-%m-%d %H:%M:%S' # e.g. 2008-08-17 00:00:00
    match_date_2_date = dt.datetime.strptime(match_date, date_format).date()
    birthday_2_date = dt.datetime.strptime(birthday, date_format).date()
    age_in_days = (match_date_2_date - birthday_2_date).days
    age_in_years = age_in_days / 365.25
    return math.floor(age_in_years)



def convert_one_hot_to_index(outputs):
    result = np.zeros(shape=(len(outputs)), dtype=int)
    i = 0
    for index, row in outputs.iterrows():
        if (row['resultsOfMatch_hw']) > 0.1:
            result[i] = 0
        if (row['resultsOfMatch_aw']) > 0.1:
            result[i] = 1
        if (row['resultsOfMatch_draw']) > 0.1:
            result[i] = 2
        i = i + 1
    return result



# Some cells in the Soccer database are NULL
def replace_nan_with_zero(inputs):
    nan_indices = torch.nonzero(torch.isnan(inputs))
    for row, col in nan_indices:
        inputs[row][col] = 0


def merge_attributes(main_df, attributes, suffix):

    attributes = attributes.add_suffix(suffix)
    attributes = attributes.rename(columns=lambda cln: cln[:-len(suffix)] if cln.endswith(suffix + suffix) else cln)
    attributes = attributes.rename(columns={'id_match' + suffix: 'id_match'})
    return main_df.merge(attributes, how='inner', on='id_match')

def merge_all_data(matches_df, team_attributes_home, team_attributes_away, player_attributes_home, player_attributes_away):
    
    matches_with_players = matches_df

    results = []
    for h in range(1):
        temp_matches_with_players = deepcopy(matches_with_players)
        indices = list(range(1,12))
        random.shuffle(indices)

        for i in indices:
            temp_matches_with_players = merge_attributes(temp_matches_with_players, player_attributes_home[i-1], '_hp_' + str(i))

        for i in indices:
            # add suffixes and clean up columns
            temp_matches_with_players = merge_attributes(temp_matches_with_players, player_attributes_away[i-1], '_ap_' + str(i))

        results.append(temp_matches_with_players)

        
    result = pd.concat(results, sort=True)
    
    columns_to_drop = filter(lambda cln: False if cln == "home_team_goal" or cln == "away_team_goal" or "date_match" else True, matches_df.columns.values)
    result = result.drop(columns_to_drop, 1)
    
    print(len(result.columns))
    
    return result
    


def get_data():
    print("querying the database.", end="")
    # Select all matches from the English league
    matches_df = sql_query_2_dataframe("SELECT * FROM MATCH WHERE Match.country_id in (SELECT id from Country WHERE name = 'England');")

    # let's drop player positions because they will not be true when we shuffle
    player_positions = []
    for i in range(1,12):
        player_positions = player_positions + ['home_player_X' + str(i)]
        player_positions = player_positions + ['away_player_X' + str(i)]
        player_positions = player_positions + ['home_player_Y' + str(i)]
        player_positions = player_positions + ['away_player_Y' + str(i)]


    matches_df = matches_df.drop(['B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD','IWA','LBH','LBD',
                                  'LBA','PSH','PSD','PSA', 'WHH','WHD','WHA','SJH','SJD','SJA','VCH','VCD',
                                  'VCA','GBH','GBD','GBA','BSH','BSD','BSA', 'country_id', 'league_id','goal',
                                  'shoton','shotoff','foulcommit','card','cross','corner','possession',
                                  'season', 'stage', 'match_api_id'] + player_positions, 1)

    # Join all players with their attributes
    player_and_attribute_df = sql_query_2_dataframe("SELECT Player.birthday, Player.height, Player.weight, Player_Attributes.* FROM Player JOIN Player_Attributes ON Player.player_api_id = Player_Attributes.player_api_id;") # England Players + Attributes
    player_attribute_df = player_and_attribute_df.drop(['player_fifa_api_id', 'id'], axis = 1)

    team_attribute_df = sql_query_2_dataframe("SELECT * FROM Team_Attributes;")
    team_attribute_df = team_attribute_df.drop(['team_fifa_api_id', 'id'], axis = 1)
    print(u"\u2713")

    print("getting team attributes dataframes.", end="")
    (team_attributes_home, team_attributes_away) = get_team_attributes(matches_df, team_attribute_df)
    print(u"\u2713")

    print("getting player attributes dataframes.", end="")
    player_attributes_home = [get_player_attributes(matches_df, player_and_attribute_df, True, i) for i in range(1,12)]
    player_attributes_away = [get_player_attributes(matches_df, player_and_attribute_df, False, i) for i in range(1,12)]
    print(u"\u2713")

    print("slicing out training data.", end="")
    # get the training dataframes for data augmentation
    matches_df_training = matches_df[:-training_cut_off]

    team_attributes_home_training = team_attributes_home[:-training_cut_off]
    team_attributes_away_training = team_attributes_away[:-training_cut_off]

    player_attributes_home_training = list(map(lambda df: df[:-training_cut_off], player_attributes_home))
    player_attributes_away_training = list(map(lambda df: df[:-training_cut_off], player_attributes_away))
    print(u"\u2713")

    # TODO: Trennung von den verschiedenen Daten hier ausgeben, sonst geht es nicht

    return merge_all_data(matches_df, team_attributes_home, team_attributes_away, player_attributes_home, player_attributes_away)




def get_preprocessed_data():
    matches_with_players = get_data()

    print("preprocessing the data", end="")
    preprocessed = preprocess_player_data(matches_with_players)
    normalization(preprocessed)
    print(u"\u2713")


    print(len(preprocessed))
    print(len(preprocessed.columns.values))

    return preprocessed

preprocessed = get_preprocessed_data()


# In[ ]:


# preprocessed.columns.values

for i in range(1, 12):
    preprocessed = preprocessed.drop('home_player_' + str(i), 1)
    preprocessed = preprocessed.drop('away_player_' + str(i), 1)
    


# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

BATCH_SIZE = 1700
stoppage_count = 0
preprocessed = shuffle(preprocessed)



# For cross validation while training; this implements early stopping
def cross_validate(validation_error):    
    if len(validation_error) >= 2:
        if (validation_error[-1] > validation_error[-2]):
            global stoppage_count
            stoppage_count = stoppage_count + 1
            
            if stoppage_count >= 5:
                stoppage_count = 0    
                return True
    
    return False


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 24, 42)
        self.pool = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(24 * 220, 3)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 24 * 220)
        x = self.fc1(x)
        return x

def main():
    
    
    # Use GPU for speedup on training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    inputs = preprocessed.drop(['resultsOfMatch_hw', 'resultsOfMatch_aw', 'resultsOfMatch_draw'], 1)[:-training_cut_off]

    outputs = preprocessed[['resultsOfMatch_hw', 'resultsOfMatch_aw', 'resultsOfMatch_draw']][:-training_cut_off]

    validation_start = len(preprocessed) - training_cut_off
    validation_end = len(preprocessed) - TEST_DATA_SIZE

    inputs_validation = preprocessed.drop(['resultsOfMatch_hw', 'resultsOfMatch_aw', 'resultsOfMatch_draw'], 1)[validation_start:validation_end]
    labels_validation = preprocessed[['resultsOfMatch_hw', 'resultsOfMatch_aw', 'resultsOfMatch_draw']][validation_start:validation_end]

    outputs = convert_one_hot_to_index(outputs)
    labels_validation = convert_one_hot_to_index(labels_validation)


    # Create Tensors to hold inputs and outputs (outputs are labels)
    x = torch.from_numpy(inputs.values.astype(np.float32)).to(device)
    y = torch.from_numpy(outputs).to(device)
    

    x_validation = torch.from_numpy(inputs_validation.values.astype(np.float32)).to(device)
    y_validation = torch.from_numpy(labels_validation).to(device)
    

    # There are some nan values in the input and they should be replaced by 0
    replace_nan_with_zero(x)
    replace_nan_with_zero(x_validation)
    
    
    # processing stuff
    x = x.unsqueeze(1)
    x_validation = x_validation.unsqueeze(1)
    
    print(x.size())
    print(x_validation.size())
    
    model = Model()
    model.to(device)

    # We use Cross-Entropy error because we are dealing with a classification task.
    loss_fn = torch.nn.CrossEntropyLoss()


    # Print some information and capture time to measure training duration
    training_start = time.time()
    print("Starting training on " + str(device) + "... ")

    learning_rate = 1e-4 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_report = {}
    validation_error = []
    epoch_step = 0
    
    
    for t in range(500):
        for idx in range(0, len(x), BATCH_SIZE):
            x_train = x[idx:idx+BATCH_SIZE]
            y_train = y[idx:idx+BATCH_SIZE]
            
            
            epoch_step = t
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = model(x_train)


            # Compute loss.
            loss = loss_fn(y_pred, y_train)
            loss_report[t] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check accurracy on validation data (cross validation)
        with torch.no_grad():
            validation_outputs = model(x_validation)
            
            
            validation_loss = loss_fn(validation_outputs, y_validation)
            validation_error.append(validation_loss.item())
            
            
            if cross_validate(validation_error) and t >= 75:
                break # if the validation error rises, we stop training

    print('Training finished after ' + f"{(time.time() - training_start):.1f}" + ' seconds.' +       ' Stopped after ' + str(epoch_step) + ' epochs.')

    plt.title('Loss')
    plt.plot(loss_report.keys(), loss_report.values())
    plt.plot(loss_report.keys(), validation_error)
    plt.legend(['Training Cross Entropy Error', 'Validation Error'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Error')

    inputs_test = preprocessed.drop(['resultsOfMatch_hw', 'resultsOfMatch_aw', 'resultsOfMatch_draw'], 1)[-TEST_DATA_SIZE:]
    outputs_test = preprocessed[['resultsOfMatch_hw', 'resultsOfMatch_aw', 'resultsOfMatch_draw']][-TEST_DATA_SIZE:]

    outputs_test = convert_one_hot_to_index(outputs_test)

    x = torch.from_numpy(inputs_test.values.astype(np.float32)).to(device)
    x = x.unsqueeze(1)
    y = torch.from_numpy(outputs_test).to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).double().sum().item()

    print('Accuracy of the network on the test football matches: %d %%' % (100 * correct / total))
    
    classes = ('home_win', 'away_win', 'draw')
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == y).squeeze()
            for i in range(len(y)):
                label = y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print(class_total)
                
    for i in range(3):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

        
if __name__ == '__main__':
    main()


# In[ ]:




