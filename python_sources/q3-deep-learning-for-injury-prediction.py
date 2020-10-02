#!/usr/bin/env python
# coding: utf-8

# # Using Deep Learning on Movement Data for Injury Prediction
# Now that we have an understanding of what variables correlate with acute injuries amongst NFL players, we investigate how repeated movement patterns can lead to physical injury.
# 
# Given just historical movement and play data of a player over the course of several seasons, **our method is able to predict with almost 70% accuracy whether or not that player was injured over the course of those seasons.**
# 
# Our pipeline can be broken down into three stages:
# 1. Pre-processing: loads, cleans and stores the data that will be used in our model
# 2. Timeseries feature extraction: extracts relevant data from our cleaned timeseries data
# 3. Model training: uses features extracted in the previous step to train a deep learning model

# ## Step 1: Preprocessing
# 
# ### Loading data
# We begin by reading in our data files and transforming them into a format that we can process

# In[ ]:


import pandas as pd
from tqdm import tqdm


# In[ ]:


# load in our data
tracking_df = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
injury_df = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")
play_df = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")


# In[ ]:


players = play_df.PlayerKey.unique()


# In[ ]:


tracking_df.head()


# In[ ]:


play_df.head()


# ### Data Cleaning
# First we take just a subset of our columns to avoid training our model on redundant features.

# In[ ]:


cat_names = ['StadiumType', 'FieldType', 'Weather', 'PlayType', 'PositionGroup', 'RosterPosition']

model_data_df = play_df[cat_names + ['PlayerKey', 'PlayKey']]
model_data_df.head()


# Next we transform columns representing categorical data into numpy's categorical data type. This allows us to treet the column labels as numbers when building deep learning models. This is particularly useful for embeddings.

# In[ ]:


# convert all categories to integers and normalize such that the minimum value is 0
from fastai.tabular.transform import Categorify

tfm = Categorify(cat_names, [])
tfm(model_data_df)
for cat_name in cat_names:
    codes = model_data_df[cat_name].cat.codes
    model_data_df[cat_name] = codes - min(codes)
model_data_df.head()


# ### Exposing data by player
# Here we define helper functions that make it easy to access all of the information relevant to a particular player

# In[ ]:


def get_plays_for_player(player):
    """
    Returns data for all of the plays associated with the player
    """
    return model_data_df[model_data_df.PlayerKey == player].drop(['PlayerKey'], axis=1)


# `tracking_df` is massive and incredibly slow to access. To speed up our calculations we cache movement data associated with every player to disk

# In[ ]:


import os
PLAYER_DATA_CACHE_PATH = "../input/tracking-cache/player_data_tracking"

# makes sure the local directory for the cache exists
if not os.path.exists(PLAYER_DATA_CACHE_PATH):
    os.makedirs(PLAYER_DATA_CACHE_PATH)

def get_tracking_for_player(player):
    """
    Returns tracking data stored in a local json file. If the file
    doesn't exist the function creates it
    """
    file_cache_path = f"{PLAYER_DATA_CACHE_PATH}/{player}.csv"
    if not os.path.exists(file_cache_path):
        small_tracking = tracking_df[tracking_df.PlayKey.str.startswith(str(player))]
        small_tracking.to_csv(file_cache_path)
    return pd.read_csv(file_cache_path)


# In[ ]:


get_tracking_for_player(players[0]).head()


# In[ ]:


def get_data_for_player(player):
    """
    Combines all of the data associated with the plays for an individual player
    """
    plays = get_plays_for_player(player)
    tracking = get_tracking_for_player(player)
    
    group = tracking.groupby("PlayKey")
    avg_speed = group.mean()['s'] * 100
    total_distance = group.sum()['dis']
    
    data = plays.merge(avg_speed, on="PlayKey").merge(total_distance, on="PlayKey").drop('PlayKey', axis=1)
    data['id'] = player
    data['time'] = data.index
    
    return data

get_data_for_player(players[1])


# Iterate over all of the players in our dataset and populate the local cache

# ## Step 2: Timeseries Feature Extraction
# We use [TSFresh](https://tsfresh.readthedocs.io/en/latest/) to automatically extract relevant features from our dataset. This converts our 3-dimensional data to 2-dimensional data.
# 
# We begin by converting our data into the format needed by TSFResh.

# In[ ]:


fresh_data = pd.concat([get_data_for_player(player) for player in tqdm(players)])
fresh_data


# In[ ]:


from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import MinimalFCParameters


# We approach the problem as one of binary classification, labeling players that were injured with "1" and players that were not injured with "0".

# In[ ]:


injured_players = set(injury_df.PlayerKey)
y = pd.DataFrame(index=players)
y['target'] = [int(player in injured_players) for player in players]
y.head()


# We use the `extract_relevant_features` method provided by the `TSFresh` library to automatically extract hundreds of features from our timeseries data. `extract_relevant_features` automatically runs correlation tests to determine which features are relevant and retains only the important features.

# In[ ]:


extracted_features = extract_relevant_features(
    fresh_data,
    y.target,
    column_id="id",
    column_sort="time",
    default_fc_parameters=MinimalFCParameters()
)
extracted_features['target'] = y.target
extracted_features.head()


# # Part 3: Model Training
# We use tabular deep learning to build a binary classifier. Our model is built on top of [FastAI](https://www.fast.ai/)'s tabular deep learning architecture.

# In[ ]:


from fastai.tabular import *
from fastai.callbacks import SaveModelCallback
from sklearn.model_selection import train_test_split


# ## Creating our model
# First we create a `DataBunch`, a concept in the fastai library that acts as a wrapper around your dataset. We choose a random subset of our players to be the validation set

# In[ ]:


_, valid = train_test_split(range(len(players)))
data = TabularDataBunch.from_df("./models", extracted_features, 'target', valid_idx=valid)


# In[ ]:


# this choice of layer sizes comes from experimental success in similar tabular deep learning problems
LAYERS = [200, 100]

def create_learner():
    """
    We define this as a function as it allows us to later load models from disk easily
    later in the notebook
    """
    return tabular_learner(data, layers=LAYERS, metrics=accuracy)


# In[ ]:


learn = create_learner()


# We follow [Leslie Lamport's technique](https://arxiv.org/pdf/1506.01186.pdf) for learning rate finding to determine the optimal hyperparamters.
# 
# We follow common practice in the data science community of:
# 1. Finding the lowest point in the graph below
# 2. Dividing that value by 10
# 3. Using that value as our learning rate
# In the graph below we see that the minimum occurs around 1e-1, so we set our learning rate to 1e-2.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


lr = 1e-2
N_CYCLES = 10


# In[ ]:


# clear the existing model cache
get_ipython().system('rm -rf ./models/models')


# ## Training our model

# In[ ]:


# train our model and save the best result
learn.fit_one_cycle(
    N_CYCLES,
    lr,
    callbacks=[SaveModelCallback(learn, every="improvement", monitor="accuracy")]
)


# We get our best result on epoch 6 with a model that achieves 70% accuracy.

# In[ ]:


best_model = create_learner()
best_model.load("bestmodel")


# ## Analyzing the results
# Looking at our confusion matrix we come to the following conclusions:
# * Our model has very few false positives
# * Our model is much better at predicting when a player will not get injured
# * We need a lot more data before we can definitively say that this appraoch effectively predicts injury likelihoods

# In[ ]:


interpretation = best_model.interpret()
interpretation.plot_confusion_matrix()


# In[ ]:




