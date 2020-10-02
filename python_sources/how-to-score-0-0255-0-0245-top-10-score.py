#!/usr/bin/env python
# coding: utf-8

# # How to Score (0.0255 - 0.0245)

# Before I start explaining the code, I have to say that the most of this code was taken from [[This kernel](https://www.kaggle.com/anycode/simple-nn-baseline-3)] by [anycode](https://www.kaggle.com/anycode) Kaggler

# **Fisrt :** Let's load the needed dependencies :

# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt  

from timeit import default_timer as timer
from sklearn import preprocessing

get_ipython().system('pip install ultimate')
from ultimate.mlp import MLP 

import gc, sys
gc.enable()


# In[ ]:


def state(message,start = True, time = 0):
    if(start):
        print(f'Working on {message} ... ')
    else :
        print(f'Working on {message} took ({round(time , 3)}) Sec \n')


# In[ ]:


INPUT_DIR = "../input/"


# ** I explained the following code with comments, I hope you understand it well**

# In[ ]:


def feature_engineering(is_train=True):
    # When this function is used for the training data, load train_V2.csv :
    if is_train: 
        print("processing train_V2.csv")
        df = pd.read_csv(INPUT_DIR + 'train_V2.csv')
        
        # Only take the samples with matches that have more than 1 player 
        # there are matches with no players or just one player ( those samples could affect our model badly) 
        df = df[df['maxPlace'] > 1]
    
    # When this function is used for the test data, load test_V2.csv :
    else:
        print("processing test_V2.csv")
        df = pd.read_csv(INPUT_DIR + 'test_V2.csv')
        
    # Make a new feature indecating the total distance a player cut :
    state('totalDistance')
    s = timer()
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    e = timer()
    state('totalDistance', False, e - s)
          

    state('rankPoints')
    s = timer()
    # Process the 'rankPoints' feature by replacing any value of (-1) to be (0) :
    df['rankPoints'] = np.where(df['rankPoints'] <= 0 ,0 , df['rankPoints'])
    e = timer()                                  
    state('rankPoints', False, e-s)
    

    target = 'winPlacePerc'
    # Get a list of the features to be used
    features = list(df.columns)
    
    # Remove some features from the features list :
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchDuration")
    features.remove("matchType")
    
    y = None
    
    # If we are processing the training data, process the target
    # (group the data by the match and the group then take the mean of the target) 
    if is_train: 
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        # Remove the target from the features list :
        features.remove(target)
    
    # Make new features indicating the mean of the features ( grouped by match and group ) :
    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    
    # If we are processing the training data let df_out = the grouped  'matchId' and 'groupId'
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    # If we are processing the test data let df_out = 'matchId' and 'groupId' without grouping 
    else: df_out = df[['matchId','groupId']]
    
    # Merge agg and agg_rank (that we got before) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the max value of the features for each group ( grouped by match )
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the minimum value of the features for each group ( grouped by match )
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    # Put the new features into a rank form ( max value will have the highest rank)
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    # Merge the new (agg and agg_rank) with df_out :
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the number of players in each group ( grouped by match )
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
     
    # Merge the group_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    # Make new features indicating the mean value of each features for each match :
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    
    # Merge the new agg with df_out :
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # Make new features indicating the number of groups in each match :
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    
    # Merge the match_size feature with df_out :
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    # Drop matchId and groupId
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)
    
    # X is the output dataset (without the target) and y is the target :
    X = np.array(df_out, dtype=np.float64)
    
    
    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y


# ### Process and scale the training data  :

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Process the training data :\nx_train, y = feature_engineering(True)\n# Scale the data to be in the range (-1 , 1)\nscaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False).fit(x_train)')


# In[ ]:


print("x_train", x_train.shape, x_train.max(), x_train.min())
scaler.transform(x_train)
print("x_train", x_train.shape, x_train.max(), x_train.min())


# ### Scale the target to be in the range (-1 , 1)

# In[ ]:


y = y * 2 - 1
print("y", y.shape, y.max(), y.min())


# ## Define an MLP model and train it :

# In[ ]:


get_ipython().run_cell_magic('time', '', 'epoch_train = 15\nmlp = MLP(layer_size=[x_train.shape[1], 28, 28, 28, 1], regularization=1, output_shrink=0.1, output_range=[-1,1], loss_type="hardmse")\nmlp.train(x_train, y, verbose=0, iteration_log=20000, rate_init=0.08, rate_decay=0.8, epoch_train=epoch_train, epoch_decay=1)\n\ndel x_train, y\ngc.collect()')


# ## Process the test data and scale it :

# In[ ]:


x_test, _ = feature_engineering(False)
scaler.transform(x_test)
print("x_test", x_test.shape, x_test.max(), x_test.min())
np.clip(x_test, out=x_test, a_min=-1, a_max=1)
print("x_test", x_test.shape, x_test.max(), x_test.min())


# ### Predict the target using the test data : 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'pred = mlp.predict(x_test)\ndel x_test\ngc.collect()')


# ### Reshape the predictions and put them in the right range(0 , 1 )

# In[ ]:


pred = pred.reshape(-1)
pred = (pred + 1) / 2


# In[ ]:


df_test = pd.read_csv(INPUT_DIR + 'test_V2.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print("fix winPlacePerc")\nfor i in range(len(df_test)):\n    winPlacePerc = pred[i]\n    maxPlace = int(df_test.iloc[i][\'maxPlace\'])\n    if maxPlace == 0:\n        winPlacePerc = 0.0\n    elif maxPlace == 1:\n        winPlacePerc = 1.0\n    else:\n        gap = 1.0 / (maxPlace - 1)\n        winPlacePerc = round(winPlacePerc / gap) * gap\n    \n    if winPlacePerc < 0: winPlacePerc = 0.0\n    if winPlacePerc > 1: winPlacePerc = 1.0    \n    pred[i] = winPlacePerc')


# In[ ]:


df_test['winPlacePerc'] = pred


# ## Create the submission file : 

# In[ ]:


submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)


#  Finally, I hope you understand every line of this kernel, also if you have any note don't hesitate to put it in a comment .
