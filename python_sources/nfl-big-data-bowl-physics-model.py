#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install wandb')

wandb login


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gc
import sys

# Any results you write to the current directory are saved as output.

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from  keras.losses import categorical_crossentropy
from keras.regularizers import l2
keras.backend.set_floatx('float32')

from kaggle.competitions import nflrush


# In[ ]:


def error_correcting_codes(df):
    df = df.replace('BLT', 'BAL')
    df = df.replace('HST', 'HOU')
    df = df.replace('ARZ', 'ARI')
    df = df.replace('CLV', 'CLE')
    return df
  

def organize_positions(df):
    return (df.loc[(df['PossessionTeam']==df['HomeTeamAbbr'])&(df['Team']=='away') | (df['PossessionTeam']==df['VisitorTeamAbbr'])&(df['Team']=='home')].copy().reset_index(),
      df.loc[((df['PossessionTeam']==df['HomeTeamAbbr'])&(df['Team']=='home') | (df['PossessionTeam']==df['VisitorTeamAbbr'])&(df['Team']=='away'))&(df['NflId']!=df['NflIdRusher'])].copy().reset_index(),
      df.loc[df['NflId']==df['NflIdRusher']].copy().reset_index())
    

def doubledown(X, doublings=1):
    np.random.seed(3)
    for w in range(doublings):
        X_dupe2 = np.concatenate((X.copy(), X.copy()), axis=0)
        for i in range(X.shape[0]):
            X_dupe2[2*i, :] = X[i, :]
            X_dupe2[2*i+1, :] = X[i, :]
        X = X_dupe2

    return X


def physics_init(df):
    way = -2*(df['PlayDirection']=='left') + 1
    theta = way*df['Dir']*np.pi/180
    df['X'] = (df['PlayDirection']=='right')*df['X'] + (df['PlayDirection']=='left')*(120 - df['X'])
    df['Sx'] = np.sin(theta)*df['S']
    df['Sy'] = np.cos(theta)*df['S']
    df['Ax'] = np.sin(theta)*df['A']
    df['Ay'] = np.cos(theta)*df['A']
    df['EquivYardLine'] = (df['PossessionTeam']==df['FieldPosition'])*(df['YardLine']+10) + (df['PossessionTeam']!=df['FieldPosition'])*(110-df['YardLine'])

    defn, off, RBs = organize_positions(df)

    defn['X'] -= RBs.loc[[i//11 for i in defn.index], 'X'].values
    defn['Y'] -= RBs.loc[[i//11 for i in defn.index], 'Y'].values
    defn = defn.loc[:, ('PlayId', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]
    defn.fillna(0, inplace=True)
    defn['Infl'] = defn['PlayerWeight']/(np.square(defn['X']) + np.square(defn['Y']))**0.5
    defn['AngularMomentum'] = -defn['PlayerWeight']*(defn['X']*defn['Sx'] + defn['Y']*defn['Sy'])/(np.square(defn['X']) + np.square(defn['Y']))

    off['X'] -= RBs.loc[[i//10 for i in off.index], 'X'].values
    off['Y'] -= RBs.loc[[i//10 for i in off.index], 'Y'].values
    off = off.loc[:, ('PlayId', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]
    off.fillna(0, inplace=True)
    off['Infl'] = off['PlayerWeight']/(np.square(off['X']) + np.square(off['Y']))**0.5
    off['AngularMomentum'] = -off['PlayerWeight']*(off['X']*off['Sx'] + off['Y']*off['Sy'])/(np.square(off['X']) + np.square(off['Y']))

    RBs['YardsBehindScrimmage'] = RBs['EquivYardLine'] - RBs['X']
    RBs['X'] = 0
    RBs['Y'] = 0
    RBs = RBs.loc[:, ('PlayId', 'YardsBehindScrimmage', 'PlayerWeight', 'Sx', 'Sy', 'Ax', 'Ay', 'X', 'Y')]
    RBs.fillna(0, inplace=True)
    
    return defn, off, RBs


def action(defn, off, RBs, timestep=0.1):
    t = 0.0
    while t<timestep:
        for X in (defn, off, RBs):
            X['X'] += X['Sx']*0.01 +X['Ax']*0.01**2/2
            X['Y'] += X['Sy']*0.01 +X['Ay']*0.01**2/2
            X['Sx'] += X['Ax']*0.01
            X['Sy'] += X['Ay']*0.01
            X['Ax'] *= 0.99
            X['Ay'] *= 0.99
        t += 0.01

        defn['X'] -= RBs.loc[[i//11 for i in defn.index], 'X'].values
        defn['Y'] -= RBs.loc[[i//11 for i in defn.index], 'Y'].values
        defn['Infl'] = defn['PlayerWeight']/(np.square(defn['X']) + np.square(defn['Y']))**0.5
        defn['AngularMomentum'] = -defn['PlayerWeight']*(defn['X']*defn['Sx'] + defn['Y']*defn['Sy'])/(np.square(defn['X']) + np.square(defn['Y']))

        off['X'] -= RBs.loc[[i//10 for i in off.index], 'X'].values
        off['Y'] -= RBs.loc[[i//10 for i in off.index], 'Y'].values
        off['Infl'] = off['PlayerWeight']/(np.square(off['X']) + np.square(off['Y']))**0.5
        off['AngularMomentum'] = -off['PlayerWeight']*(off['X']*off['Sx'] + off['Y']*off['Sy'])/(np.square(off['X']) + np.square(off['Y']))

        RBs['X'] = 0
        RBs['Y'] = 0

    return defn, off, RBs


def physics_doubledown(X, doublings, width):
    np.random.seed(3)
    for w in range(doublings):
        X_dupe = X.copy()
        X_dupe2 = np.concatenate((X.copy(), X.copy()), axis=0)
        numpy_sucks = np.arange(11)
        np.random.shuffle(numpy_sucks)
        for (i,j) in enumerate(numpy_sucks):
            X_dupe[:, width*i:width*i+width] = X[:, width*j:width*j+width]
        numpy_sucks = np.arange(10)
        np.random.shuffle(numpy_sucks)
        for (i,j) in enumerate(numpy_sucks):
            X_dupe[:, width*(i+11):width*(i+12)] = X[:, width*(j+11):width*(j+12)]
        for i in range(X.shape[0]):
            X_dupe2[2*i, :] = X[i, :]
            X_dupe2[2*i+1, :] = X_dupe[i, :]
        X = X_dupe2

    return X


def generate_physics(df, forward_action=0, timestep=0.1, doublings=0):
    d, o, r = physics_init(df)
    df = None

    defn = [d.copy()]
    off = [o.copy()]
    RBs = [r.copy()]

    for a in range(forward_action):
        d, o, r = action(d, o, r, timestep)
        defn.append(d.copy())
        off.append(o.copy())
        RBs.append(r.copy())
    d, o, r = None, None, None

    for X in (defn, off, RBs):
        for i in range(len(defn)):
            X[i]['Px'] = X[i]['Sx']*X[i]['PlayerWeight']
            X[i]['Py'] = X[i]['Sy']*X[i]['PlayerWeight']
            X[i]['Fx'] = X[i]['Ax']*X[i]['PlayerWeight']
            X[i]['Fy'] = X[i]['Ay']*X[i]['PlayerWeight']

    for i in range(len(defn)):
        if i==0:
            bigD = defn[i].loc[:, ('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum')].astype(np.float32)
            bigO = off[i].loc[:, ('Px', 'Py', 'X', 'Y', 'Infl', 'AngularMomentum')].astype(np.float32)
            backs = RBs[i].loc[:, ('YardsBehindScrimmage', 'Px', 'Py', 'Fx', 'Fy')].astype(np.float32)
            inst = np.concatenate((np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*6)), 
                                            np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*6)), 
                                            backs.copy().values), axis=1)
            summary = physics_doubledown(inst, doublings, 6)
        else:
            bigD = defn[i].loc[:, ('Infl', 'AngularMomentum')].astype(np.float32)
            bigO = off[i].loc[:, ('Infl', 'AngularMomentum')].astype(np.float32)
            inst = np.concatenate((np.reshape(bigD.copy().values, (bigD.shape[0]//11, 11*2)), 
                                            np.reshape(bigO.copy().values, (bigO.shape[0]//10, 10*2))), axis=1)
            summary = np.concatenate((summary, physics_doubledown(inst, doublings, 2)), axis=1)
    defn, off, RBs = None, None, None

    return summary


def down_situation(df):
    X = df.loc[::22, ('Down', 'Distance')].copy().astype(np.float32)
    X.fillna(-1, inplace=True)
    framer = pd.DataFrame(columns=(1.0, 2.0, 3.0, 4.0, 'Distance'), dtype=np.float32)
    concatenation = pd.concat((pd.get_dummies(X['Down']), X['Distance']), axis=1, join='outer')
    concatenation = pd.concat((framer, concatenation), axis=0, join='outer')
    concatenation.fillna(0, inplace=True)
    return concatenation.values

def stats(array):
    return array.mean(axis=0), array.std(axis=0)


def normalize(array, mn, stand):
    return (array - mn) / stand


def generate_yardudge(df):
    Y = np.zeros((df.shape[0]//22, 199))
    for (i, yerds) in enumerate(df['Yards'][::22]):
        Y[i, yerds+99] = 1
    return Y.astype(np.float32)


# In[ ]:


global meen, sigma, PCs, doublings, forward_actions, timestep

train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

doublings = 6
forward_actions = 8
timestep = 0.1

Y = doubledown(generate_yardudge(train_df), doublings)

train_df = error_correcting_codes(train_df)
X = np.concatenate(
                (doubledown(down_situation(train_df), doublings), 
                 generate_physics(train_df, forward_actions, timestep, doublings)), 
            axis=1)
train_df = None

meen, sigma = stats(X)
X = normalize(X, meen, sigma)

PCs = np.linalg.eig(np.dot(np.transpose(X), X))[1]
X = np.dot(X, PCs)

gc.collect()


# In[ ]:


import wandb
wandb.init(project="kaggle-competition-NFLBigDataBowl")

stuff = [0.01541, 0.00000173, 0.8858, 0.3867, 0.6044, 14, 180, 1024, 512, 256]

wandb.config.learning_rate = stuff[0]
wandb.config.decay = stuff[1]
wandb.config.beta = stuff[2]
wandb.config.clipse = stuff[3]
wandb.config.dropout_rate = stuff[4]
wandb.config.epochs = stuff[5]
wandb.config.batch_size = stuff[6]
wandb.config.hidden_layer1_size = int(round(stuff[7]))
wandb.config.hidden_layer2_size = int(round(stuff[8]))
wandb.config.hidden_layer3_size = int(round(stuff[9]))

np.random.seed(1729)
model = Sequential()
model.add(Dense(units=wandb.config.hidden_layer1_size, activation='relu'))
model.add(Dropout(wandb.config.dropout_rate))
model.add(Dense(units=wandb.config.hidden_layer2_size, activation='relu'))
model.add(Dropout(wandb.config.dropout_rate))
model.add(Dense(units=wandb.config.hidden_layer3_size, activation='relu'))
model.add(Dropout(wandb.config.dropout_rate))
model.add(Dense(units=199, activation='softmax'))
model.compile(loss=categorical_crossentropy, optimizer=SGD(lr=wandb.config.learning_rate, decay=wandb.config.decay, 
                                                           momentum=wandb.config.beta, nesterov=True, clipnorm=wandb.config.clipse))
model.fit(X, Y, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size, callbacks=[WandbCallback()])


# In[ ]:


X, Y = None, None
gc.collect()


# In[ ]:


# create an nfl environment
env = nflrush.make_env()


# In[ ]:


def make_my_predictions(model, test_df, train_df, sample_predictions):
    test_df = error_correcting_codes(test_df)
    X_downer = doubledown(down_situation(test_df), doublings)
    X_phys = generate_physics(test_df, forward_actions, timestep, doublings)
    X = np.concatenate((X_downer, X_phys), axis=1)
    X = normalize(X, meen, sigma)
    X = np.dot(X, PCs)

    my_predictions = np.mean(model.predict(X), axis=0)
    
    cucumber = 0
    for i in range(1, my_predictions.shape[0]-1):
        cucumber += np.maximum(my_predictions[i], 0)
        sample_predictions.iloc[0, i] = np.minimum(cucumber, 1)
        
    return sample_predictions


# In[ ]:


for (test_df, sample_prediction_df) in env.iter_test():
    predictions_df = make_my_predictions(model, test_df, train_df, sample_prediction_df)
    env.predict(predictions_df)
        
env.write_submission_file()


# In[ ]:




