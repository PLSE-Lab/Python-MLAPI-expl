# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

atp = pd.read_csv("../input/ATP.csv", dtype = object) # str
atp.index[atp['surface'] == 'surface'].tolist() # [160636]
atp.drop(atp.index[160636], inplace = True) # remove the anomalous line with repeated header names
atp = atp.convert_objects(convert_numeric = True)
atp = atp.drop(columns = ['winner_entry', 'loser_entry', 'winner_seed', 'loser_seed'], axis = 1) # vars with majority missing vals
atp['year'] = atp.tourney_date.astype(str).str[:4]
atp['date'] = pd.to_datetime(atp['tourney_date'], format = '%Y%m%d', errors = 'coerce')

# EDA: grand slam winners' countries count, descending
grandSlamIOCWinner = atp[(atp['tourney_level'] == 'G') & (atp['round'] == 'F')]\
.groupby(['winner_ioc'])\
.count()\
.sort_values('year', ascending = False)\
.iloc[:, 0]
plt.figure(figsize = (10, 10))
plt.xticks(fontsize = 15)
plt.xticks(np.arange(0, 61, step = 10))
grandSlamIOCWinner.plot(kind = 'barh')

# EDA: grand slam winners count, descending
grandSlamWinner = atp[(atp['tourney_level'] == 'G') & (atp['round'] == 'F')]\
.groupby(['winner_name'])\
.count()\
.sort_values('year', ascending = False)\
.iloc[:, 0]
plt.figure(figsize = (10, 20))
plt.xticks(fontsize = 15)
plt.xticks(np.arange(0, 19, step = 3))
grandSlamWinner.plot(kind = 'barh')

# replicate dataset and swap variables for winners & losers -> stack
atp2 = atp.copy()
atp2.rename(columns = {'winner_hand' : 'loser_hand',
                       'winner_ioc' : 'loser_ioc',
                       'winner_rank' : 'loser_rank',
                       'winner_rank_points' : 'loser_rank_points',
                       'winner_ht' : 'loser_ht',
                       'winner_age' : 'loser_age',
                       'loser_hand' : 'winner_hand',
                       'loser_ioc' : 'winner_ioc',
                       'loser_rank' : 'winner_rank',
                       'loser_rank_points' : 'winner_rank_points',
                       'loser_ht' : 'winner_ht',
                       'loser_age' : 'winner_age'},
            inplace = True)
print(atp[['winner_hand', 'loser_hand', 'winner_ioc', 'loser_ioc', 'winner_rank', 'loser_rank',
           'winner_rank_points', 'loser_rank_points', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age']].tail(3))
print(atp2[['winner_hand', 'loser_hand', 'winner_ioc', 'loser_ioc', 'winner_rank', 'loser_rank',
            'winner_rank_points', 'loser_rank_points', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age']].tail(3))

atp['won'] = 1
atp2['won'] = 0

# stack
data = atp.append(atp2, ignore_index = True, sort = True)

# drop variables with at least one of them having missing predictor values
data.dropna(axis = 0,
            subset = ['surface', 'tourney_level', 'round',
                      'winner_hand', 'loser_hand', 'winner_ioc', 'loser_ioc', 'winner_rank', 'loser_rank',
                      'winner_rank_points', 'loser_rank_points', 'winner_ht', 'loser_ht', 'winner_age', 'loser_age'],
            how = 'any',
            inplace = True)

# drop unused variables for deep network
data = data.drop(['tourney_id', 'tourney_name', 'draw_size', 'tourney_date', 'match_num',
                  'winner_id', 'winner_name', 'loser_id', 'loser_name', 'score', 'best_of',
                  'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
                  'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn',
                  'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'year', 'date'],
                 axis = 1)

# feature engg.: dummy variables
dummy_fields = ['surface', 'tourney_level', 'round', 'winner_hand', 'winner_ioc', 'loser_hand', 'loser_ioc']
for each in dummy_fields:
    dummies = pd.get_dummies(data[each], prefix=each, drop_first = True)
    data = pd.concat([data, dummies], axis = 1)

data = data.drop(dummy_fields, axis = 1)

# Scaling predictor quantitative variables
# Alternative to standardisation is min-max normalisation with range [0, 1]
quant_features = ['loser_age', 'loser_ht', 'loser_rank', 'loser_rank_points',
                  'winner_age', 'winner_ht', 'winner_rank', 'winner_rank_points']

# Store scalings in a dictionary, so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# Separate the data into features and targets
target_fields = ['won']
X, y = data.drop(target_fields, axis = 1), data[target_fields]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Deep network: tournament year could be added as a variable in the future
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
# from keras.constraints import max_norm
# from keras import regularizers
    ## kernel_regularizer = regularizers.l2(0.01), activity_regularizer = regularizers.l1(0.01)

n_cols = X_train.shape[1]
model = Sequential()
model.add(Dropout(0.1, seed = 42, input_shape = (n_cols, )))
# model.add(Dense(51, activation = 'relu', input_shape = (n_cols, ), kernel_constraint = max_norm(2.)))
model.add(Dense(26, activation = 'relu'))
model.add(Dense(13, activation = 'relu'))
model.add(Dense(7, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)
history = model.fit(X_train, y_train, epochs = 20, validation_split = 0.1, batch_size = 128, callbacks = [early_stopping])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc = 'lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

# Deep network: evaluate
# Model performance
print(model.evaluate(X_test, y_test))
mod_wts = model.get_weights()
model.summary()
