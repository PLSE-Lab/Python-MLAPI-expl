# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import model_to_dot

import datetime
import time

#print(tf.__version__)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

#----------------------------Read in Data------------------------------------------------------------
#Read in the race results data
results_raw = pd.read_csv('../input/formula-1-race-data-19502017/results.csv', na_values = "?", comment='\t')

#Read in the races data
races_raw = pd.read_csv('../input/formula-1-race-data-19502017/races.csv', na_values = "?", comment='\t')

#Read in the last races of season data
last_race_raw = pd.read_csv('../input/last-race/last_race.csv', na_values = "?", comment='\t')
last_race_raw.rename(columns={'Year':'year', 'Race ID': 'raceId'}, inplace=True)

#Read in the list of driver standings
driver_standings = pd.read_csv('../input/formula-1-race-data-19502017/driverStandings.csv')

#------------------------------------------------------------------------------------------------------

#Remove year column from last race data
#last_race_raw = last_race_raw.drop(columns='Year')


#-------------------- Filtering Races Data ---------------------
#Remove uneccesary race columns
races_raw = races_raw.drop(columns=['date', 'time', 'url'])

#Remove all races that aren't the first of the season
first_race_index = races_raw[races_raw['round'] != 1].index
races_raw = races_raw.drop(first_race_index)

#Keep only first round races from 2000 to 2017
race_year_index_low = races_raw[races_raw['year'] < 2000].index
race_year_index_high = races_raw[races_raw['year'] > 2017].index
races_raw = races_raw.drop(index=race_year_index_low) 
races_raw = races_raw.drop(index=race_year_index_high)
races_raw = races_raw.sort_values(by=['year'])


#Remove uneccesary race columns
races_filtered = races_raw.drop(columns=['round', 'circuitId', 'name'])
#print(races_filtered.to_string())
#----------------------------------------------------------------------------------


#---------------------Filtering Results Data--------------------------------------
#Clean the Raw results data to only include columns that we want
#Get a list of the first races ID's and filter the results table to only those races
firstraceId_list = races_filtered['raceId'].tolist()
raceId_filter = results_raw['raceId'].isin(firstraceId_list)
results_filtered = results_raw[raceId_filter]

#Dropping columns that we aren't interested in
results_filtered = results_filtered.drop(columns=['number','position', 'positionText', 'points','time', 'statusId'])

#print(results_raw.isna().sum())

results_filtered = pd.merge(results_filtered, races_filtered, on='raceId')
drivers_start_list = results_filtered['driverId'].unique()
#print(results_filtered.count())
#--------------------------Finding the standing for each driver as of the last race of the year-------------------------------

#Get a list of the IDs of the last races of every year
lastraceId_list = last_race_raw['raceId'].tolist()
#print(lastraceId_list)

#print(results_filtered.count())
#print(standings_filtered.count())
#Filter the drivers standings to only include the last races of the year
lastRaceId_filter = driver_standings['raceId'].isin(lastraceId_list)
standings_filtered = driver_standings[lastRaceId_filter]

#Drop standings columns that we don't want
standings_filtered = standings_filtered.drop(columns=['driverStandingsId', 'positionText', 'points', 'wins'])
#Sort the standings by raceId and by driverId
standings_filtered = standings_filtered.sort_values(["raceId","driverId"])
standings_filtered = pd.merge(standings_filtered, last_race_raw, on='raceId')
standings_filtered.rename(columns={'position':'seasonEndPos'}, inplace=True)
drivers_end_list = standings_filtered['driverId'].unique()


#---------------------------------------------------------------------------------------------------------------------------------------

#for driver in drivers_start_list:
#    standings_filtered = standings_filtered[standings_filtered.driverId != driver]

#print(standings_filtered.count())
#print(standings_filtered.to_string())
#print(results_filtered.to_string())
#In the last race CSV change the race ID to the first race of the season so that we can identify it in the results CSV. 
#All of this spaghet is to make sure that we can merge the last place standings with the results page. 

#print(results_filtered.to_string())
#driver_standings.tail()

#Clean the driver standings dataset
#driver_standings.isna().sum()
#driver_standings.drop(columns = ['positionText','wins'])

#----------------------------Merge the two tables together to get the drivers year-end position beside their position--------------------
full_results = pd.merge(results_filtered, standings_filtered, how='inner', on=['driverId', 'year'])
full_results = full_results.drop(columns=['resultId', 'raceId_x', 'driverId', 'constructorId', 'milliseconds', 'fastestLap', 'fastestLapSpeed', 'rank', 'year', 'raceId_y'])
dataset = full_results.dropna()



#Change the fastest lap time to miliseconds
for index, row in dataset.iterrows():
    timestring = row['fastestLapTime']
    pt = datetime.datetime.strptime(timestring,'%M:%S.%f')
    total_miliseconds = pt.microsecond/1000000 + pt.second + pt.minute*60
    dataset.at[index,'fastestLapTime'] = total_miliseconds

dataset["fastestLapTime"] = pd.to_numeric(dataset["fastestLapTime"])


#---------------------------------------Begin Regression----------------------------------------------------------------------------------
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["grid", "positionOrder", "laps", "fastestLapTime", "seasonEndPos"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("seasonEndPos")
train_stats = train_stats.transpose()
print(train_stats)

#Split Features from labels
train_labels = train_dataset.pop('seasonEndPos')
test_labels = test_dataset.pop('seasonEndPos')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

#Build the model
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

print(model.summary())

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
#print(example_result)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    #print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Final Position^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} Final Position".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Final Position]')
plt.ylabel('Predictions [Final Position]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
#-----------------------------------------------------------------------------------------------------------------------------
