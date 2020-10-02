import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import seaborn as sns

def AnalyzeData(input, output):

    dataToShow = np.ndarray((len(input),2))
    for i in range(0,len(input)):
        dataToShow[i][0] = sum(input[i])
        dataToShow[i][1] = output[i][0]
    df = pd.DataFrame(data=dataToShow, columns=['sum','total'])
    sns.pairplot(df, diag_kind="kde")
    plt.show()
def build_model(input_len):
  model = keras.models.Sequential([
      keras.layers.Dense(32, activation=tf.nn.relu, input_shape=[input_len]),
      keras.layers.Dense(32, activation=tf.nn.relu),
      keras.layers.Dense(2)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

def getTeamsEnum():
    return {'ATL'	:1,
            'BOS'	:2,
            'CHH'	:3,
            'CHI'	:4,
            'CLE'	:5,
            'DAL'	:6,
            'DEN'	:7,
            'DET'	:8,
            'GSW'	:9,
            'HOU'	:10,
            'IND'	:11,
            'LAC'	:12,
            'LAL'	:13,
            'MEM'	:14,
            'MIA'	:15,
            'MIL'	:16,
            'MIN'	:17,
            'NJN'	:18,
            'NYK'	:19,
            'ORL'	:20,
            'PHI'	:21,
            'PHO'	:22,
            'POR'	:23,
            'SAC'	:24,
            'SAS'	:25,
            'SEA'	:26,
            'TOR'	:27,
            'UTA'	:28,
            'WAS'	:29,
            'NOH'	:30,
            'CHA'	:31,
            'NOK'	:32,
            'OKC'	:33}

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()



def flattenPlayersStats(df):
    flatTable = []
    flatTable = df.loc[:,['points']]
    flatTable = flatTable/1000
    # for row in df.values:
    #      flatTable.append(row[7])  # minutes
    #      flatTable.append(row[8]) # points
    #      flatTable.append(row[11])  # rebounds
    #      flatTable.append(row[12])  # assists
    #      flatTable.append(row[15])  # turnovers
    #      flatTable.append(row[16])  # PF
    # return flatTable
    return list(flatTable.values.flatten())

def flatInputAndOutput(dfInput, dfOutput):
    flatInput = []
    flatOutput = []
    dfOutput = dfOutput.filter(items=['year', 'tmID', 'rank', 'o_pts', 'won', 'games'])
    playersGroups = dfInput.groupby(['year', 'tmID'])

    for k, v in playersGroups:
        teamVals = dfOutput.values[(dfOutput['year'] == k[0]) & (dfOutput['tmID'] == k[1])]
        if len(teamVals) > 0:
            # flatOutput.append([teamVals[0][3]/1000, teamVals[0][4]])
            if teamVals[0][5] == 0:
                wonPrec = 0
            else:
                wonPrec = teamVals[0][4]/teamVals[0][5]
            flatOutput.append([teamVals[0][3]/1000, wonPrec ])
            v = v.sort_values(by=['minutes'], ascending=False)
            flatInput.append(flattenPlayersStats(v.head(10)))
    return flatInput, flatOutput

dfPlayersDetails = pd.read_csv('basketball_master.csv')
dfPlayersStats = pd.read_csv('basketball_players.csv')
dfTeamsStats = pd.read_csv('basketball_teams.csv')

lastTeamsStats = dfTeamsStats[(dfTeamsStats.year > 2000)&(dfTeamsStats.year < 2011)]
#lastTeamsStats = lastTeamsStats.filter(items=['year', 'tmID', 'rank', 'pts', 'won'])

lastPlayersStats = dfPlayersStats[(dfPlayersStats.year > 2000)&(dfPlayersStats.year < 2011)]
testPlayersStats = dfPlayersStats[(dfPlayersStats.year == 2000)]

testTeamsStats = dfTeamsStats[(dfTeamsStats.year == 2000)]


#lastPlayersStats = lastPlayersStats.replace(getTeamsEnum())
#lastTeamsStats = lastTeamsStats.replace(getTeamsEnum())


flatPlayersInput = []
flatTeamsOutput = []
flatTestInput = []
flatTestOutput = []
EPOCHS = 4000



flatPlayersInput, flatTeamsOutput = flatInputAndOutput(lastPlayersStats, lastTeamsStats)
flatTestInput, flatTestOutput = flatInputAndOutput(testPlayersStats, testTeamsStats)

#AnalyzeData(flatTestInput, flatTestOutput)

model = build_model(10)
print(model.summary())

history = model.fit(
    flatPlayersInput, flatTeamsOutput,
  epochs=EPOCHS, validation_split = 0.2)

loss, mae, mse = model.evaluate(flatTestInput, flatTestOutput)
plot_history(history)
#print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
test_predictions = model.predict(flatTestInput)
# for i in range(0,len(test_predictions)):
#     print("Input:{:5.2f} output:{:5.2f} predicted:{:5.2f}".format(flatTestInput[i],flatTestOutput[i],test_predictions[i]))
print(test_predictions)
print('End')