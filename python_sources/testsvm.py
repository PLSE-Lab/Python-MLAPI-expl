# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm

import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


gameresults = []
X = []
keys = ['B365A','B365H','B365D']
# keys = ['home_team_api_id','away_team_api_id']
with sqlite3.connect('../input/database.sqlite') as con:
    #countries = pd.read_sql_query("SELECT * from Country", con)
    matches_ger = pd.read_sql_query("SELECT * from Match where country_id=7775", con)
    #matches_all = pd.read_sql_query("SELECT * from Match", con)
    #print('bundesliga: ' + str(len(matches_ger)) + ' / all: ' + str(len(matches_all)))
    #leagues = pd.read_sql_query("SELECT * from League ", con)
    #teams = pd.read_sql_query("SELECT * from Team", con)
    # id , home_team_api_id, away_team_api_id, home_team_goal, away_team_goal
    
    X = matches_ger[keys].values
    matches_ger = matches_ger[['id','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']]

    for i in range(len(matches_ger)):
        if matches_ger.values[i][3]==matches_ger.values[i][4]:
            gameresults.append(0)
        elif matches_ger.values[i][3]>matches_ger.values[i][4]:
            gameresults.append(1)
        elif matches_ger.values[i][3]<matches_ger.values[i][4]:
            gameresults.append(2)
        else:
            print('error: unrecognized result at resultset ' + str(element))
    print('gameresults: ' + str(len(gameresults)))
    print(gameresults)
    
x=np.array(gameresults)
unique, counts = np.unique(x, return_counts=True)
b=dict(zip(unique, counts))
print(b)
#clf = svm.SVC()
#clf.fit(X, gameresults)
#predictions = clf.predict(X)
#print(predictions)
