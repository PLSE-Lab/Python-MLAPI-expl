# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn import preprocessing
#from randomferns import *
from sklearn.svm import SVC # "Support vector classifier"
from sklearn import datasets, linear_model, metrics 
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] #copying test index for later

df_train.head()

train_X = df_train.loc[:, 'V1':'V16']
#test_1
#train_X['V1'] = preprocessing.normalize(train_X)
#train_X['V6'] = preprocessing.scale(train_X)

#test_2
#train_X['V1'] = preprocessing.normalize(train_X['V1'])
#train_X['V6'] = preprocessing.scale(train_X['V6'])
#train_X['V12'] = preprocessing.normalize(train_X['V12'])

#test_3
#train_X = abs(train_X)
#train_X = preprocessing.normalize(train_X)

#test_4
#x = preprocessing.scale(train_X)



train_y = df_train.loc[:, 'Class']

#test_5
#rf = SVC(kernel='linear', C=1E10, random_state = 42, max_iter = 10000)
#rf = linear_model.LinearRegression()
#x = preprocessing.scale(train_X)

#test6

#r = csv.reader(open('../input/webclubrecruitment2019/TRAIN_DATA.csv')) # Here your csv file
#lines = list(r)
#for i in len(lines.index):
#    for j in len(lines.columns):
#        if lines[0][j] == 'V6':
#            if lines[i][j] > 1000:
#                lines[i][j] = 1000
#            iflines[i][j] < -1000:
#                lines[i][j] = -1000     
#        if lines[0][j] == 'V12':
#            if lines[i][j] > 500:
#                lines[i][j] = 500
#            iflines[i][j] < -500:
#                lines[i][j] = -500

#with open('../input/webclubrecruitment2019/TRAIN_DATA.csv', 'w') as writeFile:
#    writer = csv.writer(writeFile)
#    writer.writerows(lines)
#writeFile.close()          

#, max_features = 0.2, min_samples_leaf = 30
rf = RandomForestClassifier(n_estimators=2000, random_state = 42, max_features = 0.7, min_samples_leaf = 2000)

#increase n_estimators? ... 
#rf = RandomFerns( depth=10, n_estimators=50, test_class=getattr( weakLearner, learner)() )
rf.fit(train_X, train_y)

df_test = df_test.loc[:, 'V1':'V16']
pred = rf.predict_proba(df_test)


result=pd.DataFrame()
result['Id'] = test_index
result['PredictedValue'] = pd.DataFrame(pred[:,1])
result.head()

result.to_csv('output.csv', index=False)