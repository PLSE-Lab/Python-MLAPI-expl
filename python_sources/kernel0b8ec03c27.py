# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.multiclass import OneVsOneClassifier
from datetime import datetime

train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
Y = train_data['label']
train_data = train_data.drop(['label'], axis=1)

# remove the features that are all zeros
total_data = train_data
total_data = total_data.append(test_data)
cols = []

for col in total_data.columns:
   if len(total_data[total_data[col] == 0]) == len(total_data[col]):
       cols.append(col)
       train_data = train_data.drop([col], axis=1)
       test_data = test_data.drop([col], axis=1)

# min-max scaling
train_data = (train_data - train_data.min()) / (train_data.max() - train_data.min())
test_data = (test_data - test_data.min()) / (test_data.max() - test_data.min())

def countMissing(data):
   missing = data.columns[data.isnull().any()].tolist()
   return missing
misTrain = countMissing(train_data)
misTest = countMissing(test_data)
misTotal = list(set().union(misTrain, misTest))
def imputation(data, column, value):
   data.loc[data[column].isnull(), column] = value
for ele in misTrain:
   imputation(train_data, ele, 1)
for ele in misTest:
   imputation(test_data, ele, 1)


#from sklearn import cross_validation
#alphas = [0.01, 0.1, 1.0, 10, 20, 50]
alphas = [0.1, 1.0, 5]
regs = ["l1", "l2"]
scores_1 = []
param = []
scores_2 = []

start = datetime.now()
print('start time: ', start)

for alpha in alphas:
   for reg in regs:
       
       lm1 = OneVsRestClassifier(linear_model.LogisticRegression(penalty = reg, C = alpha))
       scores_1.append(cross_val_score(lm1, train_data, Y, scoring="accuracy", cv = 10).mean())
       
       lm2 = OneVsOneClassifier(linear_model.LogisticRegression(penalty = reg, C = alpha))
       scores_2.append(cross_val_score(lm2, train_data, Y, scoring="accuracy", cv = 10).mean())
       
       param.append([alpha, reg])

finish = datetime.now()
print('finish time: ', finish)
print('time elapsed in seconds: ', (finish - start).seconds)

scores_1 = pd.DataFrame({'parameter': param, 'score': scores_1})
print(scores_1.sort_values(by = 'score', ascending = False))
scores_2 = pd.DataFrame({'parameter': param, 'score': scores_2})
print(scores_2.sort_values(by = 'score', ascending = False))


'''
for alpha in alphas:
   for reg in regs:
       lm = OneVsOneClassifier(linear_model.LogisticRegression(penalty = reg, C = alpha))
       scores.append(cross_val_score(lm, train_data, Y, scoring="accuracy", cv = 10).mean())
       param.append([alpha, reg])
scores = pd.DataFrame({'parameter': param, 'score': scores})
print(scores.sort_values(by = 'score', ascending = False))
'''