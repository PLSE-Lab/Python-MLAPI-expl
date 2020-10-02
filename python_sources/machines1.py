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

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict

X = pd.read_csv('X')
print('Length X: {}'.format(len(X)))
Y = pd.read_csv('Y')
print('Length Y: {}'.format(len(Y)))

# x_treino, x_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.2)

# Linear Regression
lg = LinearRegression().fit(X, Y)
print('LinearRegression: coef. det.: {:.4f}'.format(lg.score(X, Y)))

teste = pd.read_csv('../input/data/test')

Id = teste['Id']
teste = teste.iloc[:, 1:]

Y_hat = lg.predict(teste)
# print('Length Y: {}'.format(len(Y_hat)))

Y_hat = pd.Series(Y_hat.reshape(len(Y_hat,)))
submission = pd.concat([Id, Y_hat], axis=1)
submission.columns=['Id', 'SalePrice']
submission.to_csv('submission.csv', sep=',', index=False)

# # Decision Tree
# dt = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=6, presort=True).fit(x_treino, y_treino)
# Y_hat = dt.predict(x_teste)
# print('DecisionTree: acc. score: {:.4f}'.format(metrics.accuracy_score(y_teste, Y_hat)))
# print('Length Y: {}'.format(len(Y_hat)))
#
# # Random Forest
# rf = RandomForestClassifier(n_estimators=100).fit(x_treino, np.ravel(y_treino))
# Y_hat = rf.predict(x_teste)
# print('RandomForest: acc. score: {:.4f}'.format(metrics.accuracy_score(y_teste, Y_hat)))
# print('Length Y: {}'.format(len(Y_hat)))