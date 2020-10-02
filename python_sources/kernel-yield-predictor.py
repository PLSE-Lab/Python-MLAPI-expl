# This Python 3 environment comes with many helpful analytics libraries installed
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

# Any results you write to the current directory are saved as output.



# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:48:01 2019

@author: Nico
"""

####################################################################################################
## Trying to predict whether the yield of the german gov' bond will fall or rise in the next year ## 
####################################################################################################

## The change of the yield is already coded as a binary variable
## I am sorry but i cannot share the data
## thus, no data manipulation had to be done

## different machine learning algorithms will be applied

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


dfa = pd.read_excel(r"C:\Users\Nico\datastream_3.xlsm", sheet_name = 'Tabelle1')
df = dfa.drop('time', axis = 1)
df.info()

###########################
##  logistic Regression ###
###########################
    

X1 = df.drop(['10_gov_bond', '10_gov_bond_change', '10_gov_bond_change_binary', '30_gov_bond', '30_gov_bond_change', '30_gov_bond_change_binary'] , axis =1)
Y1 = df['10_gov_bond_change_binary']

logreg = LogisticRegression()
logreg.fit(X1, Y1)
logreg.score(X1, Y1)
# 'nur' 64 %

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size = 0.2, random_state= 42)

logreg.fit(X1_train, y1_train)

logreg.score(X1_test, y1_test)
#--> cannot be interpreted!

from sklearn.metrics import roc_curve

y1_predict = logreg.predict(X1_test)

fpr, tpr, tresholds = roc_curve(y1_test, y1_predict)

print(confusion_matrix(y1_test, y1_predict))

print(classification_report(y1_test, y1_predict))
## not good!

y1_pred_prob = logreg.predict_proba(X1_test) [:,1] 

print(y1_pred_prob)
y1_pred_prob.mean()

#####################
## Classifications ##
#####################

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

n_list = [2,3,4,5,6,7,8,9,10]

for n in n_list:
    list_fits = []
    knn = KNeighborsClassifier(n_neighbors=n)
    X1 = df.drop(['10_gov_bond', '10_gov_bond_change', '10_gov_bond_change_binary', '30_gov_bond', '30_gov_bond_change', '30_gov_bond_change_binary'] , axis =1)
    Y1 = df['10_gov_bond_change_binary']
    knn.fit(X1,Y1)
    a = knn.score(X1,Y1)
    list_fits.append(a)
    print(list_fits)
    
########################
# mit train test split #
########################

from sklearn.model_selection import train_test_split

X1 = df.drop(['10_gov_bond', '10_gov_bond_change', '10_gov_bond_change_binary', '30_gov_bond', '30_gov_bond_change', '30_gov_bond_change_binary'] , axis =1)
Y1 = df['10_gov_bond_change_binary']

X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size = 0.3, random_state = 21)


## again with a loop:

n_list = [2,3,4,5,6,7,8,9,10]

for n in n_list:
    list_fits = []
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    a = knn.score(X_test, y_test)
    list_fits.append(a)
    print(list_fits)

## better, yet still disappointing score
    
######################################
##### per support vector machine #####
######################################
    

from sklearn.svm import SVC 

X1 = df.drop(['10_gov_bond', '10_gov_bond_change', '10_gov_bond_change_binary', '30_gov_bond', '30_gov_bond_change', '30_gov_bond_change_binary'] , axis =1)
Y1 = df['10_gov_bond_change_binary']

model1 = SVC(kernel = 'linear' , C=1)
model1.fit(X1, Y1)
model1.score(X1, Y1)

model1 = SVC(kernel = 'linear' , C=10)
model1.fit(X1, Y1)
model1.score(X1, Y1)

model1 = SVC(kernel = 'linear' , C=0.4)
model1.fit(X1, Y1)
model1.score(X1, Y1)

######################
## non - linear SVM ##
######################

model1 = SVC(kernel = 'linear' , C=0.4, gamma = 1.)
model1.fit(X1, Y1)
model1.score(X1, Y1)


######################
### neural network ###
######################

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from keras.layers import Input


X1 = df.drop(['10_gov_bond', '10_gov_bond_change', '10_gov_bond_change_binary', '30_gov_bond', '30_gov_bond_change', '30_gov_bond_change_binary'] , axis =1)
Y1 = pd.DataFrame(df['10_gov_bond_change_binary'])

model1 = Sequential()

n_cols = X1.shape[1]

model1.add(Dense(28, activation = "relu", input_shape = (n_cols,)))
#second layer
model1.add(Dense(100, activation = "relu"))
# third layer
model1.add(Dense(100, activation = "relu"))
# fourth layer
model1.add(Dense(100, activation = "relu"))
# fifth layer
model1.add(Dense(100, activation = "relu"))
#output layer
model1.add(Dense(1))

model1.compile(optimizer = 'adam', loss = 'mean_squared_error')

model1.summary()

early_stopping_monitor = EarlyStopping(patience = 10)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size = 0.3, random_state=42)

model1.fit(X1_train, y1_train, epochs = 900,
          callbacks = [early_stopping_monitor])


pred1 = model1.predict(X1_test)

list_pred_1 = []

for a in pred1:
    if a >= 0:
        list_pred_1.append(1)
    else:
        list_pred_1.append(0)
    
print(list_pred_1)

y1_test["pred"] = list_pred_1

y1_test.head() 

y1_test["dev"] = y1_test["10_gov_bond_change_binary"] - y1_test["pred"]

deviation = np.std(y1_test["10_gov_bond_change_binary"] - y1_test["pred"])

y1_test.head()
print(np.mean(y1_test["dev"]))
print(deviation)
## finally good results!

####################
## decision trees ##
####################

# regressors are used here
# for numeric data, trees seperate nodes by specific benchmarks
# if the end leaf nodes are tightly distributed, then the tree is a good one!


X1 = df.drop(['10_gov_bond', '10_gov_bond_change', '10_gov_bond_change_binary', '30_gov_bond', '30_gov_bond_change', '30_gov_bond_change_binary'] , axis =1)
Y1 = df['10_gov_bond_change_binary']

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


decision_tree = DecisionTreeRegressor(max_depth = 6)

decision_tree.fit(X1,Y1)

print(decision_tree.score(X1,Y1))

##  train &  test data 

X4_train, X4_test, y4_train, y4_test = train_test_split(X1, Y1, test_size = 0.2, random_state=42)

decision_tree = DecisionTreeRegressor(max_depth = 6)

decision_tree.fit(X4_train, y4_train)

print(decision_tree.score(X4_test, y4_test))

### looping over maximal depths

depth_list = [4,5,6,7,8,8,9,10]

for depth in depth_list:
    list_results = []
    decision_tree = DecisionTreeRegressor(max_depth = depth)
    decision_tree.fit(X4_train, y4_train)
    result = decision_tree.score(X4_test, y4_test)
    list_results.append(result)
    
best_depth_dict = { 'number': depth_list, 'score': list_results }
print(best_depth_dict)

## best fit for a depth of 5!

