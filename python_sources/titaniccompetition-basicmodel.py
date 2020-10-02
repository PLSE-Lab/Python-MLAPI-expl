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

pd.set_option('display.max_colwidth',1000)
#print(os.listdir(".."))
dtrain = pd.read_csv('../input/train.csv', sep = ',')
dtest = pd.read_csv('../input/test.csv', sep=',')
ddata = dtrain.append(dtest)
Submission = pd.DataFrame()
Submission['PassengerId'] = dtest['PassengerId']

##Describing the data numerically
dtrain.describe(include='all')
print(dtrain.columns)
print(dtest.columns)
print(dtrain['SibSp'].value_counts())
print(dtrain.isnull().values.any())
for col in dtrain.isnull().columns.values.tolist():
    if dtrain.isnull()[col].any():
        print(col , '\n' , dtrain.isnull()[col].value_counts(), '\n')
print(dtrain.shape)
print(dtrain.dtypes)
print(dtrain.info())
print(pd.isnull(dtrain).sum())

#Exploration of data graphically
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Survival by Age,Class, Gender
grid = sns.FacetGrid(dtrain, col='Pclass', row='Sex', hue='Survived', palette='seismic')
grid = grid.map(plt.scatter, 'PassengerId','Age')
grid.add_legend()
grid

#Survival by Age,Port of Embarkation, Gender
grid = sns.FacetGrid(dtrain, col='Embarked', row='Sex', hue='Survived',  palette='seismic')
grid = grid.map(plt.scatter, 'PassengerId', 'Age')
grid.add_legend()
grid

#Survival by Age, Number of Siblings and Gender
grid = sns.FacetGrid(dtrain, col='SibSp', row='Sex', hue='Survived', palette = 'seismic')
grid = grid.map(plt.scatter, 'PassengerId', 'Age')
grid.add_legend()
grid

#Survival by Age, Number of parch and Gender
grid = sns.FacetGrid(dtrain, col = "Parch", row = "Sex", hue = "Survived", palette = 'seismic')
grid = grid.map(plt.scatter, "PassengerId", "Age")
grid.add_legend()
grid

#Pairplots
g = sns.pairplot(dtrain[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']], hue='Survived', 
                 palette = 'seismic',size=4,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=50) )
g.set(xticklabels=[])

#Creating simple model
from sklearn.model_selection import (cross_val_score,cross_val_predict,cross_validate,train_test_split,
GridSearchCV,KFold,learning_curve,RandomizedSearchCV,StratifiedKFold)
from sklearn.svm import SVC, LinearSVC

NUMERIC_COLUMNS=['Pclass','Age','SibSp','Parch','Fare']
#To train and test with only the above columns as predictors
#Filling each NaN values with 0 in the stated columns
dtest01 = dtest[NUMERIC_COLUMNS].fillna(0)
dtrain01 = dtrain[NUMERIC_COLUMNS].fillna(0)
target01 = dtrain['Survived']
x_train,x_test,y_train,y_test = train_test_split(dtrain01, target01, test_size=0.3, random_state=2, stratify = target01)

#Fitting model on SVC 
clf = SVC()
clf.fit(x_train,y_train)
# Print the accuracy
print("Accuracy: {}".format(clf.score(x_test, y_test)))

#predicting values
Submission['PassengerID'] = clf.predict(dtest01)
print(Submission.head())

#Create index and write to csv file
Submission.set_index('PassengerId', inplace=True)
Submission.to_csv('basic_model.csv', sep=',' )