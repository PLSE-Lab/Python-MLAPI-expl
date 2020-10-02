# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# import machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# read training and test data into a pandas dataframe
data_train = pd.read_csv('../input/train.csv')
data_test  = pd.read_csv('../input/test.csv')

for i in [1,2,3]:
    femaleAgeMean = data_train.loc[(data_train['Pclass']==i) & (data_train['Sex']=='female')].Age.mean()
    data_train.ix[(data_train['Age'].isnull()) & (data_train['Pclass']==i) & (data_train['Sex']=='female'), 'Age'] = femaleAgeMean
    maleAgeMean = data_train.loc[(data_train['Pclass']==i) & (data_train['Sex']=='male')].Age.mean()
    data_train.ix[(data_train['Age'].isnull()) & (data_train['Pclass']==i) & (data_train['Sex']=='male'), 'Age'] = maleAgeMean
    fareMean = data_train.loc[data_train['Pclass']==i].Fare.mean()
    data_train.ix[(data_train['Fare'].isnull()) & (data_train['Pclass']==i), 'Fare'] = fareMean

for i in [1,2,3]:
    femaleAgeMean = data_test.loc[(data_test['Pclass']==i) & (data_test['Sex']=='female')].Age.mean()
    data_test.ix[(data_test['Age'].isnull()) & (data_test['Pclass']==i) & (data_test['Sex']=='female'), 'Age'] = femaleAgeMean
    
    maleAgeMean = data_test.loc[(data_train['Pclass']==i) & (data_test['Sex']=='male')].Age.mean()
    data_test.ix[(data_test['Age'].isnull()) & (data_test['Pclass']==i) & (data_test['Sex']=='male'), 'Age'] = maleAgeMean
    
    fareMean = data_test.loc[data_test['Pclass']==i].Fare.mean()
    data_test.ix[(data_test['Fare'].isnull()) & (data_test['Pclass']==i), 'Fare'] = fareMean


""" UNDERSTANDING THE DATA """

#print(data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
#print(data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# initialise modified dataframe
modified_training_data = data_train[['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age']].copy()

# make a function to create age groups
def age_grp(row):
    if row['Age'] <= 1.0:
        return 0
    elif row['Age'] > 1.0 and row['Age'] <= 5.0:
        return 1
    elif row['Age'] > 5.0 and row['Age'] <= 10.0:
        return 2
    elif row['Age'] > 10.0 and row['Age'] <= 15.0:
        return 3
    elif row['Age'] > 15.0 and row['Age'] <= 20.0:
        return 4
    elif row['Age'] > 20.0 and row['Age'] <= 25.0:
        return 5
    elif row['Age'] > 25.0 and row['Age'] <= 30.0:
        return 6
    elif row['Age'] > 30.0 and row['Age'] <= 35.0:
        return 7
    elif row['Age'] > 35.0 and row['Age'] <= 40.0:
        return 8
    elif row['Age'] > 40.0 and row['Age'] <= 45.0:
        return 9
    elif row['Age'] > 45.0 and row['Age'] <= 50.0:
        return 10
    elif row['Age'] > 50.0:
        return 11

# make new dataframe column with age group
modified_training_data['ageGroup'] = modified_training_data.apply (lambda row: age_grp (row),axis=1)

print(modified_training_data[['ageGroup', 'Survived']].groupby(['ageGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False))



#%%

""" DATA PRE-PROCESSING """

# make a subset of the training data:
training = modified_training_data[['PassengerId', 'Survived', 'Pclass', 'Sex', 'ageGroup']].copy()

# convert gender to numericals
training.Sex = training.Sex.map({'male': 0, 'female': 1})

# only pick rows with finite numbers. no 'nan' values accepted.
# training = training[np.isfinite(training['ageGroup'])]

""" TRAINING NEURAL NETWORK(s) """

# X is inputs Y is output - this is for generating model
X = training[['Pclass', 'Sex', 'ageGroup']].copy()
y = training['Survived']

# split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_cv, X_test, y_cv, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# scale training and test data before running NN: initialise scaler function
scaler = StandardScaler()

# Fit scaler function only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_cv = scaler.transform(X_cv)

# initialise NN: MLP classifier
#hiddenLayerSize = 13 # size of hidden layer
#mlp = MLPClassifier(hidden_layer_sizes=(hiddenLayerSize,hiddenLayerSize,hiddenLayerSize),max_iter=1000)



NN_arch = [ (10,), (20,), (30,),
            (10,10,), (20,20,), (30,30,)]
            
regularization = [0.00001, 0.0001, 0.01]

accuracy = np.zeros((len(NN_arch),len(regularization)))
for i in range(len(NN_arch)):
    for j in range(len(regularization)):
        clf = MLPClassifier(hidden_layer_sizes=NN_arch[i], activation='logistic', alpha=regularization[j],max_iter=10000)

        clf.fit(X_cv, y_cv)

        accuracy[i,j] = clf.score(X_train,y_train)

print(accuracy)

# Choose the most accurate combination of NN_arch/regularization
id = np.unravel_index(accuracy.argmax(),accuracy.shape)
# Define the classifier
mlp = MLPClassifier(hidden_layer_sizes=NN_arch[id[0]], activation='logistic', alpha=regularization[id[1]],max_iter=20000)

# fit NN to training data:
mlp.fit(X_train,y_train)

""" PREDICT ON TRAINING-TEST DATA USING NEURAL NETWORK """

# use neural network to predict on testing part of training data:
predictions = mlp.predict(X_test)

# print prediction report
print(classification_report(y_test,predictions))


""" PREDICTING ON TEST DATA """

# prep data -----------------------
modified_testData = data_test[['PassengerId', 'Pclass', 'Sex', 'Age']].copy()

# convert gender to numericals
modified_testData.Sex = modified_testData.Sex.map({'male': 0, 'female': 1})

modified_testData['ageGroup'] = modified_testData.apply (lambda row: age_grp (row),axis=1)
#modified_testData = modified_testData[np.isfinite(modified_testData['ageGroup'])]

X_test_data = modified_testData[['Pclass', 'Sex', 'ageGroup']].copy()
X_test_data = scaler.transform(X_test_data)

# run prediction -----------------------

t = mlp.predict(X_test_data)

surv = pd.DataFrame({'PassengerId': modified_testData['PassengerId'], 'Survived': t})

surv.to_csv('NN_Titanic.csv',index=False)















