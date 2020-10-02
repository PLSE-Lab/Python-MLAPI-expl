# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tpot import TPOTClassifier 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


def data():
    
    # Importing the dataset
    X_train = pd.read_csv("../input/titanic/train.csv")
    X_test = pd.read_csv("../input/titanic/train.csv")
    
    # Replace names with titles
    X_train['Name'] = X_train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = X_train['Name'].unique()
    
    X_test['Name'] = X_test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
    titles = X_test['Name'].unique()
    
    # Replace missing ages with title median
    X_train['Age'].fillna(-1, inplace=True)
    X_test['Age'].fillna(-1, inplace=True)
    
    medians = dict()
    for title in titles:
        median = X_train.Age[(X_train["Age"] != -1) & (X_train['Name'] == title)].median()
        medians[title] = median
        
    for index, row in X_train.iterrows():
        if row['Age'] == -1:
            X_train.loc[index, 'Age'] = medians[row['Name']]
    
    for index, row in X_test.iterrows():
        if row['Age'] == -1:
            X_test.loc[index, 'Age'] = medians[row['Name']]
            
    # Replace titles with numerical values
    replacement = {
        'Don': 0,
        'Rev': 0,
        'Jonkheer': 0,
        'Capt': 0,
        'Mr': 1,
        'Dr': 2,
        'Col': 3,
        'Major': 3,
        'Master': 4,
        'Miss': 5,
        'Mrs': 6,
        'Mme': 7,
        'Ms': 7,
        'Mlle': 7,
        'Sir': 7,
        'Lady': 7,
        'the Countess': 7
    }
    
    X_train['Name'] = X_train['Name'].apply(lambda x: replacement.get(x))        
    X_test['Name'] = X_test['Name'].apply(lambda x: replacement.get(x))  
     
    # Replace missing fare with class median
    X_train['Fare'].fillna(-1, inplace=True)
    X_test['Fare'].fillna(-1, inplace=True)
    
    medians = dict()
    for pclass in X_train['Pclass'].unique():
        median = X_train.Fare[(X_train["Fare"] != -1) & (X_train['Pclass'] == pclass)].median()
        medians[pclass] = median
        
    for index, row in X_train.iterrows():
        if row['Fare'] == -1:
            X_train.loc[index, 'Fare'] = medians[row['Pclass']]
    
    for index, row in X_test.iterrows():
        if row['Fare'] == -1:
            X_test.loc[index, 'Fare'] = medians[row['Pclass']]
    
    replacement = {
        6: 0,
        4: 0,
        5: 1,
        0: 2,
        2: 3,
        1: 4,
        3: 5
    }
    X_train['Parch'] = X_train['Parch'].apply(lambda x: replacement.get(x))
    X_test['Parch'] = X_test['Parch'].apply(lambda x: replacement.get(x))
    
    X_train['Embarked'] = X_train['Embarked'].fillna('S')
    X_test['Embarked'] = X_test['Embarked'].fillna('S')
    
    replacement = {
        'S': 0,
        'Q': 1,
        'C': 2
    }
    
    X_train['Embarked'] = X_train['Embarked'].apply(lambda x: replacement.get(x))
    X_test['Embarked'] = X_test['Embarked'].apply(lambda x: replacement.get(x))
    
    replacement  = {
        5: 0,
        8: 0,
        4: 1,
        3: 2,
        0: 3,
        2: 4,
        1: 5
    }
    
    X_train['SibSp'] = X_train['SibSp'].apply(lambda x: replacement.get(x))
    X_test['SibSp'] = X_test['SibSp'].apply(lambda x: replacement.get(x))
    
    X_train['Cabin'] = X_train['Cabin'].fillna('U')
    X_test['Cabin'] = X_test['Cabin'].fillna('U')
    
    # Retain first letter only of cabin
    X_train['Cabin'] = X_train['Cabin'].map(lambda x: x[0])
    X_test['Cabin'] = X_test['Cabin'].map(lambda x: x[0])
    
    replacement = {
        'T': 0,
        'U': 1,
        'A': 2,
        'G': 3,
        'C': 4,
        'F': 5,
        'B': 6,
        'E': 7,
        'D': 8
    }
    
    X_train['Cabin'] = X_train['Cabin'].apply(lambda x: replacement.get(x))
    X_test['Cabin'] = X_test['Cabin'].apply(lambda x: replacement.get(x))
    
    X_train['Sex'] = LabelEncoder().fit_transform(X_train['Sex'])
    X_test['Sex'] = LabelEncoder().fit_transform(X_test['Sex'])

    y_train = X_train.iloc[:, 1].values
    submission = X_test.iloc[:, 0].values
    submission = pd.DataFrame(submission) 
    submission.columns = ['PassengerId'] 
    
    # Delete redundant features
    X_train = X_train.drop(X_train.columns[[1, 8]], axis=1)
    X_test = X_test.drop(X_test.columns[[7]], axis=1)

    return X_train, y_train, X_test, submission

X_train, y_train, X_test, submission = data()
# Any results you write to the current directory are saved as output.


tpot = TPOTClassifier(verbosity=2, max_time_mins=2)
tpot.fit(X_train, y_train)

y_pred = tpot.predict(X_test)
y_pred = pd.DataFrame(y_pred) 
y_pred.columns = ['Survived'] 
submission = submission.join(y_pred) 

# Exporting dataset to csv
submission.to_csv("Titanic_Submission_new.csv", index=False, sep=',')