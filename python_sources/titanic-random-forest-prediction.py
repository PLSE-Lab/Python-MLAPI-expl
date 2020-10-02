# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:10:36 2017

Kaggle project - Titanic Passenger Survivability
Result
@author: Ade Kurniawan
Note: My work here was inspired by Ahmed Besbes (http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html)
"""

import pandas as pd
import numpy as np

def combine_data():
    # load data from train.csv and test.csv
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    # extract response from train
    target = train.Survived
    
    # drop Survived column from train
    train.drop('Survived', axis = 1, inplace = True) # argument 1 in drop() indicates column dropping
    
    # combine train and test data
    combined = pd.concat([train, test]).reset_index().drop('index', axis = 1)
    
    return combined, target

def get_title(combined):
	
	combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
	title_dict = {
	"Capt": "Officer",
	"Col": "Officer",
	"Major": "Officer",
	"Jonkheer": "Royalty",
	"Don": "Royalty",
	"Sir" : "Royalty",
	"Dr": "Officer",
	"Rev": "Officer",
	"the Countess":"Royalty",
	"Dona": "Royalty",
	"Mme": "Mrs",
	"Mlle": "Miss",
	"Ms": "Mrs",
	"Mr" : "Mr",
	"Mrs" : "Mrs",
	"Miss" : "Miss",
	"Master" : "Master",
	"Lady" : "Royalty"
	}
	
	combined['Title'] = combined.Title.map(title_dict)
	return combined

def fill_age(combined, gm_train, gm_test):
    grouped_median_train = gm_train
    grouped_median_test = gm_test
    
    # filling missing ages in train data
    combined.iloc[:891].Age = combined.apply(lambda r: grouped_median_train.loc[r.Sex,
             r.Pclass, r.Title].Age if np.isnan(r.Age) else r.Age, axis = 1)
    # filling missing ages in test data
    combined.iloc[891:].Age = combined.apply(lambda r: grouped_median_test.loc[r.Sex,
                 r.Pclass, r.Title].Age if np.isnan(r.Age) else r.Age, axis = 1)
    
    return combined

def fill_fare(combined, gm_train, gm_test):
    grouped_median_train = gm_train
    grouped_median_test = gm_test
    
    # filling missing ages in train data
    combined.iloc[:891].Fare = combined.apply(lambda r: grouped_median_train.loc[r.Sex,
             r.Pclass, r.Title].Fare if np.isnan(r.Fare) else r.Fare, axis = 1)
    # filling missing ages in test data
    combined.iloc[891:].Fare = combined.apply(lambda r: grouped_median_test.loc[r.Sex,
                 r.Pclass, r.Title].Fare if np.isnan(r.Fare) else r.Fare, axis = 1)
    
    return combined

def process_ticket(combined):
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

def process_family(X):
    X['FamilySize'] = X['Parch'] + X['SibSp'] + 1
	# introducing other features based on the family size
    def famSize(x):
        if x == 1:
            return 1	# means that he/she was alone
        elif 2 <= x <= 4:
            return 2	# means that he/she had a pretty small family
        else:
            return 3	# means that he/she had a large family (more than 4 members)
    X['FamilySize'] = X['FamilySize'].map(lambda s: famSize(s))
    return X

def process_data():
    X, y = combine_data()
    
    # extracting passengers' titles from column Name
    X = get_title(X)
    
    # obtaining median of training data based on certain features
    grouped_train = X.head(891).groupby(['Sex','Pclass','Title'])
    grouped_median_train = grouped_train.median()
    
    # obtaining median of test data based on certain features
    grouped_test = X.iloc[891:].groupby(['Sex','Pclass','Title'])
    grouped_median_test = grouped_test.median()
    
    # processing age
    X = fill_age(X, grouped_median_train, grouped_median_test)
    
    # processing fare
    X = fill_fare(X, grouped_median_train, grouped_median_test)
    
    # processing embarked
    X['Embarked'].fillna('S', inplace = True)
    
    # processing cabin
    X['Cabin'].fillna('U', inplace = True)
    X['Cabin'] = X['Cabin'].map(lambda c: c[0])
    
    # processing family
    X = process_family(X)
    
    X_train = X.iloc[:891]
    X_test = X.iloc[891:]
    
    return X_train, X_test, y

def encode(X_train, X_test, cols, dummify = True):
    from sklearn.preprocessing import LabelEncoder
    X = pd.concat([X_train, X_test])
    enc = {var: LabelEncoder() for var in cols}
    
    for var in cols:
        if not dummify:
            X[var] = enc[var].fit_transform(X[var])
        # dummify the column (if needed by the user)
        elif dummify:
            temp = pd.get_dummies(X[var], prefix = var)
            X = pd.concat([X, temp], axis = 1).drop(var, axis = 1)
        else:
            print('dummify argument must be either y or n')
            
    X_train = X.iloc[:891]
    X_test = X.iloc[891:]
    
    return X_train, X_test, enc if not dummify else None
X_train, X_test, y_train = process_data()

# encode the data
X_train, X_test, enc = encode(X_train,
                        X_test, 
                        ['Sex','Cabin','Embarked','Title'],
                        dummify = True)
# beginning to train the model, drop some variables
PassengerId = X_test.PassengerId
X_train.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = True)
X_test.drop(['PassengerId','Name','Ticket'], axis = 1, inplace = True)

# importing some useful machine learning libraries
from sklearn.ensemble import RandomForestClassifier

params = {'bootstrap': True, 'max_depth': 6, 
          'max_depth': 6, 
          'max_features': 'log2',
          'min_samples_leaf': 3,
          'min_samples_split': 2,
          'n_estimators': 10}

clf = RandomForestClassifier(**params)

clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)

y_predicted = pd.DataFrame(clf.predict(X_test), 
                           index = PassengerId, 
                           columns = ['Survived'])

y_predicted.to_csv('submission.csv', header = True)