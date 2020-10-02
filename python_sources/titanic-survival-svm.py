"""
Titanic prediction script for kaggle.

Author: Faiyaz Hasan
Data: October 13, 2016
"""
import pandas as pd
import numpy as np

from pandas import read_csv
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.lda import LDA

# load training and test datasets
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64},)
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},)

# column names, i.e. features based on which survival is given. Change all
# column names to lower case first for test, train and testsurvival datasets.
train.columns = [x.lower() for x in train.columns]
test.columns = [x.lower() for x in test.columns]
                
# test set passengerid
test_passengerid = test['passengerid']                

# remove inconsequential feature columns from test and training data sets.
# remove - passengerid, name, ticket, fare, cabin, embarked.
train.drop(['passengerid', 'name', 'ticket', 'fare', 'cabin', 
              'embarked'], inplace=True, axis=1)
test.drop(['passengerid', 'name', 'ticket', 'fare', 'cabin', 
             'embarked'], inplace=True, axis=1)

# Transform nominal (gender) values to binary numerical 0/1 representation
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# create training and test data frames with features only (X_train, X_test)
# and data frame with survival values only (y_train, y_test).
y_train = train.iloc[:, 0].values
X_train = train.iloc[:, 1:].values
X_test = test.iloc[:, 0:].values

# The age feature has some missing values. Impute the data using mean. 
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(X_train)
X_train = imr.transform(X_train)
X_test = imr.transform(X_test)

# Standardize data before fitting - I am not sure what it would do to 
# standardize the male female and passenger class data.
# It would have made sense to use the StandardScaler function from the 
# sklearn.preprocessing module but it leads to some issues. So, I will only
# normalize the age data which has a pretty big spread.
X_train_std = X_train
X_test_std = X_test
X_test_std[:, 1] = (X_test[:, 1]-X_test[:, 1].mean())/X_test[:, 1].std() 
X_train_std[:, 1] = (X_train[:, 1]-X_train[:, 1].mean())/X_train[:, 1].std()

# Fit data using hyperparameter tuned svm algorithm
svm = SVC(kernel='rbf', gamma=0.01, C=1000.0, random_state=0)
svm.fit(X_train_std, y_train)                             

# predicted survival values for the test set
y_pred = svm.predict(X_test_std)

# final submission dataframe
submission = pd.DataFrame({
        "PassengerId": test_passengerid,
        "Survived": y_pred
    })

submission.to_csv('titanic.csv', index=False)
