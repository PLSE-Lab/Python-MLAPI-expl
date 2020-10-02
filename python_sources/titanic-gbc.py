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
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import  GradientBoostingClassifier

data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")

def simpilify_ages(df):
   df.Age = df.Age.fillna(-0.5)
   bins = [-1,0,5,12,18,25,35,60,120]
   group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
   categories = pd.cut(df.Age,bins,labels = group_names)
   df.Age = categories
   return df

def modify_cabin(df):
   df.Cabin = df.Cabin.fillna('N')
   df.Cabin = df.Cabin.apply(lambda x:x[0])
   return df

def simpilify_fares(df):
   df.Fare = df.Fare.fillna(-0.5)
   bins = [-1,0,8,15,31,1000]
   group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
   categories = pd.cut(df.Fare, bins, labels = group_names)
   df.Fare = categories
   return df

def simpilify_name(df):
   df['LName'] = df.Name.apply(lambda x: x.split(' ')[0])
   df['Title'] = df.Name.apply(lambda x: x.split(' ')[0])
   return df   

def drop_feature(df):
   return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_feature(df):
   df = simpilify_ages(df)
   df = simpilify_fares(df)
   df = modify_cabin(df)
   df = simpilify_name(df)
   df = drop_feature(df)
   return df
train_data = transform_feature(data_train)
test_data = transform_feature(data_test)


#encoding features
from sklearn import preprocessing
def encoding_features(df_train,df_test):
   ''' The last part of the preprocessing phase is to normalize labels.
    The LabelEncoder in Scikit-learn will convert each unique string 
    value into a number, making out data more flexible for various algorithms.'''
    
   features = ['Age','Fare','Sex','LName','Cabin','Title']
   df_combined = pd.concat([df_train[features],df_test[features]])
   for feature in features:
      le = preprocessing.LabelEncoder()
      le = le.fit(df_combined[feature])
      df_train[feature] = le.transform(df_train[feature])
      df_test[feature] = le.transform(df_test[feature])
   return df_train,df_test

train_data,test_data = encoding_features(train_data,test_data)

#creating testing and training set

X_all = train_data.drop(['Survived', 'PassengerId'], axis=1)
y_all = train_data['Survived']
num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = GradientBoostingClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['friedman_mse', 'mse', 'mae'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)



from sklearn.model_selection import KFold

def run_kfold(clf):
    kf = KFold(n_splits=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X_all):
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)


#Predict from test datasets
ids = test_data['PassengerId']
predictions = clf.predict(test_data.drop('PassengerId', axis=1))


output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
# output.to_csv('titanic-predictions.csv', index = False)
output.head()
