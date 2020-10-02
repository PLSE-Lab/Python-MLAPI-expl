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
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Reading train and test datasets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_passenger = df_test['PassengerId']

# Dropping some columns
df_train.drop(['Ticket'],axis=1, inplace=True)
df_train.drop(['Name'],axis=1, inplace=True)
df_train.drop(['PassengerId'],axis=1,inplace=True)

df_test.drop(['Ticket'],axis=1, inplace=True)
df_test.drop(['Name'],axis=1, inplace=True)
df_test.drop(['PassengerId'],axis=1,inplace=True)

y_train = df_train['Survived'].values
df_train.drop(['Survived'],axis=1, inplace=True)

# Feature handling, label encoding and filling missing values
genderEncoder = LabelEncoder()
genderEncoder.fit(df_train['Sex'].values)
df_train['Sex'] = genderEncoder.transform(df_train['Sex'].values)
df_test['Sex'] = genderEncoder.transform(df_test['Sex'].values)

df_train['Embarked'].fillna(df_train['Embarked'].mode().iloc[0], inplace=True)
df_test['Embarked'].fillna(df_test['Embarked'].mode().iloc[0], inplace=True)

embarkEncoder = LabelEncoder()
embarkEncoder.fit(df_train['Embarked'])
df_train['Embarked'] = embarkEncoder.transform(df_train['Embarked'].values)
df_test['Embarked'] = embarkEncoder.transform(df_test['Embarked'].values)

df_train['Age'].fillna(df_train['Age'].mean(),inplace=True)
df_test['Age'].fillna(df_test['Age'].mean(),inplace=True)

df_test['Fare'].fillna(df_test['Fare'].median(),inplace=True)

df_train.loc[df_train['Cabin'].isnull()==True,'Cabin'] = 0
df_train.loc[df_train['Cabin'] != 0,'Cabin'] = 1
df_test.loc[df_test['Cabin'].isnull()==True,'Cabin'] = 0
df_test.loc[df_test['Cabin'] != 0,'Cabin'] = 1

# Populating classification values on age and fare features
age_intervals = [5,10,15,20,30,40,50,60,70,80,90,120]
age_low = 0
for counter,age_up in enumerate(age_intervals):
    df_train.loc[(df_train.Age > age_low) & (df_train.Age <= age_up),'Age']=counter
    df_test.loc[(df_test.Age > age_low) & (df_test.Age <= age_up),'Age']=counter
    age_low = age_up

fare_intervals = [3,5,10,15,20,30,40,50,60,70,100,200,300,1000]
fare_low = 0
for counter,fare_up in enumerate(fare_intervals):
    df_train.loc[(df_train.Fare > fare_low) & (df_train.Fare <= fare_up),'Fare']=counter
    df_test.loc[(df_test.Fare > fare_low) & (df_test.Fare <= fare_up),'Fare']=counter
    fare_low = fare_up

x_train = df_train.values.astype('float32')
x_test = df_test.values.astype('float32')

# Model selection based on cross validation scores
#models = []
#models.append(('LogisticRegression',LogisticRegression()))
#models.append(('LinearDiscriminantAnalysis',LinearDiscriminantAnalysis()))
#models.append(('KNeighborsClassifier',KNeighborsClassifier()))
#models.append(('GaussianNB',GaussianNB()))
#models.append(('SVC',SVC()))
#models.append(('XGBClassifier',XGBClassifier()))

#for name,model in models:
#   kfold = StratifiedKFold(n_splits=5, shuffle=True)
#   results = cross_val_score(model,x_train,y_train,cv=kfold)
#   print("Model: {} Score: {}".format(name,results.mean()))

##################################################################################3   
##Model: LogisticRegression Score: 0.789984456272761
##Model: LinearDiscriminantAnalysis Score: 0.7923833608708744
##Model: KNeighborsClassifier Score: 0.7620456438731011
##Model: GaussianNB Score: 0.7800422868819524
##Model: SVC Score: 0.8159098896476934
##Model: XGBClassifier Score: 0.8215280090304695                 #BEST
###################################################################################

# XGBClassifier GridSearch trials
#model = XGBClassifier()
#kfold = StratifiedKFold(n_splits=5, shuffle=True)
#n_estimators = range(50,610,50)
#max_depth = range(1,12,2)
#learning_rate =[0.005,0.01, 0.015]
#parameters = dict(n_estimators=n_estimators,max_depth=max_depth,
#                  learning_rate=learning_rate)
#grid_search = GridSearchCV(estimator=model,param_grid=parameters,
#                           scoring="neg_log_loss",cv=kfold,verbose=1)
#grid_search.fit(x_train,y_train)

# Printing out results
#print("Best Train Score: {}".format(grid_search.best_score_))
#print("Best Train Params: {}".format(grid_search.best_params_))
#print("Accuracy Score: {}".format(grid_search.best_estimator_.score(x_train,y_train)))

#####################################################################################
##Best Train Score: -0.4172691165581429
##Best Train Params: {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 350}
##Accuracy Score: 0.8372615039281706
#####################################################################################

# Redefinition of the model based on best params
model = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=350)

# Fitting the model 
model.fit(x_train,y_train)
print("Accuracy: {}".format(model.score(x_train,y_train)))

# Preparing submission document
y_pred_test = model.predict(x_test)
submission = pd.DataFrame({"PassengerId": df_passenger,
                           "Survived": y_pred_test })
submission.to_csv("titanic_submission.csv", index=False)

