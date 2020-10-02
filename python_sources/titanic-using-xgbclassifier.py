# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data_train=pd.read_csv('../input/train.csv')
data_test=pd.read_csv('../input/test.csv')

data_pred=data_train.Survived           #prediction column
data_train.drop(['Survived'],axis=1,inplace=True)     #removing prediction column from dataset

# data_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
# data_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

def name_transformer(data):
    data['Lname']=data.Name.apply(lambda x: x.split(' ')[0])
    data['NamePrefix']=data.Name.apply(lambda x: x.split(' ')[1])
    return data

def simplify_fares(data):
    data.Fare = data.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(data.Fare, bins, labels=group_names)
    data.Fare = categories
    return data

def simplify_age(data):
    data.Age=data.Age.fillna(-0.5)
    bins=(-1,0,20,28,38,81)
    group=['Unknown','Young','Young Adult','Adult','Senior']
    trimmed=pd.cut(data.Age,bins,labels=group)
    data.Age=trimmed
    return data

def drop_features(data):
    return data.drop(['Ticket','Name','Cabin','Embarked'],axis=1)

def transformations(data):
    data=name_transformer(data)
    data=simplify_fares(data)
    data=simplify_age(data)
    data=drop_features(data)
    return data

data_train=transformations(data_train)
data_test=transformations(data_test)

X_train,X_valid,y_train,y_valid= train_test_split(data_train,data_pred,train_size=0.8,test_size=0.2,random_state=0)

# #identifying numerical and categorical columns
# object_cols=[col for col in X_train.columns if  X_train[col].dtype == "object"]
# num_cols=[col for col in X_train.columns if  X_train[col].dtype == "float64" or X_train[col].dtype == "int64" ]

onehot=OneHotEncoder(handle_unknown='ignore',sparse=False)

list1=['Fare','Age','Sex','Lname','NamePrefix']
OH_train=pd.DataFrame(onehot.fit_transform(X_train[list1]))
OH_valid=pd.DataFrame(onehot.transform(X_valid[list1]))

OH_train.index=X_train.index
OH_valid.index=X_valid.index

num_X_train=X_train.drop(list1,axis=1)
num_X_valid=X_valid.drop(list1,axis=1)

final_X_train=pd.concat([num_X_train,OH_train],axis=1)
final_X_valid=pd.concat([num_X_valid,OH_valid],axis=1)

model=XGBClassifier(n_estimators=1000,learning_rate=0.06,verbose=True)
model.fit(final_X_train,y_train)
predictions=model.predict(final_X_valid)
print(mean_absolute_error(predictions,y_valid))

OH_test=pd.DataFrame(onehot.transform(data_test[list1]))
OH_test.index=data_test.index
num_X_test=data_test.drop(list1,axis=1)
final_X_test=pd.concat([num_X_test,OH_test],axis=1)
my_predictions=model.predict(final_X_test)
my_predictions
# #applying imputer and onehotencoder via pipeline
# numerical_transformer=SimpleImputer(strategy='mean')

# categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])

# preprocessor=ColumnTransformer(transformers=[('num',numerical_transformer,num_cols),
#                                              ('cat',categorical_transformer,object_cols)])

#defining model
# model= RandomForestClassifier(n_estimators=100,random_state=0)

# #fitting model and processor on data
# my_pipeline=Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

# my_pipeline.fit(X_train,y_train)

# preds=my_pipeline.predict(X_valid)

# my_prediction=my_pipeline.predict(data_test)

output=pd.DataFrame({'PassengerId':data_test.PassengerId,'Survived':my_predictions})

output.to_csv('gender_submission.csv',index=False)