# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------
Kaggle Titanic Exploration
Author: Raj Saha
-----------------------------------------------------------------------------
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn import preprocessing
import seaborn as sns


#os.chdir("C:\SparkCourse\kaggletitanic")

"""
Data Engineering and Analysis
"""
#Load the dataset

#train_data = pd.read_csv("train.csv")
train_data = pd.read_csv("../input/train.csv")

#test_data = pd.read_csv("test.csv")
test_data = pd.read_csv("../input/test.csv")

"""
Data Transformations

Let us do the following transformations

1. Convert Date into separate columns - year, month, week
2. Convert all non numeric data to numeric
"""


# Dropping unnecessary columns
train_data = train_data.drop(['Name','Ticket','Embarked','Cabin'], axis=1)
test_data = test_data.drop(['Name','Ticket','Embarked','Cabin'], axis=1)
test_data.dtypes

#Check if Age has Null values - either sum or any function can be used
train_data["Age"].isnull().sum()
test_data["Age"].isnull().any()

#replace null age with median values
train_data["Age"]=train_data["Age"].fillna(train_data["Age"].median())
test_data["Age"]=test_data["Age"].fillna(train_data["Age"].median())

#Check if Age Null values are gone
train_data["Age"].isnull().any()
test_data["Age"].isnull().any()


#convert sex to numeric
train_data.loc[train_data["Sex"] == "male", "Sex"] = 0
train_data.loc[train_data["Sex"] == "female", "Sex"] = 1

test_data.loc[test_data["Sex"] == "male", "Sex"] = 0
test_data.loc[test_data["Sex"] == "female", "Sex"] = 1

#check if Sex has any null values
train_data["Sex"].isnull().any()
test_data["Sex"].isnull().any()

#convert sex to numeric type if it has no nulls
train_data["Sex"] = train_data["Sex"].astype(int)
test_data["Sex"] = train_data["Sex"].astype(int)

train_data["Age"] = train_data["Age"].astype(int)
test_data["Age"] = train_data["Age"].astype(int)

train_data["Fare"] = train_data["Fare"].astype(int)
test_data["Fare"] = train_data["Fare"].astype(int)


#Convert all strings to equivalent numeric representations
#to do correlation analysis
for f in train_data.columns:
    if train_data[f].dtype=='object':
        print(f)
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train_data[f].values)+list(test_data[f].values))
        train_data[f]=lbl.transform(list(train_data[f].values))
        test_data[f]=lbl.transform(list(test_data[f].values))


train_data.dtypes
test_data.dtypes    
#Find correlations
#train_data.corr()


"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = train_data.iloc[:,2:]
targets = train_data.iloc[:,1]


predictors.dtypes
targets.dtypes



pred_test = test_data.iloc[:,1:]
pred_test.dtypes


#Build model on training data
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=5)
classifier=classifier.fit(predictors,targets)

predictions=classifier.predict(pred_test)




"""
Saving the results in submission file
"""
#sample=pd.read_csv("sample_submission.csv")
#sample.QuoteConversion_Flag = predictions
#sample.to_csv("sample_submission_filled.csv")
test_data["Survived"]=predictions
test_data[["PassengerId","Survived"]].to_csv("titanic__myfirst_randomforest.csv", index=False)

