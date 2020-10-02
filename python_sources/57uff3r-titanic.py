"""
My first very basic Kaggle kernel for beginners:
A alassifier  for Titanic contest on Kaggle.

Includes some very basic data preprocessing and 
GradintBoosting classifier.
"""

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


def conver_data(train_data):
    """
    Conver data from csv dataset to representation for
    classifier
    """
    # Filling missing fields
    train_data.Age = train_data.Age.fillna(train_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(train_data.Fare.median())

    max_pass_embarked = train_data.groupby('Embarked').count()['PassengerId']
    train_data.Embarked =  train_data.Embarked.fillna(
        max_pass_embarked[max_pass_embarked == max_pass_embarked.max()].index[0]
    )
    
    # Let's drop useless fields
    train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

    # Conver textual fields to numeric types
    label = LabelEncoder()
    dicts = {}

    label.fit(train_data.Sex.drop_duplicates()) # List of values for encoding
    dicts['Sex'] = list(label.classes_)
    train_data.Sex = label.transform(train_data.Sex) # Replace textual values with encoded values

    label.fit(train_data.Embarked.drop_duplicates())
    dicts['Embarked'] = list(label.classes_)
    train_data.Embarked = label.transform(train_data.Embarked)
    return train_data


# Let's clean up our training and test sets
final_result = DataFrame(test_data.PassengerId)
train_data = conver_data(train_data)
test_data = conver_data(test_data)


# 'Survived' it's our class label and we should remove
# this colun from taining set.
target = train_data.Survived
train_data = train_data.drop(['Survived'], axis=1) 

kfold = 5
results = {}
    
    
model_gb = GradientBoostingClassifier(learning_rate=0.09, n_estimators=50)
model_gb.fit(train_data, target)

final_result.insert(1,'Survived', model_gb.predict(test_data))
final_result.to_csv('./answer.csv', index=False)    
