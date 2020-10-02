import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

def encode_gender(df):
    df.Sex = df.Sex.map({'female': 0, 'male': 1}).astype(int)
    return df

def add_age_ranges(df):
    df = floor_age(df)
    df['IsInfant'] = df.Age.map(lambda x: 1 if not math.isnan(x) and x == 0 else 0)
    df['IsChild'] = df.Age.map(lambda x: 1 if not math.isnan(x) and x > 0 and x <= 15 else 0)
    df.loc[(df.Name.str.find('Master') > -1) & (df.Age.isnull()), 'IsChild'] = 1
    df['IsAdult'] = df.Age.map(lambda x: 1 if not math.isnan(x) and x > 15 and x <= 45 else 0)
    df.loc[(df.Name.str.find('Master') == -1) & (df.Age.isnull()), 'IsAdult'] = 1
    df['IsMiddleAge'] = df.Age.map(lambda x: 1 if not math.isnan(x) and x > 45 and x <= 60 else 0)
    df['IsOld'] = df.Age.map(lambda x: 1 if not math.isnan(x) and x > 60 else 0)
    return df

def floor_age(df):
    df.Age = df.Age.map(lambda x: math.floor(x) if not math.isnan(x) else x)
    return df

def add_family_size(df):
    df['FamilySize'] = df.Parch + df.SibSp
    df['IsBigFamily'] = df.FamilySize.map(lambda x: 1 if x >= 4 else 0)
    return df

def drop_features(df):
    return df.drop(['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'FamilySize'], axis=1)

data_train = pd.read_csv('../input/train.csv', index_col=0)
data_test = pd.read_csv('../input/test.csv', index_col=0)

# remove outliers
data_train.drop([298, 499, 42, 178, 200, 313, 358, 773, 855, 631, 571, 484], inplace=True)

survived = data_train.Survived.values
del data_train['Survived']

data_train = encode_gender(data_train)
data_test = encode_gender(data_test)

data_train = add_age_ranges(data_train)
data_test = add_age_ranges(data_test)

data_train = add_family_size(data_train)
data_test = add_family_size(data_test)

data_train = drop_features(data_train)
data_test = drop_features(data_test)

parameters = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 10, None], 
    "min_samples_split": [2, 3, 5, 10],
    "min_samples_leaf": [1, 5, 10, 20]
}

model = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5).fit(data_train.values, survived)

predicted = model.predict(data_test.values)

submission = pd.DataFrame({'PassengerId': data_test.index, 'Survived': predicted})
submission.to_csv('submission.csv', index=False)