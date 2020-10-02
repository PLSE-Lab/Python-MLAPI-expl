import math
import pandas as pd
import numpy as np
import csv as csv
import sklearn.ensemble as sk
import sklearn.cross_validation as sk2
import sklearn.model_selection as ms

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)

targets = train.Survived
train.drop('Survived', 1, inplace=True)

# merging train data and test data for future feature engineering
data = train.append(test)
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)

################################# Training Set #################################
# (Sex) set female to 0 and male to 1
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Drop Ticket Attribute
data.drop('Ticket', 1, inplace=True)


# Extract titles
data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())


Title_Dictionary = {

    "Dr": "Officer",
    "Capt" : "Officer",
    "Major" : "Officer",
    "Rev" : "Officer",
    "Col" : "Officer",
    "Jonkheer" : "Royalty",
    "Don" : "Royalty",
    "Sir" : "Royalty",
    "Lady" : "Royalty",
    "Master" : "Royalty",
    "the Countess" : "Royalty",
    "Dona" : "Royalty",
    "Mme" : "Mrs",
    "Mlle" : "Miss",
    "Ms" : "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss"

}


# map titles
data['Title'] = data.Title.map(Title_Dictionary)


# family size calculated
data["Family Size"] = data["SibSp"] + data["Parch"] + 1

# since one ticket is bought for whole family, this has to be accounted for
data["PricePerPerson"] = data["Fare"] / data["Family Size"]

#fill missing price per person value by median of different groups
data["PricePerPerson"] = data.groupby(['Title', 'Pclass', 'Embarked'])['PricePerPerson'].transform(lambda x: x.fillna(x.median()))

data.drop('Fare', axis=1, inplace=True)

#fill missing age value by median of different groups
data["Age"] = data.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))

data.drop('Name', axis=1, inplace=True)

# encoding in dummy variable
titles_dummies = pd.get_dummies(data['Title'], prefix='Title')
data = pd.concat([data, titles_dummies], axis=1)

# removing the title variable
data.drop('Title', axis=1, inplace=True)


# (Embarked) fill empty values with mode
if len(data.Embarked[ data.Embarked.isnull()]) > 0:
    data.Embarked[ data.Embarked.isnull()] = data.Embarked.dropna().mode().values


# mapping each Cabin value with the cabin letter
data['Embarked'] = data['Embarked'].map(lambda c: c[0])

# encoding in dummy variable
embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
data = pd.concat([data, embarked_dummies], axis=1)

# removing the title variable
data.drop('Embarked', axis=1, inplace=True)




# replacing missing cabins with U (for Uknown)
data.Cabin.fillna('U', inplace=True)

# mapping each Cabin value with the cabin letter
data['Cabin'] = data['Cabin'].map(lambda c: c[0])

# encoding in dummy variable
cabin_dummies = pd.get_dummies(data['Cabin'], prefix='Cabin')
data = pd.concat([data, cabin_dummies], axis=1)

# removing the title variable
data.drop('Cabin', axis=1, inplace=True)


features = list(data.columns)
features.remove('PassengerId')
data[features] = data[features].apply(lambda x: x/x.max(), axis=0)






train0 = pd.read_csv('../input/train.csv')

targets = train0.Survived
train = data.ix[0:890]
test = data.ix[891:]


print(targets.head())
print(train.head())

forest = sk.RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = sk2.StratifiedKFold(targets, n_folds=5)

grid_search = ms.GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)

grid_search.fit(train, targets)


print("successsss")

output = grid_search.predict(test).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)





