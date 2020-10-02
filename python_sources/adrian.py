import math
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)


ids = test['PassengerId'].values

targets = train.Survived
train.drop('Survived',1,inplace=True)
    

# merging train data and test data
data = train.append(test)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)


Title_Dictionary = {

    "Dr": 3,
    "Capt" : 3,
    "Major" : 3,
    "Rev" : 3,
    "Col" : 3,
    "Jonkheer" : 4,
    "Don" : 4,
    "Sir" : 4,
    "Lady" : 4,
    "Master" : 5,
    "the Countess" : 4,
    "Dona" : 4,
    "Mme" : 2,
    "Mlle" : 2,
    "Ms" : 1,
    "Mr" : 0,
    "Mrs" : 1,
    "Miss" : 2

}


################################# Testing Set #################################\

# (Sex) set female to 0 and male to 1
data['Gender'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)

data.Embarked.fillna('S',inplace=True)

data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2 }).astype(int)


data['FamilySize'] = data["SibSp"] + data["Parch"] + 1;

data['NoFamily'] = data['FamilySize'].map(lambda s : 1 if s == 1 else 0)

data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

data['Title'] = data.Title.map(Title_Dictionary)

data["Age"] = data.groupby(['Title', 'Gender', 'SibSp', 'NoFamily'])['Age'].transform(lambda x: x.fillna(x.median()))

data['Infant'] = data['Age'].map(lambda s : 1 if s <= 7 else 0)

# replacing missing cabins with U (for Uknown)
data.Cabin.fillna('U',inplace=True)
    
# mapping each Cabin value with the cabin letter
data['Cabin'] = data['Cabin'].map(lambda c : c[0])

data['Cabin'] = data['Cabin'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'U': 8}).astype(int)


# (Age) fill missing values
# if len(data.Age[ data.Age.isnull()]) > 0:
#     median_age = data['Age'].dropna().median()
#     data.loc[(data.Age.isnull()), 'Age'] = median_age

# age_b_size = 10
# aux = 0
# age_medians = np.zeros(10)
# for a in range(1, 11):
#     age_medians[a - 1] = data.Age[(data.Age > aux) & (data.Age <= age_b_size * a)].median()
#     aux += age_b_size
# aux = 0
# data.Age = data.Age.map(lambda x: age_medians[math.ceil(x / age_b_size) - 1]).astype(int)



# (Fare) filling empty values with median of Pclass and doing binning
bin_size = 20
bin_max = 100
bins = [20, 40, 60, 80, 100]
median_fare = np.zeros(3)
for f in range(0,3):
    median_fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()
    if median_fare[f] > bin_max:
        median_fare[f] = bin_max

if len(data.Fare[ data.Fare.isnull()]) > 0:
    for f in range(0,3):
        data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = median_fare[f]
data.loc[data.Fare > bin_max, "Fare"] = bin_max

data.Fare = data.Fare.map(lambda x: bins[math.ceil(x / bin_size) - 1])

# Title


data = data.drop(['Name', 'Age', 'Sex', 'Ticket', 'PassengerId'], axis=1)

print(data)

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
    
print_full(data);    
    

features = list(data.columns)
data[features] = data[features].apply(lambda x: x/x.max(), axis=0)

train0 = pd.read_csv('../input/train.csv')
    
targets = train0.Survived

train_new = data.ix[0:890]
test_new = data.ix[891:]


    

################################# Classifier #################################

forest = RandomForestClassifier(max_features='sqrt')

parameter_grid = {
                 'max_depth' : [4,5,6,7,8],
                 'n_estimators': [200,210,240,250],
                 'criterion': ['gini','entropy']
                 }

cross_validation = StratifiedKFold(targets, n_folds=5)

grid_search = GridSearchCV(forest,
                          param_grid=parameter_grid,
                          cv=cross_validation)

grid_search.fit(train_new, targets)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

output = grid_search.predict(test_new).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)