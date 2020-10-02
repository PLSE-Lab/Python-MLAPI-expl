import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import re

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#print train.isnull().sum()
#print test.isnull().sum()

#fig = plt.figure(figsize=(12,6))
#ax = fig.add_subplot(111)
#train.boxplot(column='Fare', by=['Embarked', 'Pclass'], ax=ax)
#plt.axhline(y=80, color='green')

#print test[test.Fare.isnull()][['Embarked', 'Pclass']]
#fig = plt.figure(figsize=(12,6))
#ax = fig.add_subplot(111)
#test[(test.Embarked=='S')&(test.Pclass==3)].Fare.hist(bins=100)
#print test[(test.Embarked=='S')&(test.Pclass==3)].Fare.value_counts().head()

#Fill Embarked
train.set_value(train.Embarked.isnull(), 'Embarked', 'C')
#Fill Fare
test.set_value(test.Fare.isnull(), 'Fare', 8.05)
#Fill Cabin
train.set_value(train.Cabin.isnull(), 'Cabin', 'U0')
test.set_value(test.Cabin.isnull(), 'Cabin', 'U0')

full = pd.concat([train, test], ignore_index=True)

#Create feature Namenum
namenum = full.Name.map(lambda x: len(re.split(' ', x)))
full.set_value(full.index, 'NameNum', namenum)
#Create feature Title
title = full.Name.map(lambda x: re.compile(', (.*?)\.').findall(x)[0])
title[title=='Mme'] = 'Mrs'
title[title.isin(['Ms','Mlle'])] = 'Miss'
title[title.isin(['Don', 'Jonkheer'])] = 'Sir'
title[title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
title[title.isin(['Capt', 'Col', 'Major', 'Dr', 'Officer', 'Rev'])] = 'Officer'
full.set_value(full.index, 'Title', title)
#Create feature Deck
deck = full.Cabin.map(lambda x: re.compile('([A-Za-z]+)').findall(x)[0])
deck = pd.factorize(deck)[0]
full.set_value(full.index, 'Deck', deck)
#Create feature Room
def getRoomNum(x):
    roomNum = re.compile('([0-9]+)').search(x)
    if roomNum:
        return int(roomNum.group())+1
    else:
        return 1
roomnum = full.Cabin.map(getRoomNum)
full.set_value(full.index, 'Room', roomnum)
#Create feature GroupNum
full.set_value(full.index, 'GroupNum', full.Parch+full.SibSp+1)
#Create featurn GroupSize
full.set_value(full.index, 'GroupSize', 'Middle')
full.set_value(full.GroupNum==1, 'GroupSize', 'Single')
full.set_value(full.GroupNum>4 , 'GroupSize', 'Large')
#Delete parameters
del deck, namenum, roomnum, title

#Normalize Fare
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
farenormal = pd.Series(scaler.fit_transform(full.Fare.reshape(-1,1)).reshape(-1))
full.set_value(full.index, 'NorFare', farenormal)
del farenormal

#Give train and test new feature
trainLength = train.shape[0]
def setValue(features):
    for ft in features:
        train.set_value(train.index, ft, full[:trainLength][ft].values)
        test.set_value(test.index, ft, full[trainLength:][ft].values)
setValue(['NameNum', 'Title', 'Deck', 'Room', 'GroupNum', 'GroupSize'])

#Predict Age
full.drop(labels=['PassengerId', 'Name', 'Cabin', 'Survived', 'Ticket', 'Fare'], axis=1, inplace=True)
#full.drop(labels=['Parch', 'Pclass', 'SibSp', 'GroupNum'], axis=1, inplace=True)
full = pd.get_dummies(full, columns=['Embarked', 'Sex', 'Title', 'GroupSize'])
X = full[~full.Age.isnull()].drop('Age', axis=1)
y = full[~full.Age.isnull()].Age
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=42)
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
def get_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_
xgb_reg1 = xgb.XGBRegressor(seed=42)
scoring = metrics.make_scorer(metrics.mean_absolute_error, greater_is_better=False)
# parameters = {
#               'learning_rate': np.linspace(0.05,0.2,5),
#               'max_depth': [3,4,5],
#               'reg_alpha': np.linspace(0.1,1.0,5),
#               'reg_lambda': np.linspace(1.0,3.0,5)
#               }
parameters = {
              'learning_rate': [0.05],
              'max_depth': [4],
              'reg_alpha': [0.1],
              'reg_lambda': [3.0]
              }
xgb_reg1 = get_model(xgb_reg1, parameters, X_train, y_train, scoring)
print(xgb_reg1)
print('Mean Absolute Error of test data: {}'.format(metrics.mean_absolute_error(y_test, xgb_reg1.predict(X_test))))
#joblib.dump(xgb_reg1, 'xgb_reg1.pkl')
#xgb_reg1 = joblib.load('xgb_reg1.pkl')
pred = xgb_reg1.predict(full[full.Age.isnull()].drop('Age', axis=1))
full.set_value(full.Age.isnull(), 'Age', pred)

#Normalize Age, NameNum, GroupNum
agenormal = pd.Series(scaler.fit_transform(full.Age.reshape(-1,1)).reshape(-1))
namenumnormal = pd.Series(scaler.fit_transform(full.NameNum.reshape(-1,1)).reshape(-1))
groupnumnormal = pd.Series(scaler.fit_transform(full.GroupNum.reshape(-1,1)).reshape(-1))
full.set_value(full.index, 'NorAge', agenormal)
full.set_value(full.index, 'NorNameNum', namenumnormal)
full.set_value(full.index, 'NorGroupNum', groupnumnormal)
del agenormal, namenumnormal, groupnumnormal
setValue(['NorFare', 'NorAge', 'NorNameNum', 'NorGroupNum'])

#Encode feature Sex
train.Sex = np.where(train.Sex=='female', 0, 1)
test.Sex = np.where(test.Sex=='female', 0, 1)

#Handle train and test dataframe
train.drop(labels=['PassengerId', 'Name', 'NameNum', 'Cabin', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)
train.drop(labels=['SibSp', 'Parch','GroupNum'], axis=1, inplace=True)
test.drop(labels=['Name', 'NameNum', 'Cabin', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)
test.drop(labels=['SibSp', 'Parch','GroupNum'], axis=1, inplace=True)
train = pd.get_dummies(train, columns=['Embarked', 'Pclass', 'Title', 'GroupSize'])
test = pd.get_dummies(test, columns=['Embarked', 'Pclass', 'Title', 'GroupSize'])
test['Title_Sir'] = pd.Series(0, index=test.index)

#Get CV
X = train.drop(['Survived'], axis=1)
y = train.Survived
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=42)

#Model
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', seed=77)
scoring = metrics.make_scorer(metrics.accuracy_score, greater_is_better=True)
# parameters = {
#               'learning_rate': np.linspace(0.05,0.2,5),
#               'max_depth': [3,4],
#               'n_estimators': [400,600,800,1000]
#               }
parameters = {
              'learning_rate': [0.05],
              'max_depth': [3],
              'n_estimators': [400]
              }
xgb_clf = get_model(xgb_clf, parameters, X_train, y_train, scoring)
print(xgb_clf)
print(metrics.accuracy_score(y_test, xgb_clf.predict(X_test)))

PassengerId = test.PassengerId
test.drop('PassengerId', axis=1, inplace=True)
tmp = test[['GroupSize_Large', 'GroupSize_Middle', 'GroupSize_Single']]
test.drop(['GroupSize_Large', 'GroupSize_Middle', 'GroupSize_Single'], axis=1, inplace=True)
test = test.join(tmp)
survived_prediction = xgb_clf.predict(test)
result = pd.DataFrame(columns=['PassengerId', 'Survived'])
result.PassengerId = PassengerId
result.Survived = survived_prediction
result.to_csv('results.csv', index=False)