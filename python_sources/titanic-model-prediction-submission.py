# coding: utf-8

# import libraries
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("../input/train.csv",header=0)
test_df = pd.read_csv("../input/test.csv",header=0)

# gender
df['Gender'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_df['Gender'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Gives the length of the name
df['Name_length'] = df['Name'].apply(len)
test_df['Name_length'] = test_df['Name'].apply(len)

# Feature that tells whether a passenger had a cabin on the Titanic
df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_df['Has_Cabin'] = test_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Define function to extract titles from passenger names
# shamefilly adapted from:
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)
alldata = [df, test_df]
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in alldata:
    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"
for dataset in alldata:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', \
    'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# replacing NaN in the Age with its median
train_median_ages = np.zeros((2,3))
test_median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        train_median_ages[i,j] = df[(df["Gender"] == i) & (df["Pclass"] == j + 1)].Age.dropna().median()
        test_median_ages[i,j] = test_df[(test_df["Gender"] == i) & (test_df["Pclass"] == j + 1)].Age.dropna().median()

# filled Age data column
df['AgeFill'] = df['Age']
test_df['AgeFill'] = test_df['Age']
for i in range(0,2):
    for j in range(0,3):
        df.loc[(df["Age"].isnull()) & (df["Gender"]==i) & (df["Pclass"] == j+1),'AgeFill'] = train_median_ages[i,j]
        test_df.loc[(test_df["Age"].isnull()) & (test_df["Gender"]==i) & (test_df["Pclass"] == j+1),'AgeFill'] = test_median_ages[i,j]

# child < 16
def ischild(age):
    if age < 16:
        return 1
    else:
        return 0

df['ischild'] = df['AgeFill'].apply(ischild)
test_df['ischild'] = test_df['AgeFill'].apply(ischild)

# replacing NaN in the Fare column by binning fares
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
def farebin(fare):
    if fare <= 10.0:
        return 0
    elif 10 < fare <= 20:
        return 1
    elif 20 < fare <= 30:
        return 2
    else:
        return 3

df['farebin'] = df.Fare.apply(farebin)
test_df['farebin'] = test_df.Fare.apply(farebin)

# replacing nan in the Embarked with the most ocurring one "S"
df["Embarked"] = df["Embarked"].fillna("S")

# convert Embarked into values
df["Embarked"] = df["Embarked"].map({"S": 2, "C": 1, "Q": 0}).astype(int)
test_df["Embarked"] = test_df["Embarked"].map({"S": 2, "C": 1, "Q": 0}).astype(int)

# alone or with somebody
df['withsomebody'] = df['SibSp'] + df['Parch']
df["isalone"] = df['withsomebody']
df["isalone"].loc[df['withsomebody'] > 0] = 0
df["isalone"].loc[df['withsomebody'] == 0] = 1
test_df['withsomebody'] = test_df['SibSp'] + test_df['Parch']
test_df['isalone'] = test_df['withsomebody']
test_df["isalone"].loc[test_df['withsomebody'] > 0] = 0
test_df["isalone"].loc[test_df['withsomebody'] == 0] = 1

# interaction between class and age
df['Age*Class'] = df["AgeFill"]*df["Pclass"]
test_df['Age*Class'] = test_df["AgeFill"]*test_df["Pclass"]

# interaction between class and child
df["Child*Class"] = df["ischild"]*df["Pclass"]
test_df["Child*Class"] = test_df["ischild"]*test_df["Pclass"]

# interaction between class and gender
df["Gender*Class"] = df["Gender"]*df["Pclass"]
test_df["Gender*Class"] = test_df["Gender"]*test_df["Pclass"]

# drop columns which we do not need
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Age', 'Fare', 'Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Age', 'Fare', 'Parch', 'SibSp'], axis=1)

# print(df.head())
# print(test_df.head())

# training and testing sets
y_train = df["Survived"]
X_train = df.drop(["PassengerId","Survived"], axis=1)
X_test = test_df.drop("PassengerId", axis=1).copy()

#print(X_train.head())

# zscoring
X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()

print(X_train.columns.values)
print('the number of features used for modeling: ' + str(X_train.shape[1]))

# pandas to numpy
y_train = y_train.values
X_train = X_train.values
X_test = X_test.values

# # XGB
# import xgboost as xgb
#
# params = {'eta': 0.04, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,\
#           'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

# y_pred = np.zeros((np.shape(X_test)[0],5))
# for i, (train_index, test_index) in enumerate(kf.split(X_train)):
#     print('kfold: ' + str(i))
#     Xcv_train, Xcv_test = X_train[train_index], X_train[test_index]
#     ycv_train, ycv_test = y_train[train_index], y_train[test_index]
    # dtrain = xgb.DMatrix(Xcv_train, ycv_train)
    # dtest = xgb.DMatrix(Xcv_test, ycv_test)
    # watchlist = [(dtrain, 'train'),(dtest, 'test')]
    # xgb_model = xgb.train(params, dtrain, 2000, watchlist, early_stopping_rounds=100,\
    #     feval=None, maximize=True, verbose_eval=100)
    # y_pred[:,i] = xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit+50)

# y_pred = np.mean(y_pred, axis=1)
# y_pred = np.rint(y_pred).astype(int)
# print("model prediction using cross-validation done")

# # random forest classifier
# from sklearn.ensemble import RandomForestClassifier

# random_forest = RandomForestClassifier(n_estimators=100)
# random_forest.fit(X_train, y_train)
# y_pred = random_forest.predict(X_test)

# keras libraries for deep learning
from keras.models import Sequential
from keras.layers import Dense, Dropout

# MLP model
def keras_model(n_hidden1=64,n_hidden2=64,n_hidden3=64):
    model = Sequential()
    model.add(Dense(n_hidden1, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(n_hidden2, activation='relu'))
    model.add(Dense(n_hidden3, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
        optimizer="rmsprop",
        metrics=["accuracy"])

    return model

# grid search to find the best combination of hyperparameters
from sklearn import grid_search
from keras.wrappers.scikit_learn import KerasClassifier

n_hidden1_cand = np.array([64,128,256])
n_hidden2_cand = np.array([64,128,256])
n_hidden3_cand = np.array([64,128,256])
param_grid = dict(n_hidden1=n_hidden1_cand, n_hidden2=n_hidden2_cand, n_hidden3=n_hidden3_cand)

model = KerasClassifier(build_fn=keras_model, nb_epoch=10, batch_size=128, verbose=0)
clf = grid_search.GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

clf.fit(X_train, y_train)
print(clf.best_params_)

y_pred = clf.predict(X_test)
y_pred = np.round(y_pred).astype(int)
y_pred = y_pred.ravel()

# for submission
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred
    })
submission.to_csv('titanic.csv', index=False)
