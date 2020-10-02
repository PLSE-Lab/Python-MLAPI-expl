# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 22:45:08 2017

@author: sonaa
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import log_loss, accuracy_score


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df['Title'] = ''
train_df['LastName'] = ''
test_df['Title'] = ''
test_df['LastName'] = ''

title_map = {}
title_map['Mr.'] = 'Mr'
title_map['Mrs.'] = 'Mrs'
title_map['Miss.'] = 'Ms'
title_map['Mme.'] = 'Ms'
title_map['Mlle.'] = 'Ms'
title_map['Lady.'] = 'Ms'
title_map['Master.'] = 'Master'
title_map['Capt.'] = 'Mr'
title_map['Col.'] = 'Mr'
title_map['Don.'] = 'Mr'
title_map['Dr.'] = 'Mr'
title_map['Major.'] = 'Mr'
title_map['Rev.'] = 'Mr'
title_map['Sir.'] = 'Mr'
train_df.loc[:, 'Cabin'] = train_df.loc[:, 'Cabin'].fillna('0')
test_df.loc[:, 'Cabin'] = test_df.loc[:, 'Cabin'].fillna('0')
train_df['Deck'] = '0'
test_df['Deck'] = '0'
for i in train_df.index:
    s =  train_df.Name[i].split(',')
    f = s[1]
    l = s[0]
    train_df.loc[i, 'LastName'] = l
    t = f.split(' ')[1]
    train_df.loc[i, 'Deck'] = train_df.loc[i, 'Cabin'][0]
    if t in title_map.keys():
        train_df.loc[i, 'Title'] = title_map[t]
    else:
        train_df.loc[i, 'Title'] = 'Rare'
for i in test_df.index:
    s =  test_df.Name[i].split(',')
    f = s[1]
    l = s[0]
    test_df.loc[i, 'LastName'] = l
    t = f.split(' ')[1]
    test_df.loc[i, 'Deck'] = test_df.loc[i, 'Cabin'][0]
    if t in title_map.keys():
        test_df.loc[i, 'Title'] = title_map[t]
    else:
        test_df.loc[i, 'Title'] = 'Rare'

train_df["TicketNumber"] = train_df["Ticket"].str.extract('(\d{2,})')
test_df["TicketNumber"] = test_df["Ticket"].str.extract('(\d{2,})')

filler = {}
for i in train_df.columns:
    if not train_df[i].dtype == object:
        filler[i] = np.nanmedian(train_df[i])
        train_df[i] = train_df[i].fillna(filler[i])
        if not i == 'Survived':
            test_df[i] = test_df[i].fillna(filler[i])
    else:
        train_df[i] = train_df[i].fillna('0')
        test_df[i] = test_df[i].fillna('0')
        lbl = LabelEncoder()
        train_df[i] = lbl.fit_transform(train_df[i])
        test_df[i] = lbl.fit_transform(test_df[i])
        
relevant_columns = ['Pclass','Sex','Age','SibSp','Parch','TicketNumber','Fare','Title','Deck','Embarked']
X = train_df[relevant_columns]    
X_test = test_df[relevant_columns]    
scaler = preprocessing.RobustScaler()
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)
Y = train_df['Survived']

m1 = linear_model.LogisticRegressionCV()
m2 = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
m3 = KNeighborsClassifier(3)
m4 = SVC(kernel="linear", C=0.025)
m5 = SVC(gamma=2, C=1)
m6 = DecisionTreeClassifier(max_depth=5)
m7 = AdaBoostClassifier()
m8 = QuadraticDiscriminantAnalysis()
m9 = LinearDiscriminantAnalysis()
m10 =  BaggingClassifier(AdaBoostClassifier(),max_samples=0.5, max_features=0.5)

from sklearn.ensemble import VotingClassifier
m11 = VotingClassifier(estimators=[
        ('m1', m1), ('m2', m2), ('m6', m6), ('m7', m7)], voting='hard')
m12 = GradientBoostingClassifier()
models = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12]
for i in range(len(models)):
    model = models[i]
    kf = KFold(X.shape[0], n_folds=3, random_state=1)
    scores = []
    for train_i, test_i in kf:
        model.fit(X[train_i], Y[train_i])
        Y_pred = model.predict(X[test_i])
        scores.append(accuracy_score(Y[test_i], Y_pred))
    print (i+1, np.mean(scores))

pred = m12.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })
submission.to_csv("titanic_submission.csv", index=False)





    
