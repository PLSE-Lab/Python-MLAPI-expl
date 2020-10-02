# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rcf
from sklearn.model_selection import cross_val_score
import csv

def train():
    data = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
    targets = data['Survived']
    del data['Survived']
    data, ids, dummies = process(data)
    #depth_candidates = [5,10,15,20]
    #n_estimators_candidates = [5, 10, 15, 20, 25, 30, 35, 40, 100, 200]
    clf = rcf(max_depth = 10, n_estimators = 100)
    clf.fit(dummies,targets)
    sc = cross_val_score(clf, dummies, targets)
    print(sc.mean())
    return data,clf, targets, dummies

def test(clf, dummies):
    data = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
    data, ids, dummiesT = process(data)
    #######################
    #Courtesy of Hume on Stackoverflow. Makes sure the test and train matrix have
    #the same columns
    col_difference = np.setdiff1d(dummies.columns, dummiesT.columns)
    for c in col_difference:
        dummiesT[c] = 0
    dummiesT = dummiesT[dummies.columns]
    ######################
    survive_list = clf.predict(dummiesT)
    return survive_list, ids, data, dummiesT

def cabinExtract(vec):
    #separates and bags the cabin number and letter to reduce the feature space
    numVec = []
    for i in range(len(vec)):
        try:
            numbag = num_stract(vec[i]) // 10
            numVec.append(numbag)
            vec.set_value(i,ord(vec[i][0])-65)
        except Exception:
            vec.set_value(i, 0)
            numVec.append(0)
    return vec, numVec

#Extracts numbers from the Cabin strings    
def num_stract(string):
    ret = ""
    for i in string[1:]:
        if i.isalpha():
            return int(ret)
        ret += i
    return int(ret)

#Removes unneeded features, replaces NaN values and one hot encodes categorical features
def process(data):
    ids = data['PassengerId']
    del data['PassengerId']
    del data['Name']
    mode_vec = data.mode()
    names = list(data)
    cab_vec, num_vec = cabinExtract(data['Cabin'])
    #For all missing values, use the mean for age and fare. Use the mode for other categories
    #This would be the next area to improve on
    for i in range(len(data)):
        if np.isnan(data.loc[i]['Age']):
            data.set_value(i,'Age',data['Age'].mean())
        if np.isnan(data.loc[i]['Fare']):
            data.set_value(i,'Fare',data['Fare'].mean())
        for k in range(len(data.loc[i])):
            if pd.isnull(data.iloc[i][k]):
                data.set_value(i,names[k], mode_vec[names[k]].loc[0])
    del data['Ticket']
    data['Cabin'] = cab_vec
    data.loc[data['Sex']=="male", 'Sex'] = 1
    data.loc[data['Sex']=="female", 'Sex'] = 0
    data['Cabin Number'] = num_vec
    dummies = pd.get_dummies(data, columns = ['Cabin Number','Embarked', 'Cabin', 'Parch'])
    return data, ids, dummies

#function for writing results to csv
def write_submission(survive_list, ids):
    with open('submissionT.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator = '\n')
        writer.writerow(('PassengerId','Survived'))
        for i in range(len(ids)):
            writer.writerow((str(ids[i]), str(survive_list[i])))
            
res,clf, targets, dummies= train()
survive_list, ids, data, dummiesT = test(clf, dummies)
#write_submission(survive_list, ids)