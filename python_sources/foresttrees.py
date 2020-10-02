# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:38:35 2019

@author: Vadivelan Palanichamy
"""


from sklearn.tree import DecisionTreeClassifier
import pandas as pd


dataset = pd.read_csv("../input/training.csv")

print ("--------------------------------")

X=dataset.drop("class", axis=1)
y= dataset["class"]


dataset = pd.read_csv("../input/testing.csv")
X_test=dataset.drop("class", axis=1)
y_test= dataset["class"]


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC


rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
        estimators=[ ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
        )
voting_clf.fit(X, y)


from sklearn.metrics import accuracy_score
for clf in (rnd_clf, svm_clf, voting_clf):
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    #print ("*************")
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


   
