import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

def getSex(x):
    if x["Sex"]==np.nan:
        if "Mr." in x["Name"]:
            x["Sex"]="male"
        else:
            x["Sex"]="female"
    return

train.apply(getSex,axis=1)
test.apply(getSex,axis=1)

train = train.drop("Name",1)
test = test.drop("Name",1)
train = train.drop("Ticket",1)
test = test.drop("Ticket",1)
train = train.drop("Cabin",1)
test = test.drop("Cabin",1)

trainS = train.Survived
train = train.drop("Survived",1)

train = train.drop("PassengerId",1)
testId = test.PassengerId
test = test.drop("PassengerId",1)

def getEmb(x):
    if x["Embarked"]==np.nan:
        x["Embarked"]="S"
    return

train.apply(getEmb,axis=1)
test.apply(getEmb,axis=1)


numEmb = {"C":0,"Q":1,"S":2}
train.Embarked = train.Embarked.apply(numEmb.get).astype("category")
test.Embarked = test.Embarked.apply(numEmb.get).astype("category")


numSex = {"male":0,"female":1}
train.Sex = train.Sex.apply(numSex.get).astype("category")
test.Sex = test.Sex.apply(numSex.get).astype("category")
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
train = imp.fit_transform(train)
test = imp.fit_transform(test)

#bclf = RandomForestClassifier(n_estimators=100,random_state=1);

#clf = AdaBoostClassifier(base_estimator=bclf,n_estimators=1000)
clf = GradientBoostingClassifier()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi = SelectKBest(chi2, k=3)
train = chi.fit_transform(train, trainS)
test = chi.transform(test)

clf.fit(train,trainS)
dt = clf.predict(test);

res = pd.DataFrame()
res["PassengerId"] = testId
res["Survived"] = dt

res.to_csv("titanic_rf.csv",index=False)
