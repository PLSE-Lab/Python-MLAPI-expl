import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import re as re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from subprocess import check_output

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics.classification import classification_report,accuracy_score

import xgboost as xgb

sns.set(style='white', context='notebook', palette='deep')
print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train_rows= train.shape[0]
test_rows = test.shape[0]
PassengerIDs = test["PassengerId"].values

complete_data = [train,test]
for data in complete_data:
    data["FamilySize"] = data["Parch"]+data["SibSp"]+1
    data["IsAlone"] = 0
    data.loc[data["FamilySize"] == 1,'IsAlone'] = 1
    #filling NA values in Age Fare Embarked
    mean_age = data["Age"].mean()
    std_age = data["Age"].std()
    mean_fare = data["Fare"].mean()
    data.loc[np.isnan(data["Age"]),"Age"] = np.random.randint(mean_age-std_age,mean_age+std_age,size=data["Age"].isnull().sum())    
    data.loc[np.isnan(data["Fare"]),"Fare"] = mean_fare
    data["Embarked"] = data.Embarked.fillna("S")
    data["Sex"] = data["Sex"].map({'female':0,'male':1}).astype(int)
    data["Embarked"] = data["Embarked"].map({'S':0,'C':
        1,'Q':2}).astype(int)
    
for data in complete_data:
    data["CategoricalFare"] = pd.qcut(train["Fare"],3,labels=[0,1,2]) 
    data["CategoricalAge"] = pd.cut(train["Age"],5,labels=[0,1,2,3,4])
    
def get_tit_reg(name):
    query = re.search('([A-Za-z]+)\.',name)
    if query:
        return query.group(1)
    return ""
for data in complete_data:
    data["Title"] = data["Name"].apply(get_tit_reg)
    data["Title"] = data["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data["Title"] = data["Title"].map({
        "Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare":5
    }).astype(int)
    data["Title"] = data["Title"].fillna(0)

for data in complete_data:
    data = data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1,inplace=True)
    #print(data.columns)
Y_train = train["Survived"].values
X_train = train.drop("Survived",axis=1).values
X_test = test.values

xtrainSP,xcvSP,ytrainSP,ycvSP = train_test_split(
    X_train,
    Y_train,
    test_size=0.4,
    random_state=1
    )
forest_clf = RandomForestClassifier(max_depth = 6,min_samples_split=2,n_estimators=100,random_state=1)
forest_clf = forest_clf.fit(xtrainSP,ytrainSP)

print("Score for training set",forest_clf.score(xtrainSP,ytrainSP))
print("Score for cross validation",forest_clf.score(xcvSP,ycvSP))

predictions = forest_clf.predict(X_test)

#gbm = xgb.XGBClassifier(max_depth=6,n_estimators=575,learning_rate=0.05,gamma=1,subsample=0.8,objective='binary:logistic')
#gbm = gbm.fit(xtrainSP,ytrainSP)

#print("Score for training set",gbm.score(xtrainSP,ytrainSP))
#print("Score for cross validation",gbm.score(xcvSP,ycvSP))

#predictions = gbm.predict(X_test)
folds = 5
kfold  = KFold(n_splits=folds)
#print([forest_clf.fit(X_train[train], Y_train[train]).score(X_train[cv],Y_train[cv]) for train,cv in kfold.split(X_train)])
#max_score_i = 0
#max_score = 0;
#for i ,(train,cv) in enumerate(kfold.split(X_train)):
#    curr_score = forest_clf.fit(X_train[train],Y_train[train]).score(X_train[cv],Y_train[cv])
#    if(max_score < curr_score):
#        max_score = curr_score
#        max_score_i = i
#        print(curr_score,":",i)

class clfcomponent(object):
    def __init__(self,classifier,params=None):
        self.clf = classifier(**params)
    def train(self,x_train,y_train):
        return self.clf.fit(x_train,y_train)
    def predict(self,x):
        return self.clf.predict(x)

def get_predictions(clf,x_train,y_train,x_test):
    kf_train = np.zeros((train_rows,))
    kf_test = np.zeros((test_rows,))
    kf_test_clf = np.empty((folds, test_rows))
    
    for i, (train,cv) in enumerate(kfold.split(x_train)):
        x_kf_train = x_train[train]
        y_kf_train = y_train[train]
        x_kf_cv = x_train[cv]
        
        clf.train(x_kf_train,y_kf_train)
        
        kf_train[cv] = clf.predict(x_kf_cv)
        kf_test_clf[i,:] = clf.predict(x_test)
    kf_test[:] = kf_test_clf.mean(axis=0)
    return kf_train.reshape(-1,1),kf_test.reshape(-1,1)

rf_params = {
    'n_jobs': 4,
    'n_estimators': 575,
     'warm_start': True, 
     'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'min_samples_split':2,
    'random_state':1,
    'verbose': 1
}
et_params = {
    'n_jobs': -1,
    'n_estimators':575,
    #'max_features': 0.5,
    'max_depth': 5,
    'min_samples_leaf': 3,
    'verbose': 1
}
ada_params = {
    'n_estimators': 575,
    'learning_rate' : 0.95
}

gb_params = {
    'n_estimators': 575,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 3,
    'verbose': 1
}
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
rfc = clfcomponent(classifier=RandomForestClassifier,params=rf_params)
rf_kf_train, rf_kf_test = get_predictions(rfc,X_train,Y_train,X_test)
print(rf_kf_train.shape,rf_kf_test.shape)
gbc = clfcomponent(classifier=GradientBoostingClassifier,params=gb_params)
gb_kf_train,gb_kf_test = get_predictions(gbc,X_train,Y_train,X_test)
etc = clfcomponent(classifier=ExtraTreesClassifier,params=et_params)
et_kf_train,et_kf_test = get_predictions(etc,X_train,Y_train,X_test)
abc = clfcomponent(classifier=AdaBoostClassifier,params=ada_params)
ab_kf_train,ab_kf_test = get_predictions(abc,X_train,Y_train,X_test)
svc = clfcomponent(classifier=SVC,params=svc_params)
sv_kf_train,sv_kf_test = get_predictions(svc,X_train,Y_train,X_test)

meta_X_train = np.concatenate((rf_kf_train,gb_kf_train,et_kf_train,ab_kf_train,sv_kf_train),axis=1)
meta_X_test = np.concatenate((rf_kf_test,gb_kf_test,et_kf_test,ab_kf_test,sv_kf_test),axis=1)

print(meta_X_train.shape,meta_X_test.shape)

gbm = xgb.XGBClassifier(max_depth=6,n_estimators=575,learning_rate=0.05,gamma=1,subsample=0.8,objective='binary:logistic')
gbm = gbm.fit(meta_X_train,Y_train)
predictions = gbm.predict(meta_X_test)

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerIDs,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

