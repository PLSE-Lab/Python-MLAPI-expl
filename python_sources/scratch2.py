# -*- coding: utf-8 -*-
"""
Created on Mon May 30 20:36:12 2016

@author: ak_yang
"""

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import xgboost as xgb
#import seaborn as sns
#sns.set(style="whitegrid", color_codes=True)
#%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier   # used to get feature importance
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

#%%
from sklearn.model_selection import GridSearchCV


#%%
#from sklearn.modelselection import KFold, cross_val_score

# load both training set and test set
#path=os.getcwd()
#train_set = pd.read_csv(path+"\\train.csv")
#test_set = pd.read_csv(path+"\\test.csv")
train_set = pd.read_csv("../input/train.csv")
test_set  = pd.read_csv("../input/test.csv")
# Add a empyt Survived column on the test set
# Later on we will separate two sets using this column
test_set['Survived'] = np.NaN
alldata = pd.concat([train_set, test_set], ignore_index=True)

drop_list = ['Cabin']
# Create a Title column
alldata['Title'] = alldata['Name'].apply(lambda x: re.sub('(.*, )|(\\..*)','', x))

# Create a new Age2 column
alldata['Age2'] = alldata['Age']

# There are 8 empty Age column with title Master in the Name column
# We will randomly pick between 0 and 14 as the ages for these 4 records
kids_no_age = (alldata['Title'] == 'Master') & alldata.Age.isnull()
alldata.ix[kids_no_age, 'Age2'] = np.random.randint(0, 14, 8)

# Only one Dr has missing value, lets use the mean from doctor with ages
dr_no_age = (alldata['Title'] == 'Dr') & alldata.Age.isnull()
dr_with_age = (alldata['Title'] == 'Dr') & alldata.Age.notnull()
alldata.ix[dr_no_age, 'Age2'] = alldata[dr_with_age]['Age'].mean()

# Only one Ms has missing value and one with value
ms_no_age = (alldata['Title'] == 'Ms') & alldata.Age.isnull()
ms_with_age = (alldata['Title'] == 'Ms') & alldata.Age.notnull()
alldata.ix[ms_no_age, 'Age2'] = int(alldata[ms_with_age]['Age'].mean())

# Use average for each title grop to fill up the rest of null values
# We could also combine SibSp and Parch columns to make more educated guess
# But lets see how we do without that
# Mr
mr_with_age = (alldata['Title'] == 'Mr') & alldata.Age.notnull()
min_mr_age = min(alldata[mr_with_age]['Age'])
max_mr_age = max(alldata[mr_with_age]['Age'])               
mr_no_age = (alldata['Title'] == 'Mr') & alldata.Age.isnull()
# alldata.ix[mr_no_age, 'Age2'] =  np.random.randint(min_mr_age, max_mr_age, len(alldata.ix[mr_no_age]))
alldata.ix[mr_no_age, 'Age2'] = np.median(alldata[mr_with_age]['Age'])
# Miss
miss_with_age = (alldata['Title'] == 'Miss') & alldata.Age.notnull()
miss_no_age = (alldata['Title'] == 'Miss') & alldata.Age.isnull()
min_miss_age = min(alldata[miss_with_age]['Age'])
max_miss_age = max(alldata[miss_with_age]['Age'])
# alldata.ix[miss_no_age, 'Age2'] = np.random.randint(min_miss_age, max_miss_age, len(alldata.ix[miss_no_age]))
alldata.ix[miss_no_age, 'Age2'] = np.median(alldata[miss_with_age]['Age'])
# Mrs
mrs_with_age = (alldata['Title'] == 'Mrs') & alldata.Age.notnull()
mrs_no_age = (alldata['Title'] == 'Mrs') & alldata.Age.isnull()
min_mrs_age = min(alldata[mrs_with_age]['Age'])
max_mrs_age = max(alldata[mrs_with_age]['Age'])
# alldata.ix[mrs_no_age, 'Age2'] = np.random.randint(min_mrs_age, max_mrs_age, len(alldata.ix[mrs_no_age]))
alldata.ix[mrs_no_age, 'Age2'] = np.median(alldata[mrs_with_age]['Age'])

alldata['Age2'] = alldata.Age2.astype(int)

# Lets drop Age column late
drop_list.append('Age')

alldata["Fare"].fillna(alldata["Fare"].median(), inplace=True)


# As we see, Children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as adult males, adult female, and Child
def get_who(who):
    return 'Child' if who.Age2 < 16 else who.Sex
    
alldata['Who'] = alldata[['Age2','Sex']].apply(get_who,axis=1)

# adding a text Class column
class_text = {1: 'First', 2: 'Second', 3: 'Third'}
alldata['Class'] = alldata['Pclass'].map(class_text)

'''
g = sns.factorplot(x="Who", y="Survived", col="Class", 
    data=alldata[alldata['Survived'].notnull()], saturation=.5,
    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
.set_xticklabels(["Men", "Women", "Children"])
.set_titles("{col_name} {col_var}")
.despine(left=True))  
'''
# Add Sex column to the drop list since we created Who column
drop_list.append('Sex')

# create dummy variables for Who column
# drop Male as it has the lowest average of survived passengers
who_dummies  = pd.get_dummies(alldata['Who'])
who_dummies.columns = ['Child','Adult_Female','Adult_Male']
who_dummies.drop(['Adult_Male'], axis=1, inplace=True)

alldata = alldata.join(who_dummies.astype(int))

drop_list.append('Class')
drop_list.append('Who')


alldata['Fare'] = alldata['Fare'].astype(int)

# get fare for survived & not-survived passengers 
fare_not_survived = alldata[alldata['Survived'] == 0]['Fare']
fare_survived     = alldata[alldata['Survived'] == 1]['Fare']

# get average and std for fare of survived/not survived passengers
#avgerage_fare = pd.DataFrame([fare_not_survived.mean(), fare_survived.mean()])
#std_fare      = pd.DataFrame([fare_not_survived.std(), fare_survived.std()])



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies  = pd.get_dummies(alldata['Pclass'])
pclass_dummies.columns = ['Class_1','Class_2','Class_3']
pclass_dummies.drop(['Class_3'], axis=1, inplace=True)

drop_list.append('Pclass')

alldata = alldata.join(pclass_dummies.astype(int))

# Instead of having two columns Parch & SibSp
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
# So we are going to keep 
alldata['Family'] =  alldata["Parch"] + alldata["SibSp"]

alldata['Family'] = alldata['Family'].apply(lambda x: 1 if x > 0 else 0)

alldata['FamilySize'] =  alldata["Parch"] + alldata["SibSp"] + 1

# plot the survivor count by family size and survivor rate by family size side-by-side
#fig, (axis1,axis2) = plt.subplots(ncols=2, figsize= (10,5))

#sns.countplot(x="FamilySize", hue="Survived", ax=axis1, data=alldata[alldata['Survived'].notnull()])
#sns.factorplot(x="FamilySize", y="Survived",kind='bar', ax=axis2, ci=False, 
#                data=alldata[alldata['Survived'].notnull()])

# Use Family size (test score = 0.789) vs Family (test score = 0.756)
drop_list.append('Family')

le = preprocessing.LabelEncoder()

alldata['Title2'] = le.fit_transform(alldata['Title'])

drop_list += ['Title', 'Name','Ticket','Embarked', 'SibSp','Parch']
alldata.drop(drop_list, axis=1, inplace=True)

train = alldata[alldata['Survived'].notnull()].copy()
test = alldata[alldata['Survived'].isnull()].copy()

train.drop('PassengerId', axis=1, inplace = True)

X_train = train.drop('Survived', axis=1)
y_train= train['Survived'].astype('int') # when concatting train set and test set, the Survived column changed to float


X_test  = test.drop(['Survived', 'PassengerId'], axis=1).copy()

# Random Forests with OOB

#%% load training data
#log_reg = LogisticRegression(penalty='l1')
#log_reg.fit(X_train, y_train)
#print(log_reg.score(X_train, y_train))
#y_test=log_reg.predict(X_test)
#%% xgboost test
#xgb_params = {
#    'seed': 0,
#    'colsample_bytree': 0.7,
#    'silent': 1,
#    'subsample': 0.7,
#    'learning_rate': 0.075,
#    'objective': 'binary:logistic',
#    'max_depth': 4,
##    'num_parallel_tree': 1,
#    'min_child_weight': 1,
##    'eval_metric': 'rmse',
##    'nrounds': 500
#}
#

SEED=1
X_train1=np.array(X_train)
y_train1=np.array(y_train)
# data frame carries index it will cause error in the later split for index 
X_train_train,X_train_valid,y_train_train,y_train_valid=train_test_split(X_train1,y_train1,test_size=0.2,random_state=SEED)
ntrain=X_train_train.shape[0]
ntest=X_train_valid.shape[0]
nfolds=10
kfold=StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=1)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None,cv=None):
        print('seed ', seed)
        self.clf=clf(random_state=seed)
        self.cv=cv
        self.params=params
    def search(self,x_train,y_train):
        self.clf = GridSearchCV(self.clf,self.params,cv=self.cv)
        self.clf.fit(x_train,y_train)
        pritn(self.clf.best_params_)
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)



def get_oof(clf):
    oof_train = np.zeros(ntrain)
    oof_test=np.zeros(ntest)
    oof_test_skf = np.empty((nfolds,ntest))
    cnt=0
    for train_index, test_index in kfold.split(X_train_train,y_train_train):
        # X_train[train_index] will return the indexed columns 
#        x_tr = X_train_train.ix[train_index]
#        y_tr = y_train_train.ix[train_index]
#        x_te = X_train_train.ix[test_index]
        x_tr=X_train_train[train_index]
        y_tr=y_train_train[train_index]
        x_te=X_train_train[test_index]
        clf.train(x_tr, y_tr)

        #train and test should both be arrays or data frames
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[cnt,:] = clf.predict(X_train_valid)
    
        ++cnt
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)

svc_params={
            'C':[1 ,10, 100,1000],
            'kernel':('rbf','linear')
            }

svc = SklearnWrapper(clf=svm.SVC, seed=SEED, params=svc_params,cv=kfold)
svc.search(X_train,y_train)

 


#submission = pd.DataFrame({
#        "PassengerId": test_set["PassengerId"],
#        "Survived": y_test
#    })
#submission.to_csv('titanic.csv', index=False)
#%%
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#params = clf.cv_results_['params']
#for mean, stdev, param in zip(means, stds, params):
#	print("%f (%f) with: %r" % (mean, stdev, param))
# # plot
#pyplot.errorbar(subsample, means, yerr=stds)
#pyplot.title("XGBoost colsample_bytree vs Log Loss")
#pyplot.xlabel('colsample_bytree')
#pyplot.ylabel('Log Loss')
#best_parameters, score, _ = max(.grid_scores_, key=lambda x: x[1])
#print('score:', score)
