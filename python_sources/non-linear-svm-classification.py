# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#There are missing fields in Embarked column. Most of the fields are S .
datasets = [train,test]
for dataset in datasets:
    dataset.Embarked.fillna('S', inplace=True)
#Converting categorical data into numerics
train = pd.concat([train,pd.get_dummies(train.Embarked, prefix='EMAP')], axis=1)
test = pd.concat([test,pd.get_dummies(test.Embarked, prefix='EMAP')], axis=1)

trainages = np.random.randint(train.Age.mean() - train.Age.std(),train.Age.mean() + train.Age.std(),size = sum(pd.isnull(train.Age)))
testages = np.random.randint(test.Age.mean() - test.Age.std(),test.Age.mean() + test.Age.std(),size = sum(pd.isnull(test.Age)))
train.Age.loc[pd.isnull(train.Age)] = trainages
test.Age.loc[pd.isnull(test.Age)] = testages

train["Male"] = 0
train.Male.loc[train.Sex == 'male'] = 1
train["Title"] = train.Name.str.extract('( [A-Za-z]+)\.',expand = False)
#train["Agebins"] = pd.qcut(train.Age,4,labels = False)
train.Title.loc[train.Title == ' Mlle'] = ' Miss'
train.Title.loc[train.Title == ' Mme'] = ' Mrs'
train.Title.loc[train.Title == ' Dona'] = ' Mrs'
test["Male"] = 0
test.Male.loc[test.Sex == 'male'] = 1
test["Title"] = test.Name.str.extract('( [A-Za-z]+)\.',expand = False)
#train["Agebins"] = pd.qcut(train.Age,4,labels = False)
test.Title.loc[test.Title == ' Mlle'] = ' Miss'
test.Title.loc[test.Title == ' Mme'] = ' Mrs'
test.Title.loc[test.Title == ' Dona'] = ' Mrs'


train = pd.concat([train,pd.get_dummies(train.Title, prefix='TMAP')], axis=1)
test = pd.concat([test,pd.get_dummies(test.Title, prefix='TMAP')], axis=1)

for col in train.drop(['Survived'],1).columns:
    if col not in test.columns:
        test[col] = 0

train = pd.concat([train,pd.get_dummies(train.Pclass, prefix='PMAP')], axis=1)
test = pd.concat([test,pd.get_dummies(test.Pclass, prefix='PMAP')], axis=1)		
		
test.Fare.fillna(test.Fare.mean(),inplace = True)

train = pd.concat([train,pd.get_dummies(train.SibSp, prefix='SibSp')], axis=1)
test = pd.concat([test,pd.get_dummies(test.SibSp, prefix='SibSp')], axis=1)

train = pd.concat([train,pd.get_dummies(train.Parch, prefix='Parch')], axis=1)
test = pd.concat([test,pd.get_dummies(test.Parch, prefix='Parch')], axis=1)

train.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','Title','Pclass','SibSp','Parch'],1,inplace = True)
test.drop(['Name','Sex','Ticket','Cabin','Embarked','Title','Pclass','SibSp','Parch'],1,inplace = True)

train['Parch_9'] = 0
    
#--------Feature Standardisation----------#

train.Fare = (train.Fare - train.Fare.mean())/train.Fare.std()
test.Fare = (test.Fare - test.Fare.mean())/test.Fare.std()
train.Age = (train.Age - train.Age.mean())/train.Age.std()
test.Age = (test.Age - test.Age.mean())/test.Age.std()


#---------Training set splitting---------#
#from sklearn.model_selection import train_test_split

#train1,train2 = train_test_split(train,test_size = 0.20,random_state = 40)


#--------Applying Backward Elimination--------#
'''
import statsmodels.formula.api as sm

X = np.append(arr = np.ones((712,1)).astype(int), values = train1.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False), axis = 1)
X_opt = X[:, [0, 1, 2, 3, 6, 7, 9, 11, 12, 16]]
clf_new = sm.OLS(endog = train1.Survived, exog = X_opt).fit()
clf_new.summary()

X1_opt = train2[:,[]]
'''
#----------Applying kernel PCA---------------#
'''
from sklearn.decomposition import KernelPCA

pca = KernelPCA(n_components = 2, kernel = 'rbf')
train1_pca = pca.fit_transform(train1.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False))
train2_pca = pca.transform(train2.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False))
test_pca = pca.transform(test.drop(['PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace = False))
#explained_variance = pca.explained_variance_ratio_
'''
#----------Apply ML Algo now----------#
from sklearn import svm
#clf = svm.SVC(gamma = 1e-8, C = 1e6)
clf = svm.SVC(C = 1000, kernel = 'rbf', gamma = 0.001)

#clf.fit(train1.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False), train1.Survived)

#clf.fit(train1_pca, train1.Survived)
#clf.score(train2_pca, train2.Survived)
clf.fit(train.drop(['Survived'],1,inplace=False), train.Survived)
#clf.score(train2.drop(['Survived','PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace=False), train2.Survived)


#train2_pred = clf.predict(train2.drop(['Survived','PassengerId','Pclass','EmbarkedMap','TMap','Title'],1,inplace = False))


#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(train2.Survived,train2_pred)

#-------K-fold cross validation-----#
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = clf, X = train.drop(['Survived'],1,inplace = False), y = train.Survived, cv = 10)
print(accuracies.mean())

'''
#-----Applying grid search------#
from sklearn.model_selection import GridSearchCV
#parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']}, {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.5, 0.1, 0.01, 0.001, 0.0001]}]
parameters = [{'C' : [1000, 10000, 100000], 'kernel' : ['rbf'], 'gamma' : [0.001, 0.003, 0.005, 0.007, 0.009]}]
gs = GridSearchCV(estimator = clf, param_grid = parameters, scoring = 'accuracy', cv = 10)
gs = gs.fit(train.drop(['Survived','PassengerId','TMap','Pclass','EmbarkedMap','Title'],1,inplace = False),train.Survived)

best_acc = gs.best_score_
best_params = gs.best_params_
'''
#arr = pd.Series(clf.predict(test_pca),name = 'Survived')
arr = pd.Series(clf.predict(test.drop(['PassengerId'],1,inplace = False)),name = 'Survived')
df = pd.concat([test.PassengerId,arr],axis = 1)

#np.savetxt("pred1.csv",df,delimiter=',',fmt='%d')

df.to_csv('pred.csv',sep = ',',index = False)
