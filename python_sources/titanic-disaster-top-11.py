#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
import warnings
warnings.simplefilter('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/titanic/train.csv')
data.info()
print('Data Visualisation')


# In[ ]:


# Pclass vis.
fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Survived', data=data, hue='Pclass')
ax.set_ylim(0,400)
plt.title("Impact of Pclass on Survived")
plt.show()


# In[ ]:


# Sex vis.
fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Survived', data=data, hue='Sex')
ax.set_ylim(0,400)
plt.title("Impact of Sex on Survived")
plt.show()


# In[ ]:


missing_features = data.columns[data.isnull().any()]
print(data[missing_features].isnull().sum())


# In[ ]:


data.drop(['PassengerId'],axis=1,inplace=True)
data['Age'].fillna(data['Age'].mean(), inplace=True)


# In[ ]:


data['Cabin'].unique()
data['Cabin'] = data['Cabin'].str[:1]
data['Cabin'].fillna('NN',inplace=True)


# In[ ]:


data['Embarked'].describe()
data['Embarked'] = data['Embarked'].fillna('S')


# In[ ]:


sexDict = {"male" :1 , "female" : 0}
data = data.replace({"Sex" : sexDict})
data['Sex'] = data['Sex'].astype('float64')


# In[ ]:


data['Name'].head()
data['Name'].tail()


# In[ ]:


new = data["Name"].str.split(", ", n = 1, expand = True) 
data['Family'] = new[0]
data['Family'].head()
new = data["Name"].str.split(", ", n = 1, expand = True) 
data["TitleName"]= new[1].str.split().str.get(0)


# In[ ]:


data['TitleName'].head()


# In[ ]:


data.drop(['Name','Ticket'],axis=1,inplace=True)
le = preprocessing.LabelEncoder()
# data['Cabin'] = le.fit_transform(data['Cabin'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data['Family'] = le.fit_transform(data['Family'])


# In[ ]:


print(data['SibSp'].corr(data['Survived']))
print(data['Parch'].corr(data['Survived']))
data['FamilySize'] = data['SibSp'] + data['Parch']+1
data.drop(['SibSp','Parch'],axis=1,inplace=True)
data['FamilySize'] = data['FamilySize'].apply(lambda x : 'Single' if x==1 else 'SmallFam' if x>1 and x<5 else 'BigFam')


# In[ ]:


data['FamilySize'] = le.fit_transform(data['FamilySize'])
print(data['FamilySize'].corr(data['Survived']))
data['FamilySize'] = le.fit_transform(data['FamilySize'])
sns.barplot(x='FamilySize', y='Survived',  data=data)


# In[ ]:


# data.plot(x='Age' , y='Survived',style='o')
# sns.barplot(x='Age', y='Survived',  data=data)
print(data['Age'].describe())
data['Age'].corr(data['Survived'])
data['AgeApp'] = data['Age'].apply(lambda x :1 if x<7 else 0)
data.drop(['Age'],axis=1,inplace=True)
# data['AgeApp'] = le.fit_transform(data['AgeApp'])
data['AgeApp'].corr(data['Survived'])


# In[ ]:


sns.barplot(x='TitleName', y='Survived',   data=data)
data['TitleName1'] = le.fit_transform(data['TitleName'])
data['TitleName1'].corr(data['Survived'])


# In[ ]:


data.drop(['TitleName1'],axis=1,inplace=True)


# In[ ]:


dictTitle = {"Mr." : 1 , "Mrs." : 3 , "Miss." : 3 , "Master." : 3 , "Dona." : 1 ,"Don." : 1, "Rev." : 1 , "Dr." : 2 , "Mme." : 3 , "Ms." : 3, "Major." : 2 , "Lady." : 3 , "Sir." : 3 , "Mlle." : 3 , "Col." : 2 , "Capt." : 1 , "the" : 3 , "Jonkheer." : 1}
data = data.replace({"TitleName" : dictTitle})
data['TitleName'] = data['TitleName'].astype('int64')
data['TitleName'].corr(data['Survived'])


# In[ ]:


sns.barplot(x='Cabin', y='Survived',   data=data)
data['Cabin'].unique()
dictCabinloc = {"A": 2, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 2, "T": 2 ,"NN":2}
data = data.replace({"Cabin" : dictCabinloc})
data['Cabin'].unique()


# In[ ]:


data.info()
plt.figure(figsize=(20,20))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


Y=data['Survived']
X=data.loc[:, ~data.columns.isin(['Survived'])]
xtrain , xvalid , ytrain , yvalid = train_test_split(X,Y,test_size =0.15)


# In[ ]:





# In[ ]:


#'min_child_weight': [1, 5, 10],
#        'gamma': [0.5, 1, 1.5, 2, 5],
#        'subsample': [0.6, 0.8, 1.0],
#        'colsample_bytree': [0.6, 0.8, 1.0],
#       'max_depth': [3, 4, 5],


# In[ ]:


params = {
        
        'learning_rate' : [0.005 , 0.01 , 0.03 , 0.05 , 0.07 , 0.1],
        'n_estimators' : [1000,2000,3000,4000,5000]
        }

model = GridSearchCV(XGBClassifier(objective='binary:logistic',
                    silent=True, nthread=1),params,verbose=1,cv=3,n_jobs=-1)


model.fit(xtrain,ytrain)
ypred  = model.predict(xvalid)
from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,yvalid))


# In[ ]:


grid_params = {
    'n_neighbors':[3,5,11,19,21,25,31,35,41],
    'weights' : ['uniform' , 'distance'],
    'metric' : ['euclidean' , 'manhattan'],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    
}
gs = GridSearchCV(KNeighborsClassifier(),grid_params,verbose=1,cv=3,n_jobs=-1)
gs.fit(xtrain, ytrain)
preds=gs.predict(xvalid)
from sklearn.metrics import accuracy_score
print(accuracy_score(preds,yvalid))
print(gs.best_params_)


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
grid_values = {'penalty': ['l2'],'C':[0.001,.009,0.01,.09,1,5,10,25],'solver':['lbfgs'],'multi_class':['multinomial']}
grid_lr_acc = GridSearchCV(lr, param_grid = grid_values,scoring = 'accuracy')
grid_lr_acc.fit(xtrain,ytrain)
predlr = grid_lr_acc.predict(xvalid)
print(accuracy_score(predlr,yvalid))


# In[ ]:


clf2=RandomForestClassifier(random_state=42 , max_features = 'auto' , max_depth = 8 , criterion = 'entropy')
param_grid = { 
    'n_estimators': [3200 , 3500 , 4000 , 5000],
}
rfc = GridSearchCV(estimator=clf2, param_grid=param_grid, cv= 5)
rfc.fit(xtrain,ytrain)
predsrf = rfc.predict(xvalid)
print(accuracy_score(predsrf,yvalid))


# In[ ]:


rfc.best_params_
clf2 = RandomForestClassifier(**rfc.best_params_)
clf2.fit(xtrain,ytrain)
predo= clf2.predict(xvalid)
print(accuracy_score(predo,yvalid))


# In[ ]:



sclf = StackingClassifier(classifiers=[lr, model], 
                          meta_classifier=clf2)
sclf.fit(xtrain,ytrain)
predd = sclf.predict(xvalid)
print(accuracy_score(predd,yvalid))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:


testdata = pd.read_csv('../input/titanic/test.csv')
Sub = pd.DataFrame()
Sub['PassengerId'] = testdata['PassengerId']
testdata.drop(['PassengerId'],axis=1,inplace=True)
testdata['Age'].fillna(testdata['Age'].mean(), inplace=True)
testdata['Cabin'] = testdata['Cabin'].str[:1]
testdata['Cabin'].fillna('NN',inplace=True)
testdata['Embarked'] = testdata['Embarked'].fillna('S')
testdata = testdata.replace({"Sex" : sexDict})
testdata['Sex'] = testdata['Sex'].astype('float64')
new = testdata["Name"].str.split(", ", n = 1, expand = True) 
testdata['Family'] = new[0]
new = testdata["Name"].str.split(", ", n = 1, expand = True) 
testdata["TitleName"]= new[1].str.split().str.get(0)
testdata.drop(['Name','Ticket'],axis=1,inplace=True)
le = preprocessing.LabelEncoder()
# testdata['Cabin'] = le.fit_transform(testdata['Cabin'])
testdata['Embarked'] = le.fit_transform(testdata['Embarked'])
testdata['Family'] = le.fit_transform(testdata['Family'])
testdata['FamilySize'] = testdata['SibSp'] + testdata['Parch']+1
testdata.drop(['SibSp','Parch'],axis=1,inplace=True)
testdata['FamilySize'] = testdata['FamilySize'].apply(lambda x : 'Single' if x==1 else 'SmallFam' if x>1 and x<5 else 'BigFam')
testdata['FamilySize'] = le.fit_transform(testdata['FamilySize'])

# data.plot(x='Age' , y='Survived',style='o')
# sns.barplot(x='Age', y='Survived',  data=data)
print(testdata['Age'].describe())

testdata['AgeApp'] = testdata['Age'].apply(lambda x :1 if x<7 else 0)
testdata.drop(['Age'],axis=1,inplace=True)



testdata['TitleName1'] = le.fit_transform(testdata['TitleName'])
testdata.drop(['TitleName1'],axis=1,inplace=True)
dictTitle = {"Mr." : 1 , "Mrs." : 3 , "Miss." : 3 , "Master." : 3 , "Dona." : 1 , "Rev." : 1 , "Dr." : 2 , "Mme." : 3 , "Ms." : 3, "Major." : 2 , "Lady." : 3 , "Sir." : 3 , "Mlle." : 3 , "Col." : 2 , "Capt." : 1 , "the" : 3 , "Jonkheer." : 1}
testdata = testdata.replace({"TitleName" : dictTitle})
testdata['TitleName'] = testdata['TitleName'].astype('int64')
dictCabinloc = {"A": 2, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 2, "T": 2 ,"NN":2}
testdata = testdata.replace({"Cabin" : dictCabinloc})













# testdata = pd.get_dummies(testdata)


testdata.info()
xyz=testdata.loc[:, ~testdata.columns.isin([])]
missing_cols = set( xtrain.columns ) - set( xyz.columns )
for c in missing_cols:
    xyz[c] = 0
xyz = xyz[xtrain.columns]

missing_features = xyz.columns[xyz.isnull().any()]
print(xyz[missing_features].isnull().sum())
xyz['Fare'] = xyz['Fare'].fillna(xyz['Fare'].median())


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from vecstack import stacking
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pickle


models = [
    
    GradientBoostingClassifier(random_state=0, 
                               n_estimators=2500,learning_rate=0.01, max_depth=3),
    
    LogisticRegression(**grid_lr_acc.best_params_),
            
        
    XGBClassifier(**model.best_params_),
                  
    AdaBoostClassifier(random_state=0,  learning_rate=0.1, 
                      n_estimators=2500),
                       
    RandomForestClassifier(**rfc.best_params_),
    BaggingClassifier(tree.DecisionTreeClassifier(random_state=0))

]

S_train, S_test = stacking(models,                   
                           xtrain, ytrain,  xvalid,  
                           regression=True, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=10, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=0,    
         
                           verbose=2)
########################################################33

model = KNeighborsClassifier(n_neighbors=3,
                        n_jobs=-1)

#model = AdaBoostClassifier(random_state=0,  learning_rate=0.1, 
#                  n_estimators=100)

#model=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=100, max_depth=3)
model.fit(S_train, ytrain)
y_pred = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(yvalid, y_pred))
score=model.score(S_test, yvalid)


# In[ ]:





# In[ ]:




S_train, S_test = stacking(models,                   
                           xtrain, ytrain, xyz,  
                           regression=True, 
     
                           mode='oof_pred_bag', 
       
                           needs_proba=False,
         
                           save_dir=None, 
            
                           metric=accuracy_score, 
    
                           n_folds=10, 
                 
                           stratified=True,
            
                           shuffle=True,  
            
                           random_state=42, verbose=2)


# In[ ]:



sclf.fit(S_train,ytrain)
predFinal = sclf.predict(S_test)
Sub['Survived'] = predFinal
Sub.head()

Sub.to_csv('StackingC.csv',index=False)
model.fit(S_train,ytrain)
predFinal = model.predict(S_test)
Sub['Survived'] = predFinal


Sub.to_csv('Stacking.csv',index=False)
clf2.fit(S_train,ytrain)
predFinal = clf2.predict(S_test)
Sub['Survived'] = predFinal


Sub.to_csv('RFClassifier.csv',index=False)


# In[ ]:


Sub.head()

