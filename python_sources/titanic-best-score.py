#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
dataset =  pd.concat(objs=[train, test], axis=0,sort=False).reset_index(drop=True)


# In[ ]:



dataset['Ticket_Frequency'] = dataset.groupby('Ticket')['Ticket'].transform('count')
dataset['Cabin_n'] = dataset['Cabin'].str[0]
dataset['Cabin_n'].fillna('M',inplace=True)
dataset['Deck'] = dataset['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
dataset['Deck'] = dataset['Deck'].replace(['A', 'B', 'C'], 'ABC')
dataset['Deck'] = dataset['Deck'].replace(['D', 'E'], 'DE')
dataset['Deck'] = dataset['Deck'].replace(['F', 'G'], 'FG')
dataset['Deck'] = dataset['Deck'].replace(['T'], 'A')
dataset.drop(labels = ["Cabin",'Ticket'], axis = 1, inplace = True)

dataset["TFamily"] = dataset["Parch"] + dataset["SibSp"] + 1

dataset.loc[dataset['TFamily']==1 ,'IsAlone']=1
dataset.loc[dataset['TFamily']!=1 ,'IsAlone']=0
dataset['IsAlone']= dataset['IsAlone'].astype(int)
dataset["Embarked"] = dataset["Embarked"].fillna("S")
dataset['Age'] = dataset.groupby(["Sex", "Pclass"])["Age"].apply(lambda x: x.fillna(x.median()))
med_fare = dataset.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
# Filling the missing value in Fare with the median Fare of 3rd class alone passenger
dataset['Fare'] = dataset['Fare'].fillna(med_fare)

#age_bining=np.histogram_bin_edges(dataset['Age'], bins='auto')
dataset['Age_bin']=pd.cut(dataset['Age'], bins=10)

#TFamily_bining=np.histogram_bin_edges(dataset['TFamily'], bins='auto')
dataset['TFamily_bin']=pd.cut(dataset['TFamily'], bins=2)


#fare_bining=np.histogram_bin_edges(dataset['Fare'], bins='auto')
dataset['Fare_bin']=pd.cut(dataset['Fare'], bins=13)


dataset['Sex']=dataset['Sex'].map({'male':1,'female':0})
dataset = pd.get_dummies(dataset, columns=['Embarked','Age_bin','Cabin_n','Fare_bin','TFamily_bin','Deck'])
dataset.columns = dataset.columns.str.replace(',','')
dataset.columns = dataset.columns.str.replace(']','')
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

Ticket = []
for i in list(train.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
train["Ticket"] = Ticket

dataset.drop(labels = ['Name'], axis = 1, inplace = True)
train_n=dataset[dataset['Survived'].notnull()]
x_train=train_n.drop(labels=['Survived','PassengerId'],axis=1)
y_train=train_n['Survived'].astype(int)
X_train,X_test,Y_train,Y_test = train_test_split(x_train,y_train,test_size=0.25, random_state=0)



# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[ ]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(XGBClassifier())
classifiers.append(LGBMClassifier())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis",'XGBClassifier','LGBMClassifier']})
plt.figure(figsize=(20,10))
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
plt.axvline(0.81)
plt.axvline(0.84)
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[ ]:


gbm = GradientBoostingClassifier(random_state=2)
gbm.fit(X_train,Y_train)
print('Score: ',gbm.score(X_test,Y_test))

feature_importances = pd.DataFrame(gbm.feature_importances_,index = X_test.columns,columns=['importance']).sort_values('importance', ascending=False)
feature_importances.head(34)


# In[ ]:



test_n=dataset[dataset['Survived'].isnull()]
test_n.drop(labels = ["Survived"], axis = 1, inplace = True)

submission=pd.DataFrame(columns=['PassengerId','Survived'])
submission['PassengerId']=test_n['PassengerId']
test_n.drop(labels = ["PassengerId"], axis = 1, inplace = True)
Y_pred=gbm.predict(test_n)
submission['Survived']=Y_pred.astype(int)
submission.to_csv('submission_gbm.csv',header=True,index=False)

