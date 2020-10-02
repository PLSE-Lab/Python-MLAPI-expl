#!/usr/bin/env python
# coding: utf-8

# This exercise will 

# In[ ]:


# Import all the necessary modules
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, tree, metrics, ensemble


# In[ ]:


# Read training data and prepare predictor and target variables
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

# Encode gender
le = preprocessing.LabelEncoder()
df_train['Sex_Coded'] = le.fit_transform(df_train['Sex'])

# Encode title (Mr, Miss etc.)
df_train['title'] = "Others"
df_train.loc[df_train.Name.str.rfind("Mr.")>0,"title"] = "Mister"
df_train.loc[df_train.Name.str.rfind("Mrs.")>0,"title"] = "Missus"
df_train.loc[df_train.Name.str.rfind("Miss.")>0,"title"] = "Miss"
df_train.loc[df_train.Name.str.rfind("Master")>0,"title"] = "Master"
df_train['Title_Coded'] = le.fit_transform(df_train['title'])

# Fill out missing age values and encode them
df_train.loc[df_train.loc[df_train.Sex_Coded==0].loc[df_train.Age.isna()].index,'Age']=27.92
df_train.loc[df_train.loc[df_train.Sex_Coded==1].loc[df_train.Age.isna()].index,'Age']=30.73
age_bins = pd.cut(df_train.Age,bins=[0,20,50,100],include_lowest=True,labels=[1,-1,2])
df_train['Age_Coded'] = age_bins[0]

# Code Family data (Parent/Child + Sibling/Spouse) i.e. Family is more likely to survive cause of prefrence
df_train['Family'] = df_train.Parch + df_train.SibSp
df_train['Family_Coded'] = pd.cut(df_train.Family,bins=[0,1,4,11],labels=[-1,1,2],include_lowest=True)

# Code Embarkation points 
df_train['Embarked'] = df_train.Embarked.fillna(-1,inplace=True)
df_train['Embarked_Coded'] = le.fit_transform(df_train.Embarked)

X_train = df_train[['Sex_Coded','Title_Coded','Age','Pclass','Family_Coded','Embarked_Coded']]
y_train = df_train['Survived']


# In[ ]:


clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(metrics.confusion_matrix(y_train,clf.predict(X_train)))


# In[ ]:


# Read test data and preprocess
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test['Sex_Coded'] = le.fit_transform(df_test['Sex'])

df_test['title'] = "Others"
df_test.loc[df_test.Name.str.rfind("Mr.")>0,"title"] = "Mister"
df_test.loc[df_test.Name.str.rfind("Mrs.")>0,"title"] = "Missus"
df_test.loc[df_test.Name.str.rfind("Miss.")>0,"title"] = "Miss"
df_test.loc[df_test.Name.str.rfind("Master")>0,"title"] = "Master"
df_test['Title_Coded'] = le.fit_transform(df_test['title'])

df_test.loc[df_test.loc[df_test.Sex_Coded==0].loc[df_test.Age.isna()].index,'Age']=30.27
df_test.loc[df_test.loc[df_test.Sex_Coded==1].loc[df_test.Age.isna()].index,'Age']=30.27

age_bins = pd.cut(df_test.Age,bins=[0,20,50,100],include_lowest=True).factorize()
df_test['Age_Coded'] = age_bins[0]

# Code Family data (Parent/Child + Sibling/Spouse) i.e. Family is more likely to survive cause of prefrence
df_test['Family'] = df_test.Parch + df_test.SibSp
df_test['Family_Coded'] = pd.cut(df_test.Family,bins=[0,1,4,11],labels=[-1,1,2],include_lowest=True)

# Code Embarkation points 
df_test['Embarked'] = df_test.Embarked.fillna(-1,inplace=True)
df_test['Embarked_Coded'] = le.fit_transform(df_test.Embarked)

X_test = df_test[['Sex_Coded','Title_Coded','Age','Pclass','Family_Coded','Embarked_Coded']]


# In[ ]:


y_pred = clf.predict(X_test)
pd.DataFrame(data={'PassengerId':df_test['PassengerId'],'Survived':y_pred}).to_csv("my_submission_7.2.csv",index=False)


# In[ ]:


clf = ensemble.RandomForestClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_train,y_train))
print(metrics.confusion_matrix(y_train,clf.predict(X_train)))


# In[ ]:


y_pred = clf.predict(X_test)
pd.DataFrame(data={'PassengerId':df_test['PassengerId'],'Survived':y_pred}).to_csv("/kaggle/working/my_submission_7.3.csv",index=False)

