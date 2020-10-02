#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import both train and test data...
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()
test.info()


# In[ ]:


# check missing values
print(train.isnull().sum().sort_values(ascending=False))
print(20*'_')
test.isnull().sum().sort_values(ascending=False)


# In[ ]:


train.columns


# In[ ]:


# drop Cabin field from both train and test dataset as there are lots of missing value 
#drop_Columns=['Cabin','PassengerId','Ticket']
drop_Columns=['Cabin','PassengerId','Ticket']
train.drop(drop_Columns,axis=1, inplace=True)
test_passengerid=test['PassengerId']
test.drop(drop_Columns,axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


# Impute Age value with mean value
train['Age'].fillna(train['Age'].median(),inplace=True)
test['Age'].fillna(test['Age'].median(),inplace=True)


# In[ ]:


train.isnull().sum().sort_values(ascending=False)


# In[ ]:


# impute missing value of fare column with mean value 
train['Fare'].fillna(train['Fare'].median(),inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)


# In[ ]:


# Impute missing values from the Embarked column from both train and test dataset.
train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)


# In[ ]:


test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)


# In[ ]:


test.isnull().sum().sort_values(ascending=False)


# In[ ]:


train.groupby('Survived').count()


# In[ ]:


# Exploratory Data Analysis.
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


# Countplot of Survived column.
sns.countplot(x='Survived', data=train)


# In[ ]:


train.columns


# In[ ]:


# Survived against Sex ratio.
sns.countplot(x='Survived',hue='Sex', data=train)
# By looking at plot, it clearly says that more women survived that men.


# In[ ]:


# Check survived rate against PClass
sns.countplot(x='Survived', hue='Pclass', data=train)


# In[ ]:


# Check distribution ratio of Age.
sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[ ]:


train['Embarked'].value_counts()


# **#Feature Engineering**

# In[ ]:


train['Embarked'].value_counts()


# In[ ]:


# Feature Encoding:
train['Sex']=train['Sex'].map({'male':1,'female':0}).astype(int)
  


# In[ ]:


train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


# In[ ]:


test['Sex']=test['Sex'].map({'male':1,'female':0}).astype(int)
test['Embarked']=test['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)


# In[ ]:


# for first submission drop 'Name'
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


# Chec correlation with Target variable
corr=train.corr()
plt.figure(figsize=(10,10))   
sns.heatmap(corr,annot=True,fmt=".0%")
plt.show()


# In[ ]:


train.columns


# In[ ]:


# 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X=train.drop('Survived',axis=1)
y=train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=101)

logr=LogisticRegression()


# In[ ]:


def models(X_train,y_train):
    ## Logistic Regression Model
        from sklearn.linear_model import LogisticRegression
        logis= LogisticRegression(C=50)
        logis.fit(X_train, y_train)
        train_score1 =logis.score(X_train,y_train)
        test_score1 =logis.score(X_test,y_test)

        ## Random Forest Model
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10,criterion="entropy",random_state=5)
        rf.fit(X_train,y_train)
        train_score2 =rf.score(X_train,y_train)
        test_score2 =rf.score(X_test,y_test)

        ## KNN
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier(n_neighbors=11)
        knc.fit(X_train,y_train)
        train_score3 =knc.score(X_train,y_train)
        test_score3 =knc.score(X_test,y_test)

        ## SVC
        from sklearn.svm import SVC
        sv = SVC()
        sv.fit(X_train, y_train)
        train_score4 =sv.score(X_train,y_train)
        test_score4 =sv.score(X_test,y_test)

        ## XgBoost
        from xgboost import XGBClassifier
        boost= XGBClassifier(learning_rate=0.01)
        boost.fit(X_train,y_train)
        train_score5 =boost.score(X_train,y_train)
        test_score5 =boost.score(X_test,y_test)

        ## Print Accuracy
        print("Logistic train score: ", train_score1, "Test score : ",test_score1)
        print("Random Forest train score: ", train_score2, "Test score : ",test_score2)
        print("KNN train score: ", train_score3, "Test score : ",test_score3)
        print("SVC train score: ", train_score4, "Test score : ",test_score4)
        print("Xgboost train score: ", train_score5, "Test score : ",test_score5)
        
        return logis,rf,knc,sv,boost


# In[ ]:


model=models(X_train,y_train)


# In[ ]:


model


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


for i in range(len(model)):
    print("Model ", i)
    Report = classification_report(y_test,model[i].predict(X_test))
    print(Report)


# In[ ]:


# Evaluate model
#from sklearn.metrics import accuracy_score

#print (accuracy_score(y_test,pred))


# In[ ]:


pred_test=model[4].predict(test)  


# In[ ]:


final_predFile=pd.DataFrame({ 'PassengerId': test_passengerid,
                               'Survived': pred_test})


# In[ ]:


# create final CSV file.
final_predFile.to_csv(r'ResultSubmission.csv',index=False)

