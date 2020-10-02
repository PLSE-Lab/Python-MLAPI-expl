#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/titanic/train.csv')
df1=pd.read_csv('/kaggle/input/titanic/test.csv')
df3=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


df3


# In[ ]:


df.dtypes


# In[ ]:


df.corr()


# In[ ]:


df.drop(['Ticket','Fare'],axis=1,inplace=True)


# In[ ]:


df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Pclass',hue='Survived',data=df)


# In[ ]:


#above plot shows that the survival decreasing rate is in order of 3<2<1 class


# In[ ]:


sns.countplot(x='Sex',hue='Survived',data=df)


# In[ ]:


#Above plot shows the survival rate is less in male


# In[ ]:


sns.countplot(x='Embarked',hue='Survived',data=df)


# In[ ]:


#Above plot shows that the survival rate of people boarded at Southampton is very less


# In[ ]:


sns.countplot(x='Survived',hue='SibSp',data=df)


# In[ ]:


#the above plot shows that the people who came alone have less survival rate


# In[ ]:


sns.countplot(df['Survived'])


# In[ ]:


df['Survived'].value_counts()


# In[ ]:


df.drop('Name',axis=1,inplace=True)


# In[ ]:


df


# In[ ]:


#converted the Gender to numerics
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Sex']=lb.fit_transform(df['Sex'])
##test data preprocessing
df1['Sex']=lb.fit_transform(df1['Sex'])


# In[ ]:


#target variable is 'Survived';Features identified is 'Sex','Age',Pclass'
X = df.iloc[:,2:5]
Y = df.iloc[:,1]


# In[ ]:


#filling the null values of Age with 0 and coverting to integer
X['Age']=X['Age'].fillna(0)
X['Age']=X['Age'].astype(int)


# In[ ]:


#test data preprocessing
df1['Age']=df1['Age'].fillna(0)
df1['Age']=df1['Age'].astype(int)


# In[ ]:


test=df1.iloc[:,1:5]
test


# In[ ]:


test.drop('Name',axis=1,inplace=True)
#test data prepared based on trained features


# In[ ]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X=imp_mean.fit_transform(X)
X


# In[ ]:


#test data preprocessing
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
test=imp_mean.fit_transform(test)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 42)


# In[ ]:


from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train,Y_train)


# In[ ]:


predicted = xg.predict(X_test)
print(predicted)


# In[ ]:


from sklearn.metrics import accuracy_score,mean_squared_error,classification_report
from math import sqrt
rmse = sqrt(mean_squared_error(predicted,Y_test))
accuracy = accuracy_score(Y_test,predicted)
print(rmse)
print(accuracy)
#XGBoost classifier has an accuracy of 78% on survival preidictions 


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)
rand_predicted=model.predict(X_test)
print(accuracy_score(rand_predicted,Y_test))
print(sqrt(mean_squared_error(rand_predicted,Y_test)))
print(classification_report(Y_test, rand_predicted))


# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,Y_train)


# In[ ]:


svc_predict=grid.predict(X_test)


# In[ ]:


print(sqrt(mean_squared_error(svc_predict,Y_test)))
print(accuracy_score(svc_predict,Y_test))
print(classification_report(Y_test, svc_predict))


# In[ ]:


#SVM gave an accuracy of 82%
#RandomForest gave an accuacy of 76%
#XGBoost gave an accuarcy of 78%


# In[ ]:


test_svm_predicted=grid.predict(test)


# In[ ]:


gender_submission = pd.DataFrame(data=test_svm_predicted,columns = ['Survived'])


# In[ ]:


gender_submission['PassengerId'] = df1['PassengerId']
gender_submission = gender_submission[['PassengerId','Survived']]
gender_submission


# In[ ]:


gender_submission.to_csv("gender_submission.csv" , index = False)

