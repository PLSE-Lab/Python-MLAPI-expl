#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In this kernel I am not going to do any visualization or playing around with unnecessarily columns.
# Based on my assumptions and general idea, I am directly jumping into data preprocessing and building models.
# 
# ALSO,
# Please NOTE : I am not tuning parameter in this as it is base model predection. I am going to work on param tuning in the next kernal

# In[ ]:


# Import necessory libraries
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Explore
df_train = pd.read_csv('/kaggle/input/titanic/train.csv',sep=',')
df_train.head()


# In[ ]:


# Shape of the dataset
df_train.shape


# In[ ]:


# Any nulls?
df_train.isnull().sum()


# In the below function, I am taking care of filling null values based on class and gender, removing unwanted columns.

# In[ ]:


def titanic(df):
# STEP1  After analysing Columns such as 'PassengerId','Cabin','Ticket','Name','Fare','Embarked' do not add values, so we are dropping.    
    df.drop(columns=['PassengerId','Cabin','Ticket','Name','Fare','Embarked'],axis=1,inplace=True)

# STEP 2   Converting Categorical into Numerical      
    df['Sex'].replace('male', 1,inplace=True)
    df['Sex'].replace('female', 0,inplace=True)
    
# STEP 3  Since Age is important factor and taking average age in each class w.r.t Age for further imputation    
    class_one_male = df.Age[(df['Pclass'] == 1) & (df['Sex'] == 1 )].mean()
    class_one_female = df.Age[(df['Pclass'] == 1) & (df['Sex'] == 0)].mean()
    class_two_male = df.Age[(df['Pclass'] == 2) & (df['Sex'] == 1)].mean()
    class_two_female = df.Age[(df['Pclass'] == 2) & (df['Sex'] == 0)].mean()
    class_three_male = df.Age[(df['Pclass'] == 3) & (df['Sex'] == 1)].mean()
    class_three_female = df.Age[(df['Pclass'] == 3) & (df['Sex'] == 0)].mean()

    
# STEP 4  We are seperating passenger of each class w.r.t gender so that we can impute missing age accordingly     
    df_male_o = df[(df['Pclass'] == 1) & (df['Sex'] == 1)]
    df_female_o= df[(df['Pclass'] == 1) & (df['Sex'] == 0)]
    df_male_t = df[(df['Pclass'] == 2) & (df['Sex'] == 1)]
    df_female_t= df[(df['Pclass'] == 2) & (df['Sex'] == 0)]
    df_male_th = df[(df['Pclass'] == 3) & (df['Sex'] == 1)]
    df_female_th= df[(df['Pclass'] == 3) & (df['Sex'] == 0)]

# STEP 5 Null Imputation for Age
    df_male_o['Age'].fillna(value=class_one_male,inplace=True)
    df_female_o['Age'].fillna(value=class_one_female,inplace=True)
    df_male_t['Age'].fillna(value=class_two_male,inplace=True)
    df_female_t['Age'].fillna(value=class_two_female,inplace=True)
    df_male_th['Age'].fillna(value=class_three_male,inplace=True)
    df_female_th['Age'].fillna(value=class_three_female,inplace=True)

# STEP 6 Cancating above datasets   
    dataframes = [df_male_o,df_female_o,df_male_t,df_female_t,df_male_th,df_female_th]
    df_structured = pd.concat(dataframes)
    return df_structured


# In[ ]:


# Calling function
df = titanic(df = pd.read_csv('/kaggle/input/titanic/train.csv',sep=','))

# Check if any null exists
print (df.isnull().sum())


# In[ ]:


# Check for any data loss
print ('Final Shape = >',df.shape)

# training data sets
X_train,y_train = df.drop('Survived',axis=1),df.Survived


# In[ ]:


# scoring metrics
from sklearn.metrics import scorer


# In[ ]:


# Logistic model for training
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)


# In[ ]:


# Importing test set for prediction
x_test= titanic(pd.read_csv('/kaggle/input/titanic/test.csv',sep=','))
log_pred = log.predict(x_test)
log_score = round(log.score(X_train,y_train)* 100,2)
print ('Logistic training Score ==>', log_score)


# In[ ]:


# Knn Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn_pred = knn.predict(x_test)
knn_score = round(knn.score(X_train,y_train)* 100,2)
print ('Knn training Score ==>', knn_score)


# In[ ]:


# Support Vector Machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(x_test)
svc_score = round(svc.score(X_train, y_train) * 100, 2)
print ('SVC training Score ==>',svc_score)


# In[ ]:


# Sthocastic Gradiant Descent Classifier
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(x_test)
sgd_score = round(sgd.score(X_train, y_train) * 100, 2)
print ('SGDC training Score ==>',sgd_score)


# In[ ]:


# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(x_test)
dt_score = round(decision_tree.score(X_train, y_train) * 100, 2)
print ('Dic Tree training Score ==>',dt_score)


# In[ ]:


# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=50,criterion='entropy')
random_forest.fit(X_train, y_train)
rm_pred = random_forest.predict(x_test)
random_forest.score(X_train, y_train)
rm_score = round(random_forest.score(X_train, y_train) * 100, 2)
print ('Random Forest training Score ==>',rm_score)


# In[ ]:


models = pd.DataFrame({'Classifier':['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Stochastic Gradient Decent', 'Decision Tree'],
                      'Score':[svc_score,knn_score,log_score,rm_score,sgd_score,dt_score]})
models.sort_values(by='Score',ascending=False)


# In[ ]:


# Decision tree and Random forest seems to be a best classifier, so will be submitting the top results.
titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv',sep=',')
submission = pd.DataFrame({'PassengerId': titanic_test['PassengerId'],
                          'Survived': dt_score})


# In[ ]:


#submission.to_csv('/kaggle/input/titanic/gender_submission.csv',index=False)


# In[ ]:




