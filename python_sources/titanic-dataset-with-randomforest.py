#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train['Cabin'].isnull().sum()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_train['Survived'].unique()


# In[ ]:


df_train['Age'].unique()


# In[ ]:


df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())


# In[ ]:


df_train['Embarked'].unique()


# In[ ]:


df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].median)


# In[ ]:


#df_train['Cabin'] = df_train['Cabin'].fillna(df_train['Cabin'].median)


# In[ ]:


df_train.head()


# In[ ]:


df_train=df_train.drop(labels='Cabin',axis=1)


# In[ ]:


df_test


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_test = df_test.drop(labels='Cabin',axis=1)


# In[ ]:


df_test['Age'] = df_test['Age'].fillna(df_test['Age'].mean())


# In[ ]:


df_test.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()


# In[ ]:


df_train['Name'] = label.fit_transform(df_train['Name'])


# In[ ]:


df_train['Name']


# In[ ]:


df_train['Age'] = label.fit_transform(df_train['Age'])


# In[ ]:


df_train['Sex'] = label.fit_transform(df_train['Sex'])
df_train['SibSp'] = label.fit_transform(df_train['SibSp'])
df_train['Parch'] = label.fit_transform(df_train['Parch'])
df_train['Ticket'] = label.fit_transform(df_train['Ticket'])
df_train['Fare'] = label.fit_transform(df_train['Fare'])
#df_train['Cabin'] = label.fit_transform(df_train['Cabin'])


# In[ ]:


df_train['Embarked'] = label.fit_transform(df_train['Embarked'].astype(str))


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_test['Name'] = label.fit_transform(df_test['Name'])
df_test['Age'] = label.fit_transform(df_test['Age'])
df_test['Sex'] = label.fit_transform(df_test['Sex'])
df_test['SibSp'] = label.fit_transform(df_test['SibSp'])
df_test['Parch'] = label.fit_transform(df_test['Parch'])
df_test['Ticket'] = label.fit_transform(df_test['Ticket'])
df_test['Fare'] = label.fit_transform(df_test['Fare'])
df_test['Embarked'] = label.fit_transform(df_test['Embarked'].astype(str))


# In[ ]:


df_test.head()


# In[ ]:


x = df_train
target= df_test


# In[ ]:


x.head()


# In[ ]:


target.head()


# In[ ]:


X = x.drop(labels=["PassengerId",'Survived'],axis=1)
y = x['Survived']


# In[ ]:


X_scaled = scaler.fit_transform(X)
X_scaled


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.7,random_state=120)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
xg=xgb.XGBClassifier(random_state=1,learning_rate=0.01)
xg.fit(X_train, y_train)
xg.score(X_test,y_test)
y_pred=xg.predict(X_test)
accuracy_score(y_pred,y_test)


# In[ ]:


from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(xg,X_test,y_test,cmap=plt.cm.Blues,normalize=None)
#disp = plot_confusion_matrix(lg,X_test,y_test,cmap='viridis',normalize=None)
disp.confusion_matrix


# In[ ]:


x.head()


# In[ ]:


k = xg.fit(X, y)
k


# In[ ]:


# random_grid = {'n_estimators': [1,2,3,4,5,,6,20],
#                'max_features': ['auto', 'sqrt'],
#                'max_depth': [10, 20, 30, 40, 50, 60,80,90,100],
# #                'min_samples_split': [2, 5, 10,15,20],
#                'min_samples_leaf': [1, 2, 4,5],
#                'bootstrap': [True, False]}


# In[ ]:


# from sklearn.model_selection import RandomizedSearchCV
# randomcv= RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 80, cv =5, verbose=4, random_state=132, n_jobs = -1)


# In[ ]:


# randomcv.fit(X,y)


# In[ ]:


# model = RandomForestClassifier(n_estimators=20,min_samples_split=3,min_samples_leaf=6,max_features='sqrt',max_depth=20,bootstrap=True)


# In[ ]:


# model.fit(X,y)


# In[ ]:


test_data = df_test.drop("PassengerId", axis=1).copy()
#prediction = rf.predict(test_data)
test_data


# In[ ]:


prediction = xg.predict(test_data)


# In[ ]:


df_test['PassengerId']


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": prediction
    })


# In[ ]:


submission.to_csv('gender_submission.csv')


# In[ ]:




