#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV


# In[ ]:


train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
combine = [train, test]


# In[ ]:


train.head()


# # Exploratory Data Analysis
# 
# 
# ## Missing Data

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ___
# ## Data Cleaning
# 

# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


for df in combine:
    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# In[ ]:


train.dropna(inplace=True)


# In[ ]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Converting Titles in Names into numbers
# 

# In[ ]:


for df in combine:
    df['Title'] = df['Name'].apply(lambda x:x.split(',')[1])
    df['Title'] = df['Title'].apply(lambda x:x.split()[0])
    df['Title'] = df['Title'].map(lambda x: x.replace('.',''))
train.Title.value_counts()[:6]


# In[ ]:


for df in combine:
    df['Title'] = df['Title'].replace(['Don', 'Rev', 'Dr', 'Mme',       'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'the',       'Jonkheer'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for df in combine:
    df['Title'] = df['Title'].map(titles)
    df['Title'] = df['Title'].fillna(0)


# # Converting Sex feature into numbers

# In[ ]:


for df in combine:
    df['Sex'] = df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# # Creating Dummy variables

# In[ ]:


embark_train = pd.get_dummies(train['Embarked'],drop_first=True)
embark_test = pd.get_dummies(test['Embarked'],drop_first=True)


# In[ ]:


for df in combine:
    df.drop(['Embarked','Name','Ticket'],axis=1, inplace=True)


# In[ ]:


train = pd.concat([train,embark_train],axis=1)
test = pd.concat([test,embark_test],axis=1)


# In[ ]:


test.Fare = test.Fare.replace(np.nan, test['Fare'].median())


# In[ ]:


test.head()


# # Creating intervals for 'Age'

# In[ ]:


train['Age_int'] =pd.cut(train['Age'], 5)
train['Age_int'].value_counts()


# In[ ]:


frames = [train,test]
for df in frames:    
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4


# # Creating intervals for 'Fare'

# In[ ]:


train['Fare_int'] =pd.cut(train['Fare'], 4)
train['Fare_int'].value_counts()


# In[ ]:


for df in frames:    
    df.loc[ df['Fare'] <= 128, 'Fare'] = 0
    df.loc[(df['Fare'] > 128) & (df['Fare'] <= 256), 'Fare'] = 1
    df.loc[(df['Fare'] > 256) & (df['Fare'] <= 384), 'Fare'] = 2
    df.loc[ df['Fare'] > 384, 'Fare'] = 3


# In[ ]:


cols = ['Age_int','Fare_int']
train = train.drop(cols, axis=1)


# # Creating New feature for 'Parch' and 'SibSp'

# In[ ]:


train['FamSize'] = train['SibSp'] + train['Parch'] + 1
test['FamSize'] = test['SibSp'] + test['Parch'] + 1


# In[ ]:


train['IsAlone'] = train['FamSize']
test['IsAlone'] = test['FamSize']


# In[ ]:


train['IsAlone'].unique()


# In[ ]:


train['IsAlone'] = train['IsAlone'].replace([ 2,  5,  3,  7,  6,  4,  8, 11], 0)
test['IsAlone'] = test['IsAlone'].replace([ 2,  5,  3,  7,  6,  4,  8, 11], 0)


# In[ ]:


train.head()


# 
# 
# # Building a first level model
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=0)


# In[ ]:


rf = RandomForestClassifier(n_jobs= -1,n_estimators= 500,warm_start=True,max_depth= 6,min_samples_leaf= 2,max_features= 'sqrt',
    verbose=0)
et = ExtraTreesClassifier(n_jobs= -1,n_estimators= 500,max_depth= 8,min_samples_leaf= 2,verbose=0)
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.75)
gb = GradientBoostingClassifier(n_estimators=500,max_depth= 5,min_samples_leaf= 2)
svc = SVC()


# In[ ]:


models = [rf,et,ada,gb,svc]
for model in models:
    model.fit(X_train, y_train)
    


# In[ ]:


preds_rf=rf.predict(X_test)
preds_et=et.predict(X_test)
preds_ada=ada.predict(X_test)
preds_gb=gb.predict(X_test)
preds_svc=svc.predict(X_test)


# In[ ]:


#Creating meta-features to train second level model
x_first = pd.DataFrame( {'RandomForest': preds_rf,
     'ExtraTrees': preds_et,
     'AdaBoost': preds_ada,
      'GradientBoost': preds_gb,
        'SVC': preds_svc
    })


# In[ ]:


#Creating meta-features to predict second level model
preds_rf=rf.predict(test)
preds_et=et.predict(test)
preds_ada=ada.predict(test)
preds_gb=gb.predict(test)
preds_svc=svc.predict(test)


# In[ ]:


x_second = pd.DataFrame( {'RandomForest': preds_rf,
     'ExtraTrees': preds_et,
     'AdaBoost': preds_ada,
      'GradientBoost': preds_gb,
       'SVC': preds_svc
    })


# In[ ]:


#adding alorithms like logreg and KNN, or excluding SVC didn't help to improve result
sns.heatmap(x_first.corr(), annot=True)


# # Building meta-model
# 

# In[ ]:


gbm = xgb.XGBClassifier(learning_rate=0.001,n_estimators=2500,
                                max_depth=4, min_child_weight=0,
                                gamma=0, subsample=0.7,
                                colsample_bytree=0.7,
                                scale_pos_weight=1, seed=27,
                                reg_alpha=0.00006)


# In[ ]:


gbm.fit(x_first,y_test)
predictions_final = gbm.predict(x_second)


# # Submission

# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_final})
submission.to_csv('gender_submission.csv', index=False)

