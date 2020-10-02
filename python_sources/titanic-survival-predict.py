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
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')


# **Statement: Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster. Can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.****

# In[ ]:


train_df.shape , test_df.shape


# In[ ]:


train_df.head()


# **Any missing values or nan in our dataset ? lets find out**

# In[ ]:


print(f'There are {train_df.isnull().any().sum()} columns in train dataset with missing values.')
print(f'There are {test_df.isnull().any().sum()} columns in test dataset with missing values.')


# In[ ]:


sns.pairplot(train_df)


# In[ ]:


print('Training Data')
ms = train_df['Age'].isnull().sum()
ms1 = train_df['Cabin'].isnull().sum()
ms2 = train_df['Embarked'].isnull().sum()
print(f'There are {ms} missing values in train Age column ')
print(f'There are {ms1} missing values in train Cabin column ')
print(f'There are {ms2} missing values in train Embarked column ')
print('--------------------------------------------------------------------')
print('Test Data')
mst = test_df['Age'].isnull().sum()
mst1 = test_df['Cabin'].isnull().sum()
mst2 = test_df['Embarked'].isnull().sum()
print(f'There are {mst} missing values in train Age column ')
print(f'There are {mst1} missing values in train Cabin column ')
print(f'There are {mst2} missing values in train Embarked column ')


# In[ ]:


train_df.Age = train_df.Age.fillna(-999)
train_df['Age'].isna().sum()


# In[ ]:


test_df['Age'].isna().sum()


# In[ ]:


test_df.Age = test_df.Age.fillna(test_df.Age.mean())
test_df.Fare = test_df.Fare.fillna(test_df.Fare.mean())
test_df['Age'].isna().sum()


# In[ ]:


test_df['Fare'].isna().sum()


# In[ ]:


print(train_df.isna().sum())
print('**************************')
print(test_df.isna().sum())


# In[ ]:


train_df.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
train_df['Embarked']= train_df['Embarked'].fillna('S')


# In[ ]:


train_df.head()


# In[ ]:


data = [train_df, test_df]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset["Fare"].astype(int)


# In[ ]:


test_df.drop(['Cabin'], axis=1,inplace=True)


# In[ ]:


test_df.drop(['Name','Ticket'],axis=1,inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


sex_dummies_train = pd.get_dummies(train_df['Sex'])
sex_dummies_test = pd.get_dummies(test_df['Sex'])
emb_dummies_train = pd.get_dummies(train_df['Embarked'])
emb_dummies_test = pd.get_dummies(test_df['Embarked'])


# In[ ]:


train_df = pd.concat([train_df,sex_dummies_train,emb_dummies_train],axis=1)
test_df = pd.concat([test_df,sex_dummies_test,emb_dummies_test],axis=1)


# In[ ]:


test_df.info()


# In[ ]:


test_df.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[ ]:


X_train = train_df.drop(['Sex','Embarked'],axis=1)


# In[ ]:


data = [X_train, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[ ]:


final_X = X_train.drop('Survived',axis=1)
final_y = X_train['Survived']


# In[ ]:


final_X.shape,final_y.shape


# # LogisticRegression

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_X,final_y, test_size=0.3,random_state=15)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(final_X,final_y)
Y_pred = model.predict(test_df)
acc_log = round(model.score(final_X,final_y) * 100, 2)
acc_log


# # SVC

# In[ ]:


# import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
clf = XGBClassifier()
# eval_set = [(X_train, y_train), (X_test, y_test)]
# eval_metric = ["auc","error"]

model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)
model.fit(final_X,final_y,  verbose=True)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[ ]:





# # Random Forest

# In[ ]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(final_X,final_y)
Y_pred = random_forest.predict(test_df)
random_forest.score(final_X,final_y)
acc_random_forest = round(random_forest.score(final_X,final_y) * 100, 2)
acc_random = print(round(acc_random_forest,2,), "%")
acc_random


# # Decision Tree
# #### DecisionTreeClassifier

# In[ ]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(final_X,final_y)

Y_predit = decision_tree.predict(test_df)

acc_decision_tree = round(decision_tree.score(final_X,final_y) * 100, 2)
acc_rando = print(round(acc_decision_tree,2,), "%")
acc_rando


# #### DecisionTreeRegressor

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(final_X,final_y)
Y_pred = regressor.predict(X_test)
reg_decision_tree = round(regressor.score(final_X,final_y) * 100, 2)
print(round(reg_decision_tree,2,), "%")


# # Gaussian Naive Bayes

# In[ ]:


# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(final_X,final_y)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(final_X,final_y) * 100, 2)
acc_gaussia = print(round(acc_gaussian,2,), "%")
acc_gaussia


# In[ ]:


test_df.shape,train_df.shape


# In[ ]:


test_df.head()


# In[ ]:


# models = pd.DataFrame({
#     'Model': ['Logistic Regression', 
#               'SVC', 'Random forest', 'Gaussian',  
#               ],
#     'Score': [acc_log, acc_svc, acc_rando, 
#              acc_gaussia 
#               ]})
# models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
        "Survived": Y_predit
})


# In[ ]:




