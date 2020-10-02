#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import all the required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender_submission = pd.read_csv('../input/gender_submission.csv')
train.tail(5)


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


import pandas_profiling
pandas_profiling.ProfileReport(train)


# In[ ]:


train_df_corr = train.corr()
fig,ax = plt.subplots(figsize=(12,10))
sns.heatmap(train_df_corr,annot=True);


# In[ ]:


#let's remove the unwanted columns
train_df = train.drop(['Name','Ticket','Cabin'],axis=1)
test_df = test.drop(['Name','Ticket','Cabin'],axis=1)
train_df.head()


# In[ ]:


#Check for missing values
train_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


#Check for missing values
test_df.isnull().sum().sort_values(ascending=False)


# In[ ]:


# Replace Null Values (np.nan) with meadian & mode
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# Fit and transform to the parameters
train_df['Embarked'] = imputer.fit_transform(train_df[['Embarked']])

# train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode())


# In[ ]:


print("Training Dataset :\n",train_df.isnull().sum().sort_values(ascending=False))
print()
print("Test Dataset :\n",test_df.isnull().sum().sort_values(ascending=False))

# train_df = train_df.dropna()
# test_df = test_df.dropna()


# In[ ]:


train_df.describe()


# In[ ]:



Counter(train_df['Survived'])


# In[ ]:


fx,ax = plt.subplots(figsize=(12,10))
plt.scatter(train_df['Age'],train_df['Survived'])
plt.xlabel('Age')
plt.ylabel('Survived');


# In[ ]:


fig,ax = plt.subplots(figsize=(12,10))
plt.scatter(train_df['Fare'],train_df['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived');


# In[ ]:


pd.crosstab(train_df['Sex'],train_df['Survived'])


# In[ ]:


pd.crosstab(train_df['Pclass'],train_df['Survived'])


# In[ ]:


from sklearn.preprocessing import LabelEncoder

label_encoding = LabelEncoder()
train_df['Sex'] = label_encoding.fit_transform(train_df['Sex'].astype(str))
train_df.sample(5)


# In[ ]:


test_df['Sex'] = label_encoding.fit_transform(test_df['Sex'].astype(str))
test_df.head()


# In[ ]:


train_df = pd.get_dummies(train_df,columns=['Embarked'])
train_df.head()


# In[ ]:


test_df = pd.get_dummies(test_df,columns=['Embarked'])
test_df.head()


# In[ ]:


Pass_ID_test = test_df['PassengerId']
train_df = train_df.drop(['PassengerId'],axis=1)
test_df = test_df.drop(['PassengerId'],axis=1)


# In[ ]:


len(Pass_ID_test)


# ## Classification Model 1: Logistic Regression

# In[ ]:


features = train_df.drop(['Survived'],axis=1)
target = train_df['Survived']
print('Dimension of Fearures ;',features.shape)
print('Dimension of target column ;',target.shape)
print('Dimension of test dataset ;',test_df.shape)


# In[ ]:


logistic_model = LogisticRegression(penalty='l2',C=1.0,solver='liblinear').fit(features,target)
test_pred = logistic_model.predict(test_df)
len(test_pred)


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters = {'penalty': ['l1','l2'],
             'C': [0.1,0.4,0.8,1,2,3,5]}

grid_search = GridSearchCV(LogisticRegression(solver='liblinear'),parameters,cv=3,return_train_score=True)
grid_search.fit(features,target)

grid_search.best_params_


# In[ ]:


for i in range(12):
    print('Parameters: ',grid_search.cv_results_['params'][i])
    print('Mean Test Score: ',grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ',grid_search.cv_results_['rank_test_score'][i])


# In[ ]:


logistic_model = LogisticRegression(solver='liblinear',                                   penalty=grid_search.best_params_['penalty'],C=grid_search.best_params_['C']).                                   fit(features,target)


# In[ ]:


test2_pred = logistic_model.predict(test_df)


# In[ ]:


len(test2_pred)


# ## Classification Model 2 : KNN

# In[ ]:


## for the number of k neighbors
k = list(range(1, 60, 2))

## for the weights
weights_options = ['uniform', 'distance']

## for the algorithms applied 
algos = ['ball_tree', 'kd_tree', 'brute']

## leaf size (since i've initiated BallTree and KDTree algorithms)
leaves = list(np.arange(10, 110, 10))

## for the metrics
metric_options = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']

## for the parameters of the metrics
#metric_params=metric_param_options

# initializing the grid

params_grid = dict(n_neighbors=k, weights=weights_options, algorithm=algos, leaf_size=leaves, metric=metric_options)

# initializing the grid search with 10 cross_validation splits

model_KNN = KNeighborsClassifier() 

grid = GridSearchCV(model_KNN, params_grid, cv=10, scoring='accuracy',n_jobs=-1)

# training the model
grid.fit(features,target)


# In[ ]:


print(f'best parameters: {grid.best_params_},\nbest accuracy score: {grid.best_score_},\nbest estimator: {grid.best_estimator_}')


# In[ ]:



model_knn  = KNeighborsClassifier(algorithm='ball_tree', leaf_size=10, metric='manhattan',
                     metric_params=None, n_jobs=None, n_neighbors=15, p=2,weights='distance')
model_knn.fit(features,target)

kkn_test_pred = model_knn.predict(test_df)


# ## Logistics Regression Submission

# In[ ]:


logreg_test_submission = pd.DataFrame({'PassengerId':Pass_ID_test,'Survived':test2_pred})
logreg_test_submission.head()


# In[ ]:


logreg_test_submission.shape


# In[ ]:


logreg_test_submission.to_csv("logreg_test_submission.csv",index=False)


# ## KNN Submission

# In[ ]:


knn_submission = gender_submission.copy()

knn_submission['Survived'] = kkn_test_pred

knn_submission.shape


# In[ ]:


knn_submission.head()


# # Thank you 
