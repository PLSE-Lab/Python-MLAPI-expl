#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# 1. [Introduction](#intro)
# 2. [Imports](#Imports)
# 3. [Acquire data](#Acquire)
# 4. [Visualisations](#vis)
# 5. [Feature Scaleing and Droping unnecessary columns](#feature)
# 6. [Modeling](#model)
# 7. [Conclusion](#con)

# ## Introduction <a name="intro">
# This Kernal is from the Titanic dataset the objective is to predict if somone has survived based on some features. There is a training set with labeled data and a test set that you have to make predictions of. This Dataset is used by allot of people as an Introduction to Kaggle and I'm useing it for the same purpose
# 
# 

# ## Imports <a name="Imports">

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.ensemble import RandomForestClassifier
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
# Figures inline and set visualization style
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Acquire data <a name=Acquire>

# In[23]:


data_test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# ## Visualisations <a name=vis> 

# In[24]:


sns.countplot(x='Survived', data=train);


# In[25]:


sns.countplot(x='Sex', data=train);


# In[26]:


sns.catplot(x='Survived', col='Sex', kind='count', data=train);


# ## Feature Scaleing and Droping unnecessary columns <a name=feature>

# In[27]:


#Drop coloums that are not needed
train.drop('Name',axis = 1, inplace =True)
train.drop('Ticket', axis = 1, inplace = True)
data_test.drop('Name',axis = 1, inplace =True)
data_test.drop('Ticket', axis = 1, inplace = True)


# In[28]:


# feature scale sex column 
m = {'m' : 1, 'f' : 0}
train['Sex'] = train['Sex'].str[0].str.lower().map(m)
data_test['Sex'] = data_test['Sex'].str[0].str.lower().map(m)


# In[29]:


# feature scale Embarked
em = {'S':0, 'C': 1, 'Q' :2}
train['Embarked'] = train['Embarked'].str[0].str.upper().map(em)
train['Embarked'] = train['Embarked'].fillna(1)
data_test['Embarked'] = data_test['Embarked'].str[0].str.upper().map(em)
data_test['Embarked'] = data_test['Embarked'].fillna(1)


# In[30]:


# filled missing ages with the average age 
train['Age'] = train['Age'].fillna(train['Age'].median())
data_test['Age'] = train['Age'].fillna(train['Age'].median())


# In[31]:


# Replace missing values of fare with the average 
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
data_test['Fare'] = train['Fare'].fillna(train['Fare'].median())


# In[32]:


# change age and fare types to int
train['Age'] = train['Age'].astype('int64')
data_test['Age'] = data_test['Age'].astype('int64')
train['Fare'] = train['Fare'].astype('int64')
data_test['Fare'] = data_test['Fare'].astype('int64')


# In[33]:


# drop cabin
train.drop('Cabin', axis = 1, inplace = True)
data_test.drop('Cabin', axis = 1, inplace = True)


# In[34]:


# corrilation heat map of data after feature scaleing 
corr = train.corr()
sns.heatmap(corr, 
        xticklabels=train.columns,
        yticklabels=train.columns)


# In[35]:


'''
# normlising coloumns 
train=(train-train.min())/(train.max()-train.min())
temp = data_test['PassengerId']
data_test = (data_test-data_test.min())/(data_test.max()-data_test.min())
data_test['PassengerId'] = temp
'''


# ## Modeling  <a name=model>

# In[36]:


# seperating the data for training 
X_all = train.drop(['Survived', 'PassengerId'], axis=1)
y_all = train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23, shuffle = True)


# In[37]:


'''
# testing models 
model  = [
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    gaussian_process.GaussianProcessClassifier(),
    linear_model.LogisticRegressionCV(),
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    tree.DecisionTreeClassifier(),
    XGBClassifier(),
    SGDClassifier()
    ]
for i in model:
    print(str(i) + str(cross_val_score(i,X_test, y_test, scoring = "accuracy", cv = 10)))
'''


# In[38]:


'''
# gridsearch for paramaters 

rfc = XGBClassifier() 

param_grid = {"learning_rate"    : [0.01, 0.05] ,
              "max_depth"        : [ 3],
              "gamma"            : [ 0.1 ],
              "colsample_bytree" : [ 0.5, 0.7 ],
              'n_estimators': [1000],
              'subsample': [0.4]
              
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, scoring="accuracy")
CV_rfc.fit(X_train, y_train)
print(CV_rfc.best_params_)

'''


# In[ ]:





# In[39]:


'''
# fine tune paramaters 
cross_val = cross_val_score(ensemble.RandomForestClassifier(random_state = 90, warm_start = True, 
                                  min_samples_leaf = 1,
                                  min_samples_split = 2,
                                  n_estimators = 20,
                                  max_depth = 5, 
                                  max_features = 'sqrt'), X_test, y_test, cv=100)
print(np.mean(cross_val))
print(np.std(cross_val))
'''


# In[40]:



model = (ensemble.RandomForestClassifier(warm_start = True, 
                                  min_samples_leaf = 1,
                                  min_samples_split = 2,
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt'))
model.fit(X_all,y_all)
model.score(X_test,y_test)


# I tried ensembling a few models but got bad results I think I needed to spend more time on parameters or stacking the models 

# In[41]:


'''
model1 =  ensemble.RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
model2 =  ensemble.GradientBoostingClassifier(criterion = 'friedman_mse',
                                              learning_rate = 0.075,
                                              loss = 'deviance',
                                              max_depth = 5,
                                              max_features = 'sqrt',
                                              n_estimators = 10,
                                              subsample = 0.9)
model3 = XGBClassifier(colsample_bytree = 0.7,
                       gamma = 0.1, 
                       learning_rate = 0.05, 
                       max_depth = 3,
                       n_estimators = 1000,
                       subsample = 0.4)

model = VotingClassifier(estimators=[('rf', model1), ('Gb', model2), ('XGB', model3)], voting='soft')
model.fit(X_all,y_all)
model.score(X_test,y_test)
'''


# In[42]:


# useing model to make predictions and making the submission file 
test_pre = model.predict(data_test.drop(['PassengerId'], axis=1))
data_test["Survived"] = test_pre.astype(int)
data_test[['PassengerId', 'Survived']].to_csv('submission.csv', index = False)


# ## Conclusion <a name=con> 
# I tried RandomForestClassifier as it had the best results out of the models I tested I got reasonable results with it. I tried ensamble modeling and after optimising got similar results I would like to try a stacked the model to see if I can get better results. I had allot of fun with this dataset and learnt allot  
