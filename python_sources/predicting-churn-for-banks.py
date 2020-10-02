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


# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)

#timer
import time
from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} done in {:.0f}s".format(title, time.time() - t0))

# Importing modelling libraries
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV


# Importing Data

# In[ ]:


df = pd.read_csv('/kaggle/input/predicting-churn-for-bank-customers/Churn_Modelling.csv')


# Explarotary Data Analysis

# In[ ]:


df.info()


# In[ ]:


df = df.astype({"Age": float})
df.head()


# In[ ]:


df = df.astype({"Age": float})


# In[ ]:


df.shape


# In[ ]:


df['Exited'].value_counts()


# CATEGORICAL VARIABLES

# In[ ]:


df['Gender'].value_counts()


# In[ ]:


df['Geography'].value_counts()


# In[ ]:


df['NumOfProducts'].value_counts()


# In[ ]:


df['HasCrCard'].value_counts()


# In[ ]:


df['IsActiveMember'].value_counts()


# In[ ]:


df.groupby('IsActiveMember')['Exited'].value_counts()


# In[ ]:


print('The ratio of retention of active members',end=": ")
print(round(4416/5151*100,2))
print('The ratio of retention of passive members',end=": ")
print(round(3547/4849*100,2))

A higher part of the active members seems to stay with the company, compared to the non active members. 
# In[ ]:


df['Gender'].value_counts()


# In[ ]:


df.groupby('Gender')['Exited'].value_counts()


# In[ ]:


print('The ratio of retention of men',end=": ")
print(round(4559/5457*100,2))
print('The ratio of retention of women',end=": ")
print(round(3404/4543*100,2))


# A higher part of men seems to stay with the company, compared to women. 

# In[ ]:


df['Geography'].value_counts()


# In[ ]:


df.groupby('Geography')['Exited'].value_counts()


# In[ ]:


print('The ratio of retention of customers from France',end=": ") 
print(round(4204/5014*100,2))
print('The ratio of retention of  customers from Spain',end=": ")
print(round(2064/2477*100,2))
print('The ratio of retention of customers from Germany',end=": ")
print(round(1695/2509*100,2))


# A higher part of customers from France and Spain seems to stay with the company, compared to Germany.

# In[ ]:


df['NumOfProducts'].value_counts()


# In[ ]:


df.groupby('NumOfProducts')['Exited'].value_counts()


# In[ ]:


print('The ratio of retention of customers with 2 products',end=": ") 
print(round(4242/4590*100,2))
print('The ratio of retention of  customers with 1 product',end=": ")
print(round(3675/5084*100,2))


# A higher part of customers with 2 products seems to stay with the company, compared to customers with 1 product.

# In[ ]:


df['HasCrCard'].value_counts()


# In[ ]:


df.groupby('HasCrCard')['Exited'].value_counts()


# In[ ]:


print('The ratio of retention of customers with credit card',end=": ") 
print(round(2332/2945*100,2))
print('The ratio of retention of  customers without credit card',end=": ")
print(round(5631/7055*100,2))


# There is no significant difference between the customers who hold credit cards or not in terms of retention with the bank.

# NUMERICAL VARIABLES

# In[ ]:


df.describe().T


# In[ ]:


g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "Age", bins = 25)
plt.show()


# It seems younger customers tend to stick with the company more compared to older customers.

# In[ ]:


g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "CreditScore", bins = 25)
plt.show()


# It is interesting to see that there seems to be no relationship between credit score and exiting the company.

# In[ ]:


g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "Tenure", bins = 25)
plt.show()


# It is interesting to see that there seems to be no relationship between tenure and exiting the company.

# In[ ]:


g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "Balance", bins = 25)
plt.show()


# It is interesting to see that there seems to be no relationship between balance and exiting the company.

# In[ ]:


g = sns.FacetGrid(df, col = "Exited")
g.map(sns.distplot, "EstimatedSalary", bins = 25)
plt.show()


# It is interesting to see that there seems to be no relationship between estimated salary and exiting the company.

# In[ ]:


# Let's visualize the correlations between numerical features of data.
fig, ax = plt.subplots(figsize=(12,6)) 
sns.heatmap(df.iloc[:,1:len(df)].corr(), annot = True, fmt = ".2f", linewidths=0.5, ax=ax) 
plt.show()


# As it can be seen from the figure there is no significant correlation among numerical variables. It seems that only the variable of Age has some kind of correlation with the variable of Exited.

# Data Preparation
# 
# Dropping Certain Variables: RowNumber, Surname, CustomerId

# In[ ]:


df.drop(['RowNumber'], axis = 1, inplace = True)
df.drop(['Surname'], axis = 1, inplace = True)
df.drop(['CustomerId'], axis = 1, inplace = True)


# In[ ]:


df.head()


# Label encoding of the variable Gender to a dummy variable (0-1)

# In[ ]:


for d in [df]:
    d["Gender"]=d["Gender"].map(lambda x: 0 if x=='Female' else 1)


# In[ ]:


df.head()


# One hot encoding of Tenure, Geography and NumOfProducts

# In[ ]:


df = pd.get_dummies(df, columns=["Tenure"])


# In[ ]:


df = pd.get_dummies(df, columns=["NumOfProducts"])


# In[ ]:


df = pd.get_dummies(df, columns=["Geography"])


# In[ ]:


df.head()


#  Modeling, Evaluation and Model Tuning

# Splitting the data as train and test

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
predictors = df.drop(['Exited'], axis=1)
target = df["Exited"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.20, random_state = 0)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


y_train.head()


# In[ ]:


x_val.shape


# Accuracy Scores for the Default models

# In[ ]:


r=1309
models = [LogisticRegression(random_state=r),GaussianNB(), KNeighborsClassifier(),
          SVC(random_state=r,probability=True),DecisionTreeClassifier(random_state=r),
          RandomForestClassifier(random_state=r), GradientBoostingClassifier(random_state=r),
          XGBClassifier(random_state=r), MLPClassifier(random_state=r),
          CatBoostClassifier(random_state=r,verbose = False)]
names = ["LogisticRegression","GaussianNB","KNN","SVC",
             "DecisionTree","Random_Forest","GBM","XGBoost","Art.Neural_Network","CatBoost"]


# In[ ]:


print('Default model validation accuracies for the train data:', end = "\n\n")
for name, model in zip(names, models):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val) 
    print(name,':',"%.3f" % accuracy_score(y_pred, y_val))


# Cross Validation Accuracy Scores of the Default Models

# In[ ]:


results = []
print('10 fold cross validation accuracy scores of the default models:', end = "\n\n")
for name, model in zip(names, models):
    kfold = KFold(n_splits=10, random_state=1001)
    cv_results = cross_val_score(model, predictors, target, cv = kfold, scoring = "accuracy")
    results.append(cv_results)
    print("{}: {} ({})".format(name, "%.3f" % cv_results.mean() ,"%.3f" %  cv_results.std()))


# Model tuning using crossvalidation

# In[ ]:


# Tuning by Cross Validation  
rf_params = {"max_features": ["log2","Auto","None"],
                "min_samples_split":[2,3,5],
                "min_samples_leaf":[1,3,5],
                "bootstrap":[True,False],
                "n_estimators":[50,100,150],
                "criterion":["gini","entropy"]}
rf = RandomForestClassifier()
rf_cv_model = GridSearchCV(rf, rf_params, cv = 5, n_jobs = -1, verbose = 2)
rf_cv_model.fit(x_train, y_train)
rf_cv_model.best_params_


# In[ ]:


rf = RandomForestClassifier(bootstrap = True, criterion = 'entropy' , max_features = 'log2', min_samples_leaf = 3, min_samples_split = 3,
 n_estimators = 100)
rf_tuned = rf.fit(x_train,y_train)
y_pred = rf_tuned.predict(x_val) 
acc_rf = round(accuracy_score(y_pred, y_val) * 100, 2) 
print(acc_rf)


# In[ ]:


predictions = y_pred


# In[ ]:


output = pd.DataFrame({ 'Exited': predictions }) 
output.to_csv('submission.csv', index=False)
output.head()


# In[ ]:


output.describe().T


# In[ ]:


output["Exited"].value_counts()


# In[ ]:


Retention_Rate = 1734/2000*100


# In[ ]:


print(str(Retention_Rate)+'%')


# In[ ]:




