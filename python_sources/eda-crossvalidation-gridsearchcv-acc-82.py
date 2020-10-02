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
import time

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## DATA DESCRIPTION
# ![](https://i.ibb.co/WF7Q1s7/Screenshot-25-02-2020-141202.jpg)

# ## **Loading data with pandas**

# In[ ]:


df = pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe().T


# * **There is no null row in dataset but numeric columns doesn't seem to be normally distributed. Generally standart deviations are much higher than averages.**

# In[ ]:


## Let's get rid of that ugly column name.
df.rename(columns={"default.payment.next.month":"Target"}, inplace = True)


# In[ ]:


print(df["MARRIAGE"].unique())
print(df["SEX"].unique())
print(df["EDUCATION"].unique())
print(df["PAY_0"].unique())


# * ** There are some poorly labeled rows in data set. There is no match between data description and data set. For example: Description say "MARRIAGE column: Marital Status 1=Married, 2-Single, 3-Others" but in real data there are 4 unique values (0-1-2-3) so I will replace 0 with 3. Same mistake made in different columns like EDUCATION and payment columns (PAY_0, PAY_2 etc. so I will replace them with logical values.
# **
# 

# In[ ]:


df['MARRIAGE'].replace({0 : 3},inplace = True)
df["EDUCATION"].replace({6 : 5, 0 : 5}, inplace = True)
df["PAY_0"].replace({-1 : 0, -2 : 0}, inplace = True)
df["PAY_2"].replace({-1 : 0, -2 : 0}, inplace = True)
df["PAY_3"].replace({-1 : 0, -2 : 0}, inplace = True)
df["PAY_4"].replace({-1 : 0, -2 : 0}, inplace = True)
df["PAY_5"].replace({-1 : 0, -2 : 0}, inplace = True)
df["PAY_6"].replace({-1 : 0, -2 : 0}, inplace = True)


# In[ ]:


print(df["PAY_0"].unique())
print(df["PAY_2"].unique())
print(df["PAY_3"].unique())
print(df["PAY_4"].unique())
print(df["PAY_5"].unique())
print(df["PAY_6"].unique())


# In[ ]:


## Original data set might be needed so let's backup it. 
df2 = df.copy()


# * **First of all, we will look at the correlation matrix. Although some columns are categorical data type,they have numeric values so we can directly use df.corr. If categorical values were string, we have to get rid of them.  **

# In[ ]:


## .corr generate correlation matrix in df columns.
corr = df.corr()
## np.zeros_like generates matrix which is same shape with correlation matrix so we can use it like mask for inner triangle matrix. 
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,vmin=-1, vmax=1, annot = True, mask = mask)


# * ** There is no strong correlation between Target and features. The highest correlation is approximately 0.4, between "Target" and "PAY_0".**
# * ** Let's get rid of categorical columns with get_dummies method. More information about get_dummies** [get_dummies.](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)

# In[ ]:


cat_col = ["SEX","MARRIAGE","EDUCATION"]
df = pd.get_dummies(df, columns = cat_col)


# *  **If SEX_2 == 1 means customer is female, if it is == 0 customer is male, on the other hand SEX_1 == 1 means customer is male, otherwise is female, so i dropped this columns, . Actually these two columns shows exactly same things. **

# In[ ]:


df.drop(columns=["SEX_2","ID"], inplace = True)


# In[ ]:


## For showing all columns 
pd.set_option('display.max_columns', 50)
df.head()


# * **Let's look at the distributions of features with seaborn pair plot. Citation: "The diagonal Axes are treated differently, drawing a plot to show the univariate distribution of the data for the variable in that column." Get more information about pair plot [Sns_Pairplot.](https://seaborn.pydata.org/generated/seaborn.pairplot.html)**

# In[ ]:


## Preparing data for pairplotting
df_vis = df[["LIMIT_BAL","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]]
df_vis2 = df[["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","AGE"]]


# In[ ]:


pp1 = sns.pairplot(df_vis)


# In[ ]:


pp2 = sns.pairplot(df_vis2)


# * **As we have guessed before, numerical features does not seem normally distributed.(Diagonal plots)**
# 

# In[ ]:


## Pre-processing for building ML model. 
X = df.loc[:,df.columns != "Target"]
Y = df["Target"].copy()


# In[ ]:


## Importing necessary libraries 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X["BILL_AMT1"] = scaler.fit_transform(np.array(X["BILL_AMT1"]).reshape(-1,1))
X["BILL_AMT2"] = scaler.fit_transform(np.array(X["BILL_AMT2"]).reshape(-1,1))
X["BILL_AMT3"] = scaler.fit_transform(np.array(X["BILL_AMT3"]).reshape(-1,1))
X["BILL_AMT4"] = scaler.fit_transform(np.array(X["BILL_AMT4"]).reshape(-1,1))
X["BILL_AMT5"] = scaler.fit_transform(np.array(X["BILL_AMT5"]).reshape(-1,1))
X["BILL_AMT6"] = scaler.fit_transform(np.array(X["BILL_AMT6"]).reshape(-1,1))
X["PAY_AMT1"] = scaler.fit_transform(np.array(X["PAY_AMT1"]).reshape(-1,1))
X["PAY_AMT2"] = scaler.fit_transform(np.array(X["PAY_AMT2"]).reshape(-1,1))
X["PAY_AMT3"] = scaler.fit_transform(np.array(X["PAY_AMT3"]).reshape(-1,1))
X["PAY_AMT4"] = scaler.fit_transform(np.array(X["PAY_AMT4"]).reshape(-1,1))
X["PAY_AMT5"] = scaler.fit_transform(np.array(X["PAY_AMT5"]).reshape(-1,1))
X["PAY_AMT6"] = scaler.fit_transform(np.array(X["PAY_AMT6"]).reshape(-1,1))
X["LIMIT_BAL"] = scaler.fit_transform(np.array(X["LIMIT_BAL"]).reshape(-1,1))


# * ** We will use ensemble ML algorithms and ,generally, these algorithms work better with scaled numbers so I use MinMax scaler method for scaling. MinMax method squeeze values between 1 and 0. Maximum value of column = 1 and minimum value of column = 0 **
# * ** P.S. I am getting warning while I was using "np.array(X["PAY_AMT1"]).reshape(-1,1)", I checked the pandas documents and I tried to use .loc but I got the same warning. If anyone knows this error, please leave a comment**

# In[ ]:


## Control of the data 
X.head()


# ## **Feature Engineering**
# * **We will use SelectKBest with both chi2 and f_classif method for feature extraction. I will extract 10 features from both methods then I will combine of two results. For more information about chi2 and f_classif [Get_info](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)**

# In[ ]:


from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=10)
selector.fit(X, Y)
k = list(X.columns[selector.get_support(indices=True)])
k


# In[ ]:


from sklearn.feature_selection import SelectKBest,f_classif
selector = SelectKBest(f_classif, k=10)
selector.fit(X, Y)
k2 = list(X.columns[selector.get_support(indices=True)])
k2


# In[ ]:


cols_4_model = set(k+k2)
cols_4_model


# In[ ]:


## Train and test split data
from sklearn.model_selection import train_test_split
X = X[cols_4_model]
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


# In[ ]:


## List of ML Algorithms, we will use for loop for each algorithms.
models = [LogisticRegression(solver = "liblinear"),
          DecisionTreeClassifier(),
          RandomForestClassifier(n_estimators =10),
          XGBClassifier(),
          GradientBoostingClassifier(),
          LGBMClassifier(),
         ]


# In[ ]:


for model in models:
    t0 = time.time()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    roc_score = roc_auc_score(y_test, proba[:,1])
    cv_score = cross_val_score(model,X_train,y_train,cv=10).mean()
    score = accuracy_score(y_test,y_pred)
    bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)
    name = str(model)
    print(name[0:name.find("(")])
    print("Accuracy :", score)
    print("CV Score :", cv_score)
    print("AUC Score : ", roc_score)
    print(bin_clf_rep)
    print(confusion_matrix(y_test,y_pred))
    print("Time Taken :", time.time()-t0, "seconds")
    print("------------------------------------------------------------")


# **TOP 2 Algorithms for both calculating time and accuracy score:**
# * **Logistic Regression: CV Accuracy: 0.81 -- Time: 1.616 secs**
# * ** Light GBM : CV Accuracy: 0.81 -- Time: 5.36 secs**
# 
# **Now we will tune these two algorithms with GridSearchCV.
# For more information about [GridSearchCV ]**(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

# In[ ]:


## LGBM_CLF Model
t0 = time.time()
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train,y_train)
y_pred = lgbm_model.predict(X_test)
proba = lgbm_model.predict_proba(X_test)
roc_score = roc_auc_score(y_test, proba[:,1])
cv_score = cross_val_score(lgbm_model,X_train,y_train,cv=10).mean()
score = accuracy_score(y_test,y_pred)
bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)
print(name[0:name.find("(")])
print("Accuracy :", score)
print("CV Score :", cv_score)
print("AUC Score : ", roc_score)
print(bin_clf_rep)
print(confusion_matrix(y_test,y_pred))
print("Time Taken :", time.time()-t0, "seconds")
lgbm_model


# In[ ]:


## Setting parameters for LGBM Model, we will use this dictionary with GridSearchCV
lgbm_params = {"n_estimators" : [100, 500, 1000],
               "subsample" : [0.6, 0.8, 1.0],
               "learning_rate" : [0.1, 0.01, 0.02],
               "min_child_samples" : [5, 10, 20]}


# * **More information about LightGBM and parameters:
# [LGBM_CLF](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)**

# In[ ]:


## n_jobs = -1 allows multicore processing for CPU
from sklearn.model_selection import GridSearchCV
lgbm_cv_model = GridSearchCV(lgbm_model, 
                             lgbm_params, 
                             cv = 5,
                             verbose = 1,
                             n_jobs = -1)


# In[ ]:


## Code works approximately 5-6 minutes
lgbm_cv_model.fit(X_train, y_train)


# In[ ]:


## Getting best parameters
lgbm_cv_model.best_params_


# In[ ]:


## Best_Params with LGBM_CLF
t0 = time.time()
lgbm_model2 = LGBMClassifier(learning_rate = 0.01,
                            min_child_samples = 20,
                            n_estimators = 500,
                            subsample = 0.6)
lgbm_model2.fit(X_train,y_train)
y_pred = lgbm_model2.predict(X_test)
proba = lgbm_model2.predict_proba(X_test)
roc_score = roc_auc_score(y_test, proba[:,1])
cv_score = cross_val_score(lgbm_model2,X_train,y_train,cv=10).mean()
score = accuracy_score(y_test,y_pred)
bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)
print(name[0:name.find("(")])
print("Accuracy :", score)
print("CV Score :", cv_score)
print("AUC Score : ", roc_score)
print(bin_clf_rep)
print(confusion_matrix(y_test,y_pred))
print("Time Taken :", time.time()-t0, "seconds")


# * **LGBM with best parameters works with 0.81 CV Accuracy and it takes 27 seconds. I think this is not the most efficient algorithm for this data. So keep trying with Logistic Regression.**

# In[ ]:


## LOG_REG_CLF
t0 = time.time()
log_reg_model = LogisticRegression(solver="liblinear")
log_reg_model.fit(X_train,y_train)
y_pred = log_reg_model.predict(X_test)
proba = log_reg_model.predict_proba(X_test)
roc_score = roc_auc_score(y_test, proba[:,1])
cv_score = cross_val_score(log_reg_model,X_train,y_train,cv=10).mean()
score = accuracy_score(y_test,y_pred)
bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)
print(name[0:name.find("(")])
print("Accuracy :", score)
print("CV Score :", cv_score)
print("AUC Score : ", roc_score)
print(bin_clf_rep)
print(confusion_matrix(y_test,y_pred))
print("Time Taken :", time.time()-t0, "saniye")


# In[ ]:


log_reg_params = {"C":[0.1, 0.5, 1.0], 
                  "penalty":["l1","l2"],
                  "solver" : ["liblinear", "lbfgs", "newton-cg"],
                  "max_iter" : [100,200,500]
                  }


# * **More information about Logistic Regression and parameters:
# [Logistic_Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**

# In[ ]:


from sklearn.model_selection import GridSearchCV
log_reg_cv_model = GridSearchCV(log_reg_model, 
                             log_reg_params, 
                             cv = 5,
                             verbose = 1,
                             n_jobs = -1)


# In[ ]:


log_reg_cv_model.fit(X_train, y_train)


# In[ ]:


log_reg_cv_model.best_params_


# In[ ]:


## Best Params with LOG_REG_CLF
t0 = time.time()
log_reg_model = LogisticRegression(solver="liblinear",
                                  C = 0.5,
                                  max_iter = 100,
                                  penalty = "l1",
                                  )
log_reg_model.fit(X_train,y_train)
y_pred = log_reg_model.predict(X_test)
proba = log_reg_model.predict_proba(X_test)
roc_score = roc_auc_score(y_test, proba[:,1])
cv_score = cross_val_score(log_reg_model,X_train,y_train,cv=10).mean()
score = accuracy_score(y_test,y_pred)
bin_clf_rep = classification_report(y_test,y_pred, zero_division=1)
print(name[0:name.find("(")])
print("Accuracy :", score)
print("CV Score :", cv_score)
print("AUC Score : ", roc_score)
print(bin_clf_rep)
print(confusion_matrix(y_test,y_pred))
print("Time Taken :", time.time()-t0, "saniye")


# * **Logistic Regression with best parameters works with 0.81 CV accuracy and it takes 1.28 seconds. For this problem, log_reg is the best algorithm.**

# In[ ]:




