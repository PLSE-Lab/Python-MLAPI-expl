#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv", sep=',')


# # Understand our data

# In[ ]:


# list first few rows
df.head()


# In[ ]:


# check datatypes
df.dtypes


# In[ ]:


df = df.rename(columns = {'Chance of Admit ': 'Chance of Admit', 'LOR ':'LOR'})


# In[ ]:


# chance of admit descriptive statistics summary
df['Chance of Admit'].describe()


# In[ ]:


# histogram and normal distribution plot
sns.distplot(df["Chance of Admit"], fit=norm)


# In[ ]:


# skewness and kurtosis
print("Skewness: {0}".format(df['Chance of Admit'].skew()))


# In[ ]:


# correlation matrix
fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(df.drop('Serial No.', axis=1).corr(), linewidths=0.3, cmap='Blues', annot=True)


# In[ ]:


# Three most important features
sns.pairplot(df, vars=["Chance of Admit", "GRE Score", "CGPA", "TOEFL Score"], hue="Research", palette="Blues")


# In[ ]:


# Are candidates from better universities more likely to be accepted?
sns.boxenplot(x="University Rating", y="Chance of Admit", scale="linear", color='b', data=df)


# In[ ]:


# are candidates from better universities more likely to do research
sns.countplot(x="University Rating", hue="Research", data=df, palette='Blues')


# In[ ]:


# are students with research experience from lower rating universities more likely to be accepted?
low_uni = df[df["University Rating"] < 3]
sns.countplot(x=low_uni["Chance of Admit"].apply(lambda x: 1 if x > .75 else 0), hue="Research", data=df, palette='Blues')


# # Data Pre-processing & Model

# ## Regression

# In[ ]:


# combine statement of purpose and letter of recommendation letter strength
# statement strength
df["SS"] = df["SOP"]*0.6 + df["LOR"]*0.4
# research experience
df["RE"] = df["Research"]*df["University Rating"]
# letter credibility
df["LC"] = df["LOR"]*df["University Rating"]*1.3
# grade credibility
df["GC"] = df["CGPA"]*df["University Rating"]


# In[ ]:


# correlation matrix
fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(df.corr(), linewidths=0.3, cmap='Blues', annot=True)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import xgboost as xgb
import lightgbm as lgba


# In[ ]:


serialNo = df["Serial No."].values
df.drop(["Serial No."],axis=1,inplace = True)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler_fit = scaler.fit(df[["GRE Score", "TOEFL Score"]])
scaled_scores = pd.DataFrame(scaler_fit.transform(df[["GRE Score", "TOEFL Score"]]), columns=["GRE Score", "TOEFL Score"])
scaled_scores.head()


# In[ ]:


df["GRE Score"] = scaled_scores["GRE Score"]
df["TOEFL Score"] = scaled_scores["TOEFL Score"]
predictor_cols = ["GRE Score", "CGPA", "TOEFL Score", "SS", "GC", "RE"]


# In[ ]:


y = df["Chance of Admit"].values
X = df[predictor_cols]
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size = 0.20,random_state = 7)
y_train_clf = [1 if p > 0.8 else 0 for p in y_train]
y_test_clf = [1 if p > 0.8 else 0 for p in y_test]


# In[ ]:


reg = RandomForestRegressor(n_estimators=100, criterion='mse')
rf = reg.fit(X_train, y_train)
feature_importances = pd.Series(rf.feature_importances_, index=predictor_cols)
feature_importances.nlargest(6).plot(kind='barh')


# In[ ]:


regressors = [
    GradientBoostingRegressor(n_estimators=100),
    xgb.XGBRegressor(booster='gbtree', eta=0.5, max_depth=3),
    LinearRegression(),
    RandomForestRegressor(n_estimators=100, criterion='mse'),
    DecisionTreeRegressor(max_depth=4)
]


# In[ ]:


from sklearn.metrics import r2_score
log_cols = ["Regressor", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
for reg in regressors:
    reg.fit(X_train, y_train)
    name = reg.__class__.__name__
    print('='*30)
    print(name)
    
    print('****Results****')
    train_predictions = reg.predict(X_train)
    acc = r2_score(y_train, train_predictions)
    print("Accuracy (train): {:3f}".format(acc))
    test_predictions = reg.predict(X_test)
    acc = r2_score(y_test, test_predictions)
    print("Accuracy (test): {:3f}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc]], columns=log_cols)
    log = log.append(log_entry)
    
print('='*30)


# ## Classification

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


classifiers = [
    LogisticRegression(random_state=7),
    SVC(random_state=7),
    GaussianNB(),
    DecisionTreeClassifier(random_state=7),
    RandomForestClassifier(random_state=7),
    GradientBoostingClassifier(random_state=7),
    KNeighborsClassifier()
]


# In[ ]:


from sklearn.metrics import accuracy_score
log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)
for clf in classifiers:
    clf.fit(X_train, y_train_clf)
    name = clf.__class__.__name__
    print('='*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_train)
    acc = accuracy_score(y_train_clf, train_predictions)
    print("Accuracy (train): {:3f}".format(acc))
    test_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test_clf, test_predictions)
    print("Accuracy (test): {:3f}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc]], columns=log_cols)
    log = log.append(log_entry)
    
print('='*30)


# In[ ]:





# In[ ]:




