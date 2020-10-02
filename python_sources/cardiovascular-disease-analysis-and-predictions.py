#!/usr/bin/env python
# coding: utf-8

# ## Cardiovascular Disease Project Introduction

# **This is one of my first data science projects. <br>
# <br>
# In this notebook, I take a look at a dataset containing 70,000 entries of patients' medical information such as height, weight, age, blood pressure, glucose levels, and cholesterol levels. <br>
# <br>
# The goal of this project is to use these features in order to make predictions if a patient has cardiovascular disease. <br>
# Several classification models will be investigated in this project. <br>
# <br>
# Any constructive feedback is welcomed and upvotes would be greately appreciated!!

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', sep=';')
df.head()


# In[ ]:


df.shape


# In[ ]:


# Check for missing values
df.isnull().sum()


# In[ ]:


df['cardio'].unique()


# In[ ]:


# Convert age from days to years
df['age'] =  df['age'] / 365


# In[ ]:


df.head()


# In[ ]:


# Rename columns to make features more clearly understood
df.rename(columns={'ap_hi': 'systolic', 'ap_lo': 'diastolic', 'gluc': 'glucose', 'alco': 'alcohol', 'cardio': 'cardiovascular disease'}, inplace=True)


# In[ ]:


df.head()


# In[ ]:


sns.lmplot(x='weight', y='height', hue='gender', data=df, fit_reg=False, height=6)
plt.show()


# In[ ]:


sns.countplot(x='gender', data=df, hue='cardiovascular disease')
plt.show()

# Not much of a difference between females (1) and males (2) and the chance of getting cardiovascular disease.


# In[ ]:


df.describe()


# **Note:** Very strange observations in the height and weight. Minimum and maximum values do not look realistic.

# In[ ]:


df_train = df.drop('id', axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


# 24 Duplicated entries
df_train.duplicated().sum()


# In[ ]:


df_train[df_train.duplicated()]


# In[ ]:


df_train.drop_duplicates(inplace=True)


# In[ ]:


df_train.count()


# In[ ]:


df_train.isnull().sum()


# ## Exploratory Data Analysis

# In[ ]:


df_train.head()


# In[ ]:


sns.countplot(x='gender', hue='cardiovascular disease', data=df_train)
plt.show()


# In[ ]:


sns.countplot(x='cholesterol', hue='cardiovascular disease', data=df_train)
plt.show()
# There appears to be a correlation between higher cholesterol levels and cardiovascular disease
# chloesterol levels: 1 = normal, 2 = above normal, 3 = well above normal


# In[ ]:


sns.countplot(x='glucose', hue='cardiovascular disease', data=df_train)
plt.show()
# There appears to be another correlation between higher glucose levels and cardiovascular disease
# glucose levels: 1 = normal, 2 = above normal, 3 = well above normal


# In[ ]:


sns.countplot(x='active', hue='cardiovascular disease', data=df_train)
plt.show()


# In[ ]:


sns.countplot(x='smoke', hue='cardiovascular disease', data=df_train)
plt.show()


# In[ ]:


sns.countplot(x='alcohol', hue='cardiovascular disease', data=df_train)
plt.show()


# In[ ]:


sns.distplot(df_train['weight'], kde=False)
plt.show()


# In[ ]:


df_train['weight'].sort_values().head()


# In[ ]:


sns.distplot(df_train['height'], kde=False)
plt.show()


# In[ ]:


df_train['height'].max()


# This maximum height of 250 cm/8.2 ft seems unlikely 

# In[ ]:


df_train['height'].sort_values().head()


# The minimum height of 55 cm/1.8 ft also seems unlikely and unrealistic. <br>
# This dataset may not be legitimate; however, we will continue on with the data analysis and model selection.

# ## Feature Engineering

# **Body Mass Index (BMI)** is a common metric used for medical evaluation and heart health <br>
# BMI can be calculated by the following: BMI = weight(kg) / height (cm) / height (cm) x 10,000 <br>
# <br>
# **Pulse Pressure** is another indicator of heart health <br>
# Pulse Pressure can be calculated by the following: Pulse Pressure = systolic - diastolic <br>
# Typically, a pulse pressure greater than 60 can be a useful predictor of heart attacks or other cardiovascular diseases

# In[ ]:


df_train['BMI'] = df_train['weight'] / df_train['height'] / df_train['height'] * 10000
df_train['pulse pressure'] = df_train['systolic'] - df_train['diastolic']


# In[ ]:


df_train.head()
# Quick look at the dataframe to make sure these new features have been added


# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df_train['BMI'], bins=50, kde=False)
plt.show()


# In[ ]:


df_train[df_train['BMI'] > 100].head(10)


# Quick observation to see if extremely high BMI values correlate to cardiovascular disease

# In[ ]:


df_train[(df_train['pulse pressure'] >= 60 ) & (df_train['cholesterol'] == 3)].head(15)


# Cursory glance at individuals who have both high pulse pressure (>=60) *and* well above normal cholesterol levels (3). <br>
# Upon inspection of the first several entries, having both high pulse pressure and well above normal cholesterol levels correlate to a higher likelihood of having cardiovascular disease.

# In[ ]:


plt.figure(figsize=(8,4))
sns.distplot(df_train['height'], kde=False)
plt.show()


# In[ ]:


# Splitting data into training and testing datasets
X = df_train.drop(['weight', 'height', 'cardiovascular disease'], axis=1)
y = df_train['cardiovascular disease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


# In[ ]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


len(X_train)


# In[ ]:


len(y_train)


# ## Model Selection

# We will investigate different classification models and evaluate each to select the best performer.
# The models that will be evaluted are the following:
# - Random Forest
# - SVM
# - KNN
# - Naive Bayes
# - XGBoost

# **Random Forest Model Investigation**

# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[ ]:


y_pred_rfc = rfc.predict(X_test)


# In[ ]:


# Random Forest Model Evaluation
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred_rfc))
print(classification_report(y_test, y_pred_rfc))


# In[ ]:


rfc.score(X_test, y_test)


# **K-Fold cross-valuidation of Random Forest Model**

# In[ ]:


#Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies_rfc = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)


# In[ ]:


accuracies_rfc


# In[ ]:


accuracies_rfc.mean()


# In[ ]:


accuracies_rfc.std()


# **SVM Model Investigation**

# In[ ]:


# SVM
from sklearn.svm import SVC
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)


# In[ ]:


y_pred_svc = svc.predict(X_test)


# In[ ]:


# SVM Model Evaluation
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))


# In[ ]:


svc.score(X_test, y_test)


# **K-Fold cross-valuidation of SVM model**

# In[ ]:


#Applying k-Fold Cross Validation
accuracies_svc = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=10, n_jobs=4)


# In[ ]:


accuracies_svc


# In[ ]:


accuracies_svc.mean()


# In[ ]:


accuracies_svc.std()


# **K-Nearest Neighbor Model Investigation**

# In[ ]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train, y_train)


# In[ ]:


y_pred_knn = knn.predict(X_test)


# In[ ]:


# KNN Model Evaluation
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))


# In[ ]:


knn.score(X_test, y_test)


# **K-Fold cross-valuidation of KNN model**

# In[ ]:


#Applying k-Fold Cross Validation
accuracies_knn = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=10)


# In[ ]:


accuracies_knn


# In[ ]:


accuracies_knn.mean()


# In[ ]:


accuracies_knn.std()


# **Naive Bayes Model Investigation**

# In[ ]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nbc = GaussianNB()
nbc.fit(X_train, y_train)


# In[ ]:


y_pred_nbc = nbc.predict(X_test)


# In[ ]:


# Naive Bayes Model Evaluation
print(confusion_matrix(y_test, y_pred_nbc))
print(classification_report(y_test, y_pred_nbc))


# In[ ]:


nbc.score(X_test, y_test)


# **K-Fold cross-valuidation of Naive Bayes model**

# In[ ]:


#Applying k-Fold Cross Validation
accuracies_nbc = cross_val_score(estimator=nbc, X=X_train, y=y_train, cv=10)


# In[ ]:


accuracies_nbc


# In[ ]:


accuracies_nbc.mean()


# In[ ]:


accuracies_nbc.std()


# **XGBoost Model Investigation**

# In[ ]:


# XGBoost Model
from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.02, n_estimators=600)
xgb.fit(X_train, y_train)


# In[ ]:


y_pred_xgb = xgb.predict(X_test)


# In[ ]:


# XGBoost Model Evaluation
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# In[ ]:


xgb.score(X_test, y_test)


# **K-Fold cross-valuidation of XGBoost model**

# In[ ]:


#Applying k-Fold Cross Validation
accuracies_xgb = cross_val_score(estimator=xgb, X=X_train, y=y_train, cv=10)


# In[ ]:


accuracies_xgb


# In[ ]:


accuracies_xgb.mean()


# In[ ]:


accuracies_xgb.std()


# **Grid Search for the top two models** <br>
# Based on this investigation so far, the two best performers are the *XGBoost* and *SVM* models with mean accuracy scores of *73.7%* and *72.7%*, respectively. <br>
# **Note:** Grid search has been commented out due to length of processing time.

# In[ ]:


#Applying Grid Search to find the best model and best parameters (XGBoost)
#from sklearn.model_selection import GridSearchCV

#define set of parameters that will be investigated by Grid Search
#parameters = {
#            'learning_rate': [0.01, 0.02, 0.05, 0.1],
#            'n_estimators': [100, 200, 300, 500],
#            'min_child_weight': [1, 5, 10],
#            'gamma': [0.5, 1, 1.5, 2, 5],
#            'subsample': [0.6, 0.8, 1.0],
#            'colsample_bytree': [0.6, 0.8, 1.0],
#            'max_depth': [3, 4, 5]
#            }


# In[ ]:


#grid_search = GridSearchCV(estimator=xgb,
#                          param_grid = parameters,
#                          scoring = 'accuracy',
#                          cv = 10,
#                          n_jobs = -1)


# In[ ]:


#grid_search = grid_search.fit(X_train, y_train)


# In[ ]:


#Applying Grid Search to find the best model and best parameters (SVM)
#from sklearn.model_selection import GridSearchCV

#define set of parameters that will be investigated by Grid Search
#parameters = {'C': [1, 10, 100, 1000], 'kernel': ['rbf']}


# In[ ]:


#grid_search = GridSearchCV(estimator=svc,
#                          param_grid = parameters,
#                          scoring = 'accuracy',
#                          cv = 10,
#                          n_jobs = -1)


# In[ ]:


#grid_search = grid_search.fit(X_train, y_train)


# ## Conclusion

# - The XGBoost model was the best performer out of the five models giving us a mean accuracy score of 73.7%. <br>
# - K-Fold cross validation was used to ensure no overfitting was done. <br>
# - Grid search can be further performed on the best model candidates; however, this step can be time intensive, potentially demanding a great deal of CPU usage.

# In[ ]:


model = ['Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost']
scores = [accuracies_rfc.mean(),accuracies_svc.mean(),accuracies_knn.mean(),accuracies_nbc.mean(),accuracies_xgb.mean()]

summary = pd.DataFrame(data=scores, index=model, columns=['Mean Accuracy'])
summary.sort_values(by='Mean Accuracy', ascending=False)

