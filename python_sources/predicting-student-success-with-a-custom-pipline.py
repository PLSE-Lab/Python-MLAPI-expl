#!/usr/bin/env python
# coding: utf-8

# # Predicting Student Success
# 
# In this notebook, we attempt to predict the final grade of students using different classification criteria.
# -  Binary classification (pass/fail)
# -  Multi-class classification (fail/average/good)
# 
# This notebook was inspired by the following paper published by Paulo Cortez and Alice Silva from the University of Minho:
# [Usind Data Mining to Predict Secondary School Student Performance](http://www3.dsi.uminho.pt/pcortez/student.pdf)
# 
# Comments and criticism welcome.
# 
# **Please upvote if you found this notebook useful :)**

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')


# Taking a first look at the data

# In[ ]:


#import data into pandas dataframe
data = pd.read_csv('../input/student-mat.csv')

#display first 5 lines
display(data.head())

#print data properties
print('Data Shape: {}'.format(data.shape))
display(data.describe())
display(data.info())


# We see that we have no NaN or missing values.

# In[ ]:


plt.figure()
sns.countplot(data['G1'],label='Count')
plt.figure()
sns.countplot(data['G2'],label="Count")
plt.figure()
sns.countplot(data['G3'],label="Count")


# The above histograms indicate a well balanced grade distribution. 

# ## Correlation Matrix

# In[ ]:


#Feature correlations
plt.figure(figsize=(15,10))
corr_mat=sns.heatmap(data.corr(),annot=True,cbar=True,
            cmap='viridis', vmax=1,vmin=-1)
corr_mat.set_xticklabels(corr_mat.get_xticklabels(),rotation=90)


# Notice how the G1, G2 (midterm grades) and G3 (Final grade) curves are quite similiar and very highly correlated. This is tells us that students who perform well year long are also likey to perform similairly on the final.
# 
# Some correlations to note:
# -  Mother education and G1,G2,G3
# -  Father education and mother education
# -  Weekend alcohol consumption and going out
# -  Weekend alcohol consumption and weekday alcohol consumption
# 
# Before training our models, we will need to preprocess the data. Each sample contains both numerical and categorical features. Each would need to be processed differently. Numerical features will need to be scaled. Binary categorical features will need to be encoded into 0 or 1 while multi-category features will need to be one-hot encoded.
# 
# To be able to perform proper cross validation and grid search later, we'll need to create a cusotm transformer to handle the above preprocessing step and feed it into a pipeline.

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        return X[self.feature_names]
    

class CategoricalTranformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, ):
        self.binary_features = ['school','sex','address','famsize',
                                   'Pstatus','schoolsup','famsup','paid',
                                   'activities','nursery','higher','internet',
                                   'romantic','G1','G2']
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):
        
        le = LabelEncoder()
        
        for col in np.intersect1d(self.binary_features,X.columns):
            le.fit(X[col])
            X[col] = le.transform(X[col])

        X = pd.get_dummies(X)
        
        return X       


# # 1. Classification using all Features
# ## 1.1 Binary Classificaton

# In[ ]:


X = data.iloc[:,0:32]
y = data['G3']

bins = (-1, 9.5, 21)
performance_level = ['fail', 'pass']
X['G1'] = pd.cut(X['G1'], bins = bins, labels = performance_level)
X['G2'] = pd.cut(X['G2'], bins = bins, labels = performance_level)
y = pd.cut(y, bins = bins, labels = performance_level)

plt.figure()
sns.countplot(X['G1'],label="Count")
plt.figure()
sns.countplot(X['G2'],label="Count")
plt.figure()
sns.countplot(y,label="Count")

#encode target
le_target = LabelEncoder()
le_target.fit(y)
y = le_target.transform(y)


# Split data into training and test sets and prepare pipeline:

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)

categorical_features = X.select_dtypes(exclude=['int64']).columns
numerical_features = X.select_dtypes(include=['int64']).columns


categorical_preproc = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),
                                         ('cat_transformer', CategoricalTranformer())])
    
numerical_preproc = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),         
                                       ('scaler', StandardScaler())])

full_preproc = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_preproc),
                                                 ('numerical_pipeline', numerical_preproc)])
# Temporary variable to extract column names
X_temp = categorical_preproc.transform(X)
col = X_temp.columns.tolist()+numerical_features.tolist()


# Train and predict using SVC model

# In[ ]:


kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10],
              'model__gamma': [0.001, 0.01, 0.1, 1, 10],
              'model__kernel': ['linear', 'rbf']}

pipe = Pipeline(steps = [('proproc', full_preproc),
                         ('model', SVC(random_state=0))])
                
grid = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
                
print("Best cross-validation accuracy: {:.3f}".format(grid.best_score_))
print("Test set score: {:.3f}".format(grid.score(X_test,y_test)))
print("Best parameters: {}".format(grid.best_params_))

conf_mat = confusion_matrix(y_test, grid.predict(X_test))
sns.heatmap(conf_mat, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le_target.classes_, xticklabels=le_target.classes_)

svm = SVC(random_state=0, C = grid.best_params_['model__C'],
          gamma = grid.best_params_['model__gamma'],
          kernel = grid.best_params_['model__kernel'],
          probability=True)


# ## 1.2 Multi-Class Classification

# In[ ]:


X = data.iloc[:,0:32]
y = data['G3']

bins = (-1, 9.5, 16.5, 21)
performance_level = ['fail', 'average', 'good']
X['G1'] = pd.cut(X['G1'], bins = bins, labels = performance_level)
X['G2'] = pd.cut(X['G2'], bins = bins, labels = performance_level)
y = pd.cut(y, bins = bins, labels = performance_level)

plt.figure()
sns.countplot(X['G1'],label="Count")
plt.figure()
sns.countplot(X['G2'],label="Count")
plt.figure()
sns.countplot(y,label="Count")

#encode target
le_target = LabelEncoder()
le_target.fit(y)
y = le_target.transform(y)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)

categorical_features = X.select_dtypes(exclude=['int64']).columns
numerical_features = X.select_dtypes(include=['int64']).columns


categorical_preproc = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),
                                         ('cat_transformer', CategoricalTranformer())])
    
numerical_preproc = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),         
                                       ('scaler', StandardScaler())])

full_preproc = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_preproc),
                                                 ('numerical_pipeline', numerical_preproc)])


X_temp = categorical_preproc.transform(X)
col = X_temp.columns.tolist()+numerical_features.tolist()


# In[ ]:


kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10],
              'model__gamma': [0.001, 0.01, 0.1, 1, 10],
              'model__kernel': ['linear', 'rbf']}

pipe = Pipeline(steps = [('proproc', full_preproc),
                         ('model', SVC(random_state=0))])
                
grid = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
                
print("Best cross-validation accuracy: {:.3f}".format(grid.best_score_))
print("Test set score: {:.3f}".format(grid.score(X_test,y_test)))
print("Best parameters: {}".format(grid.best_params_))

conf_mat = confusion_matrix(y_test, grid.predict(X_test))
sns.heatmap(conf_mat, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le_target.classes_, xticklabels=le_target.classes_)

svm = SVC(random_state=0, C = grid.best_params_['model__C'],
          gamma = grid.best_params_['model__gamma'],
          kernel = grid.best_params_['model__kernel'],
          probability=True)


# # 2. Classification without G1 & G2
# 
# ## 2.1 Binary Classification

# In[ ]:


X = data.iloc[:,0:30]
y = data['G3']

bins = (-1, 9.5, 21)
performance_level = ['fail', 'pass']
y = pd.cut(y, bins = bins, labels = performance_level)

#encode target  and preprocessing
le = LabelEncoder()
le_target.fit(y)
y = le_target.transform(y)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)

categorical_features = X.select_dtypes(exclude=['int64']).columns
numerical_features = X.select_dtypes(include=['int64']).columns
categorical_preproc = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),
                                         ('cat_transformer', CategoricalTranformer())])
    
numerical_preproc = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),         
                                       ('scaler', StandardScaler())])

full_preproc = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_preproc),
                                                 ('numerical_pipeline', numerical_preproc)])
# Temporary variable to extract column names
X_temp = categorical_preproc.transform(X)
col = X_temp.columns.tolist()+numerical_features.tolist()


# In[ ]:


kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10],
              'model__gamma': [0.001, 0.01, 0.1, 1, 10],
              'model__kernel': ['linear', 'rbf']}

pipe = Pipeline(steps = [('proproc', full_preproc),
                         ('model', SVC(random_state=0))])
                
grid = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
                
print("Best cross-validation accuracy: {:.3f}".format(grid.best_score_))
print("Test set score: {:.3f}".format(grid.score(X_test,y_test)))
print("Best parameters: {}".format(grid.best_params_))

conf_mat = confusion_matrix(y_test, grid.predict(X_test))
sns.heatmap(conf_mat, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le_target.classes_, xticklabels=le_target.classes_)

svm = SVC(random_state=0, C = grid.best_params_['model__C'],
          gamma = grid.best_params_['model__gamma'],
          kernel = grid.best_params_['model__kernel'],
          probability=True)


# ## 2.2 Multi-Class Classification

# In[ ]:


X = data.iloc[:,0:30]
y = data['G3']

bins = (-1, 9.5, 16.5, 21)
performance_level = ['fail', 'average', 'good']
y = pd.cut(y, bins = bins, labels = performance_level)

#encode target
le_target = LabelEncoder()
le_target.fit(y)
y = le_target.transform(y)

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=0,stratify=y)

categorical_features = X.select_dtypes(exclude=['int64']).columns
numerical_features = X.select_dtypes(include=['int64']).columns


categorical_preproc = Pipeline(steps = [('cat_selector', FeatureSelector(categorical_features)),
                                         ('cat_transformer', CategoricalTranformer())])
    
numerical_preproc = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),         
                                       ('scaler', StandardScaler())])

full_preproc = FeatureUnion(transformer_list = [('categorical_pipeline', categorical_preproc),
                                                 ('numerical_pipeline', numerical_preproc)])


X_temp = categorical_preproc.transform(X)
col = X_temp.columns.tolist()+numerical_features.tolist()


# In[ ]:


kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
param_grid = {'model__C': [0.001, 0.01, 0.1, 1, 10],
              'model__gamma': [0.001, 0.01, 0.1, 1, 10],
              'model__kernel': ['linear', 'rbf']}

pipe = Pipeline(steps = [('proproc', full_preproc),
                         ('model', SVC(random_state=0))])
                
grid = GridSearchCV(pipe, param_grid=param_grid, cv=kfold, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
                
print("Best cross-validation accuracy: {:.3f}".format(grid.best_score_))
print("Test set score: {:.3f}".format(grid.score(X_test,y_test)))
print("Best parameters: {}".format(grid.best_params_))

conf_mat = confusion_matrix(y_test, grid.predict(X_test))
sns.heatmap(conf_mat, annot=True, cbar=False, cmap="viridis_r",
            yticklabels=le_target.classes_, xticklabels=le_target.classes_)

svm = SVC(random_state=0, C = grid.best_params_['model__C'],
          gamma = grid.best_params_['model__gamma'],
          kernel = grid.best_params_['model__kernel'],
          probability=True)


# ## Summary
# 
# As expected, the high correlation between G1, G2 & G3 played a big role in increasing the model's performance when taken into consideration. As well, the model performed better with binary classification than multi-class classification. This may be attributed to the low number of targets labeled 'good' in the dataset when considering multi-class classification.
