#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import category_encoders as ce
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
import numpy as np
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve
import catboost as cb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style('darkgrid')


# # Importing Files

# In[ ]:


df = pd.read_csv('../input/terrabluext-intern-test-data/Test_Data.csv')


# # Exploring data

# There are total 43 columns inclduing Target columns and 5839 rows, Target is a categorical value and all others are numerical data. No null data in dataset so we don't need to impute missing data.
# Class is not symetrical hence we need to split data startified.

# In[ ]:


df.shape


# In[ ]:


df.head(20)


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


pd.set_option('display.max_columns', 90)


# In[ ]:


df.describe()


# # Dealing with outliers

# we will clear outliers one by one a long and tedious process but this will ensure minimum data loss.
# This Dataset have high value of outliers in multiple columns

# In[ ]:


def describe_plot(label):
    print(df[label].describe())
    print(df.shape)
    sns.boxplot(df[label])


# In[ ]:


def remover_plot(label, outlier):
    df.drop(df[df[label] > outlier].index, inplace=True)
    print(df.shape)
    sns.boxplot(df[label])


# In[ ]:


# Class A
describe_plot('A')


# In[ ]:


# Class E
describe_plot("E")


# In[ ]:


remover_plot('E', 280)


# In[ ]:


# Class F
describe_plot('F')


# In[ ]:


remover_plot('F', 138)


# In[ ]:


# Class G
describe_plot('G')


# In[ ]:


remover_plot('G', 105)


# In[ ]:


# Class j
describe_plot('J')


# In[ ]:


remover_plot('J', 32)


# In[ ]:


# Class L
describe_plot('L')


# In[ ]:


remover_plot('L', 325)


# In[ ]:


# Class O
describe_plot('O')


# In[ ]:


remover_plot('O', 2.2)


# In[ ]:


# Class P
describe_plot('P')


# In[ ]:


remover_plot('P', 0.78)


# In[ ]:


#  Class
describe_plot('R')


# In[ ]:


remover_plot('R', 35000)


# In[ ]:


# Class X1
describe_plot('X1')


# In[ ]:


remover_plot('X1', 18000)


# In[ ]:


# Class
describe_plot('X2')


# In[ ]:


remover_plot('X2', 10.0)


# In[ ]:


# Class X
describe_plot('X9')


# In[ ]:


remover_plot('X9', 0.6)


# In[ ]:


# Class Y1
describe_plot('Y1')


# In[ ]:


remover_plot('Y1', 4.7)


# In[ ]:


# Class Y2
describe_plot('Y2')


# In[ ]:


df = df.drop(df[df['Y2'] < -3.480000e-05].index)
sns.boxplot(df['Y2'])


# In[ ]:


# Class Y4
describe_plot('Y4')


# In[ ]:


df = df.drop(df[df['Y4'] < 0.012].index)
sns.boxplot(df['Y4'])


# In[ ]:


# Class y5
describe_plot('Y5')


# In[ ]:


remover_plot('Y5', 0.021)


# In[ ]:


df = df.drop(df[df['Y5'] < -0.02].index)
sns.boxplot(df['Y5'])


# In[ ]:


# Class Y6
describe_plot('Y6')


# In[ ]:


remover_plot('Y6', 0.6)


# In[ ]:


# Class Y7
describe_plot('Y7')


# In[ ]:


df = df.drop(df[df['Y7'] < -0.35].index)
sns.boxplot(df['Y7'])


# In[ ]:


# Class Y8
describe_plot('Y8')


# In[ ]:


remover_plot('Y8', 0.020)


# In[ ]:


# Class Y9
describe_plot('Y9')


# In[ ]:


remover_plot('Y9', 0.25)


# In[ ]:


# Class Z2
describe_plot('Z2')


# In[ ]:


remover_plot('Z2', 0.035)


# In[ ]:


# Class Z3
describe_plot('Z4')


# In[ ]:


remover_plot('Z4', 0.08)


# In[ ]:


# Class Z5
describe_plot('Z6')


# In[ ]:


remover_plot('Z6', 0.0003)


# # Correlation

# Converting class in one-hot values to find correlation

# In[ ]:


corr_mat = df.corr(method='pearson')


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr_mat, vmax=0.99, vmin=0.5, ax=ax, square=True, cmap='Blues')


# In[ ]:


sns.catplot(x='Class', kind='count', data=df)


# In[ ]:


df.shape


# # Separating label and feature and taking log to decrease skewness in data

# In[ ]:


Y = df['Class']


# In[ ]:


X = df.drop('Class', axis=1)


# In[ ]:


print('Size of label and target',X.shape, Y.shape)


# In[ ]:


X = np.log1p(X)


# # Train , Test Split stratify on Class due to imbalanced distibution

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, stratify=Y)


# In[ ]:


sns.countplot(y_train.sort_values(ascending=True))


# In[ ]:


sns.countplot(y_test.sort_values(ascending=True))


# # Scaling data

# In[ ]:


Rs = RobustScaler()


# In[ ]:


x_train = Rs.fit_transform(x_train)


# In[ ]:


x_test = Rs.transform(x_test)


# # Grid Search

# In[ ]:


def grid_search(estimator, param):

    grid = GridSearchCV(estimator, param, n_jobs=-1, cv=5)
    grid.fit(x_train, y_train)
    print('Best parameter', grid.best_params_)
    model = grid.best_estimator_
    pred = model.predict(x_test)
    return model, pred


# # Metrics

# In[ ]:


def metrics_evaluate(y_pred):
    con_mat = confusion_matrix(y_test, y_pred)
    print(con_mat)
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(con_mat, ax=ax, square=True, vmax=500, vmin=60)
    
    # Classification report
    class_report = pd.DataFrame(data=classification_report(y_test, y_pred, output_dict=True))
    print(class_report.head())
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)    
    print('Accuracy Test', accuracy*100)


# # Models

# CATBOOST
# Catboost is an advanced boosting algorithm which can easily work on Categorical feature. It also requires almost no hyperparameter tunning as it optimize itself to approximately best solution according to dataset.
# Precision, recall and accuracy provided by catboost is good compared to random forests 91%

# In[ ]:


cat_boost = cb.CatBoostClassifier()


# In[ ]:


cat_param = {}


# In[ ]:


model_cat, pred_cat = grid_search(cat_boost, cat_param)


# In[ ]:


metrics_evaluate(pred_cat)


# # Reason not to use PCA and Feature selection

# 1. Pca reduces dimesnionality of the dataset but it also discards some valuable info hence using Pca on such small dataset is not viable
# 2. Feature selection using Random forest decreased Precision and recall of the model as it also removed some usefull features

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




