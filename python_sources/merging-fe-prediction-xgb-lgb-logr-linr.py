#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# 
# # Merging feature importance diagrams and solution of 4 models (XGB, LGB, LogReg, LinReg) with equal weights 
# ## For the example of competition "Titanic: Machine Learning from Disaster"
# ### The choice of the optimum weights for merging of model solutions is making expertly based on the analysis of Confusion matrix and FI diagram of these models
# ### Feature importance diagrams has 6 options:
# * XGB
# * LGBM
# * LogisticRegression (with Eli5)
# * LinearRegression (with Eli5)
# * Mean
# * Merging

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [FE & EDA](#3)
# 1. [Preparing to modeling](#4)
# 1. [Tuning models, building the feature importance diagrams and prediction](#5)
#     -  [LGBM](#5.1)
#     -  [XGB](#5.2)
#     -  [Logistic Regression](#5.3)
#     -  [Linear Regression](#5.4)
# 1. [Showing Confusion Matrices](#6)
# 1. [Comparison and merging of all feature importance diagrams](#7)
# 1. [Merging solutions and submission](#8)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import eli5

import lightgbm as lgbm
import xgboost as xgb

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
submission = pd.read_csv('../input/titanic/gender_submission.csv')


# In[ ]:


traindf.head(3)


# In[ ]:


traindf.info()


# In[ ]:


testdf.info()


# In[ ]:


submission.head()


# ## 3. FE & EDA <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#Thanks to:
# https://www.kaggle.com/mauricef/titanic
# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code
#
df = pd.concat([traindf, testdf], axis=0, sort=False)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]
family = df.groupby(df.LastName).Survived
df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())
df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)
df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())
df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount -                                     df.Survived.fillna(0), axis=0)
df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)
df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)
df['Alone'] = (df.WomanOrBoyCount == 0)

#Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster
#"Title" improvement
df['Title'] = df['Title'].replace('Ms','Miss')
df['Title'] = df['Title'].replace('Mlle','Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')
# Embarked
df['Embarked'] = df['Embarked'].fillna('S')
# Cabin, Deck
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'

# Thanks to https://www.kaggle.com/erinsweet/simpledetect
# Fare
med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df['Fare'] = df['Fare'].fillna(med_fare)
#Age
df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
# Family_Size
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis
cols_to_drop = ['Name','Ticket','Cabin', 'IsWomanOrBoy', 'WomanOrBoyCount', 'FamilySurvivedCount']
df = df.drop(cols_to_drop, axis=1)

df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)
df.Alone = df.Alone.fillna(0)

target = df.Survived.loc[traindf.index]
df = df.drop(['Survived'], axis=1)
train, test = df.loc[traindf.index], df.loc[testdf.index]


# In[ ]:


pd.set_option('max_columns',100)


# In[ ]:


train.head(5)


# In[ ]:


train.info()


# In[ ]:


test.info()


# ## 4. Preparing to modeling <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to : https://www.kaggle.com/aantonova/some-new-risk-and-clusters-features
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train.columns.values.tolist()
for col in features:
    if train[col].dtype in numerics: continue
    categorical_columns.append(col)
for col in categorical_columns:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values)) 


# In[ ]:


train = reduce_mem_usage(train)


# In[ ]:


train.info()


# In[ ]:


test = reduce_mem_usage(test)


# In[ ]:


test.info()


# ## 5. Tuning models, building the feature importance diagrams and prediction<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 5.1 LGBM <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


X = train
z = target


# In[ ]:


#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
train_set = lgbm.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgbm.Dataset(Xval, Zval, silent=False)


# In[ ]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }

modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50, verbose_eval=10, valid_sets=valid_set)


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgbm.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


feature_score = pd.DataFrame(train.columns, columns = ['feature']) 
feature_score['score_lgb'] = modelL.feature_importance()


# In[ ]:


# Prediction
y_train_lgb = modelL.predict(train, num_iteration=modelL.best_iteration).astype('int')
y_preds_lgb = modelL.predict(test, num_iteration=modelL.best_iteration)


# ### 5.2 XGB<a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
data_train = xgb.DMatrix(train)
data_test  = xgb.DMatrix(test)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:logistic',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=10)

print('score = %1.5f, n_boost_round =%d.'%(modelx.best_score,modelx.best_iteration))


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
feature_score


# In[ ]:


# Prediction
y_train_xgb = modelx.predict(data_train).astype('int')
y_preds_xgb = modelx.predict(data_test)


# ### 5.3 Logistic Regression <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Standardization for regression models
Scaler_train = preprocessing.MinMaxScaler()
train = pd.DataFrame(
    Scaler_train.fit_transform(train),
    columns=train.columns,
    index=train.index
)


# In[ ]:


test = pd.DataFrame(
    Scaler_train.fit_transform(test),
    columns=test.columns,
    index=test.index
)


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train, target)
coeff_logreg = pd.DataFrame(train.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


len(coeff_logreg)


# In[ ]:


# the level of importance of features is not associated with the sign
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()
feature_score = pd.merge(feature_score, coeff_logreg, on='feature')


# In[ ]:


# Eli5 visualization
eli5.show_weights(logreg)


# In[ ]:


# Prediction
y_train_logreg = logreg.predict(train).astype('int')
y_preds_logreg = logreg.predict(test)


# ### 5.4 Linear Regression <a class="anchor" id="5.4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Linear Regression

linreg = LinearRegression()
linreg.fit(train, target)
coeff_linreg = pd.DataFrame(train.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


# Eli5 visualization
eli5.show_weights(linreg)


# In[ ]:


# the level of importance of features is not associated with the sign
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()


# In[ ]:


feature_score = pd.merge(feature_score, coeff_linreg, on='feature')
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')
feature_score


# In[ ]:


# Prediction
y_train_linreg = linreg.predict(train).astype('int')
y_preds_linreg = linreg.predict(test)


# ### 6. Showing Confusion Matrices<a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Showing Confusion Matrix
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title, figsize=(5,4)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[ ]:


# Showing Confusion Matrix for LGB model
plot_cm(y_train_lgb, z, 'Confusion matrix for LGB model', figsize=(7,7))


# In[ ]:


# Showing Confusion Matrix for XGB model
plot_cm(y_train_xgb, z, 'Confusion matrix for XGB model', figsize=(7,7))


# In[ ]:


# Showing Confusion Matrix for Logistic Regression
plot_cm(y_train_logreg, z, 'Confusion matrix for Logistic Regression', figsize=(7,7))


# In[ ]:


# Showing Confusion Matrix for Linear Regression
plot_cm(y_train_linreg, z, 'Confusion matrix for Linear Regression', figsize=(7,7))


# ### 7. Comparison and merging of all feature importance diagrams <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
# MinMax scale all importances
feature_score = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(feature_score),
    columns=feature_score.columns,
    index=feature_score.index
)

# Create mean column
feature_score['mean'] = feature_score.mean(axis=1)

# Plot the feature importances
plot_title = "Consolidation feature importance diagrams (by mean values)"
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10), title = plot_title)


# In[ ]:


feature_score.sort_values('mean', ascending=False)


# In[ ]:


# Set weight of models
w_lgb = 0.48
w_xgb = 0.48
w_logreg = 0.03
w_linreg = 1 - w_lgb - w_xgb - w_logreg
w_linreg


# In[ ]:


# Merging FI diagram
# Create merging column with different weights
feature_score['merging'] = w_lgb*feature_score['score_lgb'] + w_xgb*feature_score['score_xgb']                        + w_logreg*feature_score['score_logreg'] + w_linreg*feature_score['score_linreg']

# Plot the feature importances
plot_title = "Consolidation feature importance diagrams (by merging values)"
feature_score.sort_values('merging', ascending=False).plot(kind='bar', figsize=(20, 10), title = plot_title)


# In[ ]:


feature_score.sort_values('merging', ascending=False)


# ### 8. Merging solutions and submission<a class="anchor" id="8"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


y_preds = w_lgb*y_preds_lgb + w_xgb*y_preds_xgb + w_logreg*y_preds_logreg + w_linreg*y_preds_linreg


# In[ ]:


submission['Survived'] = [1 if x>0.5 else 0 for x in y_preds]
submission.head()


# In[ ]:


submission['Survived'].hist()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0)
