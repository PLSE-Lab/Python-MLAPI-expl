#!/usr/bin/env python
# coding: utf-8

# # Titanic with 3 models

# ## **Acknowledgements**
# #### This kernel uses such good kernels:
#    - https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
#    - https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
#    - https://www.kaggle.com/ash316/eda-to-prediction-dietanic
#    - https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
#    - https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
#    - https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic
#    - https://www.kaggle.com/headsortails/pytanic

# <a class="anchor" id="0.1"></a>
# ## **Table of Contents**
# 1. [Import libraries](#1)
# 2. [Download datasets](#2)
# 3. [Data research](#3)
# 4. [FE](#4)
# 5. [EDA & Visualization](#5)
# 6. [Preparing data for building the feature importance diagrams](#6)  
# 7. [FE: building the feature importance diagrams](#7)
#   -  [LGBM](#7.1)
#   -  [XGB](#7.2)
#   -  [Logistic Regression](#7.3)
#   -  [Linear Reagression](#7.4)
# 8. [Comparison of the all feature importance diagrams](#8)
# 9. [Preparing to modeling](#9)
#   - [Data for modeling](#9.1)
#   - [Encoding categorical features](#9.2)
# 10. [Tuning models](#10)
#   -  [Random Forest](#10.1)
#   -  [XGB](#10.2)
#   -  [GradientBoostingClassifier](#10.3)
# 11. [Models evaluation](#11)
# 12. [Prediction & Output data](#11)

# <a class="anchor" id="1"></a>
# ## 1. Import libraries 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


import copy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# WordCloud
from wordcloud import WordCloud


# preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# models
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier


# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
# Setting up visualisations
sns.set_style(style='white') 
sns.set(rc={
    'figure.figsize':(12,7), 
    'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '.9',
    'axes.linewidth': 1.0,
    'grid.linestyle': u'-'},font_scale=1.5)
custom_colors = ["#3498db", "#95a5a6","#34495e", "#2ecc71", "#e74c3c"]
sns.set_palette(custom_colors)


# <a class="anchor" id="2"></a>
# ## 2. Download datasets 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId') # Train data
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId') # Test data
td = pd.concat([copy.deepcopy(traindf), copy.deepcopy(testdf)], axis=0, sort  = False) # Train & Test data
submission = pd.read_csv('../input/titanic/gender_submission.csv') # Form for answers


# <a class="anchor" id="3"></a>
# ## 3. Data research
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
# We check the train sample for balance
traindf['Survived'].value_counts(normalize=True)


# In[ ]:


td.nunique()


# In[ ]:


td.describe()


# <a class="anchor" id="4"></a>
# ## 4. FE 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


#Thanks to:
# https://www.kaggle.com/mauricef/titanic
# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code
#
td = pd.concat([traindf, testdf], axis=0, sort=False)
td['Title'] = td.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
td['Title'] = td.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
td['IsWomanOrBoy'] = ((td.Title == 'Master') | (td.Sex == 'female'))
td['LastName'] = td.Name.str.split(',').str[0]
family = td.groupby(td.LastName).Survived
td['WomanOrBoyCount'] = family.transform(lambda s: s[td.IsWomanOrBoy].fillna(0).count())
td['WomanOrBoyCount'] = td.mask(td.IsWomanOrBoy, td.WomanOrBoyCount - 1, axis=0)
td['FamilySurvivedCount'] = family.transform(lambda s: s[td.IsWomanOrBoy].fillna(0).sum())
td['FamilySurvivedCount'] = td.mask(td.IsWomanOrBoy, td.FamilySurvivedCount -                                     td.Survived.fillna(0), axis=0)
td['WomanOrBoySurvived'] = td.FamilySurvivedCount / td.WomanOrBoyCount.replace(0, np.nan)
td.WomanOrBoyCount = td.WomanOrBoyCount.replace(np.nan, 0)
td['Alone'] = (td.WomanOrBoyCount == 0)

#Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster
#"Title" improvement
td['Title'] = td['Title'].replace('Ms','Miss')
td['Title'] = td['Title'].replace('Mlle','Miss')
td['Title'] = td['Title'].replace('Mme','Mrs')
# Embarked
td['Embarked'] = td['Embarked'].fillna('S')
# Cabin, Deck
td['Deck'] = td['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
td.loc[(td['Deck'] == 'T'), 'Deck'] = 'A'

# Thanks to https://www.kaggle.com/erinsweet/simpledetect
# Fare
med_fare = td.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
td['Fare'] = td['Fare'].fillna(med_fare)
#Age
td['Age'] = td.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
# Family_Size
td['Family_Size'] = td['SibSp'] + td['Parch'] + 1

# Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis
cols_to_drop = ['Name','Ticket','Cabin']
td = td.drop(cols_to_drop, axis=1)

td.WomanOrBoySurvived = td.WomanOrBoySurvived.fillna(0)
td.WomanOrBoyCount = td.WomanOrBoyCount.fillna(0)
td.FamilySurvivedCount = td.FamilySurvivedCount.fillna(0)
td.Alone = td.Alone.fillna(0)


# In[ ]:


target = td.Survived.loc[traindf.index]
td = td.drop(['Survived'], axis=1)
train, test = td.loc[traindf.index], td.loc[testdf.index]


# <a class="anchor" id="5"></a>
# ## 5. EDA & Visualization
# ##### [Back to Table of Contents](#0.1)

# #### Correlation Between The Features

# In[ ]:


# Thanks to: https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(td.corr(), dtype=np.bool)
#mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (15,12))
sns.heatmap(td.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='gray',
            fmt='.2g',
            center = 0,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);


# #### Survived

# In[ ]:


# Thanks to: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
f,ax=plt.subplots(1,2,figsize=(18,8))
traindf['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=traindf,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# #### Pclass

# In[ ]:


# Thanks to: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
f,ax=plt.subplots(1,2,figsize=(18,8))
traindf['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=traindf,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
f,ax=plt.subplots(1,2,figsize=(20,10))
traindf[traindf['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
traindf[traindf['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


# Thanks to: https://www.kaggle.com/headsortails/pytanic
dummy = td[td['Title'].isin(['Mr','Miss','Mrs','Master'])]
foo = dummy['Age'].hist(by=dummy['Title'], bins=np.arange(0,81,1))


# In[ ]:


# Thanks to: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
f,ax=plt.subplots(2,2,figsize=(20,15))
sns.countplot('Embarked',data=traindf,ax=ax[0,0])
ax[0,0].set_title('No. Of Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=traindf,ax=ax[0,1])
ax[0,1].set_title('Male-Female Split for Embarked')
sns.countplot('Embarked',hue='Survived',data=traindf,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=traindf,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# In[ ]:


td['Family'] = td.Parch + td.SibSp
td['Is_Alone'] = td.Family == 0


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
traindf['Fare_Category'] = pd.cut(traindf['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid', 'High_Mid','High'])


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
p = sns.countplot(x = "Embarked", hue = "Survived", data = traindf, palette=["C1", "C0"])
p.set_xticklabels(["Southampton","Cherbourg","Queenstown"])
p.legend(labels = ["Deceased", "Survived"])
p.set_title("Training Data - Survival based on embarking point.")


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
traindf.Embarked.fillna(traindf.Embarked.mode()[0], inplace = True)


# In[ ]:


traindf.info()


# <a class="anchor" id="6"></a>
# ## 6. Preparing data for building the feature importance diagrams 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Clone data for FE
train_fe = copy.deepcopy(traindf)
target_fe = train_fe['Survived']
del train_fe['Survived']


# In[ ]:


train_fe = train_fe.fillna(train_fe.isnull())


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train_fe.columns.values.tolist()
for col in features:
    if train_fe[col].dtype in numerics: continue
    categorical_columns.append(col)
indexer = {}
for col in categorical_columns:
    if train_fe[col].dtype in numerics: continue
    _, indexer[col] = pd.factorize(train_fe[col])
    
for col in categorical_columns:
    if train_fe[col].dtype in numerics: continue
    train_fe[col] = indexer[col].get_indexer(train_fe[col])


# In[ ]:


train_fe.info()


# <a class="anchor" id="7"></a>
# ## 7. FE: building the feature importance diagrams
# ##### [Back to Table of Contents](#0.1)

# <a class="anchor" id="7.1"></a>
# ### 7.1 LGBM
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
X = train_fe
z = target_fe


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
train_set = lgb.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgb.Dataset(Xval, Zval, silent=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
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

modelL = lgb.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgb.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
feature_score = pd.DataFrame(train_fe.columns, columns = ['feature']) 
feature_score['score_lgb'] = modelL.feature_importance()


# <a class="anchor" id="7.2"></a>
# ### 7.2 XGB
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
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


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
feature_score


# <a class="anchor" id="7.3"></a>
# ### 7.3 Logistic Regression
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Standardization for regression models
train = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(train_fe),
    columns=train_fe.columns,
    index=train_fe.index
)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train_fe, target_fe)
coeff_logreg = pd.DataFrame(train_fe.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


len(coeff_logreg)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# the level of importance of features is not associated with the sign
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()
feature_score = pd.merge(feature_score, coeff_logreg, on='feature')


# <a class="anchor" id="7.4"></a>
# ### 7.4 Linear Reagression
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Linear Regression

linreg = LinearRegression()
linreg.fit(train_fe, target_fe)
coeff_linreg = pd.DataFrame(train_fe.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# the level of importance of features is not associated with the sign
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
feature_score = pd.merge(feature_score, coeff_linreg, on='feature')
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')
feature_score


# <a class="anchor" id="8"></a>
# ## 8. Comparison of the all feature importance diagrams
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
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
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
feature_score.sort_values('mean', ascending=False)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
# Create total column with different weights
feature_score['total'] = 0.5*feature_score['score_lgb'] + 0.3*feature_score['score_xgb']                        + 0.1*feature_score['score_logreg'] + 0.1*feature_score['score_linreg']

# Plot the feature importances
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


#Thanks to: https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
feature_score.sort_values('total', ascending=False)


# <a class="anchor" id="9"></a>
# ## 9. Preparing to modeling
# ##### [Back to Table of Contents](#0.1)

# <a class="anchor" id="9.1"></a>
# ### 9.1 Data for modeling 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
target = traindf.Survived.loc[traindf.index]
train, test = td.loc[traindf.index], td.loc[testdf.index]


# In[ ]:


train.head(3)


# In[ ]:


test.head(3)


# In[ ]:


target[:3]


# <a class="anchor" id="9.2"></a>
# ### 9.2 Encoding categorical features
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train.columns.values.tolist()
for col in features:
    if train[col].dtype in numerics: continue
    categorical_columns.append(col)
categorical_columns


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
# Encoding categorical features
for col in categorical_columns:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values)) 


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
#%% split training set to validation set
SEED = 100
Xtrain, Xval, Ztrain, Zval = train_test_split(train, target, test_size=0.3, random_state=SEED)


# <a class="anchor" id="10"></a>
# ## 10. Tuning models
# ##### [Back to Table of Contents](#0.1)

# <a class="anchor" id="10.1"></a>
# ### 10.1 Random Forest 
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
# Random Forest

random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)
random_forest.fit(train, target)
Y_pred = random_forest.predict(test).astype(int)
random_forest.score(train, target)
acc_random_forest = round(random_forest.score(train, target) * 100, 2)
print(acc_random_forest,random_forest.best_params_)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
random_forest_submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})


# <a class="anchor" id="10.2"></a>
# ### 10.2 XGB
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
def hyperopt_xgb_score(params):
    clf = XGBClassifier(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_xgb = {
            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'eta': hp.quniform('eta', 0.025, 0.5, 0.005),
            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),
            'subsample': hp.quniform('subsample', 0.5, 1, 0.005),
            'gamma': hp.quniform('gamma', 0.5, 1, 0.005),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),
            'eval_metric': 'auc',
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'tree_method': 'exact',
            'silent': 1,
            'missing': None
        }
 
best = fmin(fn=hyperopt_xgb_score, space=space_xgb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
params = space_eval(space_xgb, best)
params


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
XGB_Classifier = XGBClassifier(**params)
XGB_Classifier.fit(train, target)
Y_pred = XGB_Classifier.predict(test).astype(int)
XGB_Classifier.score(train, target)
acc_XGB_Classifier = round(XGB_Classifier.score(train, target) * 100, 2)
acc_XGB_Classifier


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
xgb_submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(XGB_Classifier,ax = axes,height =0.5)
plt.show();
plt.close()


# <a class="anchor" id="10.3"></a>
# ### 10.3 GradientBoostingClassifier
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
def hyperopt_gb_score(params):
    clf = GradientBoostingClassifier(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_gb = {
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
        }
 
best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
params = space_eval(space_gb, best)
params


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
# Gradient Boosting Classifier

gradient_boosting = GradientBoostingClassifier(**params)
gradient_boosting.fit(train, target)
Y_pred = gradient_boosting.predict(test).astype(int)
gradient_boosting.score(train, target)
acc_gradient_boosting = round(gradient_boosting.score(train, target) * 100, 2)
acc_gradient_boosting


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
submission_gradient_boosting = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})


# <a class="anchor" id="11"></a>
# ## 11. Models evaluation
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
models = pd.DataFrame({
    'Model': ['Random Forest',  'XGBClassifier', 'GradientBoostingClassifier'],
    
    'Score': [acc_random_forest, acc_XGB_Classifier, acc_gradient_boosting]})


# In[ ]:


models


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models
# Plot
plt.figure(figsize=[15,6])
xx = models['Model']
plt.tick_params(labelsize=14)
plt.plot(xx, models['Score'], label = 'Score')
plt.legend()
plt.title('Score  for 3 popular models')
plt.xlabel('Models')
plt.ylabel('Score, %')
plt.xticks(xx, rotation='vertical')
plt.savefig('graph.png')
plt.show()


# <a class="anchor" id="12"></a>
# ## 12. Prediction & Output data
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


random_forest_submission.to_csv('submission_RandomForest.csv', index=False)


# In[ ]:


xgb_submission.to_csv('submission_XGB_Classifier.csv', index=False)


# In[ ]:


submission_gradient_boosting.to_csv('submission_gradient_boosting.csv', index=False)

