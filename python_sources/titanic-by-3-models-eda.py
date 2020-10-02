#!/usr/bin/env python
# coding: utf-8

# # Titanic by 3 models + EDA

# ## **Acknowledgements**
# #### This kernel uses such good kernels:
#    - https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
#    - https://www.kaggle.com/vbmokin/feature-importance-xgb-lgbm-logreg-linreg
#    - https://www.kaggle.com/ash316/eda-to-prediction-dietanic
#    - https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
#    - https://www.kaggle.com/vbmokin/used-cars-price-prediction-by-15-models

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
#   -  [LGBM](#10.3)
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
from sklearn.ensemble import RandomForestClassifier
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


# ## Preview settings

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
# Missing values
td.isnull().sum()
sns.heatmap(td.isnull(), cbar = False).set_title("Missing values heatmap")


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


td['Title'] = td.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
td['Title'] = td.Name.str.split(',').str[1].str.split('.').str[0].str.strip()

td['Title'] = td['Title'].replace('Ms','Miss')
td['Title'] = td['Title'].replace('Mlle','Miss')
td['Title'] = td['Title'].replace('Mme','Mrs')

td['Age'] = td.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

med_fare = td.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
td['Fare'] = td['Fare'].fillna(med_fare)


# In[ ]:


# Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis
cols_to_drop = ['Name','Ticket','Cabin']
td = td.drop(cols_to_drop, axis=1)          
traindf = traindf.fillna(traindf.isnull())


# <a class="anchor" id="5"></a>
# ## 5. EDA & Visualization
# ##### [Back to Table of Contents](#0.1)

# #### Correlation Between The Features

# In[ ]:


# Thanks to: https://www.kaggle.com/ash316/eda-to-prediction-dietanic
sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) 
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# #### Survived

# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
(traindf.Survived.value_counts(normalize=True) * 100).plot.barh().set_title("Training Data - Percentage of people survived and Deceased")


# #### Pclass

# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
fig_pclass = traindf.Pclass.value_counts().plot.pie().legend(labels=["Class 3","Class 1","Class 2"], loc='center right', bbox_to_anchor=(2.25, 0.5)).set_title("Training Data - People travelling in different classes")


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
pclass_1_survivor_distribution = round((traindf[traindf.Pclass == 1].Survived == 1).value_counts()[1]/len(traindf[traindf.Pclass == 1]) * 100, 2)
pclass_2_survivor_distribution = round((traindf[traindf.Pclass == 2].Survived == 1).value_counts()[1]/len(traindf[traindf.Pclass == 2]) * 100, 2)
pclass_3_survivor_distribution = round((traindf[traindf.Pclass == 3].Survived == 1).value_counts()[1]/len(traindf[traindf.Pclass == 3]) * 100, 2)
pclass_perc_df = pd.DataFrame(
    { "Percentage Survived":{"Class 1": pclass_1_survivor_distribution,"Class 2": pclass_2_survivor_distribution, "Class 3": pclass_3_survivor_distribution},  
     "Percentage Not Survived":{"Class 1": 100-pclass_1_survivor_distribution,"Class 2": 100-pclass_2_survivor_distribution, "Class 3": 100-pclass_3_survivor_distribution}})
pclass_perc_df.plot.bar().set_title("Training Data - Percentage of people survived on the basis of class")


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
pclass_perc_df


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
fig_sex = (traindf.Sex.value_counts(normalize = True) * 100).plot.bar()
male_pr = round((traindf[traindf.Sex == 'male'].Survived == 1).value_counts()[1]/len(traindf.Sex) * 100, 2)
female_pr = round((traindf[traindf.Sex == 'female'].Survived == 1).value_counts()[1]/len(traindf.Sex) * 100, 2)
sex_perc_df = pd.DataFrame(
    { "Percentage Survived":{"male": male_pr,"female": female_pr},  "Percentage Not Survived":{"male": 100-male_pr,"female": 100-female_pr}})
sex_perc_df.plot.barh().set_title("Percentage of male and female survived and Deceased")
fig_sex


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
traindf['Age_Range'] = pd.cut(traindf.Age, [0, 10, 20, 30, 40, 50, 60,70,80])
sns.countplot(x = "Age_Range", hue = "Survived", data = traindf, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
sns.distplot(td['Age'].dropna(),color='darkgreen',bins=30)


# In[ ]:


td['Family'] = td.Parch + td.SibSp
td['Is_Alone'] = td.Family == 0


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
traindf['Fare_Category'] = pd.cut(traindf['Fare'], bins=[0,7.90,14.45,31.28,120], labels=['Low','Mid', 'High_Mid','High'])


# In[ ]:


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
x = sns.countplot(x = "Fare_Category", hue = "Survived", data = traindf, palette=["C1", "C0"]).legend(labels = ["Deceased", "Survived"])
x.set_title("Survival based on fare category")


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


# Thanks to: https://www.kaggle.com/sumukhija/tip-of-the-iceberg-eda-prediction-0-80861
traindf['Salutation'] = traindf.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip()) 
traindf.Salutation.nunique()
wc = WordCloud(width = 1000,height = 450,background_color = 'white').generate(str(traindf.Salutation.values))
plt.imshow(wc, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

traindf.Salutation.value_counts()


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
target = td.Survived.loc[traindf.index]
td = td.drop(['Survived'], axis=1)
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
# ### 10.3 LGBM
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
def hyperopt_lgb_score(params):
    clf = LGBMClassifier(**params)
    current_score = cross_val_score(clf, train, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_lgb = {
            'learning_rate': hp.quniform('learning_rate', 0, 0.05, 0.0001),
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth':  hp.choice('max_depth', np.arange(2, 12, dtype=int)),
            'num_leaves': hp.choice('num_leaves', 2*np.arange(2, 2**11, dtype=int)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 9, 0.025),
            'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.005),
            'objective': 'binary',
            'boosting_type': 'gbdt',
            }
 
best = fmin(fn=hyperopt_lgb_score, space=space_lgb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
params = space_eval(space_lgb, best)
params


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
LGB_Classifier = LGBMClassifier(**params)
LGB_Classifier.fit(train, target)
Y_pred = LGB_Classifier.predict(test).astype(int)
LGB_Classifier.score(train, target)
acc_LGB_Classifier = round(LGB_Classifier.score(train, target) * 100, 2)
acc_LGB_Classifier


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
lgb_submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": Y_pred})


# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgb.plot_importance(LGB_Classifier,ax = axes,height = 0.5)
plt.show();
plt.close()


# <a class="anchor" id="11"></a>
# ## 11. Models evaluation
# ##### [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to: https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models
models = pd.DataFrame({
    'Model': ['Random Forest',  'XGBClassifier', 'LGBMClassifier'],
    
    'Score': [acc_random_forest, acc_XGB_Classifier, acc_LGB_Classifier]})


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


lgb_submission.to_csv('submission_LGB_Classifier.csv', index=False)

