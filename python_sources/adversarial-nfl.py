#!/usr/bin/env python
# coding: utf-8

# In this notebook we'll try to use adversarial validation in order to see how similar/different the train and test sets are. Since this is not a "regular" Kaggle competition, a bit of ingenuity was needed to get test data into the form that would allow for this to be done. Furthermore, using a few extra engineered features is alwasy a good ide. For both of those purposes we will use [Tomonori Masui](https://www.kaggle.com/tnmasui)'s [excellent kernel](https://www.kaggle.com/tnmasui/nfl-eda-on-train-test-fe-correlation-anova).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn import model_selection, preprocessing, metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import shap
import os
print(os.listdir("../input"))
from sklearn import preprocessing
import xgboost as xgb
import gc


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/nfl-eda-on-train-test-fe-correlation-anova/train.csv')
test = pd.read_csv('../input/nfl-eda-on-train-test-fe-correlation-anova/test.csv')


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


object_columns = train.select_dtypes(include='object').columns


# In[ ]:


test.shape


# In[ ]:


features = test.columns


# In[ ]:


train = train[features]


# In[ ]:


train['target'] = 0
test['target'] = 1


# In[ ]:


train_test = pd.concat([train, test], axis =0)

target = train_test['target'].values
del train, test
gc.collect()


# In[ ]:


# Label Encoding
for f in object_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_test[f].values) )
    train_test[f] = lbl.transform(list(train_test[f].values))


# In[ ]:


train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)
del train_test
gc.collect()


# In[ ]:


train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()


# In[ ]:


train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)


# In[ ]:


param = {'num_leaves': 50,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 5,
         'learning_rate': 0.2,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 44,
         "metric": 'auc',
         "verbosity": -1}


# In[ ]:


num_round = 50
clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# Well, folks, doesn't get any better than the AUC of 1.0! Let's see which ones are the most responsibel columns.

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# Guess season and Defense Personnel change significantly between train and test. let's remove them and see what happens.

# In[ ]:


del train, test, clf
gc.collect()

train = pd.read_csv('../input/nfl-eda-on-train-test-fe-correlation-anova/train.csv')

test = pd.read_csv('../input/nfl-eda-on-train-test-fe-correlation-anova/test.csv')

del train['Season']
del train['DefensePersonnel']
del test['Season']
del test['DefensePersonnel']

object_columns = train.select_dtypes(include='object').columns

features = test.columns
train = train[features]
test = test[features]

train['target'] = 0
test['target'] = 1

train_test = pd.concat([train, test], axis =0)

del train, test

target = train_test['target'].values

# Label Encoding
for f in object_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_test[f].values) )
    train_test[f] = lbl.transform(list(train_test[f].values))
    
train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)

train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()

train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# OK, AUC of 0.999261 is not too shabby eaither. Let's see what's going on here.

# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# So temperature and humidity are the top two factors. In the light of the previous finding this is not surprising at all: both are very good proxies for the season. Let's remove those two features as well adn see what happens.

# In[ ]:


del train, test, clf
gc.collect()

train = pd.read_csv('../input/nfl-eda-on-train-test-fe-correlation-anova/train.csv')

test = pd.read_csv('../input/nfl-eda-on-train-test-fe-correlation-anova/test.csv')

del train['Season']
del train['DefensePersonnel']
del train['Temperature']
del train['Humidity']
del test['Season']
del test['DefensePersonnel']
del test['Temperature']
del test['Humidity']

object_columns = train.select_dtypes(include='object').columns

features = test.columns
train = train[features]
test = test[features]

train['target'] = 0
test['target'] = 1

train_test = pd.concat([train, test], axis =0)

del train, test

target = train_test['target'].values

# Label Encoding
for f in object_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_test[f].values) )
    train_test[f] = lbl.transform(list(train_test[f].values))
    
train, test = model_selection.train_test_split(train_test, test_size=0.33, random_state=42, shuffle=True)

train_y = train['target'].values
test_y = test['target'].values
del train['target'], test['target']
gc.collect()

train = lgb.Dataset(train, label=train_y)
test = lgb.Dataset(test, label=test_y)

clf = lgb.train(param, train, num_round, valid_sets = [train, test], verbose_eval=50, early_stopping_rounds = 50)


# In[ ]:


feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),features)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(20))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:




