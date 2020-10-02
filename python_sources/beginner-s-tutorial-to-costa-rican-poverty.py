#!/usr/bin/env python
# coding: utf-8

# # Beginner's Tutorial to Costa Rican Household Poverty Level Prediction
# ![](https://www.habitatforhumanity.org.uk/wp-content/uploads/2017/10/Housing-poverty-Costa-Rica--1200x600-c-default.jpg)
# ![](https://www.habitatforhumanity.org.uk/wp-content/uploads/2017/10/where-we-work-costa-rica--800x400-c-default.jpg)
#  ## How we address housing poverty in Costa Rica

# ### Import Libraries

# In[ ]:


import datetime
import gc
import numpy as np
import os
import operator
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# ## Shape of the data

# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


print("Costa Rican Household Poverty Level Prediction -  rows:",train_df.shape[0]," columns:", train_df.shape[1])
print("Costa Rican Household Poverty Level Prediction -  rows:",test_df.shape[0]," columns:", test_df.shape[1])


# In[ ]:


test_df.head()


# ## Missing Values

# In[ ]:


train_df.isnull().values.any()


# In[ ]:


test_df.isnull().values.any()


# In[ ]:


train_df.info()


# In[ ]:


train_df.isnull().values.sum(axis=0)


# In[ ]:


train_df_describe = train_df.describe()
train_df_describe


# In[ ]:


test_df_describe = test_df.describe()
test_df_describe


# In[ ]:


test_df.isnull().values.sum(axis=0)


# ## Distribution of Target Variable

# In[ ]:


plt.figure(figsize=(12, 5))
plt.hist(train_df.Target.values, bins=4)
plt.title('Histogram - target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[ ]:


plt.title("Distribution of Target")
sns.distplot(train_df['Target'].dropna(),color='blue', kde=True,bins=100)
plt.show()


# ### Violin distribution of target

# In[ ]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=train_df.Target.values)
plt.show()


# In[ ]:


plt.title("Distribution of log(target)")
sns.distplot(np.log1p(train_df['Target']).dropna(),color='blue', kde=True,bins=100)
plt.show()


# In[ ]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(1+train_df.Target.values))
plt.show()


# In[ ]:


np.unique(train_df.Target.values)


# In[ ]:


columns_to_use = train_df.columns[1:-1]


# In[ ]:


columns_to_use


# In[ ]:


y = train_df['Target'].values-1


# In[ ]:


train_test_df = pd.concat([train_df[columns_to_use], test_df[columns_to_use]], axis=0)
# extract columns which data type is object
object_cols = [f_ for f_ in train_test_df.columns if train_test_df[f_].dtype == 'object']


# In[ ]:


# labeling
for col in object_cols:
    le = LabelEncoder()
    print(col)
    le.fit(train_test_df[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
del le


# In[ ]:


def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) +         " ".join(map(str,args)), **kwargs)

id_name = 'Id'
target_name = 'Target'

df_all = pd.concat([train_df, test_df], axis=0)
cols = [f_ for f_ in df_all.columns if df_all[f_].dtype == 'object' and f_ != id_name]
print(cols)

for c in tqdm(cols):
    le = preprocessing.LabelEncoder()
    le.fit(df_all[c].astype(str))
    train_df[c] = le.transform(train_df[c].astype(str))
    test_df[c] = le.transform(test[c].astype(str))

    del le
gc.collect()

def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms']

extract_features(train_df)
extract_features(test_df)


# ## Identifying features that are highly correlated with target

# In[ ]:


labels = []
values = []
for col in train_df.columns:
    if col not in ["Id", "Target"]:
        labels.append(col)
        values.append(np.corrcoef(train_df[col].values, train_df["Target"].values)[0,1])
corr_df = pd.DataFrame({'columns_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
 
corr_df = corr_df[(corr_df['corr_values']>0.20) | (corr_df['corr_values']<-0.20)]
ind = np.arange(corr_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(10,6))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='black')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.columns_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# ## Correlation matrix of the most highly correlated features

# In[ ]:


temp_df = train_df[corr_df.columns_labels.tolist()]
corrmat = temp_df.corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrmat, vmax=1., square=True, cmap=plt.cm.BrBG)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# ## Predictive Model

# In[ ]:


cnt = 0
p_buf = []
n_splits = 20
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=None)
err_buf = []   

cols_to_drop = [
    id_name, 
    target_name,
]
X = train_df.drop(cols_to_drop, axis=1, errors='ignore')
feature_names = list(X.columns)
X = X.fillna(0)
X = X.values
y = train_df[target_name].values

classes = np.unique(y)
dprint('Number of classes: {}'.format(len(classes)))
c2i = {}
i2c = {}
for i, c in enumerate(classes):
    c2i[c] = i
    i2c[i] = c

y_le = np.array([c2i[c] for c in y])

X_test = test_df.drop(cols_to_drop, axis=1, errors='ignore')
X_test = X_test.fillna(0)
X_test = X_test.values
id_test = test_df[id_name].values

dprint(X.shape, y.shape)
dprint(X_test.shape)

n_features = X.shape[1]

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': -1,
    'num_leaves': 14,
    'learning_rate': 0.1,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 8,
    'colsample_bytree': 0.89,
    'min_child_samples': 90,
    'subsample': 0.96,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': len(np.unique(y)),
}


# In[ ]:


sampler = RandomUnderSampler(random_state=314)
X, y = sampler.fit_sample(X, y)
y_le = np.array([c2i[c] for c in y])

for train_index, valid_index in kf.split(X, y):
    print('Fold {}/{}*{}'.format(cnt + 1, n_splits, n_repeats))
    params = lgb_params.copy() 

    lgb_train = lgb.Dataset(
        X[train_index], 
        y_le[train_index], 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X[valid_index], 
        y_le[valid_index],
        feature_name=feature_names,
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=99999,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=400, 
        verbose_eval=100, 
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(10):
            if i < len(tuples):
                print(i, tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X[valid_index], num_iteration=model.best_iteration)

    err = f1_score(y_le[valid_index], np.argmax(p, axis=1), average='macro')

    dprint('{} F1: {}'.format(cnt + 1, err))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)

    cnt += 1

    del model, lgb_train, lgb_valid, p
    gc.collect


# In[ ]:


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.6f} +/- {:.6f}'.format(err_mean, err_std))
preds = p_buf/cnt


# In[ ]:


print(preds)
preds = np.argmax(preds, axis = 1) +1
preds


# In[ ]:


sample_submission  = pd.read_csv("../input/sample_submission.csv")
sample_submission.head()


# In[ ]:


sample_submission['Target'] = preds
sample_submission.to_csv('submission_{:.6f}.csv'.format(err_mean), index=False)
sample_submission.head()


# In[ ]:


np.mean(preds)

