#!/usr/bin/env python
# coding: utf-8

# Versions of some packages installed on my computer are different from what they have on kaggle, that's why output of this notebook has slightly different scores from what you've seen on the leaderboard. However, that's close version of notebook that gave me submission with private LB score 564855.334.

# In[ ]:


import pandas as pd
import numpy as np
import os
import datetime as dt

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm_notebook as tqdm

import lightgbm as lgbm


# In[ ]:


os.listdir('../input/')


# In[ ]:


sns.set(style = 'darkgrid', palette = 'spring')

pd.set_option('display.max_columns', 400)
pd.set_option('max_rows', 1000)


# ## Load data

# In[ ]:


DATA_FOLDER = '../input/infopulsehackathon/'
SUBMISSIONS_FOLDER = ''
os.listdir(DATA_FOLDER)


# In[ ]:


df  = pd.read_csv(DATA_FOLDER+'train.csv', index_col='Id')
df.shape


# In[ ]:


df.head()


# In[ ]:


target = 'Energy_consumption'


# ## EDA

# In[ ]:


df.isna().any().any()


# Luckily, there are no NAs :)

# In[ ]:


fig, ax = plt.subplots(1,2,figsize=(10, 25))
df.nunique()[:(df.shape[1] // 2):].sort_index(ascending=False).plot(kind='barh', ax = ax[0])
df.nunique()[(df.shape[1] // 2):].sort_index(ascending=False).plot(kind='barh', ax = ax[1])
ax[0].set_xlim(0, 4000)
ax[1].set_xlim(0, 4000)
fig.suptitle('Count unique values of each feature');
fig.tight_layout(rect=[0, 0.05, 1, 0.98])


# In[ ]:


print('Stats of unique values number:')
df.nunique().describe()


# Looks like most features have just a few unique values thus they might be categorical even if they are represented as numeric variables.

# Lets look at the features that are not numeric for sure.

# In[ ]:


df.select_dtypes(exclude = np.number)


# In[ ]:


print('Number of unique values:')
df.select_dtypes(exclude = np.number).nunique()


# ## Feature selection

# We have large amount of features, but doubt all of them should be included modeling dataset. 
# Several steps were followed to decrease number of features:
# - try to get rid of multicollinearity (high intercorrelations  between independent variables). Even though LightGBM is known to be robusted to such a thing, highly intercorrelated features make no good as well - they simply do not provide model with new information. sklearn.feature_selection.VarianceThreshold is used to remove similar variables
# - exclude variables that have only one unique value

# In[ ]:


var_cuter = VarianceThreshold(threshold=0.1)
df_var = df.drop(target, axis=1).select_dtypes(np.number)
var_mask = var_cuter.fit(df_var).get_support()
var_drop_cols = df_var.iloc[:,var_mask==False].columns.tolist()

print('low-variance features:')
var_drop_cols


# In[ ]:


print('Features that have single value')
df.nunique()[df.nunique()<2]


# In[ ]:


drop_cols = df.nunique()[df.nunique()<2].index.tolist()
drop_cols += var_drop_cols
print(f'{len(drop_cols)} features will be droped:')
drop_cols


# ## Categorical features

# We have already seen that feature_3, feature_4, feature_257, feature_258 are string type thus they are categorical for sure.
# I also assume that there are more features that are categorical but represented as numeric features. Of course, I cannot prove it and I'm not sure that my assumption is right, but I tried to treat features with low number of unique values as categorical ones and that slightly improved performance.

# In[ ]:


cat_cols = df.select_dtypes(exclude= np.number).columns.tolist() # feature_3, feature_4, feature_257, feature_258
cat_cols += df.nunique()[df.nunique() < 5].index.tolist() # features that have less that n unique values
cat_cols = [col for col in cat_cols if col not in drop_cols]
cat_cols = list(set(cat_cols))
len(cat_cols)


# ## Model training

# #### RepeatedKFold. 
# Started from simple KFold with 5 folds (also tried 4 and 10 folds), than switched to RepeatedKFold with 5 folds and 10 repeats. So, 5x10=50 models are used for prediction which is complete overkill:) But I wanted to be sure that performance of the models is stable and does not depend on random state. Also, I decided to use this approarch instead of blending because it's easier. StratifiedKfold with stratification along target quartiles did not improve public score in my case (tried with a few different random states); even though standard deviation of validation set is smaller with StratifiedKfold it might be because train and validation sets have similar distributions and not because models performance gets better. 
# #### LightGBM. 
# I didn't expect that such an exhaustive LightGBM (300000 iterations and learning rate 0.0005) will give me the best performance on such a small dataset. However, I ended up with this model.

# In[ ]:


X = df.drop([target]+drop_cols, axis=1)
y = df[target]


# In[ ]:


n_splits=5
n_repeats=10

num_iter = 300000
early_stopping_rounds = 10000
verbose = 10000

lgb_params = {
    'learning_rate': 0.0005,
    'num_leaves': 15,
    'feature_fraction': 0.1,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 8,
    'application': 'regression',
    'metric': ['mse'],
    'num_threads': -1,
    'seed': 42}


# In[ ]:


models = []

kf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

k=0
for train_ids, val_ids in tqdm(kf.split(X), desc='KFold', total=n_splits*n_repeats):
    k+=1
    print(f'\nFOLD {k}')
    print(dt.datetime.now().time())
    X_train, y_train = X.iloc[train_ids].copy(), y.iloc[train_ids]
    X_val, y_val = X.iloc[val_ids].copy(), y.iloc[val_ids]

    X_train[cat_cols] = X_train[cat_cols].astype('category')
    X_val[cat_cols] = X_val[cat_cols].astype('category')

    lgb_train = lgbm.Dataset(X_train, y_train)
    lgb_eval = lgbm.Dataset(X_val, y_val)

    lgb = lgbm.train(lgb_params,
                lgb_train,
                num_boost_round=num_iter,
                valid_sets=(lgb_train, lgb_eval),
                valid_names=('train', 'val'),
               early_stopping_rounds=early_stopping_rounds,
               verbose_eval = verbose)
    
    models.append(lgb)


# In[ ]:


def get_score_lgb(model, name='val'):
    return list(model.best_score[name].values())[0]

train_scores = np.array([get_score_lgb(model, 'train') for model in models])
val_scores = np.array([get_score_lgb(model, 'val') for model in models])

print('mean train score: {:.0f}+-{:.0f}'.format(train_scores.mean(), train_scores.std()))
print('min {:.0f}; max {:.0f}\n'.format(train_scores.min(), train_scores.max()))
print('mean validation score: {:.0f}+-{:.0f}'.format(val_scores.mean(), val_scores.std()))
print('min {:.0f}; max {:.0f}\n'.format(val_scores.min(), val_scores.max()))


# In[ ]:


# lgbm.plot_importance(models[0], importance_type='gain', figsize=(5,40), ignore_zero=False)


# ### Submission

# In[ ]:


df_test = pd.read_csv(DATA_FOLDER + 'test.csv', index_col='Id')

df_test.shape


# In[ ]:


df_test.drop(drop_cols, axis=1, inplace=True)


# In[ ]:


df_test[cat_cols] = df_test[cat_cols].astype('category')


# In[ ]:


predictions = np.array([model.predict(df_test, num_iter=model.best_iteration) for model in models]).mean(axis=0)


# In[ ]:


sns.distplot(y)


# In[ ]:


sns.distplot(predictions)


# In[ ]:


submission = pd.DataFrame({
    'Id': df_test.index,
    'Energy_consumption': predictions
})


# In[ ]:


submission.to_csv(SUBMISSIONS_FOLDER + 'submission.csv', index = False)

