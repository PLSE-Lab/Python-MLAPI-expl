#!/usr/bin/env python
# coding: utf-8

# # Visualizing each variable vs a normally distributed.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import seaborn as sns
from tqdm import tqdm

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# # Number of times the unique variable appears vs `np.random.normal` ...

# In[ ]:


for var in ['var_{}'.format(x) for x in range(0, 200)]:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))
    train_df.groupby(var)['target'].agg(['count','mean']).sort_values('count')         .plot(kind='scatter', x='mean', y='count', ax=ax1, alpha=0.1, title='Train Data')
    train_df['random_{}'.format(var)] = np.random.normal(train_df[var].mean(), train_df[var].std(), 200000).round(4)
    train_df.groupby('random_{}'.format(var))['target'].agg(['count','mean']).sort_values('count')         .plot(kind='scatter', x='mean', y='count', ax=ax2, alpha=0.1, title='Simulated Data')
    # Both together
    train_df.groupby(var)['target'].agg(['count','mean']).sort_values('count')         .plot(kind='scatter', x='mean', y='count', ax=ax3, alpha=0.1)
    train_df.groupby('random_{}'.format(var))['target'].agg(['count','mean']).sort_values('count')         .plot(kind='scatter', x='mean', y='count', ax=ax3, alpha=0.1, color='orange', title='Both')
    ax1.set_xlabel('average target')
    ax2.set_xlabel('average target')
    ax3.set_xlabel('average target')
    ax1.set_ylabel('count of unique value')
    ax2.set_ylabel('count of unique value')
    ax3.set_ylabel('count of unique value')
    fig.suptitle(var)
    plt.show()


# ## Interactive Version of the Plot for var_12
# Var 12 looks strange so I wanted to interact with the points that don't appear in the simulated data

# In[ ]:


import bokeh
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from collections import OrderedDict

output_notebook()

var = 'var_126'
train_df['random_{}'.format(var)] = np.random.normal(train_df[var].mean(), train_df[var].std(), 200000).round(4)

TOOLS="crosshair,pan,wheel_zoom,box_zoom,reset,hover,previewsave"
p = figure(tools=TOOLS, title=var)
source = ColumnDataSource(data=dict(x=train_df.groupby(var)['target'].mean(),
                 y=train_df.groupby(var)['target'].count(),
                 size=(train_df.groupby(var)['target'].mean().index / 2),
                 label=train_df.groupby(var)['target'].mean().index * 10000))

source2 = ColumnDataSource(data=dict(x=train_df.groupby('random_{}'.format(var))['target'].mean(),
                 y=train_df.groupby('random_{}'.format(var))['target'].count(),
                 size=(train_df.groupby('random_{}'.format(var))['target'].mean().index / 2),
                 label=train_df.groupby('random_{}'.format(var))['target'].mean().index))

p.circle(x='x', y='y', size='size', source=source)
p.circle(x='x', y='y', size='size', source=source2, color='orange')

hover =p.select(dict(type=HoverTool))
hover.tooltips = OrderedDict([
    ("index", "$index"),
    ("(xx,yy)", "(@x, @y)"),
    ("label", "@label"),
])
show(p)


# # Create Feature that is the difference in unique counts vs normal distribution unique counts

# In[ ]:


# Reload Train and Test
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


def transform(df, var='var_12'):
    df['random_{}'.format(var)] = np.random.normal(df[var].mean(), df[var].std(), 200000).round(4)
    var_counts = pd.DataFrame(df.groupby(var)['ID_code'].count()).reset_index()
    var_counts_random = pd.DataFrame(df.groupby('random_{}'.format(var))['ID_code'].count()).reset_index()
    merged_counts = pd.merge(var_counts, var_counts_random, left_on=var, right_on='random_{}'.format(var))
    merged_counts['diff'] = merged_counts['ID_code_x'] - merged_counts['ID_code_y']
    df['{}_diff_normal_dist'.format(var)] = df.merge(merged_counts[[var,'diff']], how='left')['diff']
    df = df.drop('random_{}'.format(var), axis=1)
    return df


# In[ ]:


# Loop and add features
for var in tqdm(['var_{}'.format(x) for x in range(0, 200)]):
    train_df = transform(train_df, var=var)
    test_df = transform(test_df, var=var)


# In[ ]:


from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

random_state = 42
params = {
    "objective" : "binary", "metric" : "auc", "boosting": 'gbdt', "max_depth" : -1, "num_leaves" : 13,
    "learning_rate" : 0.01, "bagging_freq": 5, "bagging_fraction" : 0.4, "feature_fraction" : 0.05,
    "min_data_in_leaf": 80, "min_sum_heassian_in_leaf": 10, "tree_learner": "serial", "boost_from_average": "false",
    "bagging_seed" : random_state, "verbosity" : 1, "seed": random_state
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
oof = train_df[['ID_code', 'target']].copy()
oof['predict'] = 0
predictions = test_df[['ID_code']].copy()
val_aucs = []

features = [col for col in train_df.columns if col not in ['target', 'ID_code']]
X_test = test_df[features].values

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['target'])):
    X_train, y_train = train_df.iloc[trn_idx][features], train_df.iloc[trn_idx]['target']
    X_valid, y_valid = train_df.iloc[val_idx][features], train_df.iloc[val_idx]['target']
    
    N = 3
    p_valid,yp = 0,0
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    evals_result = {}
    lgb_clf = lgb.train(params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=1000,
                        verbose_eval = 500,
                        evals_result=evals_result)
    p_valid += lgb_clf.predict(X_valid)
    yp += lgb_clf.predict(X_test)

    oof['predict'][val_idx] = p_valid
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    predictions['fold{}'.format(fold+1)] = yp
    
mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
all_auc = roc_auc_score(oof['target'], oof['predict'])
print("Mean auc: %.9f, std: %.9f. All auc: %.9f." % (mean_auc, std_auc, all_auc))

predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)
submission = pd.DataFrame({"ID_code":test_df["ID_code"].values})
submission["target"] = predictions['target']
submission.to_csv("lgb_submission2.csv", index=False)

