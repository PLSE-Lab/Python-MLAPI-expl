#!/usr/bin/env python
# coding: utf-8

# As promised in this [thread](https://www.kaggle.com/c/champs-scalar-coupling/discussion/99125#latest-571188), here is a notebook to get the [LOFO](https://github.com/aerdem4/lofo-importance) importance using [Giba's features](https://www.kaggle.com/titericz/giba-r-data-table-simple-features-1-17-lb). [](http://)

# In[ ]:


get_ipython().system('pip install lofo-importance')


# In[ ]:


import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from lofo import LOFOImportance, plot_importance
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


TARGET_COL = "scalar_coupling_constant"
GIBA_TO_DROP_COLS = ["id", "molecule_name",
                     "atom_index_0", "atom_index_1",
                     "scalar_coupling_constant",
                     "type",
                     "ID",
                     "structure_atom_0", "structure_atom_1",
                     "structure_x_0", "structure_y_0", "structure_z_0",
                     "structure_x_1", "structure_y_1", "structure_z_1",
                     "typei", "pos",
                     "R0", "R1", "E0", "E1", "Unnamed: 0",
                     "molecule_name.1", "atom_index_1.1", 
                     "dataset"]

PARAMS = {
     "boosting_type":"gbdt",
     "objective": "regression_l2",
     "learning_rate": 0.15,
     "num_leaves": 255,
     "sub_feature": 0.50,
     "sub_row": 0.75,
     "bagging_freq": 1,
     "metric": 'mae'
}

SEED = 314
N_SPLITS = 3


# In[ ]:


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
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# # Import Giba's features

# In[ ]:


df = pd.read_csv("../input/giba_features.csv.gz")
train_df = df.loc[lambda df: ~(df[TARGET_COL].isnull()), :]
train_df = reduce_mem_usage(train_df)


# In[ ]:


FEATURES_COLS = train_df.drop(GIBA_TO_DROP_COLS, axis=1, errors="ignore").columns.tolist()


# In[ ]:


# Fix the CV seed for reproducibilty
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# # Build the features' importance DataFrame

# In[ ]:


dfs = []

for _type in tqdm_notebook(train_df['type'].unique()):
    print(f'LOFO importance for {_type}')
    type_train_df = train_df.loc[lambda df: df['type'] == _type].reset_index(drop=True)
    model = LGBMRegressor(**PARAMS)
    lofo_imp = LOFOImportance(type_train_df, FEATURES_COLS, TARGET_COL, cv=CV, 
                              scoring="neg_mean_absolute_error", model=model)
    _df = lofo_imp.get_importance()
    _df['type'] = _type
    dfs.append(_df)

importance_df = pd.concat(dfs)
importance_df.to_csv('lofo_giba_features.csv', index=False)


# # Plot the results

# In[ ]:


# I have adapted the one from the LOFO lib to add a title and get the figure.
def plot_importance(importance_df, figsize=(8, 8), ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    importance_df = importance_df.copy()
    importance_df["color"] = (importance_df["importance_mean"] > 0).map({True: 'g', False: 'r'})
    importance_df.sort_values("importance_mean", inplace=True)

    importance_df.plot(x="feature", y="importance_mean", xerr="importance_std",
                       kind='barh', color=importance_df["color"], figsize=figsize, ax=ax)
    return ax


# In[ ]:


for _type in importance_df['type'].unique():
    ax = plot_importance(importance_df.loc[lambda df: df['type'] == _type].drop('type', axis=1), 
                    figsize=(12, 20))
    ax.set_title(f'LOFO importance plot for {_type}')
    ax.get_figure().savefig(f'lofo_importance_{_type}.png')

