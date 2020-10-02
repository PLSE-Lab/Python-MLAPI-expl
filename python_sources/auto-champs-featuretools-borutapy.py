#!/usr/bin/env python
# coding: utf-8

# # Auto Champs
# 
# In this notebook I aim to demonstrate automation on feature generation and feature selection. For that purpose I used featuretools to aggregate the data by `molecule_name` and `atom_index_0`/`atom_index_1` and automatically generated possible statistical features with a depth of 2.
# 
# Then having the automatically generated features, I used borutaPy to select the best features for the model. And finally I train a lightgbm model using those features.
# 
# This notebook can be seen as an automated version of this work: https://www.kaggle.com/artgor/brute-force-feature-engineering
# 
# Additional example for featuretools: https://www.kaggle.com/willkoehrsen/automated-feature-engineering-tutorial
# 
# Additional example for borutaPy: https://www.kaggle.com/rsmits/feature-selection-with-boruta

# In[ ]:


get_ipython().run_line_magic('ls', '../input')


# In[ ]:


dataset_dir = '/kaggle/input/'
download_dir = './'


# In[ ]:


is_sample = False # if True, run in test mode
boosting_rounds = 18000 # lightgbm training epochs
boruta_max_iter = 60 # max iteration number for boruta
num_boruta_rows = 8000 # use a small subsample to quickly fit with boruta feature selector


# In[ ]:


import numpy as np
import pandas as pd
import featuretools as ft
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import gc
import lightgbm
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from dask.distributed import LocalCluster

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# calculate competition metric
def competition_metric(df, preds, verbose=0):
    # log of mean absolute error, calculated for each scalar coupling type.
    df_copy = df.copy()
    df_copy["prediction"] = preds
    maes = []
    for t in df_copy.type.unique():
        y_true = df_copy[df.type == t].scalar_coupling_constant.values
        y_pred = df_copy[df.type == t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        if verbose == 1:
            print(f"{t} log(MAE): {mae}")
        maes.append(mae)
    del df_copy
    gc.collect()
    return np.mean(maes)


# ## Prepare the dataset
# 
# I will read the datasets and craete basic features like the distance measurements and then concatenate train and test.

# In[ ]:


train = pd.read_csv(f"{dataset_dir}train.csv")
train.head()


# In[ ]:


test = pd.read_csv(f"{dataset_dir}test.csv")
test.head()


# In[ ]:


concat = pd.concat([train, test])


# In[ ]:


structures = pd.read_csv(f"{dataset_dir}structures.csv")
structures.head()


# In[ ]:


# map structures dataframe into concat
def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

concat = map_atom_info(concat, 0)
concat = map_atom_info(concat, 1)

concat.head()


# In[ ]:


# create basic features like distance
def particle_distance(df):
    dist = ( (df["x_1"] - df["x_0"])**2 + (df["y_1"] - df["y_0"])**2 + (df["z_1"] - df["z_0"])**2 )**0.5
    return dist

concat["distance"] = particle_distance(concat)

# create distance values for each axis
def particle_distance_x(df):
    dist = ( (df["x_1"] - df["x_0"])**2 )**0.5
    return dist

def particle_distance_y(df):
    dist = ( (df["y_1"] - df["y_0"])**2 )**0.5
    return dist

def particle_distance_z(df):
    dist = ( (df["z_1"] - df["z_0"])**2 )**0.5
    return dist

concat["distance_x"] = particle_distance_x(concat)
concat["distance_y"] = particle_distance_y(concat)
concat["distance_z"] = particle_distance_z(concat)

concat.head()


# In[ ]:


if is_sample:
    print("\n!!! WARNING SAMPLE MODE ACTIVE !!!\n")
    concat = concat[:1000]


# ## Aggregation
# 
# In order to create stacked features with a depth more than 1, we need to define [relational entities](https://docs.featuretools.com/loading_data/using_entitysets.html). Since I want to aggregate by `molecule_name` + `atom_index_0` and `molecule_name` + `atom_index_1`, I will concat those columns and create the ids below.

# In[ ]:


le = preprocessing.LabelEncoder()
mol_atom_0 = concat.molecule_name.astype(str) + '_' + concat.atom_index_0.astype(str)
concat['molecule_atom_0_id'] = le.fit_transform(mol_atom_0)


# In[ ]:


le = preprocessing.LabelEncoder()
mol_atom_1 = concat.molecule_name.astype(str) + '_' + concat.atom_index_1.astype(str)
concat['molecule_atom_1_id'] = le.fit_transform(mol_atom_1)


# In[ ]:


concat.head()


# ## Create the Entity Set
# 
# In order to use featuretools, I need to create an entity set and define the relations. I could do this by splitting the concat dataframe into 3 dataframes with two of them are for the concatenated molecule name and atom index number features. However there is an easier way to do it by [entitiy normalization](https://docs.featuretools.com/loading_data/using_entitysets.html#creating-entity-from-existing-table).

# In[ ]:


# Create the entity set for featuretools
es = ft.EntitySet(id='concat')


# In[ ]:


# Add entites to entity set
es = es.entity_from_dataframe(
    entity_id='concat', dataframe=concat.drop(['scalar_coupling_constant'], axis=1), index='id')


# In[ ]:


es = es.normalize_entity(
    base_entity_id='concat',
    new_entity_id='molecule_atom_0',
    index='molecule_atom_0_id',
    additional_variables=['atom_0', 'x_0', 'y_0', 'z_0'])


# In[ ]:


es = es.normalize_entity(
    base_entity_id='concat',
    new_entity_id='molecule_atom_1',
    index='molecule_atom_1_id',
    additional_variables=['atom_1', 'x_1', 'y_1', 'z_1'])


# In[ ]:


es


# ## Generate Features

# In[ ]:


# It is faster when using n_jobs > 1, however kaggle kernels die if I define multiple jobs, so I comment out those lines below.
#cluster = LocalCluster()


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Perform an automated Deep Feature Synthesis with a depth of 2\n#features0, feature_names0 = ft.dfs(entityset=es, target_entity='molecule_atom_0', max_depth=2, dask_kwargs={'cluster': cluster}, n_jobs=2)\nfeatures0, feature_names0 = ft.dfs(entityset=es, target_entity='molecule_atom_0', max_depth=2)\nprint(features0.shape)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Perform an automated Deep Feature Synthesis with a depth of 2\n#features1, feature_names1 = ft.dfs(entityset=es, target_entity='molecule_atom_1', max_depth=2, dask_kwargs={'cluster': cluster}, n_jobs=2)\nfeatures1, feature_names1 = ft.dfs(entityset=es, target_entity='molecule_atom_1', max_depth=2)\nprint(features1.shape)")


# In[ ]:


feature_names0


# In[ ]:


feature_names1


# In[ ]:


# add column suffixes
def col_suffix_handler(df, suffix):
    col_dict = {col:"{}{}".format(col, suffix) for col in df.columns.values}
    df.rename(columns=col_dict, inplace=True)
    return df

# I will need unqiue feature names after feature selection with boruta
features0 = col_suffix_handler(features0, '__molecule_atom_0')
features1 = col_suffix_handler(features1, '__molecule_atom_1')


# ## Select Features

# In[ ]:


# reduce memory
def reduce_memory(df):
    num_converted_cols = 0
    for col in df.columns.values:
        if df[col].dtype == "float64":
            num_converted_cols += 1
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            num_converted_cols += 1
            df[col] = df[col].astype("int32")
    print("{} cols converted.".format(num_converted_cols))
    return df

concat = reduce_memory(concat)
features0 = reduce_memory(features0)
features1 = reduce_memory(features1)


# In[ ]:


# handle NaN values
def nan_handler(df):
    for col in df.columns.values:
        if np.any(df[col].isnull()):
            print(col)
            if df[col].dtype == 'O':
                df[col] = df[col].fillna('NO_VALUE')
            else:
                df[col] = df[col].fillna(-999)
    return df
                
features0 = nan_handler(features0)
features1 = nan_handler(features1)


# In[ ]:


# handle inf/-inf values
def inf_handler(df):
    for col in df.columns.values:
        if np.any(df[col]==np.inf) or any(df[col]==-np.inf):
            print(col)
            if df[col].dtype == 'O':
                df[df[col]==np.inf] = 'NO_VALUE'
                df[df[col]==-np.inf] = 'NO_VALUE'
            else:
                df[df[col]==np.inf] = 999
                df[df[col]==-np.inf] = 999
    return df
                
features0 = inf_handler(features0)
features1 = inf_handler(features1)


# In[ ]:


# list unnecessary columns
cols_to_remove = [
    'id',
    'scalar_coupling_constant'
]


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# feature selection using boruta\n\n# merge features with concat df\nconcat_features_ = concat.iloc[:num_boruta_rows].merge(\n    features0, left_on=[\'molecule_atom_0_id\'], right_index=True, how=\'left\')\n\nconcat_features_ = concat_features_.iloc[:num_boruta_rows].merge(\n    features1, left_on=[\'molecule_atom_1_id\'], right_index=True, how=\'left\')\n\n# label encode object type (categorical) columns\nfor col in concat_features_.columns.values:\n    if concat_features_[col].dtype == \'O\':\n        le = preprocessing.LabelEncoder()\n        concat_features_[col] = le.fit_transform(concat_features_[col])\n\nforest = RandomForestRegressor(n_jobs=-1)\n\nfeat_selector = BorutaPy(\n    forest, n_estimators=\'auto\', verbose=2, random_state=42, max_iter=boruta_max_iter, perc=90)\n\nX = concat_features_.drop(cols_to_remove, axis=1).iloc[:num_boruta_rows, :].values\ny = concat_features_[["scalar_coupling_constant"]].values[:num_boruta_rows, 0]\n\nfeat_selector.fit(X, y)\n\nfeatures = concat_features_.drop(cols_to_remove, axis=1).columns.values.tolist()\n\ndel X, y, concat_features_\ngc.collect()\n\n# list selected boruta features\nselected_features = []\nindexes = np.where(feat_selector.support_ == True)\nfor x in np.nditer(indexes):\n    selected_features.append(features[x])\n\nprint(len(selected_features))\nprint(selected_features)')


# ## Train Lightgbm Model

# In[ ]:


# merge features0 and features1 with concat df (using only selected features)

selected_features0_ = list(set(selected_features) - set(concat.columns.values.tolist()))
selected_features0_ = [f for f in selected_features0_ if '__molecule_atom_0' in f]

selected_features1_ = list(set(selected_features) - set(concat.columns.values.tolist()))
selected_features1_ = [f for f in selected_features1_ if '__molecule_atom_1' in f]

concat_features = concat.merge(
    features0[selected_features0_], 
    left_on=['molecule_atom_0_id'], 
    right_index=True, 
    how='left'
)

concat_features = concat_features.merge(
    features1[selected_features1_], 
    left_on=['molecule_atom_1_id'], 
    right_index=True,
    how='left'
)

print(concat_features.shape)
concat_features.head()


# In[ ]:


concat_features.dtypes.unique()


# In[ ]:


# label encode object type columns
for col in concat_features.columns.values:
    if concat_features[col].dtype == 'O':
        le = preprocessing.LabelEncoder()
        concat_features[col] = le.fit_transform(concat_features[col])
        
concat_features.head()


# In[ ]:


len_train = len(train)
del train, test, concat, features0, features1
gc.collect()


# In[ ]:


train = concat_features[:len_train]
test = concat_features[len_train:]


# In[ ]:


del concat_features
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', '# use selected boruta features and train a lightgbm model\n\ntrain_index, valid_index = train_test_split(np.arange(len(train)),random_state=42, test_size=0.1)\n\nX_train = train[selected_features].values[train_index]\ny_train = train[[\'scalar_coupling_constant\']].values[:, 0][train_index]\n\nvalid_df = train.iloc[valid_index]\n\ndel train\ngc.collect()\n\nX_valid = valid_df[selected_features].values\ny_valid = valid_df[[\'scalar_coupling_constant\']].values[:, 0]\n\nparams = {\'boosting\': \'gbdt\', \'colsample_bytree\': 1, \n              \'learning_rate\': 0.1, \'max_depth\': 40, \'metric\': \'mae\',\n              \'min_child_samples\': 50, \'num_leaves\': 500, \n              \'objective\': \'regression\', \'reg_alpha\': 0.8, \n              \'reg_lambda\': 0.8, \'subsample\': 0.5 }\n\nlgtrain = lightgbm.Dataset(X_train, label=y_train)\nlgval = lightgbm.Dataset(X_valid, label=y_valid)\n\nmodel_lgb = lightgbm.train(\n    params, lgtrain, boosting_rounds, valid_sets=[lgtrain, lgval], \n    early_stopping_rounds=1000, verbose_eval=500)\n\n# evaluate using validation set\nevals = model_lgb.predict(X_valid)\nlmae = competition_metric(valid_df, evals, verbose=1)\nprint("Log of MAE = {}".format(lmae))\n\ndel valid_df, X_train, y_train, X_valid, y_valid\ngc.collect()')


# ## Predict

# In[ ]:


# predict for test set
X_test = test[selected_features].values
preds = model_lgb.predict(X_test)


# In[ ]:


# save predictions
test["scalar_coupling_constant"] = preds
test[["id", "scalar_coupling_constant"]].to_csv(f"{download_dir}preds.csv", index=False)

