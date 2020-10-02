#!/usr/bin/env python
# coding: utf-8

# # Predicting Molecular Properties
# ---
# 
# The goal of this competition is to find a **coupling constant** through intermolecular bonding. Coupling Constant is constant indicating the strength of the physical interaction (here between atoms), called a complete combination when the combined constant is 1.
# 
# We receive the coupling constant information from the *'train data'* by the coupling of two atoms. And from the *'Structures data'* we get the x, y, z axis information for each atom.  
# 
# #### Training Strategy
# For model training, I did not train the entire data, but I did the train by each **'type'** of molecular.
# And I used the **LightGBM** for model training. 

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# path
path_dir = '../input/champs-scalar-coupling/'
file_list = os.listdir(path_dir)
file_list


# ## 1. Load Train/Test Data
# **Columns**
# - molecule_name
# - atom_index_0 / atom_index_1
# - type
# - Coupling Constant : A constant indicating the strength of the physical interaction (here between atoms), called a complete combination when the combined constant is 1.

# In[ ]:


train_df = pd.read_csv(path_dir+'train.csv')
test_df = pd.read_csv(path_dir+'test.csv')   # target = 'scalar_coupling_constant'

print('Length of train set: {}'.format(len(train_df)))
print('Length of test set: {}'.format(len(test_df)))


# In[ ]:


print('Unique molecule of train set: {}'.format(len(train_df['molecule_name'].unique())))
train_df.head()


# In[ ]:


print('Unique molecule of test set: {}'.format(len(test_df['molecule_name'].unique())))
test_df.head()


# ## 2. EDA

# #### 2.1 Distribution of Target ('scalar_coupling_constant')
# - Min Value : -36.2186
# - Max Value : 204.88
# - Most are between -20 and +20
# - Small distribution exists between 80 and 100

# In[ ]:


# Distribution of target
print('Min Value of Target : {}'.format(train_df['scalar_coupling_constant'].min()))
print('Max Value of Target : {}'.format(train_df['scalar_coupling_constant'].max()))

plt.figure(figsize=(11, 5))
sns.distplot(train_df['scalar_coupling_constant'])
plt.title('Distribution of scalar_coupling_constant')
plt.show()


# #### 2.2 Distribution of 'scalar_coupling_constant' by type
# - '1JHC' type is distributed in a relatively high scalar coupling range(+66.6 ~ +204.8) 
# - '2JHH' type is distributed in a relatively low scalar coupling range(-35.1 ~ +11.8

# In[ ]:


# Distribution of 'scalar_coupling_constant' by type
plt.figure(figsize=(14, 13))
for i, t in enumerate(train_df['type'].unique()):
    plt.subplot(4,2, i+1)
    sns.distplot(train_df[train_df['type'] == t]['scalar_coupling_constant'])
    plt.title('Distribution of coupling constant by type '+ t)
    plt.tight_layout()


# #### 2.3 Count by 'type'
# - High in order 3JHC, 2JHC, 1JHC, 3JHH, 2JHH, 3JHN, 2JHN, 1JHN.

# In[ ]:


# Count by 'type'
type_index = train_df['type'].value_counts().index
type_cnt = train_df['type'].value_counts()

plt.figure(figsize=(11, 4))
sns.barplot(x=type_index, y=type_cnt)
plt.xlabel('type'); plt.ylabel('Count')
plt.title('Count by type')
plt.tight_layout()


# #### 2.4 Count by atom index 0, 1
# - Atom index 0 has the most number of distributions from 9 to 18.
# - Atom index 1 has the most number of distributions from 1 to 8.

# In[ ]:


# Count by atom index 0, 1
for i in [0, 1]:
    atom_index = train_df['atom_index_'+str(i)].value_counts().index
    atom_cnt = train_df['atom_index_'+str(i)].value_counts()
    
    plt.figure(figsize=(11, 4))
    sns.barplot(x=atom_index, y=atom_cnt)
    plt.xlabel('atom index '+str(i)); plt.ylabel('Count')
    plt.title('Count by atom index '+str(i))
    plt.tight_layout()


# ## 3. Load Structures Data
# **Columns**
# - molecule_name
# - atom_index
# - atom
# - x, y, z axis of atom

# In[ ]:


structures_df = pd.read_csv(path_dir+'structures.csv')

print('Length of test set: {}'.format(len(structures_df)))
structures_df.head()


# #### 3.1. 3Dimension plot by Molecule

# In[ ]:


for name in structures_df['molecule_name'].unique()[:4]:
    structures_molecule =structures_df[structures_df['molecule_name'] == name]

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(structures_molecule['x'], structures_molecule['y'], structures_molecule['z'], s=200, edgecolors='white')
    ax.set_title(str(name)+ ' 3D plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# ## 4. Preprocessing
# #### 4.1. Merge Train&Test - Structures Data

# In[ ]:


def mapping_atom_index(df, atom_idx):
    atom_idx = str(atom_idx)
    df = pd.merge(df, structures_df,
                  left_on  = ['molecule_name', 'atom_index_'+atom_idx],
                  right_on = ['molecule_name',  'atom_index'],
                 how = 'left')
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': 'atom_'+atom_idx,
                            'x': 'x_'+atom_idx,
                            'y': 'y_'+atom_idx,
                            'z': 'z_'+atom_idx})
    return df


# In[ ]:


train_merge = mapping_atom_index(train_df, 0)
train_merge = mapping_atom_index(train_merge, 1)

test_merge = mapping_atom_index(test_df, 0)
test_merge = mapping_atom_index(test_merge, 1)


# In[ ]:


train_tmp = train_merge[['id','molecule_name','type']]
test_tmp = test_merge[['id','molecule_name','type']]

train_merge.head()


# #### 4.2. Derived variables - 'Distance'
# - distance between *x axis* of atom index
# - distance between *y axis* of atom index
# - distance between *z axis* of atom index
# - distance between *atom*

# In[ ]:


def dist_between_atom(df):
    # distance between axis of atom
    df['x_dist'] = (df['x_0'] - df['x_1'])**2
    df['y_dist'] = (df['y_0'] - df['y_1'])**2
    df['z_dist'] = (df['z_0'] - df['z_1'])**2
    
    # distance between atom
    df['atom_dist'] = (df['x_dist']+df['y_dist']+df['z_dist'])**0.5
    
    return df
    
train_dist = dist_between_atom(train_merge)
test_dist = dist_between_atom(test_merge)


# In[ ]:


train_dist.head()


# #### 4.3. Label encoding 
# - type, atom_0, atom_1

# In[ ]:


# Label encoding
categorical_features = ['type', 'atom_0', 'atom_1']
for col in categorical_features:
    le = LabelEncoder()
    le.fit(list(train_dist[col].values) + list(test_dist[col].values))
    train_dist[col] = le.transform(list(train_dist[col].values))
    test_dist[col] = le.transform(list(test_dist[col].values))


# In[ ]:


train_le = train_dist.copy()
test_le = test_dist.copy()


# In[ ]:


train_le.head()


# #### 4.4. Standardization
# - z = (x - u) / s

# In[ ]:


# train
train_data = train_le.drop(['id','molecule_name','scalar_coupling_constant'], axis=1)
train_target = train_le['scalar_coupling_constant']
# test
test_data = test_le.drop(['id','molecule_name',], axis=1)


# In[ ]:


# z-score standardization
train_scale = (train_data - train_data.mean()) / train_data.mean()
train_scale = train_scale.fillna(0)
test_scale = (test_data - train_data.mean()) / train_data.mean()


# #### 4.5. Variable Correlations

# In[ ]:


train_corr = train_scale.copy()
train_corr['scalar_coupling_constant'] = train_target
corrmat = train_corr.corr()
top_corr_features = corrmat.index[abs(corrmat['scalar_coupling_constant']) >= 0.1]

plt.figure(figsize=(10,7))
sns.heatmap(train_corr[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.title('Variable Correlations')
plt.show()


# ## 5. Training by LightGBM 

# #### 5.1. Training by 'type' through LightGBM

# In[ ]:


train_scale = train_scale.drop('type', axis=1)
train_scale['type'] = train_tmp['type']
train_scale['scalar_coupling_constant'] = train_target

test_scale = test_scale.drop('type', axis=1)
test_scale[['id', 'type']] = test_tmp[['id', 'type']]


# In[ ]:


score_by_type = []    # List of Validation score by type 
feature_importance_df = []
test_pred_df = pd.DataFrame(columns=['id', 'scalar_coupling_constant'])   # Dataframe for submission

# Extract data by type
types = train_tmp['type'].unique()
for typ in types:
    print('---Type of '+str(typ)+'---')
    
    train = train_scale[train_scale['type'] == typ]
    target = train['scalar_coupling_constant']
    train = train.drop(['type','scalar_coupling_constant'], axis=1)
    
    # Split train set / valid set
    x_train, x_val, y_train, y_val = train_test_split(train, target, random_state=42)
    
    # LightGBM
    categorical_features = ['atom_0','atom_1']
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_features)
    lgb_val = lgb.Dataset(x_val, y_val, categorical_feature=categorical_features)

    # Parameters of LightGBM
    params = {'num_leaves': 128,
              'min_child_samples': 79,
              'objective': 'regression',
              'max_depth': 9,
              'learning_rate': 0.1,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.13,
              'reg_lambda': 0.36,
              'colsample_bytree': 1.0
             }
    # Training
    lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], 
                          num_boost_round=15000,    # Number of boosting iterations.
                          early_stopping_rounds=500,    # early stopping for valid set
                          verbose_eval=2500)    # eval metric on the valid set is printed at 2500 each boosting
    
    # Feature Importances
    feature_importance = lgb_model.feature_importance()
    df_fi = pd.DataFrame({'columns':x_train.columns, 'importances':feature_importance})
    df_fi = df_fi[df_fi['importances'] > 0].sort_values(by=['importances'], ascending=False)
    feature_importance_df.append(df_fi)
    
    # Predict Validation set
    score_by_type.append(list(lgb_model.best_score['valid_1'].values()))
    
    # Predict Test set
    test = test_scale[test_scale['type'] == typ]
    test_id = test['id']
    test = test.drop(['id','type'], axis=1)
    
    test_preds = lgb_model.predict(test)
    test_pred_df = pd.concat([test_pred_df, pd.DataFrame({'id':test_id, 'scalar_coupling_constant':test_preds})], axis=0)


# #### 5.2. Validation MAE by type

# In[ ]:


for typ, score in zip(types, score_by_type):
    print('Type {} valid MAE  : {}'.format(str(typ), score))

print('\nAverage of valid MAE  : {}'.format(np.mean(score_by_type)))


# #### 5.3. Feature Importances Plot by Type

# In[ ]:


for typ, df_fi in zip(types, feature_importance_df):
    fig = plt.figure(figsize=(12, 6))
    ax = sns.barplot(df_fi['columns'], df_fi['importances'])
    ax.set_xticklabels(df_fi['columns'], rotation=80, fontsize=13)
    plt.title('Type '+str(typ)+' feature importance')
    plt.tight_layout()
    plt.show()


# #### 5.4. Save prediction of test set to *.csv 

# In[ ]:


test_pred_df.head(10)


# In[ ]:


test_pred_df.to_csv('lgb_submission.csv', index=False)

