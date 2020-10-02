#!/usr/bin/env python
# coding: utf-8

# ## Interpretable XGBoost / LGBM / Catboost
# 
# With SHAP, Let's interpret Tree-based Model.
# 
# - XGBoost 
# - LightGBM (TBD)
# - Catboost (TBD)
# 
# ### Reference
# 
# - [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/index.html)

# ## Read Files & Library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('../input/cat-in-the-dat/train.csv')\ntest = pd.read_csv('../input/cat-in-the-dat/test.csv')")


# First of all, I will separate the target value.

# In[ ]:


target = train['target']
train_id = train['id']
test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


# To make encoding easier, let's connect to concat for a while.

# In[ ]:


df = pd.concat([train, test], axis=0, sort=False )


# ## Categorical Encoding
# 
# 
# ### Binary Feature (map/apply)
# 

# In[ ]:


bin_dict = {'T':1, 'F':0, 'Y':1, 'N':0}
df['bin_3'] = df['bin_3'].map(bin_dict)
df['bin_4'] = df['bin_4'].map(bin_dict)


# ### Nominal Feature (One-Hot Encoding)

# In[ ]:


print(f'Shape before dummy transformation: {df.shape}')
df = pd.get_dummies(df, columns=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'],
                    prefix=['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4'], 
                    drop_first=True)
print(f'Shape after dummy transformation: {df.shape}')


# ### Ordinal Feature (~Label Encoding)

# In[ ]:


from pandas.api.types import CategoricalDtype 

ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 
                                     'Master', 'Grandmaster'], ordered=True)
ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',
                                     'Boiling Hot', 'Lava Hot'], ordered=True)
ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

df.ord_1 = df.ord_1.astype(ord_1)
df.ord_2 = df.ord_2.astype(ord_2)
df.ord_3 = df.ord_3.astype(ord_3)
df.ord_4 = df.ord_4.astype(ord_4)

df.ord_1 = df.ord_1.cat.codes
df.ord_2 = df.ord_2.cat.codes
df.ord_3 = df.ord_3.cat.codes
df.ord_4 = df.ord_4.cat.codes


# ### Data (Cycle Encoding)

# In[ ]:


def date_cyc_enc(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
    return df

df = date_cyc_enc(df, 'day', 7)
df = date_cyc_enc(df, 'month', 12)


# ### ETC (Label Encoding)

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.preprocessing import LabelEncoder\n\n# Label Encoding\nfor f in ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5']:\n    lbl = LabelEncoder()\n    lbl.fit(df[f])\n    df[f'le_{f}'] = lbl.transform(df[f])")


# ### Drop 'object' features (Remaining features)

# In[ ]:



df.drop(['nom_5','nom_6','nom_7','nom_8','nom_9', 'ord_5'] , axis=1, inplace=True)


# ## Reduce Memory Usage & Train_test split

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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


df = reduce_mem_usage(df)


# In[ ]:


train = df[:train.shape[0]]
test = df[train.shape[0]:]

train.shape


# ## Models
# 
# For more parameters, I recommend the following site.
# 
# - [Laurae++](https://sites.google.com/view/lauraepp/parameters)

# ### Simple XGBoost
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "from xgboost import XGBClassifier\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_val, y_train, y_val = train_test_split(\n    train, target, test_size=0.2, random_state=2019\n)\n\nxgb_clf = XGBClassifier(learning_rate=0.05,\n    n_estimators=2000,\n    seed=42,\n    eval_metric='auc',\n)\n\nxgb_clf.fit(\n    X_train, \n    y_train, \n    eval_set=[(X_train, y_train), (X_val, y_val)],\n    early_stopping_rounds=20,\n    verbose=20\n)\n")


# In[ ]:


results = xgb_clf.evals_result()
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# plot log loss
plt.figure(figsize=(15, 7))
plt.plot(x_axis, results['validation_0']['auc'], label='Train')
plt.plot(x_axis, results['validation_1']['auc'], label='Val')
plt.legend()
plt.ylabel('AUC')
plt.xlabel('# of iterations')
plt.title('XGBoost AUC')
plt.show()


# ### Feature Importance
# 
# You can extract `Feature Importance` from boosting models
# 
# **importance_type**
# 
# - `weight` - the number of times a feature is used to split the data across all trees.
# - `gain` - the average gain across all splits the feature is used in.
# - `cover` - the average coverage across all splits the feature is used in.
# - `total_gain` - the total gain across all splits the feature is used in.
# - `total_cover` - the total coverage across all splits the feature is used in.

# ### SHAP Value
# 
# The main idea of SHAP value is :
# 
# *How does this prediction $i$ change when variable $j$ is removed from this model*
# 
# ### Tree Explainer
# 
# **TreeSHAP**, a variant of SHAP for tree-based machine learning models such as decision trees, random forests and gradient boosted trees. TreeSHAP is fast, computes exact Shapley values, and correctly estimates the Shapley values when features are dependent.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import shap\n\nshap.initjs()\n\nexplainer = shap.TreeExplainer(xgb_clf)\nshap_values = explainer.shap_values(train)')


# There are many plot in `SHAP` package 
# 
# - summary plot 
# - dependence plot
# - force plot
# - decision plot 
# - etc
# 
# In this notebook, I will introduce `How to read a SHAP's plot`.

# ### SHAP Summary Plot
# 
# The **summary plot** combines feature importance with feature effects. 
# 
# Each point on the summary plot is a Shapley value for a feature and an instance. 

# **Type 1 : bar plot**

# In[ ]:


shap.summary_plot(shap_values, train, plot_type="bar")


# **Type 2 : default**

# In[ ]:


shap.summary_plot(shap_values, train)


# - `Feature importance`: Variables are ranked in descending order.
# - `Impact`: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
# - `Original value`: Color shows whether that variable is high (in red) or low (in blue) for that observation.
# - `Correlation`

# ### SHAP Dependence Plot
# 
# SHAP feature dependence might be the simplest global interpretation plot
# 
# SHAP **dependence plots** are an alternative to partial dependence plots and accumulated local effects. 

# In[ ]:


shap.dependence_plot("le_ord_5", shap_values, train)


# In[ ]:


shap.dependence_plot("ord_4", shap_values, train)


# In[ ]:


shap.dependence_plot("day", shap_values, train)


# ### SHAP Force Plot 
# 
# You can visualize feature attributions such as Shapley values as "**forces**". Each feature value is a force that either increases or decreases the prediction. 
# The prediction starts from the baseline. 
# The baseline for Shapley values is the average of all predictions. 
# 
# In the plot, each Shapley value is an arrow that pushes to increase (positive value) or decrease (negative value) the prediction. These forces balance each other out at the actual prediction of the data instance.

# In[ ]:


shap.force_plot(explainer.expected_value, shap_values[0,:], train.iloc[0,:])


# In[ ]:




