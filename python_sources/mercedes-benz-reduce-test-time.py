#!/usr/bin/env python
# coding: utf-8

# **Objective:**
# 
# This dataset contains an anonymized set of variables that describe different Mercedes cars. The ground truth is labeled 'y' and represents the time (in seconds) that the car took to pass testing.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
from sklearn import ensemble
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_columns = 999


# In[ ]:


import os
os.listdir("../input/mercedes-benz-greener-manufacturing")


# In[ ]:


train_df = pd.read_csv("../input/mercedes-benz-greener-manufacturing/train.csv.zip")
test_df = pd.read_csv("../input/mercedes-benz-greener-manufacturing/test.csv.zip")


# In[ ]:


print(f'Train Shape: {train_df.shape}')
print(f'Test Shape: {test_df.shape}')


# In[ ]:


train_df.head()


# "y" is the variable we need to predict

# In[ ]:


plt.figure(figsize=(8, 6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()


# In[ ]:


upper_limit = 180
train_df.loc[train_df['y'] > upper_limit, 'y'] = upper_limit

plt.figure(figsize=(8, 6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train_df.y.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.show()


# In[ ]:


train_df.dtypes.reset_index()


# In[ ]:


dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Column", "Column Type"]
dtype_df['Column Type'].value_counts()


# In[ ]:


dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type")["Count"].count().reset_index()


# So majority of the columns are integers with 8 categorical columns and 1 float column (target variable)

# In[ ]:


dtype_df.loc[:10, :]


# X0 to X8 are the categorical columns.

# In[ ]:


train_df.isnull().sum(axis=0).reset_index()


# In[ ]:


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count'] > 0]
missing_df = missing_df.sort_values(by="missing_count")
missing_df


# In[ ]:


unique_values_dict = {}

for col in train_df.columns:
    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        unique_value = str(np.sort(train_df[col].unique()).tolist())
        if unique_value not in unique_values_dict:
            unique_values_dict[unique_value] = [col]
        else:
            unique_values_dict[unique_value].append(col)

for unique_val in unique_values_dict:
    print("Columns containing the unique values : ",unique_val)
    print(unique_values_dict[unique_val])
    print("--------------------------------------------------")


# After Integer Columns Analysis we can see that all the integer columns are binary with some columns have only one unique value 0

# Now we have to visualize categorical cols against 'y' col

# In[ ]:


train_df['X1'].value_counts()


# In[ ]:


var_name = "X1"

plt.figure(figsize=(12,6))
sns.stripplot(x=var_name, y='y', data=train_df, order= train_df[var_name].value_counts().index)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X2"

plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train_df, order= train_df[var_name].value_counts().index)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# In[ ]:


var_name = "X3"

plt.figure(figsize=(12,6))
sns.violinplot(x=var_name, y='y', data=train_df, order= train_df[var_name].value_counts().index)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# Now exploring Binary Variables

# In[ ]:


one_count_list = []
zero_count_list = []

cols_list = unique_values_dict['[0, 1]']

# Now to store total no. of 0's & 1's in each col
for col in cols_list:
    zero_count_list.append((train_df[col] == 0).sum())
    one_count_list.append((train_df[col] == 1).sum())

N = len(cols_list)
ind = np.arange(N)
width = 0.35

plt.figure(figsize=(6,100))
p1 = plt.barh(ind, zero_count_list, width, color='red')
p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color="green")
plt.yticks(ind, cols_list)
plt.legend((p1, p2), ('Zero count', 'One Count'))
plt.title("Count Distribution", fontsize=15)
plt.show()


# Checking mean 'y' value for 0 & 1, for each binary cols

# In[ ]:


zero_mean_list = []
one_mean_list = []

cols_list = unique_values_dict['[0, 1]']

for col in cols_list:
    zero_mean_list.append(train_df.loc[train_df[col] == 0, 'y'].mean())
    one_mean_list.append(train_df.loc[train_df[col] == 1, 'y'].mean())

temp_df = pd.DataFrame({"column_name": cols_list + cols_list, "value": [0]*len(cols_list) + [1]*len(cols_list), "y_mean": zero_mean_list + one_mean_list})
temp_df = temp_df.pivot(index = 'column_name', columns = 'value', values = 'y_mean')
temp_df.head()


# Now to check if the 'y' mean values of 1s and 0's are almost same or diff wrt each col

# In[ ]:


plt.figure(figsize=(8, 80))
sns.heatmap(temp_df, cmap="YlGnBu")
plt.title("Mean of y val across binary variables", fontsize=15)
plt.show()


# Binary variables which shows a good color difference in the above graphs between 0 and 1 are likely to be more predictive given the the count distribution is also good between both the classes

# Now we will look into the 'ID' col which will give an idea of how the splits are done across train and test (random or id based) and also to help see if ID has some potential prediction capability (probably not so useful for business)

# In[ ]:


var_name = "ID"

plt.figure(figsize=(12,6))
sns.regplot(x = var_name, y = 'y', data = train_df, scatter_kws = {'alpha':0.5, 's':30})
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# The regplot performs a simple linear regression and there seems to be a slight decreasing trend with respect to ID variable.

# In[ ]:


train_df['eval_set'] = "train"
test_df['eval_set'] = "test"
train_df['eval_set'].head()


# In[ ]:


full_df = pd.concat([train_df[["ID", "eval_set"]], test_df[["ID", "eval_set"]]])
full_df.head()


#  Let's see how the IDs are distributed across train and test.

# In[ ]:


plt.figure(figsize=(12,6))
sns.stripplot(x="eval_set", y='ID', data=full_df)
plt.xlabel("eval_set", fontsize=12)
plt.ylabel('ID', fontsize=12)
plt.title("Distribution of ID variable with evaluation set", fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x="eval_set", y='ID', data=full_df)
plt.xlabel("eval_set", fontsize=12)
plt.ylabel('ID', fontsize=12)
plt.title("Distribution of ID variable with evaluation set", fontsize=15)
plt.show()


# Now let us run and xgboost model to get the important variables.

# In[ ]:


for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df['y'].values
train_X = train_df.drop(["ID", "y", "eval_set"], axis=1)

def xgb_r2_score(preds, dtrain):
    labels = dtrain.getLabel()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# Categorical occupy the top spots followed by binary variables.
# 
# Let us also build a Random Forest model and check the important variables.

# In[ ]:


model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)

feat_names = train_X.columns.values

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# Binary is now on the top spot. 
# So based on diff model there is a significant difference in important features
