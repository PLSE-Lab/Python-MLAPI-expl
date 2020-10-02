#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# I have taken a lot of code from https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes,
# Since this is my first submission for kaggle, please be patient with and allow me to improve by suggesting 
# improvements to my kernel or general understanding of ML Concepts.

import os
from subprocess import check_output

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#FOR BUILDING THE MODEL
import pylab
import scipy.stats as stats
from sklearn import preprocessing

import xgboost as xgb
from sklearn import ensemble
get_ipython().run_line_magic('matplotlib', 'inline')


# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
print(train_data.shape)
print(train_data.info())


# In[ ]:


train_data.head()


# Splitting the Data into Target and predictor variables

# In[ ]:


train_data_target = train_data.iloc[:,0:2]


# ## We will check if the target has any outliers
# - For checking outliers, I have decided to go for a Boxplot, which will give me a general understanding of what the data-set is about

# In[ ]:


plt.figure(figsize=(15,5))
sns.boxplot(train_data_target.loc[:,'y'])
plt.show()


# **As you can see, there are quite some outliers.**
# - To Remove these outliers, I have decided to use the z-score method with threshold of three

# In[ ]:


train_data_target['z'] = np.abs(stats.zscore(train_data_target.loc[:,'y']))
threshold=3

ID_OF_OUTLIERS = train_data_target[train_data_target['z']>3].ID


# Once I have all the ID's of the outliers, I will remove them from the training data-set

# In[ ]:


train_data_target = train_data_target[~train_data_target['ID'].isin(list(ID_OF_OUTLIERS))]


# In[ ]:


train_data_predictors = train_data[~train_data['ID'].isin(list(ID_OF_OUTLIERS))].iloc[:,2:]


# ## Now again lets see the distribution of the target var

# In[ ]:


plt.figure(figsize=(15,5))
sns.boxplot(train_data_target.loc[:,'y'])
plt.show()


# The Distribution is a lot better

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_data_target.shape[0]), np.sort(train_data_target.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()


# Overall the scatter plot looks good
# 
# ### The Next that we should do is to check the unique values of all the columns. If atall there are columns with only one value then we should exclude them.
# 
# - First check for all the unique values:
# - Then drop the ones in which there is oly one value
# - Then check for feature importance using XGBOOST AND RANDOM FOReST Regressor
# - USE RANDOM FOREST REGRESSOR TO PREDICT THE OUTPUT
# 

# In[ ]:


# This portion has been taken from https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-mercedes

unique_values_dict = {}
for col in train_data_predictors.columns:
    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        unique_value = str(np.sort(train_data_predictors[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist[:]
for unique_val, columns in unique_values_dict.items():
    print("Columns containing the unique values : ",unique_val)
    print(columns)
    print("--------------------------------------------------")


# In[ ]:


#Dropping the column who dont have a lot of importance

train_data_predictors_v2 = train_data_predictors.drop(columns = unique_values_dict['[0]'],axis=1)


# In[ ]:


# We will first need to do one hot encoding before we run the XGBOOST MODEL

for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
  lbl = preprocessing.LabelEncoder()

  lbl.fit(list(train_data_predictors_v2[f].values)) 

  train_data_predictors_v2[f] = lbl.transform(list(train_data_predictors_v2[f].values))


# # Lets do some feature selection using XGBoost and Random Forest.

# In[ ]:


train_y = train_data_target.y.values
train_X = train_data_predictors_v2

# Thanks to anokas for this #
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
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

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# ### Feature Selection Using Random Forest>

# In[ ]:



model_RFR = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
model_RFR.fit(train_X, train_y)
feat_names = train_X.columns.values

## plot the importances ##
importances = model_RFR.feature_importances_
std = np.std([tree.feature_importances_ for tree in model_RFR.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# As inferred by SRK, the features selected by Random Forest and XGBoost are quite different.
# Since I am more familiar with Random Forest, I will go ahead and predict the values using Random Forest Regressor

# In[ ]:


test_df = pd.read_csv("../input/test.csv")


# Dropping the non-important columns

# In[ ]:


test_df_v2 = test_df.drop(columns = unique_values_dict['[0]'],axis=1)


# One Hot encoding the categorical vars in the test data

# In[ ]:


for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
  lbl = preprocessing.LabelEncoder()

  lbl.fit(list(test_df_v2[f].values)) 

  test_df_v2[f] = lbl.transform(list(test_df_v2[f].values))


# Finally predicting the variables using Random Forest

# In[ ]:


predictions = model_RFR.predict(test_df_v2.iloc[:,1:])


# Preparing the submission file

# In[ ]:


my_submission = pd.DataFrame({'ID':test_df_v2['ID'],'y':predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




