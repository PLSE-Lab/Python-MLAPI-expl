#!/usr/bin/env python
# coding: utf-8

# **Fastai implemantation**

# **Table of contents**
# 
# Importing Necessary Libraries
# 
# Preprocessing
# 
# Model Building
# 
# Creating a validation set
# 
# Visualization
# 

# **Importing Necessary Libraries**

# In[ ]:


get_ipython().system('pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np 
import pandas as pd 
from fastai.imports import*
from fastai.structured import *
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics
from pprint import pprint
import os
import shap
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, get_dataset, info_plots

print(os.listdir("../input"))


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# **Preprocessing**

# The evaluation criteria is RMSE of log of Sales Price. So first, let's change the target variable to log

# In[ ]:


df_train['SalePrice'] = np.log(df_train['SalePrice'])


# Convert the categorical variables into numbers. We can use the train_cats function from fastai for this purpose

# In[ ]:


train_cats(df_train)#Change any columns of strings in a panda's dataframe to a column of categorical values


# We don't want to mess the catagorical values in df_test, values should be same as use in df_raw. There is a function in fast.ai to solve this problem.

# In[ ]:


apply_cats(df_test, df_train)


# We have to impute the missing values and store the data as dependent and independent part. This is done by using the fastai function proc_df. The function performs the following tasks:
# 
# *     For continuous variables, it checks whether a column has missing values or not
# *     If the column has missing values, it creates another column called columnname_na, which has 1 for missing and 0 for not missing
# *     Simultaneously, the missing values are replaced with the median of the column
# *     For categorical variables, pandas replaces missing values with -1. So proc_df adds 1 to all the values for categorical variables. Thus, we have 0 for missing while all other values are incremented by 1
# 

# In[ ]:


train_df, y_trn, nas = proc_df(df_train, 'SalePrice')
test_df, _, _ = proc_df(df_test, na_dict=nas)
train_df.head()


# In[ ]:


df_test.columns[df_test.isnull().any()]


# In[ ]:


df_train.columns[df_train.isnull().any()]


# In[ ]:


test_df.columns


# In[ ]:


train_df.columns


# In[ ]:


test_df.drop(['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na'], axis =1, inplace = True)
train_df.drop(['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na'], axis = 1, inplace = True)


# **Model Building**

# Defining function to calculate the evaluation metric

# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(train_X), train_y), rmse(m.predict(val_X), val_y),     ## RMSE of log of prices
                m.score(train_X, train_y), m.score(val_X, val_y)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# Split data

# In[ ]:


train_X, val_X, train_y, val_y = train_test_split(train_df, y_trn, test_size=0.33, random_state=42)


# Bulid a single tree

# In[ ]:


get_ipython().run_line_magic('time', '')
m = RandomForestRegressor(n_estimators=1, min_samples_leaf=3, n_jobs=-1, max_depth = 3, oob_score=True) ## Use all CPUs available
m.fit(train_X, train_y)

print_score(m)


# In[ ]:


draw_tree(m.estimators_[0], train_X, precision=3)


# Use of grid search to find best parameter for Regressor

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# In[ ]:


rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)# Fit the random search model
rf_random.fit(train_X, train_y)


# In[ ]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, val_X, val_y)


# **Visualization**

# In[ ]:


perm = PermutationImportance(rf_random, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# In[ ]:


preds = np.stack([t.predict(val_X) for t in rf_random.best_estimator_])


# In[ ]:


preds.shape


# The dimensions of the predictions is (130, 482) . This means we have 130 predictions for each row in the validation set.

# In[ ]:


preds[:,0], np.mean(preds[:,0]), val_y[0]


# The actual value is 11.94 all of our predictions comes close to this value. On taking the average of all our predictions we get 11.84, which is a good prediction.

# In[ ]:


plt.plot([metrics.r2_score(val_y, np.mean(preds[:i+1], axis=0)) for i in range(20)]);


# As expected, the r^2 becomes better as the number of trees increases

# In[ ]:


for feat_name in val_X.columns:
#for feat_name in base_features:
    #pdp_dist = pdp.pdp_isolate(model=rf_random.best_estimator_, dataset=val_X, model_features=base_features, feature=feat_name)
    pdp_dist = pdp.pdp_isolate(model = rf_random.best_estimator_, dataset=val_X, model_features=val_X.columns, feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)

    plt.show()


# In[ ]:


explainer = shap.TreeExplainer(rf_random.best_estimator_)
shap_values = explainer.shap_values(val_X)

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[1,:], val_X.iloc[1,:], matplotlib=True) ## change shap and val_X


# In[ ]:


shap.summary_plot(shap_values, val_X)


# In[ ]:


pred = rf_random.best_estimator_.predict(test_df)
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()


# In[ ]:


#submission['SalePrice'] = np.exp(pred)   ## Convert log back 
submission.to_csv('submission_v2.csv', index=False)


# Credit: 
# * https://www.analyticsvidhya.com/blog/2018/10/comprehensive-overview-machine-learning-part-1/
# * https://www.kaggle.com/dansbecker/partial-plots
# 

# **If you like my work Please UPVOTE**
