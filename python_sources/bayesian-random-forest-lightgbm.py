#!/usr/bin/env python
# coding: utf-8

# # Exploration of Forest Cover
# 
# ![Image](https://www.fs.fed.us/foresthealth/images/FS_regions.gif)
# 
# The **National Forest System,** the **Rocky Mountain Region(Region-2)** enjoys a proud heritage in the Forest Service. The Shoshone National Forest in Wyoming and the White River National Forest in Colorado are among the first National Forests Congress created from the original Forest Reserves. The Region, headquartered in Golden, Colorado, comprises 17 national forests and 7 national grasslands.
# 
# The US Forest Service Rocky Mountain Region has formally identified four overarching themes as emphasis areas on which to focus strategic long-term efforts to preserve their special values: Forest and Grassland Health, Recreation, Water and Public Service. Forests and Grasslands continue to hold in trust America's resources- timber, wildlife, water, range, recreation - to ensure their availability today and tomorrow.

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
from datetime import date
import pandas as pd
import numpy as np 
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
import random 
import warnings
import operator
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)

import pandas as pd
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import math


# <hr>
# # Part-1 Exploration Analysis
# <hr>
# ## 1. Dataset Preparation
# Lets view the snapshot of the dataset which is given for training and testing purposes.

# In[ ]:


train = pd.read_csv("../input/forest-cover-type-kernels-only/train.csv")
test = pd.read_csv("../input/forest-cover-type-kernels-only/test.csv")

print ("Train Dataset: Rows, Columns: ", train.shape)
print ("Test Dataset: Rows, Columns: ", test.shape)


# In[ ]:


print ("Glimpse of Train Dataset:... ")
train.head()


# In[ ]:


print ("Summary of Train Dataset: ")
train.describe()


# In[ ]:


print ("Top Columns having missing values")
missmap = train.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap.head()


# <hr>
# ## 2.Different Types of Cover Types
# <hr>

# In[ ]:


target = train['Cover_Type'].value_counts().to_frame()
levels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow","Aspen", "Douglas-fir", "Krummholz"]
trace = go.Bar(y=target.Cover_Type, x=levels, marker=dict(color='orange', opacity=0.6))
layout = dict(title="Cover_Type Tree Levels", margin=dict(l=10), width=1200, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


def largest_index(inlist):
    largest = -1
    largest_index = 0
    for i in range(len(inlist)):
        item = inlist[i]
        if item > largest:
            largest = item
            largest_index = i
    return largest_index


# # 1.Load Data

# In[ ]:


def load_data():
    loc_train = "../input/forest-cover-type-kernels-only/train.csv"
    loc_test = "../input/forest-cover-type-kernels-only/test.csv"
    loc_submission = "kaggle.rf200.entropy.submission.csv"
    df_train = pd.read_csv(loc_train)
    df_test = pd.read_csv(loc_test)
    return (loc_train, loc_test, loc_submission, df_train,df_test)

loc_train, loc_test, loc_submission, df_train,df_test = load_data()


# # 2.Normalize so Column

# In[ ]:


cols_to_normalize = ['Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
df_train[cols_to_normalize] = normalize(df_train[cols_to_normalize])
df_test[cols_to_normalize] = normalize(df_test[cols_to_normalize])


# In[ ]:


feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]
feature_cols.append('binned_elevation')
feature_cols.append('Horizontal_Distance_To_Roadways_Log')
feature_cols.append('Soil_Type12_32')
feature_cols.append('Soil_Type23_22_32_33')
feature_cols.append('Horizontal_Distance_To_Hydrology')


# # Feature Engineering

# In[ ]:


df_train['binned_elevation'] = [math.floor(v/50.0) for v in df_train['Elevation']]
df_test['binned_elevation'] = [math.floor(v/50.0) for v in df_test['Elevation']]

df_train['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df_train['Horizontal_Distance_To_Roadways']]
df_test['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df_test['Horizontal_Distance_To_Roadways']]

df_train['Soil_Type12_32'] = df_train['Soil_Type32'] + df_train['Soil_Type12']
df_test['Soil_Type12_32'] = df_test['Soil_Type32'] + df_test['Soil_Type12']
df_train['Soil_Type23_22_32_33'] = df_train['Soil_Type23'] + df_train['Soil_Type22'] + df_train['Soil_Type32'] + df_train['Soil_Type33']
df_test['Soil_Type23_22_32_33'] = df_test['Soil_Type23'] + df_test['Soil_Type22'] + df_test['Soil_Type32'] + df_test['Soil_Type33']

df_train['Horizontal_Distance_To_Hydrology_Log'] = [math.log(v+1) for v in df_train['Horizontal_Distance_To_Hydrology']]
df_test['Horizontal_Distance_To_Hydrology_Log'] = [math.log(v+1) for v in df_test['Horizontal_Distance_To_Hydrology']]


# In[ ]:


X_train = df_train[feature_cols]
X_test = df_test[feature_cols]
y = df_train['Cover_Type']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.10, random_state=42, stratify=y)
X_train.shape,y_train.shape,X_val.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from bayes_opt import BayesianOptimization\nimport lightgbm as lgb\n\n\ndef bayes_parameter_opt_lgb(X, y, init_round=15, opt_roun=25, n_folds=7, random_seed=42, n_estimators=10000, learning_rate=0.02, output_process=False,colsample_bytree=0.93,min_child_samples=56,subsample=0.84):\n    # prepare data\n    train_data = lgb.Dataset(data=X, label=y)\n    # parameters\n    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight, colsample_bytree,min_child_samples,subsample):\n        params = {\'application\':\'multiclass\',\'num_iterations\': n_estimators, \'learning_rate\':learning_rate, \'early_stopping_round\':300, \'metric\':\'macroF1\'}\n        params["num_leaves"] = int(round(num_leaves))\n        params["num_class"] = 8\n        params[\'feature_fraction\'] = max(min(feature_fraction, 1), 0)\n        params[\'bagging_fraction\'] = max(min(bagging_fraction, 1), 0)\n        params[\'max_depth\'] = int(round(max_depth))\n        params[\'lambda_l1\'] = max(lambda_l1, 0)\n        params[\'lambda_l2\'] = max(lambda_l2, 0)\n        params[\'min_split_gain\'] = min_split_gain\n        params[\'min_child_weight\'] = min_child_weight\n        params[\'colsample_bytree\'] = 0.93\n        params[\'min_child_samples\'] = 56,\n        params[\'subsample\'] = 0.84\n        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=[\'auc\'])\n        return max(cv_result[\'auc-mean\'])\n    # range \n    lgbBO = BayesianOptimization(lgb_eval, {\'num_leaves\': (19, 45),\n                                            \'feature_fraction\': (0.1, 0.9),\n                                            \'bagging_fraction\': (0.8, 1),\n                                            \'max_depth\': (5, 8.99),\n                                            \'lambda_l1\': (0, 5),\n                                            \'lambda_l2\': (0, 3),\n                                            \'min_split_gain\': (0.001, 0.1),\n                                            \'min_child_weight\': (5, 50),\n                                            \'colsample_bytree\' : (0.7,1.0),\n                                            \'min_child_samples\' : (40,65),\n                                            \'subsample\' : (0.7,1.0)\n                                           }, random_state=0)\n    # optimize\n    lgbBO.maximize(init_points=init_round, n_iter=opt_roun)\n    \n    # output optimization process\n    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")\n    \n    # return best parameters\n    return lgbBO.res[\'max\'][\'max_params\']\n\nopt_params = bayes_parameter_opt_lgb(X_train, y_train, init_round=10, opt_roun=10, n_folds=6, random_seed=42, n_estimators=500, learning_rate=0.02,colsample_bytree=0.93)')


# In[ ]:


opt_params = {'bagging_fraction': 0.9957236684465528,
 'colsample_bytree': 0.7953949538181928,
 'feature_fraction': 0.7333800304661316,
 'lambda_l1': 1.79753950286893,
 'lambda_l2': 1.710590311253639,
 'max_depth': 6,
 'min_child_samples': 48,
 'min_child_weight': 49,
 'min_split_gain': 0.016737988780906453,
 'num_leaves': 33,
 'subsample': 0.9033449610388691}


# # For Basic Model

# In[ ]:


from sklearn.model_selection import train_test_split
import lightgbm as lgb

# X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.20, random_state=42, stratify=y)
# X_train.shape,y_train.shape,X_val.shape
lgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=314, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced')
lgb.set_params(**opt_params)
#         n_estimators=132,#132
#         learning_rate=0.1,
#         num_leaves=64,
#         max_depth=7,
#         min_data_in_leaf=40,
#         max_bin =15,
#         reg_alpha=0.8, reg_lambda=0.6,
#         colsample_bytree=1.0,
#         min_split_gain=0.001, objective = "softmax",random_state=42,
#         stratified=True)
lgb.fit(X_train,y_train)
print(lgb.score(X_train,y_train))
y_pred = lgb.predict(X_test)


# In[ ]:


sub = pd.read_csv("../input/forest-cover-type-kernels-only/sample_submission.csv")
sub.head()


# In[ ]:


sub["Cover_Type"] = y_pred
sub.to_csv("submission_works.csv", index = False)


# In[ ]:




