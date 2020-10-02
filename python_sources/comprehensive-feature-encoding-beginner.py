#!/usr/bin/env python
# coding: utf-8

# ![Alt Text](https://media.giphy.com/media/LFiOdYoOlEKac/giphy.gif)

# # If you have any thoughts or ideas for improvements would I highly appreciate it.
# 
# # and if you liked the notebook please give it an upvote 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Datavisulizing 
import matplotlib.pyplot as plt
import seaborn as sns
# import seaborn as sn
import missingno as msno


from sklearn.impute import SimpleImputer
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm_notebook


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import category_encoders as ce

from sklearn import metrics

import lightgbm as lgb


# In[ ]:


raw_train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')
raw_test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')
raw_submission = pd.read_csv('../input/cat-in-the-dat-ii/sample_submission.csv', index_col='id')
print(raw_train.shape, raw_test.shape, raw_submission.shape)


# #  ***Agenda***
# ## Datavisualization 
#     *   Identify key features 
#     *   Identify columns with missing values
#     *   Identify Trends, Relationships and Distribution in the dataset
# ## Feature engineering
#     *   Target encoding 
#     *   Count encoding 
#     *   Catboost encoding 
#     *   Label encoding  
#     *   f_classif 
# ## Model training 
#     *   RandomForest 
#     *   LogisticRegression
#     *   XGBoost 
#     *   LightGBM 
# 

# ## Datavisualization 
# >     *   Identify key feature

# In[ ]:


raw_train.head()


# In[ ]:


raw_train.columns


# ## Description of the content in each column
# ### Binary features
# * zerosundones_0 = contain only zero (counted enteries 528377) and ones (counted enteries 53729)
# * zerosundones_1 = contain only zero (counted enteries 474018) and ones (counted enteries 107979)
# * zerosundones_2 = contain only zero (counted enteries 419845) and ones (counted enteries 162225)
# * FvsT = contain only F (counted enteries 366212) and T (counted enteries 215774)
# * NvsY = contain only N (counted enteries 312344) and Y (counted enteries 269609)
# 
# ### Nominal features
# * colors = contain  three colors Red, Blue, Green
# * trigonometry = contain Triangle, Polygon, Circle, Star, Trapezoid, Square
# * animals = contain Hamster, Axolotl, Dog, Lion, Snake, Cat
# * contries =  contain Russia, Canada, Finland, Costa Rica, Snake, India, China
# * instruments = contain Bassoon, Theremin, Theremin, Oboe, Piano, India, China
# * random_0 = contain what seems like randomized letter and number combinations 
# * random_1 = contain what seems like randomized letter and number combinations 
# * random_2 = contain what seems like randomized letter and number combinations 
# * random_3 = contain what seems like randomized letter and number combinations 
# * random_4 = contain what seems like randomized letter and number combinations 
# 
# ### Ordinal features 
# * oneTwoTree = contain the numbers 1.0(counted enteries 227917), 2.0(counted enteries 155997), 3.0(counted enteries 197798)
# * competetitions_levels = contain Contributor, Grandmaster, Novice, Expert, Master
# * temperature = contain Hot, Warm, Freezing, Lava Hot, Cold, boiling Hot
# * alpha_0 = contain the letters in the alphabet (counted enteries 582084) 
# * alpha_1 = contain the letters in the alphabet (counted enteries 582070) 
# * alpha_2 = contain the letters in the alphabet (counted enteries 582287) 
# 
# ### Potentially cyclical
# * day = contain the seven days; 1.0 (enteried 84724), 2.0 (enteries 65495), 3.0 (enteries 113835), 4.0 (enteries 23663), 5.0 (enteries 110464), 6.0 (enteries 97432), 7.0 (enteries 86435) 
# * month =contain the 12 months; 1.0 (enteries 52154), 2.0 (enteries 40700), 3.0 (enteries 70160), 4.0 (enteries 14614), 5.0 (enteries 68906), 6.0 (enteries 60478), 7.0 (enteries 53480), 8.0 (enteries 79245), 9.0 (enteries 20620), 10.0 (enteries 2150), 11.0 (enteries 51165), 12.0 (enteries 68340)
# 
# 
# * target = contain zeros (enteries 487677) and ones (enteries 112323)
# 

# In[ ]:


#Rename columns for traningset 
structuret_train = raw_train.copy();

structuret_train = structuret_train.rename(columns = {'bin_0':'zerosundones_0'});
structuret_train = structuret_train.rename(columns = {'bin_1':'zerosundones_1'});
structuret_train = structuret_train.rename(columns = {'bin_2':'zerosundones_2'});
structuret_train = structuret_train.rename(columns = {'bin_3':'FvsT'});
structuret_train = structuret_train.rename(columns = {'bin_4':'NvsY'});

structuret_train = structuret_train.rename(columns = {'nom_0': 'colors'});
structuret_train = structuret_train.rename(columns = {'nom_1': 'trigonometry'});
structuret_train = structuret_train.rename(columns = {'nom_2': 'animals'});
structuret_train = structuret_train.rename(columns = {'nom_3': 'contries'});
structuret_train = structuret_train.rename(columns = {'nom_4': 'instruments'});
structuret_train = structuret_train.rename(columns = {'nom_5': 'random_0'});
structuret_train = structuret_train.rename(columns = {'nom_6': 'random_1'});
structuret_train = structuret_train.rename(columns = {'nom_7': 'random_2'});
structuret_train = structuret_train.rename(columns = {'nom_8': 'random_3'});
structuret_train = structuret_train.rename(columns = {'nom_9': 'random_4'});

structuret_train = structuret_train.rename(columns = {'ord_0': 'oneTwoTree'});
structuret_train = structuret_train.rename(columns = {'ord_1': 'competetitions_levels'});
structuret_train = structuret_train.rename(columns = {'ord_2': 'temperature'});
structuret_train = structuret_train.rename(columns = {'ord_3': 'alpha_0'});
structuret_train = structuret_train.rename(columns = {'ord_4': 'alpha_1'});
structuret_train = structuret_train.rename(columns = {'ord_5': 'alpha_2'});


# In[ ]:


#Rename columns for testset 
structuret_test = raw_test.copy();

structuret_test = structuret_test.rename(columns = {'bin_0':'zerosundones_0'});
structuret_test = structuret_test.rename(columns = {'bin_1':'zerosundones_1'});
structuret_test = structuret_test.rename(columns = {'bin_2':'zerosundones_2'});
structuret_test = structuret_test.rename(columns = {'bin_3':'FvsT'});
structuret_test = structuret_test.rename(columns = {'bin_4':'NvsY'});

structuret_test = structuret_test.rename(columns = {'nom_0': 'colors'});
structuret_test = structuret_test.rename(columns = {'nom_1': 'trigonometry'});
structuret_test = structuret_test.rename(columns = {'nom_2': 'animals'});
structuret_test = structuret_test.rename(columns = {'nom_3': 'contries'});
structuret_test = structuret_test.rename(columns = {'nom_4': 'instruments'});
structuret_test = structuret_test.rename(columns = {'nom_5': 'random_0'});
structuret_test = structuret_test.rename(columns = {'nom_6': 'random_1'});
structuret_test = structuret_test.rename(columns = {'nom_7': 'random_2'});
structuret_test = structuret_test.rename(columns = {'nom_8': 'random_3'});
structuret_test = structuret_test.rename(columns = {'nom_9': 'random_4'});

structuret_test = structuret_test.rename(columns = {'ord_0': 'oneTwoTree'});
structuret_test = structuret_test.rename(columns = {'ord_1': 'competetitions_levels'});
structuret_test = structuret_test.rename(columns = {'ord_2': 'temperature'});
structuret_test = structuret_test.rename(columns = {'ord_3': 'alpha_0'});
structuret_test = structuret_test.rename(columns = {'ord_4': 'alpha_1'});
structuret_test = structuret_test.rename(columns = {'ord_5': 'alpha_2'});


# >     *   Identify columns with missing values

# In[ ]:


msno.matrix(raw_train, figsize = (30,5))


# In[ ]:


investigate_structuret_train = pd.DataFrame({'columns':structuret_train.columns})
investigate_structuret_train['data_type'] = structuret_train.dtypes.values
investigate_structuret_train['missing_val'] = structuret_train.isnull().sum().values 
investigate_structuret_train['uniques'] = structuret_train.nunique().values
investigate_structuret_train


# We will fill missing value in the section below (feature engineering)

# >     *   Identify Trends, Relationships and Distribution in the dataset

# ### The target value distribution

# In[ ]:


target_dist = structuret_train.target.value_counts()

barplot = plt.bar(target_dist.index, target_dist, color = 'darkred', alpha = 0.8)
barplot[0].set_color('darkblue')

plt.xlabel('Target', fontsize = 18)

plt.show()
print("percentage of the target: {}%".format(structuret_train.target.sum() / len(structuret_train.target)))


# It seems like target ratio is unbalanced.

# ### Heatmap of the trainings data

# In[ ]:


plt.figure(figsize=(12,10))
cols = raw_train.select_dtypes(exclude=['object']).columns
data = raw_train[cols].corr()
sns.heatmap(data, 
            xticklabels=data.columns.values,
            yticklabels=data.columns.values)

plt.show()


# It doesnt seems like there is any significant in the heatmap

# ## Feature engineering
#     * Defining functions inorder to validate the most effective encoding

# In[ ]:


# Diving the data into train and validation and selecting ther target value 
def feature_target(DataFrame):
    
    DataFrame.fillna(DataFrame.median(), inplace = True)
    
    y = DataFrame.target 
    X = DataFrame.drop(['target'], axis = 1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=0)
    
    return X_train, X_valid, y_train, y_valid 


# In[ ]:


def test_method(X_train, X_valid, y_train, y_valid):
    
    train = lgb.Dataset(X_train, label=y_train)
    valid = lgb.Dataset(X_valid, label=y_valid)
    
    param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    
    model_bst = lgb.train(param, train, num_boost_round=1000, valid_sets=[valid], 
                    early_stopping_rounds=10, verbose_eval=False)

    valid_pred = model_bst.predict(X_valid)
    
    mae = mean_absolute_error(y_valid, valid_pred)
    
    print(f"Validation MAE score: {mae:.4f}")
    
    return model_bst


# In[ ]:


#creating a list with all the categorical data with relatively low cardinality
categorical_data = [cat for cat in structuret_train if
                    structuret_train[cat].nunique() < 10 and 
                    structuret_train[cat].dtype == "object"]

categorical_data_test = [cat for cat in structuret_test if
                    structuret_test[cat].nunique() < 10 and 
                    structuret_test[cat].dtype == "object"]

# We want to find the numerical data
numerical_data = [num for num in structuret_train if structuret_train[num].dtype in ['int64', 'float64']]
numerical_data_test = [num for num in structuret_test if structuret_test[num].dtype in ['int64', 'float64']]


# In[ ]:


# making a copy of the renamed columns from raw_train(the original trainings data)
label_train = structuret_train.copy() 
onehot_train = structuret_train.copy()
count_train = structuret_train.copy()
target_train = structuret_train.copy()
catboost_train = structuret_train.copy()

# making a copy of the renamed columns from raw_test(the original test data)
label_test = structuret_test.copy() 
onehot_test = structuret_test.copy()
count_test = structuret_test.copy()
target_test = structuret_test.copy()
catboost_test = structuret_test.copy()


# > ***Label encoding***
# 
# Label encoding assigns each unique value to a different integer.
# We would see that in the temperature feature: "Hot" (0) < "boiling hot" (1) < "lava hot" (2) < "cold" (3) ect. 
# 

# In[ ]:


# converting all enteries to strings
label_train[categorical_data] = label_train[categorical_data].astype(str)  
label_test[categorical_data] = label_test[categorical_data].astype(str) 
    
# Label encoding
cat_features = categorical_data
encoder = LabelEncoder()

label_encoded_train = label_train[cat_features].apply(encoder.fit_transform)
label_encoded_test = label_test[cat_features].apply(encoder.fit_transform)

label_train = label_train[numerical_data].join(label_encoded_train)
label_test = label_test[numerical_data_test].join(label_encoded_test)


# In[ ]:


X_t, X_v, y_t, y_v = feature_target(label_train)
bst_results_label = test_method(X_t, X_v, y_t, y_v)


# ***Count Encoding***
# 
# Count encoding take the number of times a categorical values that has occurred and replace it with a number. 
# This happens example if the value "grandmaster" occured 10 times in the competetion_level feature, then "grandmaster" would be replaced with the number 10.
# 

# In[ ]:


count_train[categorical_data] = count_train[categorical_data].astype(str)
count_train[numerical_data] = count_train[numerical_data].astype(float)
count_test[categorical_data_test] = count_test[categorical_data_test].astype(str)
count_test[numerical_data_test] = count_test[numerical_data_test].astype(float)


# In[ ]:


# Count encoding 
cat_features = categorical_data

count_enc = ce.CountEncoder()

count_encoded_train = count_enc.fit_transform(count_train[cat_features])
count_encoded_test = count_enc.fit_transform(count_test[cat_features])

count_encoded_train = label_train.join(count_encoded_train.add_suffix("_count"))
count_encoded_test = label_test.join(count_encoded_test.add_suffix("_count"))


# In[ ]:


X_t, X_v, y_t, y_v = feature_target(count_encoded_train)
bst_results_count = test_method(X_t, X_v, y_t, y_v)


# ***Target encoding ***
# 
# Target encoding takes the average of the target value and replaces the categorical value for that value of feature 
# We see that in animals values where we would calculate the average value for all the rows with animals == "Hamster", animals == "dogs", animals == "cats" ect.   

# In[ ]:


cat_features = categorical_data

target_enc = ce.TargetEncoder(cols=cat_features)

target_encoded_train = target_enc.fit_transform(target_train[cat_features], target_train.target)
target_encoded_test = target_enc.transform(target_test[cat_features])

target_encoded_train = count_encoded_train.join(target_encoded_train.add_suffix("_target"))
target_encoded_test = count_encoded_test.join(target_encoded_test.add_suffix("_target"))


# In[ ]:


X_t, X_v, y_t, y_v = feature_target(target_encoded_train)
bst_results_target = test_method(X_t, X_v, y_t, y_v)


# ***catboost encoding***
# 
# catboost encoding is similar to target encoding since it takes the target probablity for a value. But the major difference between them is that catboost takes the probability for each row and calculate it from the rows before it

# In[ ]:


cat_features = categorical_data

catboost_enc = ce.CatBoostEncoder(cols=cat_features)

catboost_encoded_train = catboost_enc.fit_transform(catboost_train[cat_features], catboost_train.target)
catboost_encoded_test = catboost_enc.transform(catboost_test[cat_features])

catboost_encoded_train = count_encoded_train.join(catboost_encoded_train.add_suffix("_catboost"))
catboost_encoded_test = count_encoded_test.join(catboost_encoded_test.add_suffix("_catboost"))


# In[ ]:


X_t, X_v, y_t, y_v = feature_target(catboost_encoded_train)
bst_results_catboost = test_method(X_t, X_v, y_t, y_v)


# Catboost dosent seem to a positive effect on the result therefore we will continue with the target_encoded data 

# >     *   f_classif 
#         Drop feature that is not needed 

# In[ ]:


from sklearn.feature_selection import SelectKBest, f_classif


feature_cols = target_encoded_train.columns.drop('target')
X_t, X_v, y_t, y_v = feature_target(target_encoded_train)

# Keeping 14 features
selector = SelectKBest(f_classif, k=13)

X_new = selector.fit_transform(X_t, y_t)
X_new


# In[ ]:


# Get back the features we want to kept
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=X_t.index, 
                                 columns=feature_cols)
selected_features.head()


# In[ ]:


# Dropping columns that has zero values
selected_columns = selected_features.columns[selected_features.var() != 0]

# using valid for the selected features.
X_v[selected_columns].head()


# In[ ]:


# Testing the effect of dropping the feature that has no influence on the target value 
investigate = pd.DataFrame(target_encoded_train[selected_columns])
investigate['target'] = target_encoded_train.target

investigate_test = pd.DataFrame(target_encoded_test[selected_columns])

X_t, X_v, y_t, y_v = feature_target(investigate)
bst = test_method(X_t, X_v, y_t, y_v)


# It seems like the result got slightly worse, but an observation reveales that the number of columns we keep influence the result greatly 
# 
# Using target_encoded data which has 34 columns (include; count and target encoding) is an impressive amount af data to modelling on, eventhough the lightgbm model gives us a mae on 0.2824. 
# 
# The investigate only contain 14 columns (include a combination of count and target values)
# 
# The models below will be trained on investigate, but if we cant get the result under 0.2824
# 
# we will use the target_encoded data in the final submission 

# ## ***Model training***

# * RandomForest
# 
# Random forest consists of individual decision trees that operate as an ensemble. 
# Moreover it consists of leafs which is the point at the bottom where we make a prediction and the tree's depth is a measure of how many splits it makes before coming to a prediction

# In[ ]:


# optimizing model RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

RFR_model = RandomForestRegressor(n_estimators = 100, random_state = 0)

X_train_1, X_valid_1, y_train_1, y_valid_1 = feature_target(investigate)

RFR_model.fit(X_train_1, y_train_1)  
preds_1 = RFR_model.predict(X_valid_1) 
RFR_mae = mean_absolute_error(preds_1, y_valid_1) 


# In[ ]:


RFR_mae


# * LogisticRegression 
# 
# Logistic regression can describe ones data and explain the relationship between dependent binary variable and one or more nominal and ordinal vaeiables 
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression

X_train_2, X_valid_2, y_train_2, y_valid_2 = feature_target(investigate)
                                                    
LR_model = LogisticRegression(C=0.03, max_iter=300)
LR_model.fit(X_train_2, y_train_2)
preds_2 = LR_model.predict_proba(X_valid_2)[:, 1]

LR_mae = mean_absolute_error(preds_2, y_valid_2) 


# In[ ]:


LR_mae


# * XGBoost 
# 
# Gradient boosting a method that goes through cycles to iteratively add models into an ensemble.
# 
# Gradient boosting work by 
# 
# It begins by initializing the ensemble with a single model, whose predictions can be pretty naive
# 
# 1. use the current ensemble to generate predictions for each observation in the dataset
# 2. These predictions are used to calculate a loss function
# 3. use the loss function to fit a new model that will be added to the ensemble
# 4. we add the new model to ensemble, and ...
# 5. ... repeat!

# In[ ]:


import xgboost as xgb

X_train_3, X_valid_3, y_train_3, y_valid_3 = feature_target(investigate)

xgb_model = xgb.XGBClassifier(max_depth=20,n_estimators=2020,colsample_bytree=0.20,learning_rate=0.020,objective='binary:logistic', n_jobs=-1)

xgb_model.fit(X_train_3, y_train_3,eval_set=[(X_valid_3,y_valid_3)],verbose=0,early_stopping_rounds=200
) 

# preds_3 = xgb_model.predict(X_valid_3) 
preds_3 = xgb_model.predict_proba(X_valid_3)[:,1]
xgb_mae = mean_absolute_error(preds_3, y_valid_3) 


# In[ ]:


xgb_mae


# * LightGBM 
# 
# Gradient boosting uses tree based learning to optimize ones model 

# In[ ]:


import lightgbm as lgb

X_train_4, X_valid_4, y_train_4, y_valid_4 = feature_target(investigate)

train = lgb.Dataset(X_train_4, label=y_train_4)
valid = lgb.Dataset(X_valid_4, label=y_valid_4)
    
param = {'num_leaves': 64, 'objective': 'binary', 
             'metric': 'auc', 'seed': 7}
    
lgb_model = lgb.train(param, train, num_boost_round=1000, valid_sets=[valid], 
                    early_stopping_rounds=10, verbose_eval=False)

preds_4 = lgb_model.predict(X_valid_4)
    
lgb_mae = mean_absolute_error(y_valid_4, preds_4)


# In[ ]:


lgb_mae


# In[ ]:


scores = [RFR_mae, LR_mae, xgb_mae, lgb_mae]
pd.DataFrame(np.array([scores]),
                   columns=['RFR_mae', 'LR_mae', 'xgb_mae', 'lgb_mae'])


# In conclusion; The best model is lgb_model

# ## Submission 

# In[ ]:


test_pred_0 = lgb_model.predict(investigate_test)


# In[ ]:


submission_df = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')
sub_id = submission_df['id']
submission = pd.DataFrame({'id':sub_id})
submission['target'] = test_pred_0
submission.to_csv("submission5.csv",index = False)
print('Model ready for submission!')

