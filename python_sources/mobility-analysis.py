#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Project Objective

# Sigma Cab Private Limited - 
#     
# Cab aggregator service. Their customers can download their app on smartphones and book a cab from any where in the cities they operate in. They, in turn search for cabs from various service providers and provide the best option to their client across available options. They have been in operation for little less than a year now. During this period, they have captured surge_pricing_type from the service providers.
# 
# The task is to help them in predicting the surge_pricing_type pro-actively. This would in turn help them in matching the right cabs with the right customers quickly and efficiently.
# 

# In[ ]:


train = pd.read_csv('/kaggle/input/janatahack-mobility-analysis/train.csv')
test = pd.read_csv('/kaggle/input/janatahack-mobility-analysis/test.csv')
submit = pd.read_csv('/kaggle/input/janatahack-mobility-analysis/sample_submission.csv')


# # Data Dictionary

# **Trip_ID  **              =    ID for TRIP (Can not be used for purposes of modelling)
# 
# Trip_Distance          =    The distance for the trip requested by the customer
# 
# Type_of_Cab            =    Category of the cab requested by the customer
# 
# Customer_Since_Months  =    Customer using cab services since n months; 0 month means current month
# 
# Life_Style_Index       =    Proprietary index created by Sigma Cabs showing lifestyle of the customer based on their behaviour
# 
# Confidence_Life_Style_Index    =    Category showing confidence on the index mentioned above
# 
# Destination_Type       =    Sigma Cabs divides any destination in one of the 14 categories.
# 
# Customer_Rating        =    Average of life time ratings of the customer till date
# 
# Cancellation_Last_1Month    =    Number of trips cancelled by the customer in last 1 month
# 
# Var1, Var2 and Var3    =     Continuous variables masked by the company. Can be used for modelling purposes
# 
# Gender                 =        Gender of the customer
# 
# Surge_Pricing_Type     =   Predictor variable can be of 3 types
# 
# 

# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train


# In[ ]:


train.dtypes


# In[ ]:


train.isna().sum()


# Missing values are found in Type_of_cab ( object ), Customer_Since_Months ( Float ), Life_Style_Index ( Float ), Confidence_Life_Style_Index ( Float ) and var1 ( Float )

# In[ ]:


sns.distplot(train['Trip_Distance'],kde = False,norm_hist=False)


# * The distribution is not normal

# In[ ]:


train['Trip_Distance'].max()
train['Trip_Distance'].min()


# The maximum distance that the customers are travelling is 110 while minimum is 0.5

# In[ ]:


train['Type_of_Cab'].unique()


# A,B,C,D,E are the categories of cab
# 

# In[ ]:


sns.countplot(train['Type_of_Cab'])


# Majority is Type B

# In[ ]:


train['Type_of_Cab'].isna().sum()


# In[ ]:


#train['Customer_Since_Months'].astype(int)
train['Customer_Since_Months'].value_counts()
sns.countplot(train['Customer_Since_Months'])


# Customer using cab services since n months; 0 month means current month
# 
# 
# So, more use from 10 months

# In[ ]:


train['Destination_Type'].unique()


# In[ ]:


sns.countplot(train['Destination_Type'])


# In[ ]:


train['Destination_Type'].value_counts()


# In[ ]:


train.Cancellation_Last_1Month.value_counts()


# In[ ]:


sns.countplot(train.Cancellation_Last_1Month)


# In[ ]:


var = [train['Var1'],train['Var2'],train['Var3']]
for variables in var:
    sns.distplot(variables)
    plt.show()


# In[ ]:


sns.countplot(train['Gender'])


# In[ ]:


sns.countplot(train['Surge_Pricing_Type'])


# In[ ]:


train.dropna()


# In[ ]:


df = train.append(test)


# In[ ]:


test.shape


# In[ ]:


test.isna().sum()


# In[ ]:


df.isna().sum()


# In[ ]:


sns.countplot(df['Destination_Type'])


# In[ ]:


df['Destination_Type'].fillna(df['Destination_Type'].mode(),inplace = True)


# In[ ]:


sns.countplot(df['Cancellation_Last_1Month'])


# In[ ]:


df['Cancellation_Last_1Month'].isna().sum()


# In[ ]:


df[df['Type_of_Cab'].isna() == True]


# In[ ]:


sns.countplot(df['Type_of_Cab'])


# In[ ]:


df['Type_of_Cab'] = df['Type_of_Cab'].fillna('F')


# In[ ]:


sns.countplot(df['Type_of_Cab'])


# In[ ]:


df['Customer_Since_Months'].value_counts()


# In[ ]:


df[df['Customer_Since_Months'].isna() == True]


# In[ ]:


df[df['Customer_Since_Months'].isna() == False].mean()


# In[ ]:


sns.countplot(df['Customer_Since_Months'])


# In[ ]:


df['Customer_Since_Months'].fillna(10.0,inplace = True)


# In[ ]:


df.columns


# In[ ]:


print("Minimum Value",df['Life_Style_Index'].min())
print("Maximum Value",df['Life_Style_Index'].max())
print("Average Value",df['Life_Style_Index'].mean())
sns.distplot(df['Life_Style_Index'])


# In[ ]:


sns.countplot(df['Confidence_Life_Style_Index'])


# In[ ]:


df.groupby('Confidence_Life_Style_Index')['Life_Style_Index'].median()


# In[ ]:


print(df['Life_Style_Index'].mean())
df['Life_Style_Index'].median()


# In[ ]:


df['Life_Style_Index'].fillna(df['Life_Style_Index'].mean(),inplace = True)
df['Confidence_Life_Style_Index'].fillna('D',inplace = True)


# In[ ]:


print(df['Var1'].mean())
print(df['Var1'].median())


# In[ ]:


sns.distplot(df['Var1'])


# In[ ]:


df['Var1'].value_counts()


# In[ ]:


df[['Var1','Var2','Var3']].corr()


# In[ ]:


df.groupby('Var1')[['Var2','Var3']].mean()


# In[ ]:


df['Var1'].fillna(64,inplace = True)


# In[ ]:


df.isna().sum()


# In[ ]:


df[['Var1','Var2','Var3']] = df[['Var1','Var2','Var3']].astype(int)


# In[ ]:


df['Customer_Since_Months'] = df['Customer_Since_Months'].astype(int)


# In[ ]:





# In[ ]:


categorical = ['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender']


# In[ ]:


from sklearn import preprocessing 
  

label_encoder = preprocessing.LabelEncoder() 
for i in categorical:
    df[i]= label_encoder.fit_transform(df[i]) 


# In[ ]:


df


# In[ ]:


trains = df[df['Surge_Pricing_Type'].isna() == False]
tests = df[df['Surge_Pricing_Type'].isna() == True]


# In[ ]:


trains['Surge_Pricing_Type'] = trains['Surge_Pricing_Type'].astype(int)


# In[ ]:


trains


# In[ ]:


tests


# In[ ]:


trains.to_csv('train.csv',index = False)
tests.to_csv('test.csv',index = False)


# Model building

# In[ ]:


from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[ ]:


categorical = ['Type_of_Cab','Confidence_Life_Style_Index','Destination_Type','Gender']


# In[ ]:


trains.columns


# In[ ]:


train_x = trains.drop(columns=['Surge_Pricing_Type','Trip_ID'],axis=1)
train_y = trains['Surge_Pricing_Type']


# In[ ]:


test_x = tests.drop(columns=['Surge_Pricing_Type','Trip_ID'],axis=1)

trains = trains.drop(columns = ['Trip_ID'],axis = 1)
tests = tests.drop(columns = ['Trip_ID'],axis = 1)
# In[ ]:


trains


# In[ ]:


#import h2o
#h2o.init()
#train1 = h2o.H2OFrame(train)
#test1 = h2o.H2OFrame(tests)
#train1.columns
#y = 'Surge_Pricing_Type'
#x = train1.col_names
#x.remove(y)
#train1['Surge_Pricing_Type'] = train1['Surge_Pricing_Type'].asfactor()
#train1['Surge_Pricing_Type'].levels()
#from h2o.automl import H2OAutoML
#aml = H2OAutoML(max_models = 20,max_runtime_secs=2000, seed = 42)
#aml.train(x = x, y = y, training_frame = train1)
#preds = aml.predict(test1)
#ans=h2o.as_list(preds) 
#submit['target'] = ans['predict']
#submit.to_csv('Solution_of_H20_EDA.csv',index=False)


# In[ ]:


def extra_tree(Xtrain,Ytrain,Xtest):
    extra = ExtraTreesClassifier()
    extra.fit(Xtrain, Ytrain) 
    extra_prediction = extra.predict(Xtest)
    return extra_prediction

def Xg_boost(Xtrain,Ytrain,Xtest):
    xg = XGBClassifier(loss='exponential', learning_rate=0.05, n_estimators=1000, subsample=1.0, criterion='friedman_mse', 
                                  min_samples_split=2, 
                                  min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_depth=10, min_impurity_decrease=0.0, 
                                  min_impurity_split=None, 
                                  init=None, random_state=None, max_features=None, verbose=1, max_leaf_nodes=None, warm_start=False, 
                                  presort='deprecated', 
                                  validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    xg.fit(Xtrain, Ytrain) 
    xg_prediction = xg.predict(Xtest)
    return xg_prediction
def LGBM(Xtrain,Ytrain,Xtest):
    lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=40,
                            max_depth=5, learning_rate=0.05, n_estimators=1000, subsample_for_bin=200, objective='binary', 
                            min_split_gain=0.0, min_child_weight=0.001, min_child_samples=10,
                            subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0,
                            reg_lambda=0.0, random_state=None, n_jobs=1, silent=True, importance_type='split')
    #lgbm = LGBMClassifier(n_estimators= 500)
    lgbm.fit(X_train, Y_train)
    lgbm_preds = lgbm.predict(X_test)
    return lgbm_preds


# In[ ]:


#target = 'Surge_Pricing_Type'
#scoring_parameter = 'balanced-accuracy'


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator


# In[ ]:


#pred_xg = Xg_boost(X_train,Y_train,X_test)
#pred_et = extra_tree(X_train,Y_train,X_test)
#pred_l = LGBM(X_train,Y_train,X_test)


# In[ ]:


#submit['target'] = pred_xg
#print(submit['target'].unique())
#submit.to_csv('XG.csv',index = False)


# In[ ]:


#submit['target'] = pred_et
#print(submit['target'].unique())
#submit.to_csv('ET.csv',index = False)


# In[ ]:


#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0).fit(train_x, train_y)
#ans = clf.predict(test_x)


# In[ ]:


#submit['Surge_Pricing_Type'] = ans
##print(submit['Surge_Pricing_Type'].unique())
#submit.to_csv('LR.csv',index = False)


# In[ ]:


#from sklearn.ensemble import RandomForestClassifier
#rf = RandomForestClassifier(n_estimators=10).fit(train_x, train_y)
#prediction_of_rf = rf.predict(test_x)
#submit['Surge_Pricing_Type'] = prediction_of_rf
#print(submit['Surge_Pricing_Type'].unique())
#submit.to_csv('RF.csv',index = False)


# In[ ]:


#submit['Surge_Pricing_Type'] = nri
#submit.to_csv('Dknn.csv',index = False)


# MODELS

# In[ ]:


target_map = {1:0, 2:1, 3:2}
target_map_inverse = {0:1, 1:2, 2:3}


# In[ ]:


trains["Surge_Pricing_Type"] = trains["Surge_Pricing_Type"].map(target_map)


# In[ ]:


features = [col for col in trains.columns if col not in ["Trip_ID", "Surge_Pricing_Type"]]
target = trains["Surge_Pricing_Type"]


# In[ ]:


param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.5,
    'boost': 'gbdt',
    'feature_fraction': 0.7,
    'learning_rate': 0.005,
    'num_class':3,
    'metric':'multi_logloss',
    'max_depth': 8,  
    'num_leaves': 70,
    'min_data_in_leaf':40,
    'objective': 'multiclass',
    'scale_pos_weight':1,
    'device':'gpu',
    'verbosity': 1
}


# In[ ]:


import lightgbm as lgb
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# 

# In[ ]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1048)
predictions = np.zeros((len(tests), 3))
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(trains.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(trains.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(trains.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)
    predictions_val = np.argmax(clf.predict(trains.iloc[val_idx][features], num_iteration=clf.best_iteration), axis=1)
    
    print("CV score: {:<8.5f}".format(sklearn.metrics.accuracy_score(predictions_val, target.iloc[val_idx])))
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(tests[features], num_iteration=clf.best_iteration) / folds.n_splits


# In[ ]:


cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:150].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('Features importance (averaged/folds)')
plt.tight_layout()
#plt.savefig('FI.png')


# In[ ]:


predictions_test = np.argmax(predictions, axis=1)
submit["Surge_Pricing_Type"] = predictions_test
submit["Surge_Pricing_Type"] = submit["Surge_Pricing_Type"].map(target_map_inverse)
submit.to_csv("janatahack_mobility_solution.csv", index=False)

