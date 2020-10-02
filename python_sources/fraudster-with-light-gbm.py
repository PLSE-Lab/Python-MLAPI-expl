#!/usr/bin/env python
# coding: utf-8

# # Read Me Please!
# ### 1. Here, I only worked with one Dataset (Transaction) and built an end to end model with XGBoost, Light GBM, CatBoost. I still managed to achieve a significant score!
# ### 2. Dataframes are memory optimized to run GridSearchCV computations. Remember, optimize memory just before training the model. Otherwise, you can have errors while target encoding/ Catboost encoding with the optimized dfs.
# ### 3. Hyperparameter tuning is important but trust me that won't improve your Accuracy or AUC substantially. But it's very important to tune the model in case of over fitting and calculating efficiently so that run time is less and we don't suffer from our available resources.
# ### 4. The highest submission score 92.1% is a LGBMC in default parameters! As this is an imbalanced dataset problem, we may need few other tricks that I'm exploring.

# ### ***My goal is to make this Notebook an End to End Pipeline for a Novice ML Engineer to understand what are the steps are taken typically in a classification problem! If you can contribute, you are welcome! ***

# ## *** Please upvote if you like it!***

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,KFold,train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from category_encoders import CatBoostEncoder, TargetEncoder

from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
import csv
import ast
from timeit import default_timer as timer


from time import time
import datetime

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing Data sets

# In[ ]:


#df_id_tr=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
#df_id_ts=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
df_tran_tr=pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
df_tran_ts=pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
#df_sample=pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")


# # Exploratory Data Aanalysis
# ## Some Burning Questions to answer before we start building models...
# ### 1. What types of data are there? how many categorical & numeric features?
# ### 2. how much missing values?
# ### 3. is it a balanced or imbalanced classification problem?
# ### 4. how are the features related with the labelled classification (output)?
# ### 5. should we merge transaction and identity datasets? if yes/not, how do we proceed?
# ### 6. which features to give it a try?
# ### 7. to what extent we preprocess the data? how do we deal nulls?
# ### 8. how is the data distribution? Skewed? normal dist?

# In[ ]:


#Exploring data about the nature and types with memory consumption
df_tran_tr.info(verbose=True, null_counts=True, memory_usage='deep')


# In[ ]:


df_tran_ts.info(verbose=True, null_counts=True, memory_usage='deep')


# In[ ]:


df_tran_tr.describe()


# In[ ]:


# #making a joint dataframe: tran+id
# df_joint_tr=pd.merge(df_tran_tr,df_id_tr,on='TransactionID')
# df_joint_ts=pd.merge(df_tran_ts, df_id_ts, on="TransactionID")


# # Visualizing the missing values

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(18,9))
train_full_num = df_tran_tr.filter(regex='isFraud|TransactionDT|TransactionAmt|dist|C|D')
sns.heatmap(train_full_num.isnull(), cbar= False)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(18,9))
train_full_num = df_tran_tr.filter(regex='M')
sns.heatmap(train_full_num.isnull(), cbar= False)


# In[ ]:


# #this can be avoided as most often this gives memory error
# %matplotlib notebook
# %matplotlib inline
# train_full_Vesta = df_tran_tr.filter(regex='V')
# plt.figure(figsize=(18,9))
# sns.heatmap(train_full_Vesta.isnull(), cbar= False)


# In[ ]:


# %matplotlib notebook
# %matplotlib inline
# train_full_id = df_id_tr.filter(regex='id')
# plt.figure(figsize=(18,9))
# sns.heatmap(train_full_id.isnull(), cbar= False)


# In[ ]:


#balanced/imbalanced?
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,5))
sns.countplot(x='isFraud',data=df_tran_tr)


# In[ ]:


# pd.options.display.max_rows=500
# df_joint_tr.info()


# ## Categorical features and their co-relation with predictor

# In[ ]:


#checking each feature count relation with fraud
df_tran_tr.groupby('ProductCD')['isFraud'].value_counts(normalize=True).unstack()[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('P_emaildomain')['isFraud'].value_counts(normalize=True).unstack().fillna(0)[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('R_emaildomain')['isFraud'].value_counts(normalize=True).unstack().fillna(0)[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('card4')['isFraud'].value_counts(normalize=True).unstack()[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('card6')['isFraud'].value_counts(normalize=True).unstack().fillna(0)[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('M8')['isFraud'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('M9')['isFraud'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False)


# In[ ]:


df_tran_tr.groupby('M7')['isFraud'].value_counts(normalize=True).unstack().dropna()[1].sort_values(ascending=False)


# # Co-relating numeric features through Heat Maps

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
def cor_heat(df):
    cor=df.corr()
    plt.figure(figsize=(20,10),dpi=100)
    sns.heatmap(data=cor,annot=True,square=True,linewidths=0.1,cmap='YlGnBu')
    plt.title("Pearson Co-relation: Heat Map")
cor_heat(df_tran_tr.filter(regex='C|isFraud'))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
def cor_heat(df):
    cor=df.corr()
    plt.figure(figsize=(20,7),dpi=100)
    sns.heatmap(data=cor,annot=True,square=True,linewidths=0.1,cmap='YlGnBu')
    plt.title("Pearson Co-relation: Heat Map")
cor_heat(df_tran_tr.filter(regex='Tran|isFraud'))


# In[ ]:


pd.set_option('display.max_rows',400)
#using absolute to figure out features with higher co-relation irrespective of their +/- value
abs(df_tran_tr.filter(regex='V|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


# In[ ]:


abs(df_tran_tr.filter(regex='D[0-9]|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


# In[ ]:


abs(df_tran_tr.filter(regex='C|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)[0:11]


# In[ ]:


abs(df_tran_tr.filter(regex='add|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


# In[ ]:


abs(df_tran_tr.filter(regex='dist|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


# In[ ]:


abs(df_tran_tr.filter(regex='card1|card2|card3|card5|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


# # Insights so far from EDA to build initial Model...
# ### 1. M values are to be ignored for initial model due to very high number of null values and relate to fraud lesser comparatively
# ### 2. joining the transaction and identity (on TransactionID) leaves very small dataset (~144k). so, we will drop identity dataset as of now and work only with Transaction Dataset
# ### 3. Vxxx features are very good co-relator of fraud.need to maximize their utilization in model building
# ### 4. D features need to be considered due to their high correlation [needs imputations though] (top 3 initially); We drop D9,D2 even though they show high correlation due to high null values
# ### 5. TransactionDT, TransactionAMT will be considered
# ### 6. Top 5 C features (sorted by the co-relation) will be considered as no null values
# ### 7. From categorical features ProductCD, card (excluding card2), P_emaildomain (R_emaildomain excluded due to high null values) will be included and need category encoding
# ### 8. dist1,dist2 features are excluded due to high null & low co-relation
# ### 9. addr1, addr2 features should be included
# ### 10. P_emaildoamin has very high correlation and we will drop all the null values of it from main Dataset
# 
# 

# # Pre-processing & Feature Engineering

# In[ ]:


#This code will save your memory which will be required to run SVC/KNN within allocated memory and grid_search
## This cell is for advanced model building with D features selected from their null counts and correlation values
df_tran_tr=df_tran_tr.filter(regex='V|addr|C2|C8|C12|C1|C4|C10|C11|C6|C7|Tran|card1|card3|card4|card5|card6|ProductCD|P_emaildomain|R_emaildomain|isFraud|D[0-9]')
#dropping these columns due to their very low correlation with target
df_tran_tr=df_tran_tr.drop(['D7','D13','D8'],axis=1)
#df_tran_tr.info(verbose=True, null_counts=True)
df_tran_ts=df_tran_ts.filter(regex='V|addr|C2|C8|C12|C1|C4|C10|C11|C6|C7|Tran|card1|card3|card4|card5|card6|ProductCD|P_emaildomain|R_emaildomain|D[0-9]')
df_tran_ts=df_tran_ts.drop(['D7','D13','D8'],axis=1)
print("Train data set shape: {0}".format(df_tran_tr.shape))
print("Test data set shape: {0}".format(df_tran_ts.shape))


# # Use below cells if you want to manually impute the numeric missing features for models other than XGBoost/LGBM/CatBoost

# ## Just set manual_imputation= True to impute manually wherever needed

# In[ ]:


manual_imputation= False
#manual_imputation=True


# In[ ]:


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #imputing D features missing values with median
    for p in df_tran_tr.filter(regex='D[0-9]'):
        df_tran_tr[p]=df_tran_tr[p].fillna(df_tran_tr[p].median())

    for q in df_tran_ts.filter(regex='D[0-9]'):
        df_tran_ts[q]=df_tran_ts[q].fillna(df_tran_ts[q].median())


# In[ ]:


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #imputing V features missing values with median
    for x in df_tran_tr.filter(regex='V'):
        df_tran_tr[x]=df_tran_tr[x].fillna(df_tran_tr[x].median())

    for y in df_tran_ts.filter(regex='V'):
        df_tran_ts[y]=df_tran_ts[y].fillna(df_tran_ts[y].median())


# In[ ]:


# #checking the co-relation after imputations
# pd.set_option('display.max_rows',400)
# abs(df_tran_tr.filter(regex='V|isFraud').fillna(0).corr())['isFraud'].sort_values(ascending=False)


# In[ ]:


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #filling numerical card features with median
    for a in df_tran_tr.filter(regex='card1|card3|card5'):
        df_tran_tr[a]=df_tran_tr[a].fillna(df_tran_tr[a].median())

    for b in df_tran_ts.filter(regex='card1|card3|card5'):
        df_tran_ts[b]=df_tran_ts[b].fillna(df_tran_ts[b].median())


# In[ ]:


if manual_imputation==True:
    #filling null C values in test tran dataset with median
    for c in df_tran_ts.filter(regex='C2|C8|C12|C1|C4|C10|C11|C6|C7'):
        df_tran_ts[c]=df_tran_ts[c].fillna(df_tran_ts[c].median())


# In[ ]:


if manual_imputation==True:
    # use it when you don't want auto imputation for missing values
    #filling null addr values with median
    df_tran_tr.addr1=df_tran_tr.addr1.fillna(df_tran_tr.addr1.median())
    df_tran_tr.addr2=df_tran_tr.addr2.fillna(df_tran_tr.addr2.median())

    df_tran_ts.addr1=df_tran_ts.addr1.fillna(df_tran_ts.addr1.median())
    df_tran_ts.addr2=df_tran_ts.addr2.fillna(df_tran_ts.addr2.median())


# # Imputing and transforming Categorical features

# ### Missing values imputations with most common values. Catboost Encoding auto fills these.

# ## Just set the category encoding you want to proceed with 

# In[ ]:


cat_encoding="catboost"
#cat_encoding="label"


# In[ ]:


if cat_encoding=="label":
    # If you want to R_emaildomain in your model
    #dropping null values for R_emaildomain; this is a very critical indicator to fruad; so have to utilize it
    ##df_tran_tr=df_tran_tr.dropna(subset=['R_emaildomain'])
    df_tran_tr['R_emaildomain']=df_tran_tr['R_emaildomain'].fillna('gmail.com')
    #as we can't drop any row from test data, filling it with mode
    df_tran_ts['R_emaildomain']=df_tran_ts['R_emaildomain'].fillna('gmail.com')


# In[ ]:


if cat_encoding=="label":
    #df_tran_tr=df_tran_tr.dropna(subset=['P_emaildomain'])
    df_tran_tr['P_emaildomain']=df_tran_tr['P_emaildomain'].fillna('gmail.com')
    #as we can't drop any row from test data, filling it with mode
    df_tran_ts['P_emaildomain']=df_tran_ts['P_emaildomain'].fillna('gmail.com')


# In[ ]:


if cat_encoding=="label":
    #checking max present card4 type to fill in null values
    df_tran_tr.groupby('card4')['isFraud'].value_counts().unstack()
    print(df_tran_tr.card4.mode())
    print(df_tran_ts.card4.mode())


# In[ ]:


if cat_encoding=="label":
    df_tran_tr.groupby('card6')['isFraud'].value_counts().unstack()
    print(df_tran_tr.card6.mode())
    print(df_tran_ts.card6.mode())


# In[ ]:


if cat_encoding=="label":
    df_tran_tr.card4=df_tran_tr.card4.fillna('visa')
    df_tran_tr.card6=df_tran_tr.card6.fillna('debit')

    df_tran_ts.card4=df_tran_ts.card4.fillna('visa')
    df_tran_ts.card6=df_tran_ts.card6.fillna('debit')


# In[ ]:


# for x in df_tran_tr.filter(regex='M'):
#     df_tran_tr[x]=df_tran_tr[x].fillna(df_tran_tr[x].value_counts().index[0])
# for y in df_tran_ts.filter(regex='M'):
#     df_tran_ts[y]=df_tran_ts[y].fillna(df_tran_ts[y].value_counts().index[0])


# ### Label Encoding categorical features

# In[ ]:


if cat_encoding=="label":
    #Label Encoding R_emaildomain
    df_tran_tr.R_emaildomain=LabelEncoder().fit_transform(df_tran_tr.R_emaildomain)
    df_tran_ts.R_emaildomain=LabelEncoder().fit_transform(df_tran_ts.R_emaildomain)

    #Label Encoding P_emaildomain
    df_tran_tr.P_emaildomain=LabelEncoder().fit_transform(df_tran_tr.P_emaildomain)
    df_tran_ts.P_emaildomain=LabelEncoder().fit_transform(df_tran_ts.P_emaildomain)

    #Label Encoding ProductCD
    df_tran_tr.ProductCD=LabelEncoder().fit_transform(df_tran_tr.ProductCD)
    df_tran_ts.ProductCD=LabelEncoder().fit_transform(df_tran_ts.ProductCD)

    #Label encoding card features
    df_tran_tr.card4=LabelEncoder().fit_transform(df_tran_tr.card4)
    df_tran_tr.card6=LabelEncoder().fit_transform(df_tran_tr.card6)
    df_tran_ts.card4=LabelEncoder().fit_transform(df_tran_ts.card4)
    df_tran_ts.card6=LabelEncoder().fit_transform(df_tran_ts.card6)


    # for z in df_tran_tr.filter(regex="M"):
    #     df_tran_tr[z]=LabelEncoder().fit_transform(df_tran_tr[z])

    # for a in df_tran_ts.filter(regex="M"):
    #     df_tran_ts[a]=LabelEncoder().fit_transform(df_tran_ts[a])


# ### CatBoost Encoding

# In[ ]:


if cat_encoding=="catboost":
    cat_features=['R_emaildomain','P_emaildomain','ProductCD','card4','card6']

    cbe=CatBoostEncoder(cols=cat_features)
    # X= df_tran_tr.drop(['isFraud'],axis=1)
    # y= df_tran_tr[['isFraud']]
    cbe.fit(df_tran_tr[cat_features],df_tran_tr[['isFraud']])

    # #Train & Test Set transforming
    df_tran_tr=df_tran_tr.join(cbe.transform(df_tran_tr[cat_features]).add_suffix('_target'))
    df_tran_tr.drop(['R_emaildomain','P_emaildomain','ProductCD','card4','card6'],axis=1,inplace=True)

    df_tran_ts=df_tran_ts.join(cbe.transform(df_tran_ts[cat_features]).add_suffix('_target'))
    df_tran_ts.drop(['R_emaildomain','P_emaildomain','ProductCD','card4','card6'],axis=1,inplace=True)


# In[ ]:


#don't do this before category encoding
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


#Reducing memory without any data loss
df_tran_tr=reduce_mem_usage(df_tran_tr)
df_tran_ts=reduce_mem_usage(df_tran_ts)


# ### Splitting Train & Test Sets

# In[ ]:


X=df_tran_tr.drop(['isFraud'],axis=1)
y=df_tran_tr[['isFraud']]
## as this is an imbalanced problem, we need to stratify the splitting so that training is well distributed
#no need for LGB hyper model selection
# X_train, X_test, y_train, y_test = train_test_split(df_tran_tr.drop(['isFraud'],axis=1), df_tran_tr[['isFraud']], test_size=0.2, random_state=0,stratify=df_tran_tr[['isFraud']])


# # Hyperparameters Tuning

# ## Details of XGB classifier hyper parameters
# 
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# In[ ]:


# this gives memory error; so I will be trying to iterate manually with parameter values
# param_test = {'n_estimators':np.arange(2,100,1), 
#               'max_depth':np.arange(3,10,1),
#               'gamma':np.arange(0,0.5,0.1),
#               'learning_rate':np.arange(0,0.1,0.01),
#              'min_child_weight':np.arange(1,10,1)
#              }


# gs=GridSearchCV(estimator=XGBClassifier(),scoring='roc_auc',param_grid=param_test,cv=3)
# gs.fit(X_train,np.ravel(y_train)) #used ravel to convert to 1-D array
# print("best parameters are: {0} for the best score of {1}".format(gs.best_params_,gs.best_score_))


# ## Details of Light GBM hyperparameters

# In[ ]:


# hyper_tuning="hyperopt"
hyper_tuning="grid_search"


# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html#for-better-accuracy
# 
# https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
# 
# https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
# 
# https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a
# 
# https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian%20Hyperparameter%20Optimization%20of%20Gradient%20Boosting%20Machine.ipynb

# ### with lgb.cv which will require model in test data

# In[ ]:


# #you can try cv using below method
# params_lgb={'boosting_type':'gbdt',
#            'objective': 'binary',
#            'random_state':42}
# k_fold=10
# train_data=lgb.Dataset(X_train,label=y_train)
# validation_data=lgb.Dataset(X_test,label=y_test)
# time_to_train=time()
# lgbmc=lgb.cv(params_lgb,train_data,num_boost_round=10000,nfold=k_fold,metrics='auc',
#              verbose_eval=True, early_stopping_rounds=500)
# print("Training is completed!")
# print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - time_to_train))))
# print('-',30)
# print(lgbmc.best_score_)
# print(lgbmc.best_params)


# ## Bayesian optimization method with Hyperopt
# 
# It's advantage over Grid_search is that it won't iterate over some discrete values rather all the values within the search domain!

# https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a

# In[ ]:


#look carefully the default parameter; we will circle around the default values to find the optimum ones
LGBMClassifier()


# In[ ]:


# if hyper_tuning=="hyperopt":
    
#     MAX_EVALS = 500
#     N_FOLDS=5
#     train_set=lgb.Dataset(X,label=y)
#     #Objective function: part-1

#     def objective(params, n_folds = N_FOLDS):
#         """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

#         # Keep track of evals
#         global ITERATION

#         ITERATION += 1

#         # Retrieve the subsample if present otherwise set to 1.0
#         subsample = params['boosting_type'].get('subsample', 1.0)

#         # Extract the boosting type
#         params['boosting_type'] = params['boosting_type']['boosting_type']
#         params['subsample'] = subsample

#         # Make sure parameters that need to be integers are integers
#         for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
#             params[parameter_name] = int(params[parameter_name])

#         start = timer()

#         # Perform n_folds cross validation
#         cv_results = lgb.cv(params, train_set=train_set, num_boost_round = 10000, nfold = n_folds, 
#                             early_stopping_rounds = 100, metrics = 'auc', seed = 50)

#         run_time = timer() - start

#         # Extract the best score
#         best_score = np.max(cv_results['auc-mean'])

#         # Loss must be minimized
#         loss = 1 - best_score

#         # Boosting rounds that returned the highest cv score
#         n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

#         # Write to the csv file ('a' means append)
#         of_connection = open(out_file, 'a')
#         writer = csv.writer(of_connection)
#         writer.writerow([loss, params, ITERATION, n_estimators, run_time])

#         # Dictionary with information for evaluation
#         return {'loss': loss, 'params': params, 'iteration': ITERATION,
#                 'estimators': n_estimators, 
#                 'train_time': run_time, 'status': STATUS_OK}
    
#     #defining search space: part-2
#     space = {
#         'class_weight': hp.choice('class_weight', [None, 'balanced']),
#         'boosting_type': hp.choice('boosting_type', 
#                                    [{'boosting_type': 'gbdt', 
#                                         'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
#                                      {'boosting_type': 'dart', 
#                                          'subsample': hp.uniform('dart_subsample', 0.5, 1)},
#                                      {'boosting_type': 'goss'}]),
#         'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
#         'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
#         'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
#         'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
#         'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#         'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#         'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
#     }
    
#     # boosting type domain 
#     boosting_type = {'boosting_type': hp.choice('boosting_type', 
#                                                 [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)}, 
#                                                  {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
#                                                  {'boosting_type': 'goss', 'subsample': 1.0}])}

#     # Draw a sample
#     params = sample(boosting_type)
    
#     # Retrieve the subsample if present otherwise set to 1.0
#     subsample = params['boosting_type'].get('subsample', 1.0)

#     # Extract the boosting type
#     params['boosting_type'] = params['boosting_type']['boosting_type']
#     params['subsample'] = subsample
    
    
#     # File to save first results
#     out_file = 'gbm_trials.csv'
#     of_connection = open(out_file, 'w')
#     writer = csv.writer(of_connection)

#     # Write the headers to the file
#     writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
#     of_connection.close()



#     # Algorithm: part-3
#     tpe_algorithm = tpe.suggest
#     # Trials object to track progress
#     bayes_trials = Trials()
    


# In[ ]:



# %%capture
# #capture results: part-4
# # Global variable
# global  ITERATION

# ITERATION = 0

# # Run optimization
# best = fmin(fn = objective, space = space, algo = tpe.suggest, 
#             max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50),
#             verbose=50,show_progressbar=True)

# results = pd.read_csv('gbm_trials.csv')

# # Sort with best scores on top and reset index for slicing
# results.sort_values('loss', ascending = True, inplace = True)
# results.reset_index(inplace = True, drop = True)
# results.head()


# In[ ]:



# # Convert from a string to a dictionary
# ast.literal_eval(results.loc[0, 'params'])

# # Extract the ideal number of estimators and hyperparameters
# best_bayes_estimators = int(results.loc[0, 'estimators'])
# best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()
# print("Best Estimator is:{}".format(best_bayes_estimators))
# print("-"* 30)
# print("Best Parameters are {}".format(best_bayes_params))


# ## Randomized Searching
# https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search

# In[ ]:


k_fold=5
kf=StratifiedKFold(n_splits=k_fold,shuffle=True, random_state=42)
if hyper_tuning=="grid_search":
    
    params_lgb_grid={
    'boosting_type': ['gbdt', 'dart'],'num_leaves': list(range(30, 150)),
    'learning_rate': list(np.logspace(np.log(0.005), np.log(0.2), base = np.exp(1), num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),'reg_lambda': list(np.linspace(0, 1)),'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'n_estimators':list(np.arange(100,2000,50))
    }
    
    lgb_estimator=LGBMClassifier(objective='binary',random_state=42,max_depth=-1,num_boost_rounds=1000)
    rs=RandomizedSearchCV(estimator=lgb_estimator,scoring='roc_auc',param_distributions=params_lgb_grid,cv=kf,verbose=100, n_iter=20,n_jobs=2)
    rs.fit(X,np.ravel(y)) #used ravel to convert to 1-D array
    bs_score=rs.best_score_
    bs_params=rs.best_params_
    bs_est=rs.best_estimator_
    print("best parameters are: {0} for the best score of {1}".format(rs.best_params_,rs.best_score_))
    print("-"*30)
    print("best estimator is:{}".format(rs.best_estimator_))


# ## train and validation in same iteration with lgb.train

# In[ ]:


# k_fold=5
# kf=StratifiedKFold(n_splits=k_fold,shuffle=True, random_state=42)
# training_start_time = time()
# aucs=[]
# for fold, (trn_idx,val_idx) in enumerate(kf.split(X,y)):
#     start_time = time()
#     print('Training on fold {}'.format(fold + 1))
#     trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
#     val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])
#     clf = lgb.train(params_lgb, trn_data, num_boost_round=10000, valid_sets = [trn_data, val_data], 
#                     verbose_eval=200, early_stopping_rounds=100)
#     aucs.append(clf.best_score['valid_1']['auc'])
#     print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
# print('-' * 30)
# print('Training is completed!.')
# print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
# print(clf.best_params_)
# print('-' * 30)


# # Testing tuned model in Test data

# In[ ]:


# def testing_model(my_model):
#     my_model.fit(X_train,np.ravel(y_train))
#     my_model_pred=my_model.predict(X_test)
#     print("accuracy score is {0}% and roc 
#     print(classification_report(y_test,my_model_pred))
#     print(my_model)


# In[ ]:


# xgbc_test=XGBClassifier(gamma=0.5,learning_rate=0.03,max_depth=9,n_estimators=200,min_child_weight=3, reg_alpha=0.005)
# testing_model(xgbc_test)


# In[ ]:


print(df_tran_tr.shape)
print(df_tran_ts.shape)


# In[ ]:


# #saving the model parameters
# params=clf.best_params_
# best_iter=clf.best_iteration


# ## Tuning model with best parameters
# 
# Hardcoding model for minimum Loss 0.0441241 learnt from Bayesian hyperopt searching 

# In[ ]:


# tuned_lgb=LGBMClassifier(boosting_type='gbdt',class_weight='balanced', colsample_bytree=0.8970523178797932,
#                learning_rate=0.8970523178797932, min_child_samples=45,n_estimators=3323, n_jobs=-1, 
#                 num_leaves=138, objective='binary',random_state=42, reg_alpha=0.5005020213127344, 
#                 reg_lambda= 0.45121616279208887, silent=True,
#                subsample=0.6614743075688195, subsample_for_bin=260000)
tuned_lgb=LGBMClassifier(objective='binary',random_state=42,**bs_params)


# # Which algorithm to tune finally?

# In[ ]:


def model_output(your_model):
    #training the model with inp and out df
    your_model.fit(df_tran_tr.drop(['isFraud'],axis=1),np.ravel(df_tran_tr[['isFraud']]))
    your_model_pred= your_model.predict_proba(df_tran_ts)[:,1]
    your_model_df= pd.DataFrame({'TransactionID':df_tran_ts['TransactionID'],'isFraud': your_model_pred.astype(float)})
    your_model_df.to_csv('submission_fraud.csv',index=False)

rdf_model=RandomForestClassifier(warm_start=True)
xgb_model=XGBClassifier()
nbg_model=GaussianNB()
mplc_model=MLPClassifier()
adb_model= AdaBoostClassifier()
gbb_model=GradientBoostingClassifier()
svc_model= SVC()
knn_model=KNeighborsClassifier()
sgd_model=SGDClassifier()
lgbmc_model= LGBMClassifier()
catb_model=CatBoostClassifier(eval_metric = 'AUC')
model_output(tuned_lgb)


# # Observations and findings after Iterations on Model Score on test data
# ### 1. XGBoost and RandomForest give 89.3% and 83.4% accuracy on default parameters run. Seems Ensemble(checked AdaBoost, GradientBoosting) has a good prediction on this data.
# ### 2. Missing values for P & R emaildomain can be imputed by mode and missing values for numeric features can be imputed through XGBoost auto imputation (auto imputation gave .03% higher score for me. so I'm following that).
# ### 3. we will check if the id dataset can add values to prediction through Heatmap correlation
# ### 4. We can try out MLPC, KNN, SVC, NB algos. but ensemble seems to a winner here.
# ### 5. are all V rich features needed to be included in the model? which columns can we drop? we can set a certain co-relation score and allow all V features meeting that threshold...
# ### 6. tried with 137k train set ( size of R_emaildomain non-null values) and got a score of 86.5% which is less than our auto imputed XGBoost classifier score of 89.6%.

# In[ ]:




