#!/usr/bin/env python
# coding: utf-8

# # HR Analytics [Kaggle dataset](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

# #### Problem Statement : **Predict if there will be a Attrition or not**

# #### Metrics Used : AUC Score

# ##### Importing Libraries and reading data

# In[ ]:


get_ipython().system('pip install xgboost==0.90')


# In[ ]:


get_ipython().system('pip install fastai==0.7.0')


# In[ ]:


from fastai.imports import *
from fastai.structured import *
from xgboost import XGBClassifier
from pandas_summary import DataFrameSummary
from IPython.display import display
import xgboost as xgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[ ]:


PATH = "/home/shivangmathur/Downloads/"


# In[ ]:


df_raw = pd.read_csv('../input/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)


# In[ ]:


display_all(df_raw.tail().T)


# #### Feature Engineering

# #### Creating more useful features

# In[ ]:


df_raw['DailyRateAge'] = df_raw['DailyRate']/df_raw['Age']
df_raw['PerfomanceReward'] = df_raw['PercentSalaryHike']/df_raw['PerformanceRating']
df_raw['Satisfaction'] = df_raw['EnvironmentSatisfaction']*df_raw['JobSatisfaction']


# In[ ]:


df_raw['Attrition'].value_counts()


# In[ ]:


df_raw.drop('EmployeeNumber',axis=1,inplace=True)


# ##### Using train_cats and proc_df features of fastai

# In[ ]:


train_cats(df_raw)


# In[ ]:


df, y, nas = proc_df(df_raw, 'Attrition',max_n_cat=30)


# ##### Splitting into train and test set

# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_test = 280  
n_trn = len(df)-n_test
raw_train, raw_test = split_vals(df_raw, n_trn)
X_train, X_test = split_vals(df, n_trn)
y_train, y_test = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_test.shape


# #### Creating the model

# In[ ]:


params = {
    'booster': 'gbtree', 
    'objective': 'binary:hinge',#egression task
    'subsample': 0.6,#% of data to grow trees and prevent overfitting
    'colsample_bytree': 0.7, # 70% of features used
    'eta': 0.01,'max_depth':8,'eval_metric': 'auc',
    'seed': 42} # for reproducible results

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_test,y_test)


watchlist = [(dtrain, 'train')]

xgb_model = xgb.train(params, dtrain,91,evals= watchlist,verbose_eval=True)


# #### Getting predictions and evaluating results

# In[ ]:


test = xgb.DMatrix(X_test, y_test)
pred_x = xgb_model.predict(test)


# In[ ]:


confusion_matrix(y_test, pred_x)


# In[ ]:


metrics.roc_auc_score(y_test,pred_x)


# #### Feature Importance

# In[ ]:


from matplotlib import pyplot
from xgboost import plot_importance
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(xgb_model, max_num_features=50, height=0.8, ax=ax)
pyplot.show()

