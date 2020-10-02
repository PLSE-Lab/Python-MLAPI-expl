#!/usr/bin/env python
# coding: utf-8

# <font size="8">**Data Science Workshop - Forum SUPCOM 2019**</font>

# <font size="6">**Introduction**</font>

# Fraud is a billion-dollar business and it is increasing every year. The PwC global economic crime survey of 2018 found that half (49 percent) of the 7,200 companies they surveyed had experienced fraud of some kind. This is an increase from the PwC 2016 study in which slightly more than a third of organizations surveyed (36%) had experienced economic crime.  
# This problem is a binary classification problem - i.e. our target variable is a binary attribute (Is the user making the click fraudlent or not?) and our goal is to classify users into "fraudlent" or "not fraudlent" as well as possible.

# <font size="6">**Metric**</font>

# AUC score only depends on how well you well you can separate the two classes. In practice, this means that only the order of your predictions matter, as a result of this, any rescaling done to your model's output probabilities will have no effect on your score.

# <font size="6">**Libraries Importing**</font>

# Installing notebook vega for the visualization 

# In[ ]:


get_ipython().system('pip install -U vega_datasets notebook vega')


# In[ ]:


import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import roc_auc_score
import datetime
import random
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import linear_model
from sklearn import tree
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import eli5
import shap
from IPython.display import HTML
import json
import altair as alt

import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# alt.renderers.enable('notebook')

# %env JOBLIB_TEMP_FOLDER=/tmp


# <font size="6">**EDA - Exploratory Data Analysis**</font>

# In[ ]:


train = pd.read_pickle("../input/fraud-detection-forum-supcom-2019/train.pkl")
test = pd.read_pickle("../input/fraud-detection-forum-supcom-2019/test.pkl")


# In[ ]:


print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')


# In[ ]:


train.head()


# * TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
# * TransactionAMT: transaction payment amount in USD
# * ProductCD: product code, the product for each transaction
# * card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
# * addr: address
# * dist: distance
# * P_ and (R__) emaildomain: purchaser and recipient email domain
# * C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
# * D1-D15: timedelta, such as days between previous transaction, etc.
# * M1-M9: match, such as names on card and address, etc.
# * Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

# In[ ]:


print(f'There are {train.isnull().any().sum()} columns in train dataset with missing values.')
print(f'There are {test.isnull().any().sum()} columns in test dataset with missing values.')


# In[ ]:


one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
print(f'There are {len(one_value_cols)} columns in train dataset with one unique value.')
print(f'There are {len(one_value_cols_test)} columns in test dataset with one unique value.')


# Most of columns have missing data, which is normal in real world. Also there are columns with one unique value (or all missing). There are a lot of continuous variables and some categorical. Let's have a closer look at them.

# In[ ]:


for i in range(1,10): 
    print(f"The most counted values of id_0{i}")
    print(train[f'id_0{i}'].value_counts(dropna=False, normalize=True).head()*100)
    print("\n\n")
for i in range(10,38): 
    print(f"The most counted values of id_{i}")
    print(train[f'id_{i}'].value_counts(dropna=False, normalize=True).head())
    print("\n\n")


# In[ ]:


for i in range(1,10): 
    if train[f'id_0{i}'].dtypes!="O":
        plt.figure();
        plt.hist(train[f'id_0{i}']);
        plt.title(f'Distribution of id_0{i} variable');
for i in range(10,38): 
    if train[f'id_{i}'].dtypes!="O":
        plt.figure();
        plt.hist(train[f'id_{i}']);
        plt.title(f'Distribution of id_{i} variable');


# In[ ]:


import os
import time
import datetime
import json
import gc
from numba import jit

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics

from itertools import product

import altair as alt
from altair.vega import v5
from IPython.display import HTML

# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey
def prepare_altair():
    """
    Helper function to prepare altair for working.
    """

    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v5.SCHEMA_VERSION
    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
    noext = "?noext"
    
    paths = {
        'vega': vega_url + noext,
        'vega-lib': vega_lib_url + noext,
        'vega-lite': vega_lite_url + noext,
        'vega-embed': vega_embed_url + noext
    }
    
    workaround = f"""    requirejs.config({{
        baseUrl: 'https://cdn.jsdelivr.net/npm/',
        paths: {paths}
    }});
    """
    
    return workaround
    

def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay outside and 
    return wrapped

@add_autoincrement
def render(chart, id="vega-chart"):
    """
    Helper function to plot altair visualizations.
    """
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vg_embed) {{
        const spec = {chart};     
        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
        console.log("anything?");
    }});
    console.log("really...anything?");
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )


# In[ ]:


charts = {}
for i in ['id_30', 'id_31', 'id_33', 'DeviceType', 'DeviceInfo']:
    feature_count = train[i].value_counts(dropna=False)[:40].reset_index().rename(columns={i: 'count', 'index': i})
    chart = alt.Chart(feature_count).mark_bar().encode(
                x=alt.X(f"{i}:N", axis=alt.Axis(title=i)),
                y=alt.Y('count:Q', axis=alt.Axis(title='Count')),
                tooltip=[i, 'count']
            ).properties(title=f"Counts of {i}", width=800)
    charts[i] = chart

render(charts['id_30'] & charts['id_31'] & charts['id_33'] & charts['DeviceType'] & charts['DeviceInfo'])


# In[ ]:


plt.hist(train['TransactionDT'], label='train');
plt.hist(test['TransactionDT'], label='test');
plt.legend();
plt.title('Distribution of transactiond dates');


# In[ ]:


train.isFraud.hist()


# <font size="6">**Feature Engineering**</font>

# In[ ]:


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# In[ ]:


SEED = 42
seed_everything(SEED)
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


# In[ ]:


def frequency_encoding(train_df, test_df, columns, self_encoding=False):
    for col in tqdm_notebook(columns):
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()
        if self_encoding:
            train_df[col] = train_df[col].map(fq_encode)
            test_df[col]  = test_df[col].map(fq_encode)            
        else:
            train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
            test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)
    return train_df, test_df


# In[ ]:


def timeblock_frequency_encoding(train_df, test_df, periods, columns, 
                                 with_proportions=True, only_proportions=False):
    for period in periods:
        for col in tqdm_notebook(columns):
            new_col = col +'_'+ period
            train_df[new_col] = train_df[col].astype(str)+'_'+train_df[period].astype(str)
            test_df[new_col]  = test_df[col].astype(str)+'_'+test_df[period].astype(str)

            temp_df = pd.concat([train_df[[new_col]], test_df[[new_col]]])
            fq_encode = temp_df[new_col].value_counts().to_dict()

            train_df[new_col] = train_df[new_col].map(fq_encode)
            test_df[new_col]  = test_df[new_col].map(fq_encode)
            
            if only_proportions:
                train_df[new_col] = train_df[new_col]/train_df[period+'_total']
                test_df[new_col]  = test_df[new_col]/test_df[period+'_total']
            if with_proportions:
                train_df[new_col+'_proportions'] = train_df[new_col]/train_df[period+'_total']
                test_df[new_col+'_proportions']  = test_df[new_col]/test_df[period+'_total']

    return train_df, test_df


# In[ ]:


from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

# Let's add temporary "time variables" for aggregations
# and add normal "time variables"
for df in [train, test]:
    
    # Temporary variables for aggregation
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
    df['DT_M'] = ((df['DT'].dt.year-2017)*12 + df['DT'].dt.month).astype(np.int8)
    df['DT_W'] = ((df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear).astype(np.int8)
    df['DT_D'] = ((df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear).astype(np.int16)
    
    df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)
    df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)
    df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)
        
    # Possible solo feature
    df['is_december'] = df['DT'].dt.month
    df['is_december'] = (df['is_december']==12).astype(np.int8)

    # Holidays
    df['is_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

    
# Total transactions per timeblock
for col in ['DT_M','DT_W','DT_D']:
    temp_df = pd.concat([train[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train[col+'_total'] = train[col].map(fq_encode)
    test[col+'_total']  = test[col].map(fq_encode)
    


# In[ ]:


for df in [train, test]:
    df['bank_type'] = df['card3'].astype(str) +'_'+ df['card5'].astype(str)


# In[ ]:


# from final features. But we can use it for aggregations.
train['uid'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)
test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)

train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)
test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)

train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)

train['uid4'] = train['uid3'].astype(str)+'_'+train['P_emaildomain'].astype(str)
test['uid4'] = test['uid3'].astype(str)+'_'+test['P_emaildomain'].astype(str)

train['uid5'] = train['uid3'].astype(str)+'_'+train['R_emaildomain'].astype(str)
test['uid5'] = test['uid3'].astype(str)+'_'+test['R_emaildomain'].astype(str)

# Add values remove list
new_columns = ['uid','uid2','uid3','uid4','uid5']

print('#'*10)
print('Most common uIds:')
for col in new_columns:
    print('#'*10, col)
    print(train[col].value_counts()[:10])

# Do Global frequency encoding 
# i_cols = ['card1','card2','card3','card5'] + new_columns


# In[ ]:


cat_features=[]
continuous_features=[]
for col in train.columns:
    if train[col].dtypes =="O":
        cat_features.append(col)
    else:
        if col != "isFraud":
            continuous_features.append(col)


# In[ ]:


periods = ['DT_M','DT_W','DT_D']
train, test = frequency_encoding(train, test, cat_features, self_encoding=True)
print("Freq encoding is done")
train, test = timeblock_frequency_encoding(train, test, periods, cat_features, 
                                 with_proportions=False, only_proportions=True)
print("Freq encoding based on time is done")


# <font size="6">**Modeling**</font>

# In[ ]:


train.dtypes


# In[ ]:


EXCLU_COLUMNS = ["isFraud",'TransactionID',"DT"]
cols= [col for col in train.columns if col not in EXCLU_COLUMNS]


# In[ ]:


train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
depth_tree=12


# In[ ]:


gc.collect()


# Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

kf = KFold(n_splits = 5, random_state = 5168, shuffle = False)
oof_train_Decision_Tree= np.zeros(train.shape[0])
for i,(train_index, val_index) in enumerate(kf.split(train)):
    X_train, X_val = train[cols].iloc[train_index], train[cols].iloc[val_index]
    y_train, y_val = train.isFraud[train_index], train.isFraud[val_index]
    
    clf = DecisionTreeClassifier(max_depth=depth_tree)
    print(f"Training of the {i+1}th decision tree has started : ")
    clf.fit(X_train, y_train)
    oof_train_Decision_Tree[val_index] =clf.predict_proba(X_val)[:,1]
    
    
    val_score = roc_auc_score(y_val, oof_train_Decision_Tree[val_index])
    print(f"CV on the {i+1}th fold = {val_score}")
print(f"AUC after training all the folds = {roc_auc_score(train.isFraud, oof_train_Decision_Tree)}")
    


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

kf = KFold(n_splits = 5, random_state = 5168, shuffle = False)
oof_train_RF= np.zeros(train.shape[0])
for i,(train_index, val_index) in enumerate(kf.split(train)):
    X_train, X_val = train[cols].iloc[train_index], train[cols].iloc[val_index]
    y_train, y_val = train.isFraud[train_index], train.isFraud[val_index]
    
    clf = RandomForestClassifier(n_estimators=50,max_depth=depth_tree,n_jobs=4)
    print(f"Training of the {i+1}th random forest has started : ")
    clf.fit(X_train, y_train)
    oof_train_RF[val_index] =clf.predict_proba(X_val)[:,1]
    
    
    val_score = roc_auc_score(y_val, oof_train_RF[val_index])
    print(f"CV on the {i+1}th fold = {val_score}")
print(f"AUC after training all the folds = {roc_auc_score(train.isFraud, oof_train_RF)}")
    



# LightGBM

# In[ ]:


import lightgbm as lgb

kf = KFold(n_splits = 5, random_state = 5168, shuffle = False)
oof_train_LGB= np.zeros(train.shape[0])
for i,(train_index, val_index) in enumerate(kf.split(train)):
    X_train, X_val = train[cols].iloc[train_index], train[cols].iloc[val_index]
    y_train, y_val = train.isFraud[train_index], train.isFraud[val_index]
    lgb_params = {
                'objective':'binary',
                'boosting_type':'gbdt',
                'metric':'auc',
                'n_jobs':-1,
                'learning_rate':0.1,
                'max_depth':depth_tree,
                'n_estimators':1000,
                'seed': 0,
                'early_stopping_rounds':100, 
            }
    clf = lgb.LGBMClassifier(**lgb_params)
    print(f"Training of the {i+1}th fold has started : ")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_val, y_val)],
        verbose=50,
        early_stopping_rounds=100)
    oof_train_LGB[val_index] = clf.predict_proba(X_val,num_iteration=clf.best_iteration_)[:,1]
    
    
    val_score = roc_auc_score(y_val, oof_train_LGB[val_index])
    print(f"CV on the {i+1}th fold = {val_score}")
print(f"AUC after training all the folds = {roc_auc_score(train.isFraud, oof_train_LGB)}")
    



# In[ ]:




