#!/usr/bin/env python
# coding: utf-8

# # <center> Demand Forecasting</center>
# ________

# <center><img src='https://datahack-prod.s3.ap-south-1.amazonaws.com/__sized__/contest_cover/cover_1_3vEBqwk-thumbnail-1200x1200.png'/></center>

# ## About

# Demand Forecasting is the pivotal business process around which strategic and operational plans of a company are devised. Based on the Demand Forecast, strategic and long-range plans of a business like budgeting, financial planning, sales and marketing plans, capacity planning, risk assessment and mitigation plans are formulated.

# ## Problem Statement
# 
# One of the largest retail chains in the world wants to use their vast data source to __build an efficient forecasting model__ to predict the sales for each SKU in its portfolio at its __76 different stores__ using historical sales data for the __past 3 years__ on a week-on-week basis. Sales and promotional information is also available for each week - product and store wise. 
# 
# However, no other information regarding stores and products are available. Can you still forecast accurately the sales values for every such product/SKU-store combination for the __next 12 weeks accurately__? 
# 
# - If yes, then dive right in! Let's Play

# ## Data Description

# <center><img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1279142%2F4f09fa27a17b01fa40700e7b80d87add%2Fdataset_description.jpg?generation=1594430740572308&alt=media'/></center>

# ## Evaluation Metric
# - The evaluation metric for this competition is 100*RMSLE (Root Mean Squared Log Error).

# # Let's Begin
# 
# > In this notebook i provide you some hints if you implement them in your notebook,Surely gonna get better results.(because i already implemented and got much better results).
# 
# > - So keep your eye on given hints.

# In[ ]:


# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[ ]:


## Import all the required libraries

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
# % matplotlib inline


# To begin this exploratory analysis, first import libraries and define functions for plotting the data using `matplotlib`. Depending on the data, not all plots will be made. (Hey, I'm just a simple kerneling bot, not a Kaggle Competitions Grandmaster!)

# There are 3 csv files in the current version of the dataset:
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train=pd.read_csv('../input/janatahack-demand-forecasting-analytics-vidhya/train.csv')

test=pd.read_csv('../input/janatahack-demand-forecasting-analytics-vidhya/test.csv')

sample=pd.read_csv('../input/janatahack-demand-forecasting-analytics-vidhya/sample_submission.csv')


# In[ ]:


train.head(10)
print('Shape of training data is {}'.format(train.shape))

print('-------------'*5)

test.head()
print('Shape of test data is {}'.format(test.shape))

print('--------------'*5)

sample.head()


# In[ ]:


train.describe()


# In[ ]:


train.isnull().sum()


# ### Data Visualization

# In[ ]:


train['week'].unique()


# In[ ]:


# Number of units sold in accordance with the week

train.groupby('week').sum()['units_sold'].plot(figsize=(12,8))
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 26,
        }
plt.xlabel('Week',fontdict=font)
plt.ylabel('units_sold',fontdict=font)


# In[ ]:


# amount earned through sales in each week

train.groupby('week').sum()['total_price'].plot(figsize=(12,8))
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 26,
        }
plt.xlabel('Week',fontdict=font)
plt.ylabel('total_price',fontdict=font)


# In[ ]:


train['store_id'].unique()


# In[ ]:


## product sold by each of the store


train.groupby('store_id').sum()['units_sold'].plot(figsize=(15,8),kind='bar')
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 26,
        }
plt.xlabel('store_id',fontdict=font)
plt.ylabel('units_sold',fontdict=font)


# - Max number of product are sold by ```8023``` store id.

# In[ ]:


## Product was on display at a prominent place at the store

# Impact on sales on the basis of display
train.groupby(['is_display_sku','store_id']).sum()['units_sold']


# ## Data Preprocessing

# In[ ]:


# join test and train data

train['train_or_test']='train'
test['train_or_test']='test'
df=pd.concat([train,test])


# In[ ]:


df.head()
df.shape


# ### Creating Time Based Features, So further we can convert this time series problem in to a regression one.

# In[ ]:


# function to utilize date time column i.e '''week'''

def create_week_date_featues(dataframe):

    df['Month'] = pd.to_datetime(df['week']).dt.month

    df['Day'] = pd.to_datetime(df['week']).dt.day

    df['Dayofweek'] = pd.to_datetime(df['week']).dt.dayofweek

    df['DayOfyear'] = pd.to_datetime(df['week']).dt.dayofyear

    df['Week'] = pd.to_datetime(df['week']).dt.week

    df['Quarter'] = pd.to_datetime(df['week']).dt.quarter 

    df['Is_month_start'] = pd.to_datetime(df['week']).dt.is_month_start

    df['Is_month_end'] = pd.to_datetime(df['week']).dt.is_month_end

    df['Is_quarter_start'] = pd.to_datetime(df['week']).dt.is_quarter_start

    df['Is_quarter_end'] = pd.to_datetime(df['week']).dt.is_quarter_end

    df['Is_year_start'] = pd.to_datetime(df['week']).dt.is_year_start

    df['Is_year_end'] = pd.to_datetime(df['week']).dt.is_year_end

    df['Semester'] = np.where(df['week'].isin([1,2]),1,2)

    df['Is_weekend'] = np.where(df['week'].isin([5,6]),1,0)

    df['Is_weekday'] = np.where(df['week'].isin([0,1,2,3,4]),1,0)

    df['Days_in_month'] = pd.to_datetime(df['week']).dt.days_in_month
    
    return df


# In[ ]:


df=create_week_date_featues(df)


# In[ ]:


df.head(5)
df.shape


# ### Encode the categorical variable

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


col=['store_id','sku_id','Is_month_start','Is_month_end','Is_quarter_start','Is_quarter_end','Is_year_start','Is_year_end']


# In[ ]:


for i in col:
    df = pd.get_dummies(df, columns=[i])


# In[ ]:


df.head()


# In[ ]:


# col2=['Is_month_start','Is_month_end','Is_quarter_start','Is_quarter_end','Is_year_start','Is_year_end']

# for i in col2:
#     df = pd.get_dummies(df, columns=[i])


# In[ ]:


# drop the columns
df.drop(['record_ID','week'],inplace=True,axis=1)


# In[ ]:


df.head()
df.shape


# ## Treating skewed features

# In[ ]:


# Total price columns

df['total_price'].plot(kind='hist')


# In[ ]:


df['total_price']=np.log1p(df['total_price'])
df['total_price'].plot(kind='hist')


# In[ ]:


df['base_price'].plot(kind='hist')


# In[ ]:


df['base_price']=np.log1p(df['base_price'])
df['base_price'].plot(kind='hist')


# ## Hint1:  
# 
# Treat the skewness of the target variable too. 
# - In my final notebook i am using it and getting a better score.

# In[ ]:


df.head()


# In[ ]:


train_1=df.loc[df.train_or_test.isin(['train'])]
test_1=df.loc[df.train_or_test.isin(['test'])]
train_1.drop(columns={'train_or_test'},axis=1,inplace=True)
test_1.drop(columns={'train_or_test'},axis=1,inplace=True)


# In[ ]:


train_1.head()
train_1.shape
test_1.shape
test_1.head()


# In[ ]:


test_1.drop(['units_sold'],axis=1,inplace=True)


# In[ ]:


train_1.shape
test_1.shape


# In[ ]:


x=train_1.drop(['units_sold'],axis=1)
y=train_1['units_sold']


# In[ ]:


x=x.values
test_data=test_1.values

y=y.values


# In[ ]:


x.shape
test_data.shape


# ## Time to train the model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.35)


# In[ ]:


# x_train.shape


# ## Training the model with xgboost

# In[ ]:


from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb

# function to plot all features based out of its importance.
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time


# ### Creating RMSLE function 

# I am not using this ```RMSLE``` function in my model because of error. If anyone able to figure out let me know in the comment section. 

# In[ ]:


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.power(np.log1p(y_true)-np.log1p(y_pred), 2)))


# In[ ]:


# Perform cross-validation
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


# ## Hint2:
# 
# In this notebook i use ```XGboost( a gradient boosting algorithm )```. Use other available gradient boosting algorithms,gonna get better results. 
# 
# - I am also using other one.
# - Don't waste you time with ```xgboost``` as this algo is not generalzing well for this data(based on my experience).
# - I though that much hint is very much sufficient and probably you got me what i am trying to say.
# 
# - See you on Leaderboard.

# In[ ]:


model = XGBRegressor(
    max_depth=12,
    booster = "gbtree",
    n_estimators=200,
    eval_metric = 'rmse',
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,
    seed=42,
    objective='reg:linear')


# In[ ]:


kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()

for train, test in kfold.split(x):
    x_train, x_test =x[train], x[test]
    y_train, y_test = y[train], y[test]
    model.fit(x_train, y_train,verbose=True,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              early_stopping_rounds = 50)
#     preds = model.predict(x_test)
#     score = rmsle(y_test, preds)
#     scores.append(score)
#     print(score)
    
    
# print("Average: ", sum(scores)/len(scores))


# In[ ]:


# # defint the model parameters

# ts = time.time()

# model = XGBRegressor(
#     max_depth=12,
#     booster = "gbtree",
#     n_estimators=500,
#     min_child_weight=350, 
#     colsample_bytree=0.8, 
#     subsample=0.8, 
#     eta=0.3,
#     seed=42,
#     objective='reg:linear')

# model.fit(
#     x_train, 
#     y_train, 
#     eval_metric="rmse", 
#     eval_set=[(x_train, y_train), (x_valid, y_valid)], 
#     verbose=True, 
#     early_stopping_rounds = 100)

# time.time() - ts


# In[ ]:


test_data.shape


# In[ ]:


# create prediction on test data

pred=model.predict(test_data)


# In[ ]:


len(pred)


# In[ ]:


# sample['units_sold']=pred.round()

sample['units_sold']=pred


# In[ ]:


sample.head()


# In[ ]:


sample['units_sold'].unique()


# Treat the -ve predicted values.

# In[ ]:


sample['units_sold']=abs(sample['units_sold']).astype('int')


# ### Create the final submission

# In[ ]:


sample.to_csv('submission_xgb.csv',index=False,encoding='utf-8')


# ## Note: 
# - With this kernel I have no intention to ruin the competition spirit, it is created to just help you to get started.
# 
# If you like my work then do 
# 
#     - Do follow
#     - Do upvote
#     - Have doubts regarding this kernel use comment section.
#     
# 
# As this competition is still going on .So I will upload my final kernel once competition got over.
# 
# - Stay tuned for new and improved version.

# ## Note1:
# 
# My current ranking in the table is ```29```.
# - If any one want some hint let me know in the comment section. 

# > Edit-23/07/2020

# ## Trying some thing new
# 
# As this competition is over but resently i amazed with new ensemble technique ```MinMaxBestBaseStacking```. So to just check is it really effective or not. 
# 
# [Original Source Repo](https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/Kaggle-PyTorch/PyTorch-Ensembler/utils.py)
# 
# - For the sake of learning.

# ## MinMaxBestBaseStacking

# In[ ]:


sub_path = "../input/testing-minmaxbestbasestacking"
all_files = os.listdir(sub_path)
print(all_files)


# In[ ]:


for f in all_files:
    print(f)


# In[ ]:


d=pd.read_csv('../input/testing-minmaxbestbasestacking/866_126527_us_submission_lgbm_22.csv')
d.head()
d.shape


# In[ ]:


# # Read and concatenate submissions

outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files ]
concat_sub = pd.concat(outs, axis=1)
concat_sub.head()


# In[ ]:


cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))

cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols


# In[ ]:


concat_sub.head()


# In[ ]:


concat_sub.reset_index(inplace=True)


# In[ ]:


concat_sub.head()


# In[ ]:


ncol = concat_sub.shape[1]
ncol


# In[ ]:


# get the data fields ready for stacking
concat_sub['target_mean'] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
concat_sub['target_median'] = concat_sub.iloc[:, 1:ncol].median(axis=1)
concat_sub.head()


# In[ ]:


concat_sub.describe()


# In[ ]:


# Create 1st submission for the mean with round
# concat_sub['target_mean'].round()


# In[ ]:


col2=['record_ID','units_sold']


# In[ ]:


# concat_sub['target'] = concat_sub['target_mean']

concat_sub['target'] = concat_sub['target_mean'].round()
data=concat_sub[['record_ID', 'target']]
data.columns=col2


# In[ ]:


data.head()


# In[ ]:


# data.to_csv('submission_mean.csv', index=False, float_format='%.6f')

data.to_csv('submission_mean_round.csv', index=False, float_format='%.6f')


# ### Got this 
# 
# > Your private score for this submission is : 449.7631659776047, 
# 
# > Had it been a live contest, your rank would be : 24
# 
# 
# My actual ranking in this competion is 44.. Got a improvemnt of 20. Really impressed with the results.

# In[ ]:


# Create 1st submission for the median


# In[ ]:


data_med=data


# In[ ]:


data_med['units_sold']=concat_sub['target_median']
data_med


# In[ ]:


data_med.to_csv('submission_median.csv', index=False, float_format='%.6f')


# ## Got this 
# 
# > Your private score for this submission is : 453.80995302601514, 
# > Had it been a live contest, your rank would be : 33
# 
# Remember my final standing in the competion is 44. So in both cases my ranking is improve.
# 
# - So Ensemble techniqe is working.
# 
# - Be ready to use it in upcoming hackathons.

# ### Check the prediction with the Blend of mean and median 

# In[ ]:


data['units_sold']= 1.85/3 * data['units_sold'] + 1.15/3 * data_med['units_sold']


# In[ ]:


data.to_csv('submission_mean_median_blend_2.csv', index=False, float_format='%.6f')


# No improvement ... 
