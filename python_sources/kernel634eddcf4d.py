#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv",)
test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
sample = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")


# In[ ]:


train.set_index("Id",inplace=True)
train.head()


# In[ ]:


train.shape


# In[ ]:


test.set_index("ForecastId",inplace=True)
test.head()


# # Missing values

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train["Province_State"].replace(np.nan, 'Unknown', inplace= True)


# In[ ]:


test["Province_State"].replace(np.nan, 'Unknown', inplace= True)


# In[ ]:


train["County"].replace(np.nan, 'Unknown', inplace= True)


# In[ ]:


test["County"].replace(np.nan, 'Unknown', inplace= True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# # Categorical data

# In[ ]:


train.dtypes


# In[ ]:


train['Date'] = pd.to_datetime(train.Date)


# In[ ]:


from datetime import datetime as dt
train['week'] = train['Date'].dt.week
train['month'] = train['Date'].dt.month
train['year'] = train['Date'].dt.year


# In[ ]:


train.drop("Date",axis=1,inplace=True)


# In[ ]:


test['Date'] = pd.to_datetime(test.Date)
from datetime import datetime as dt
test['week'] = test['Date'].dt.week
test['month'] = test['Date'].dt.month
test['year'] = test['Date'].dt.year
test.drop("Date",axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


cat_column  = [col for col in train.columns if train[col].dtypes == 'object']


# In[ ]:


cat_column_other = [col for col in cat_column if not col == 'Target']
cat_column_other


# In[ ]:


cat_column_target = [col for col in cat_column if col == 'Target']
cat_column_target


# In[ ]:


import category_encoders as ce
target_enc = ce.CatBoostEncoder(cols=cat_column_other)
train[cat_column_other] = target_enc.fit_transform(train[cat_column_other],train['TargetValue'])
test[cat_column_other] = target_enc.transform(test[cat_column_other])


# In[ ]:


train = pd.get_dummies(train, columns=["Target"], prefix=["Type_is"] )
test = pd.get_dummies(test, columns=["Target"], prefix=["Type_is"] )


# In[ ]:


train.head()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


for col in train.columns:
    print('plot of {} is:'.format(col))
    plt.boxplot(train[col])
    plt.show()


# In[ ]:


Q1 = train.quantile(0.07)
Q3 = train.quantile(0.93)
IQR = Q3 - Q1
print(IQR)


# In[ ]:


train.shape


# In[ ]:


train1 = train[~((train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))).any(axis=1)]
train1.shape


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# for col in cat_column:
#     train[col] = labelencoder.fit_transform(train[col])
#     test[col] = labelencoder.fit_transform(test[col])


# In[ ]:


from sklearn.preprocessing import StandardScaler
std = StandardScaler()


# In[ ]:


X = train1.drop('TargetValue', axis=1).copy()
X = std.fit_transform(X)


# In[ ]:


y = train1['TargetValue'].copy()


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.80, test_size=0.20,random_state = 0)


# In[ ]:


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# rf =  RandomForestRegressor(n_jobs=-1,verbose=1)
# rf.fit(train_X , train_y)


# In[ ]:


# prediction = rf.predict(val_X)


# In[ ]:


# from sklearn.metrics import mean_absolute_error
# val_mae = mean_absolute_error(prediction,val_y)
# print(val_mae)


# In[ ]:


# import xgboost
# xgb_model = xgboost.XGBRegressor()
# xgb_model.fit(train_X , train_y)
# pred_out = xgb_model.predict(val_X)


# In[ ]:


# from sklearn.metrics import mean_absolute_error
# val_mae = mean_absolute_error(pred_out,val_y)
# print(val_mae)


# In[ ]:


# from sklearn.neighbors import KNeighborsRegressor
# # for i in range(1,10,1):
# neigh = KNeighborsRegressor()
# neigh.fit(train_X,train_y)
# predict_n = neigh.predict(val_X)
# print('Mean absolute error: %.2f'
#      % mean_absolute_error(val_y,predict_n))


# In[ ]:


import lightgbm as lgb 
lg = lgb.LGBMRegressor()
lg = lg.fit(train_X,train_y)
predict_l = lg.predict(val_X)
print('Mean absolute error: %.2f'
     % np.sqrt(mean_absolute_error(val_y,predict_l)))


# In[ ]:


# from catboost import CatBoostRegressor

# model = CatBoostRegressor()
# #train the model
# model.fit(train_X,train_y)
# # make the prediction using the resulting model
# preds = model.predict(val_X)
# print('Mean absolute error: %.2f'
#      % np.sqrt(mean_absolute_error(val_y,preds)))


# In[ ]:





# In[ ]:


test1 = std.fit_transform(test)


# In[ ]:


predict = lg.predict(test1)


# In[ ]:


prediction_list = [int(x) for x in predict]


# In[ ]:


prediction_list


# In[ ]:


sub = pd.DataFrame({'Id': test.index , 'TargetValue': prediction_list})


# In[ ]:


sub['TargetValue'].value_counts()


# In[ ]:


p=sub.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()
q=sub.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()
r=sub.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()


# In[ ]:


p.columns = ['Id' , 'q0.05']
q.columns = ['Id' , 'q0.5']
r.columns = ['Id' , 'q0.95']


# In[ ]:


p = pd.concat([p,q['q0.5'] , r['q0.95']],1)


# In[ ]:


p['q0.05']=p['q0.05'].clip(0,10000)
p['q0.05']=p['q0.5'].clip(0,10000)
p['q0.05']=p['q0.95'].clip(0,10000)
p


# In[ ]:


sub=pd.melt(p, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])
sub['variable']=sub['variable'].str.replace("q","", regex=False)
sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']
sub['TargetValue']=sub['value']
sub=sub[['ForecastId_Quantile','TargetValue']]
sub.reset_index(drop=True,inplace=True)
sub.to_csv("submission.csv",index=False)
sub


# In[ ]:





# In[ ]:




