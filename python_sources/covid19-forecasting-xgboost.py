#!/usr/bin/env python
# coding: utf-8

# This Kerenl is forked from  https://www.kaggle.com/mielek/covid19-forecasting-xgboost. Study was done for the 4th week. Improved models parameters.

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer


# **Loading Training and Testing Data**

# In[ ]:


train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
submission_csv = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')


# **Convert String Datetime to python datetime**

# In[ ]:


train_data['Date'] = pd.to_datetime(train_data['Date'], infer_datetime_format=True)
test_data['Date'] = pd.to_datetime(test_data['Date'], infer_datetime_format=True)


# In[ ]:


train_data.loc[:, 'Date'] = train_data.Date.dt.strftime('%y%m%d')
train_data.loc[:, 'Date'] = train_data['Date'].astype(int)

test_data.loc[:, 'Date'] = test_data.Date.dt.strftime('%y%m%d')
test_data.loc[:, 'Date'] = test_data['Date'].astype(int)


# In[ ]:


train_data['Province_State'] = np.where(train_data['Province_State'] == 'nan',train_data['Country_Region'],train_data['Province_State'])
test_data['Province_State'] = np.where(test_data['Province_State'] == 'nan',test_data['Country_Region'],test_data['Province_State'])


# In[ ]:


convert_dict = {'Province_State': str}
train_data = train_data.astype(convert_dict)
test_data = test_data.astype(convert_dict)


# In[ ]:


test_data.head()


# In[ ]:


train_data.head()


# In[ ]:


sns.countplot(y="Country_Region", data=train_data,order=train_data["Country_Region"].value_counts(ascending=False).iloc[:10].index)


# In[ ]:


sns.regplot(x=train_data["ConfirmedCases"], y=train_data["Fatalities"], fit_reg=True)


# In[ ]:


sns.jointplot(x=train_data["ConfirmedCases"], y=train_data["Fatalities"],kind='scatter')


# **Label Encoding Country**

# In[ ]:


#get list of categorical variables
s = (train_data.dtypes == 'object')
object_cols = list(s[s].index)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


# **Try using Label Encoder**

# In[ ]:


object_cols


# In[ ]:


label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()

train_data['Province_State'] = label_encoder1.fit_transform(train_data['Province_State'])
test_data['Province_State'] = label_encoder1.transform(test_data['Province_State'])

train_data['Country_Region'] = label_encoder2.fit_transform(train_data['Country_Region'])
test_data['Country_Region'] = label_encoder2.transform(test_data['Country_Region'])

    


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


Test_id = test_data.ForecastId


# In[ ]:


train_data.drop(['Id'], axis=1, inplace=True)
test_data.drop('ForecastId', axis=1, inplace=True)


# **Check missing value**

# In[ ]:


missing_val_count_by_column = (train_data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column>0])


# **Make model XGBRegressor**

# In[ ]:


from xgboost import XGBRegressor


# In[ ]:


train_data.head()


# In[ ]:


X_train = train_data[['Province_State','Country_Region','Date']]
y_train = train_data[['ConfirmedCases', 'Fatalities']]


# In[ ]:


y_train_confirm = y_train.ConfirmedCases
y_train_fatality = y_train.Fatalities


# In[ ]:


x_train = X_train.iloc[:,:].values
x_test = X_train.iloc[:,:].values


# In[ ]:


model1 = XGBRegressor(n_estimators=40000,random_state = 42, max_depth=20)
model1.fit(X_train, y_train_confirm)
y_pred_confirm = model1.predict(test_data)
y_pred_confirm =np.around(y_pred_confirm,decimals = 0)


# In[ ]:


model2 = XGBRegressor(n_estimators=30000,random_state = 42, max_depth=20)
model2.fit(X_train,y_train_fatality )
y_pred_fat = model2.predict(test_data)
y_pred_fat =np.around(y_pred_fat,decimals = 0)


# **Submission**

# In[ ]:


df_sub = pd.DataFrame()
df_sub['ForecastId'] = Test_id
df_sub['ConfirmedCases'] = y_pred_confirm
df_sub['Fatalities'] = y_pred_fat
df_sub.to_csv('submission.csv', index=False)


# In[ ]:


df_sub


# In[ ]:




