#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd   #data pre-processing
import numpy as np    #mathematical operation
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from math import sqrt #mathematical functions

from sklearn.model_selection import train_test_split #spliting data
from sklearn.preprocessing import StandardScaler,MinMaxScaler  #scaling data
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error #model performace checking

from sklearn.linear_model import LinearRegression #linear regression 
from sklearn.ensemble import RandomForestRegressor#randomforest regression
import xgboost as Xgb                             #Xgboost regession

import warnings
warnings.filterwarnings("ignore") #to ignore warnings


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


data = pd.read_csv("../input/car-price/train_data.csv")


# In[ ]:


# data["Global_Sales_Sub-Segment_Brand"] = data["Global_Sales_Sub-Segment"]+"_"+data["Brand"]


# In[ ]:


# data["year_cal"] =  data["year"] - data["Generation_Year"]


# In[ ]:


# data = data[data["year_cal"]>0]


# In[ ]:


data.drop(["year","Generation_Year"],axis=1,inplace=True)


# In[ ]:


data.describe()


# In[ ]:


data.drop(["Nameplate","vehicle_id","date"],inplace=True,axis=1)


# In[ ]:


numeric=data.select_dtypes(include=['float64','int64'])
categorical = data.select_dtypes(include=['object'])


# In[ ]:


numeric.columns


# In[ ]:


numeric.describe()


# In[ ]:


# #LINEARITY CHECK>>>#to check price has linear relation or not with Indep. var's
# for i, col in enumerate (numeric.columns):
#     plt.figure(i)
#     sns.regplot(x=data[col],y=data['Price_USD'])


# In[ ]:


# np.percentile(data["Price_USD"],100)


# In[ ]:


# data = data[data["Price_USD"] < np.percentile(data["Price_USD"],100)]


# In[ ]:


#LINEARITY CHECK>>>#to check price has linear relation or not with Indep. var's

for i, col in enumerate (numeric.columns):
    plt.figure(i)
    sns.regplot(x=data[col],y=data['Price_USD'])


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(24,24))

for i, column in enumerate(numeric.columns):
    sns.distplot(data[column],ax=axes[i//3,i%3])


# In[ ]:


# def remove_outlier(df_in, col_name):
#     q1 = df_in[col_name].quantile(0.25)
#     q3 = df_in[col_name].quantile(0.75)
#     iqr = q3-q1 #Interquartile range
#     fence_low  = q1-(1.5*iqr)
#     fence_high = q3+(1.5*iqr)
#     df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
#     return df_out


# In[ ]:


# numeric.drop(["Price_USD"],axis=1,inplace=True)


# In[ ]:


# for i, col in enumerate (numeric.columns):
#     print(col)
#     data = remove_outlier(data,col)


# In[ ]:


# data["Fuel_Type"].value_counts()


# In[ ]:


#LINEARITY CHECK>>>#to check price has linear relation or not with Indep. var's

for i, col in enumerate (numeric.columns):
    plt.figure(i)
    sns.regplot(x=data[col],y=data['Price_USD'])


# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=4, ncols=3,figsize=(24,24))

for i, column in enumerate(numeric.columns):
    sns.distplot(data[column],ax=axes[i//3,i%3])


# In[ ]:


numeric=data.select_dtypes(include=['float64','int64'])
categorical = data.select_dtypes(include=['object'])


# In[ ]:


corr=numeric.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap="YlGnBu")


# In[ ]:


#Correlation with output variable
cor_target = abs(corr["Price_USD"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features


# In[ ]:


data=data.drop(columns=categorical)
data.head(2)


# In[ ]:


data.columns


# In[ ]:


data = np.log(data)


# In[ ]:


y = data["Price_USD"]


# In[ ]:


data.drop(["Fuel_cons_combined","Price_USD","Length"],axis=1,inplace=True)


# In[ ]:


categorical.columns


# In[ ]:


# categorical.drop(["Fuel_Type"],axis=1,inplace=True)


# In[ ]:


categorical["Fuel_Type"].value_counts()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


ohe = OneHotEncoder(handle_unknown = 'ignore')
c = ohe.fit_transform(categorical).toarray()


# In[ ]:


ohe.categories_


# In[ ]:


categorical.columns


# In[ ]:


data.values


# In[ ]:


np.concatenate((data.values,c),axis=1).shape


# In[ ]:


X=np.concatenate((data.values,c),axis=1)


# In[ ]:


y


# In[ ]:


# #Min-Max Scaling of data range 0 to 1
# ms = MinMaxScaler()
# msy =  MinMaxScaler()
# dfX_scaled = ms.fit_transform(X.values)
# # y =  msy.fit_transform(y.values)


# In[ ]:


# from sklearn.preprocessing import scale

# cols=X.columns
# dfX_scaled=pd.DataFrame(scale(X))
# dfX_scaled.columns=cols
# dfX_scaled.columns


# In[ ]:


# split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=101)


# In[ ]:


#function to calculate mean absolute percentile error 
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


# lr = LinearRegression()
# lr.fit(X_train,y_train)
# pred = lr.predict(X_test)
# print("Mse:",mean_squared_error(y_test,pred))
# rms = sqrt(mean_squared_error(y_test,pred))
# print("Rmse:",rms)
# print("Mape:",mean_absolute_percentage_error(y_test,pred))
# print("R-square:",r2_score(y_test,pred))


# In[ ]:


# rf = RandomForestRegressor(random_state=101)
# rf.fit(X_train,y_train)
# pred = rf.predict(X_test)
# print("Mse:",mean_squared_error(y_test,pred))
# rms = sqrt(mean_squared_error(y_test,pred))
# print("Rmse:",rms)
# print("Mape:",mean_absolute_percentage_error(y_test,pred))
# print("R-square:",r2_score(y_test,pred))


# In[ ]:


# import xgboost as Xgb
# xgb = Xgb.XGBRegressor(random_state=101,learning_rate=0.3)
# xgb = xgb.fit(X_train,y_train)
# pred = xgb.predict(X_test)
# print("Mse:",mean_squared_error(y_test,pred))
# rms = sqrt(mean_squared_error(y_test,pred))
# print("Rmse:",rms)
# print("Mape:",mean_absolute_percentage_error(y_test,pred))
# print("R-square:",r2_score(y_test,pred))


# In[ ]:


import xgboost as Xgb
xgb = Xgb.XGBRegressor(random_state=101,learning_rate=0.3)
xgb = xgb.fit(X, y)
pred = xgb.predict(X)
print("Mse:",mean_squared_error(y,pred))
rms = sqrt(mean_squared_error(y,pred))
print("Rmse:",rms)
# print("Mape:",mean_absolute_percentage_error(y,pred))
print("R-square:",r2_score(y,pred))


# In[ ]:


# rf = RandomForestRegressor(random_state=101)
# rf = rf.fit(X, y)
# pred = rf.predict(X)
# print("Mse:",mean_squared_error(y,pred))
# rms = sqrt(mean_squared_error(y,pred))
# print("Rmse:",rms)
# # print("Mape:",mean_absolute_percentage_error(y,pred))
# print("R-square:",r2_score(y,pred))


# In[ ]:


test = pd.read_csv("../input/car-price/oos_data.csv")


# In[ ]:


test.columns


# In[ ]:


# test["year_cal"] =  test["year"] ,:""- test["Generation_Year"]
test.drop(["year","Generation_Year","Length"],axis=1,inplace=True)


# In[ ]:


# test["Global_Sales_Sub-Segment_Brand"] = test["Global_Sales_Sub-Segment"]+"_"+test["Brand"]
test.drop(["Nameplate","vehicle_id","date"],inplace=True,axis=1)
numeric=test.select_dtypes(include=['float64','int64'])
categorical = test.select_dtypes(include=['object'])


# In[ ]:


# test.drop(["year","Generation_Year"],axis=1,inplace=True)


# In[ ]:


test.drop(["Fuel_cons_combined"],axis=1,inplace=True)


# In[ ]:


test.columns


# In[ ]:


data.columns


# In[ ]:


categorical.columns


# In[ ]:


test=test.drop(columns=categorical)


# In[ ]:


test.columns


# In[ ]:


test = np.log(test)


# In[ ]:


len(c[0])


# In[ ]:


t = ohe.transform(categorical).toarray()


# In[ ]:


test.shape


# In[ ]:


test  = np.concatenate((test.values,t),axis=1)


# In[ ]:


X.shape


# In[ ]:


test.shape


# In[ ]:


# cols=test.columns
# test_scaled=pd.DataFrame(scale(test))
# test_scaled.columns=cols
# test_scaled.columns


# In[ ]:


vehicle_id = pd.read_csv("../input/car-price/oos_data.csv",usecols=["vehicle_id"])


# In[ ]:


vehicle_id["Price_USD"] = xgb.predict(test)
# vehicle_id["Price_USD"] = msy.inverse_transform(vehicle_id[["Price_USD"]].values)


# In[ ]:


vehicle_id["Price_USD"] = np.exp(vehicle_id["Price_USD"])


# In[ ]:


vehicle_id.to_csv("Global_Submission_f_xgb::BrandXlength_gear_ohe.csv",index=False)


# In[ ]:


vehicle_id

