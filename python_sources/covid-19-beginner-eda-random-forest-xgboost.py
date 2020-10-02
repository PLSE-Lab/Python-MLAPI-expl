#!/usr/bin/env python
# coding: utf-8

# # Beginner Attempt:
# <p>
# This is an attempt at performing some EDA and predicting ConfirmedCases and Fatalities using a RandomForest, XGBoost Regressor model. <br>
#     There is overlapping Data in the Test and Train sets, predicting without the overlap predictably results in a lower score. <br> 
#     Will appreciate any advice.
#     

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
np.set_printoptions(threshold=sys.maxsize)


# In[ ]:


path_train = 'covid19-global-forecasting-week-1/train-2.csv'
path_test = 'covid19-global-forecasting-week-1/test.csv'
path_sbumit = 'covid19-global-forecasting-week-1/submission.csv'

train_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/train.csv'
test_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/test.csv'
submit_kaggle = '/kaggle/input/covid19-global-forecasting-week-1/submission.csv'

df_train = pd.read_csv(train_kaggle)
df_test = pd.read_csv(test_kaggle)


# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# ## EDA

# ## About the Data 
# 1. Contains Daily Reports of Number of Cases and Fatalities for countries.
# 2. [Missing Data]Contains some entries with Province/State Information Missing - Dropped.
# 3. Contains latitude and longitude for entries, Can Plot on map.
# 4. Date - 22nd Feb to 23nd March. (Getting Updated Continuosly)
# 5. Country/Region - 163

# In[ ]:


# Dataset Dimesnions
print('Train shape', df_train.shape)
print('Test shape', df_test.shape)
# Missing/Null Values
print('\nTrain Missing\n', df_train.isnull().sum())
print('\nTest Missing\n', df_test.isnull().sum())


# ### Unique countries in the dataset 

# In[ ]:


lst = df_train['Country/Region'].unique()
print('Total_Countries\n:', len(lst))
for i in lst:
    print(i)


# ### Date Range for the Dataset 

# In[ ]:


print(df_train['Date'].min(), ' - ', df_train['Date'].max())


# ### Checking Daily Worldwide Confirmed Cases and Fatalities 

# In[ ]:


# GroupBy syntax (columns to group by in list)[Columns to aggregate, apply function to] . aggregation functions on it 
train_cases_conf = df_train.groupby(['Date'])['ConfirmedCases'].sum()
train_cases_conf


# In[ ]:


train_cases_conf.plot(figsize = (10,8), title = 'Worldwide Confirmed Cases')


# In[ ]:


train_fatal = df_train.groupby(['Date'])['Fatalities'].sum()
train_fatal


# In[ ]:


train_fatal.plot(figsize = (10,8), title = 'Worldwide Fatalaties')


# ### Check Confirmed cases and fatalities for a country 
# scale = "linear", "log"

# In[ ]:


def country_stats(country, df):
    country_filt = (df['Country/Region'] == country)
    df_cases = df.loc[country_filt].groupby(['Date'])['ConfirmedCases'].sum()
    df_fatal = df.loc[country_filt].groupby(['Date'])['Fatalities'].sum()
    fig, axes = plt.subplots(nrows = 2, ncols= 1, figsize=(15,15))
    df_cases.plot(ax = axes[0])
    df_fatal.plot(ax = axes[1])
    
country_stats('US', df_train)


# #### Fatalities and Confirmed Cases by Country (Log Scale)

# In[ ]:


# grouping using same Country filter to get fatalities on each date (grouped by date)
# groupby([list of columns to groupby]) [which columns to apply aggregate fx to ]. (aggregate function)
# To Do - Fix Ticks 

def country_stats_log(country, df):
    count_filt =(df_train['Country/Region'] == country)
    df_count_case = df_train.loc[count_filt].groupby(['Date'])['ConfirmedCases'].sum()
    df_count_fatal = df_train.loc[count_filt].groupby(['Date'])['Fatalities'].sum()
    plt.figure(figsize=(15,10))
    plt.axes(yscale = 'log')
    plt.plot(df_count_case.index, df_count_case.tolist(), 'b', label = country +' Total Confirmed Cases')
    plt.plot(df_count_fatal.index, df_count_fatal.tolist(), 'r', label = country +' Total Fatalities')
    plt.title(country +' COVID Cases and Fatalities (Log Scale)')
    plt.legend()
    

country_stats_log('US', df_train)


# ###  Most Affected Countries

# In[ ]:


# as_index = False to not make the grouping column the index, creates a df here instead of series, preserves
# Confirmedcases column

train_case_country = df_train.groupby(['Country/Region'], as_index=False)['ConfirmedCases'].max()

# Sorting by number of cases
train_case_country.sort_values('ConfirmedCases', ascending=False, inplace = True)
train_case_country


# In[ ]:


plt.figure(figsize=(8,6))
plt.bar(train_case_country['Country/Region'][:5], train_case_country['ConfirmedCases'][:5], color = ['red', 'yellow','black','blue','green'])


# #### No. of Cases on a Particular Day, (Not Increase, Cumulative)

# In[ ]:


# Confirmed Cases till a particular day by country

def case_day_country (Date, df):
    df = df.groupby(['Country/Region', 'Date'], as_index = False)['ConfirmedCases'].sum()
    date_filter = (df['Date'] == Date)
    df = df.loc[date_filter]
    df.sort_values('ConfirmedCases', ascending = False, inplace = True)
    sns.catplot(x = 'Country/Region', y = 'ConfirmedCases' , data = df.head(10), height=5,aspect=3, kind = 'bar')
    
    
case_day_country('2020-03-23', df_train)


# # Data Wrangling/ Pre-processing/ Cleaning 
# 1. Identifying and Handling missing values.
# 2. Data Formating.
# 3. Data Normalization(centering and scaling).
# 4. Data bining.
# 5. Turning categorical values into numerical values.

# ### Need to Exclude Leaky Data, the same Dates are in both train and test set.
# 1. First convert object to python datetime type <br>
# Using pd.to_datetime() <br>
# Check Getting converted to float, because haven't converted to date before comparison, still object.

# In[ ]:


df_train.Date = pd.to_datetime(df_train['Date'])
print(df_train['Date'].max())
print(df_test['Date'].min())


# ### Truncate df_train by date < df_test['Date'].min()

# In[ ]:


date_filter = df_train['Date'] < df_test['Date'].min()
df_train = df_train.loc[date_filter]


# In[ ]:


# Dropping ID and getting rid of Province/State with NULL values 
df_train.info()


# In[ ]:


# lets get Cumulative sum of ConfirmedCases and Fatalities for each country on each data (same as original data)
# Doing to create copy without ID and 

train_country_date = df_train.groupby(['Country/Region', 'Date', 'Lat', 'Long'], as_index=False)['ConfirmedCases', 'Fatalities'].sum()


# In[ ]:


print(train_country_date.info())
print(train_country_date.isnull().sum())


# ### Feature Engineering
# Splitting Date into day, month, day of week. <br>
# Check if Date is in python datetime format. Else, convert object to python datetime type <br>
# Using pd.to_datetime()

# In[ ]:


train_country_date.info()


# #### Using Pandas Series.dt.month
# The month as January=1, December=12.

# In[ ]:


# Adding day, month, day of week columns 

train_country_date['Month'] = train_country_date['Date'].dt.month
train_country_date['Day'] = train_country_date['Date'].dt.day
train_country_date['Day_Week'] = train_country_date['Date'].dt.dayofweek
train_country_date['quarter'] = train_country_date['Date'].dt.quarter
train_country_date['dayofyear'] = train_country_date['Date'].dt.dayofyear
train_country_date['weekofyear'] = train_country_date['Date'].dt.weekofyear


# In[ ]:


train_country_date.head()


# In[ ]:


train_country_date.info()


# #### Same Feature Engineering for Test Set

# In[ ]:


# First drop Province/State
df_test.drop('Province/State', axis = 1, inplace = True)

# Converting Date Object to Datetime type

df_test.Date = pd.to_datetime(df_test['Date'])
df_test.head(2)


# In[ ]:


# adding Month, DAy, Day_week columns Using Pandas Series.dt.month

df_test['Month'] = df_test['Date'].dt.month
df_test['Day'] = df_test['Date'].dt.day
df_test['Day_Week'] = df_test['Date'].dt.dayofweek
df_test['quarter'] = df_test['Date'].dt.quarter
df_test['dayofyear'] = df_test['Date'].dt.dayofyear
df_test['weekofyear'] = df_test['Date'].dt.weekofyear


# In[ ]:


df_test.info()


# #### Councatenating Train-Test to Label encode Country/Region Categorical Variable.
# 1. Make copy of train data without Confirmed Cases and Fatalities. Index - 0 to 17608
# 2. Concatenate train, test.
# 3. Label Encode Countries.
# 4. Add back Cofirmed Cases, Fatalities columns to clean_train_data.
# 5. Modelling
# 6. Saving Predicted Values with ForecastID

# In[ ]:


# train_country_date
# df_test
# Lets select the Common Labels and concatenate.

labels = ['Country/Region', 'Lat', 'Long', 'Date', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear']

df_train_clean = train_country_date[labels]
df_test_clean = df_test[labels]

data_clean = pd.concat([df_train_clean, df_test_clean], axis = 0)


# In[ ]:


data_clean.info()


# ## Preparing Data For Models - LabelEncode Country

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


# Label Encoder for Countries 

enc = LabelEncoder()
data_clean['Country'] = enc.fit_transform(data_clean['Country/Region'])
data_clean


# In[ ]:


# Dropping Country/Region and Date

data_clean.drop(['Country/Region', 'Date'], axis = 1, inplace=True)


# ### Splitting Back into Train and Test

# In[ ]:


index_split = df_train.shape[0]
data_train_clean = data_clean[:index_split]


# In[ ]:


data_test_clean = data_clean[index_split:]


# ### Adding Back Confirmed Cases and Fatalities
# Using original df_train, check shape is same, head, tail have same values. ORDER NEEDS TO BE SAME.

# In[ ]:


data_train_clean.tail(5)


# ### Creating Features and Two Labels

# In[ ]:


x = data_train_clean[['Lat', 'Long', 'Month', 'Day', 'Day_Week','quarter', 'dayofyear', 'weekofyear', 'Country']]
y_case = df_train['ConfirmedCases']
y_fatal = df_train['Fatalities']


# ### Train-Test Split - Confirmed Cases

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y_case, test_size = 0.3, random_state = 42)


# ### Train-Test Split - Fatalities

# In[ ]:


from sklearn.model_selection import train_test_split

x_train_fatal, x_test_fatal, y_train_fatal, y_test_fatal = train_test_split(x, y_fatal, test_size = 0.3, random_state = 42)


# ## Modeling - Regression Problem 
# Using features Country/Region, Lat, Long, Month, Day, Day_week, quarter, dayofyear, weekofyear.<br>
# To predict ConfirmedCases, Fatalities.
# ### To predict 2 Different Target Variables, Train two classifiers, one for each.

# # Modelling
# 1. Linear Regression - Worse than baseline model. 
# 2. Logistic Regression (Will need GridSearchCV for Max_iter) - Absolute Trash.
# 3. Polynomial Regression - Not Tried
# 4. SVM Regressor - Very bad performance with a poly kernel and some variation of c and eta. (read up more)
# 4. RandomForest Regressor - Gives 1.7 RMSE, With data leak removed gives - 2.18417 RMSE.
# 5. GradientBoost Regressor - Gives slightly lower performance than RF

# ## 3. RandomForest Regressor
# <p> With Leaky Data - Train MSE 284698.84113318456 <br>
# Submission RMSLE - 1.70407 </p>
# <p> Without Leaky Data - Test MSE 291078.15156607644 <br>
#     Submission RMSLE - 2.18417 </p>

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# #### For ConfirmedCases

# In[ ]:


rf = RandomForestRegressor(n_estimators =100)
rf.fit(x_train, y_train.values.ravel())


# In[ ]:


rf.score(x_train, y_train)


# In[ ]:


rf.score(x_test, y_test)


# In[ ]:


# Predicted Values and MSE
y_pred_train = rf.predict(x_train)
print(mean_squared_error(y_train, y_pred_train))


# In[ ]:


# Training on entire set and predict values.

rf.fit(x, y_case.values.ravel())


# In[ ]:


# Predicted ConfirmedCases
rf_pred_case = rf.predict(data_test_clean)


# In[ ]:


plt.figure(figsize=(15,8))
plt.plot(rf_pred_case)


# #### For Fatalities

# In[ ]:


rf.fit(x, y_fatal.values.ravel())


# In[ ]:


rf_pred_fatal = rf.predict(data_test_clean)


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot(rf_pred_fatal)


# In[ ]:


# Saving to Submission.csv

#submission = pd.read_csv(path_sbumit)
#submission['ConfirmedCases'] = rf_pred_case
#submission['Fatalities'] = rf_pred_fatal

#submission.to_csv('submission.csv', index = False)


# ## 5. XGBoost Regressor
# <p> With Leaky Data - Train MSE <br>
# Submission RMSLE -  </p>
# <p> Without Leaky Data - Train MSE 10064.67200159855, 4.047602533022124 <br>
#     Submission RMSLE - 2.27873 </p>

# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[ ]:


reg = xgb.XGBRegressor(n_estimators=1000)


# #### For ConfirmedCases

# In[ ]:


reg.fit(x_train, y_train)


# In[ ]:


reg.score(x_train, y_train)


# In[ ]:


reg_y_pred = reg.predict(x_train)


# In[ ]:


mean_squared_error(y_train, reg_y_pred)


# In[ ]:


reg.score(x_test, y_test)


# In[ ]:


# Slightly Better than Random Forest 
reg_y_test_pred = reg.predict(x_test)
mean_squared_error(y_test, reg_y_test_pred)


# ### Visualising predictions error on entire train set

# In[ ]:


reg.fit(x, y_case)


# In[ ]:


y_train_pred = reg.predict(x)


# In[ ]:


plt.plot(y_case)


# In[ ]:


plt.plot(y_train_pred)


# In[ ]:


mean_squared_error(y_case, y_train_pred)


# In[ ]:


xgb_pred_case = reg.predict(data_test_clean)


# In[ ]:


plt.plot(xgb_pred_case)


# #### For Fatalities

# In[ ]:


reg.fit(x, y_fatal)


# In[ ]:


# Checking MSE for Fatalities

print(mean_squared_error(y_fatal, reg.predict(x)))


# In[ ]:


plt.plot(reg.predict(x))


# In[ ]:


plt.plot(y_fatal)


# #### Predict on Test Set

# In[ ]:


xgb_pred_fatal = reg.predict(data_test_clean)


# In[ ]:


plt.plot(xgb_pred_fatal)


# In[ ]:


# Saving to Submission.csv

submission = pd.read_csv(submit_kaggle)
submission['ConfirmedCases'] = xgb_pred_case
submission['Fatalities'] = xgb_pred_fatal

submission.to_csv('submission.csv', index = False)

