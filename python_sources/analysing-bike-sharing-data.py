#!/usr/bin/env python
# coding: utf-8

# # Analysing Bike-Sharing Data

# **Bike sharing data from Washington DC downoaded from UCI Machine Learning Repository**
# 
# The goal is to predict the usage or total count from the daily and hourly data. 
# This notebook describes the steps taken to clean and analyse the provided datasets. 
# 
# Key questions to be answered with the help of the analysis:
# 
# - Is an hourly predication rational and necessary?
# - Is the daily predication enough?
# - Can we spot anomalies based on uncommon weather conditions? 
# - What are other helpful features that can be collected and what can they be used for?

# ## 1. Importing libraries and reading data sets

# In[ ]:


# importing librarys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor





# In[ ]:


# Laoding daily data set
bike_day = pd.read_csv('../input/day.csv')


# In[ ]:


# expand number of columns
pd.set_option('display.max_columns', 30)
# Viewing day data
bike_day.head()


# **Viewing data types**

# In[ ]:


# View shape of sets
print(bike_day.dtypes)


# ## 2. Cleaning Daily Data
# 
# - Columns will be renamed. 
# - "season","month","holiday","workingday" and "weather" should be of "categorical" data type.In the following these are converted.
# - We also keep the numerical category for mixed linear model analysis
# - Note: mistake in Read_me.txt in the asignment of the seasons: Winter =1, Spring = 2, Summer = 3; Fall = 4;

# In[ ]:


# Renaming Columns

bike_day.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'weathersit':'weather',
                        'mnth':'month',
                        'hum':'humidity',
                        'cnt':'total_cnt'},inplace=True)

import calendar
from datetime import datetime

# Creating new Columns

bike_day["weekday"] = bike_day.datetime.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
bike_day["month"] = bike_day.datetime.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
bike_day["season_num"] = bike_day.season 
bike_day["season"] = bike_day.season.map({1: 'Winter', 2 : 'Spring', 3 : 'Summer', 4 : 'Fall' })
bike_day["weather_num"] = bike_day.weather
bike_day["weather"] = bike_day.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ",                                         3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",                                         4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })

bike_day['weekday_num'] = bike_day.weekday.map({'Monday': 1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday': 7})
bike_day['month_num'] = bike_day.month.map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June' : 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})



# defining categorical variables

bike_day['season'] = bike_day.season.astype('category')
bike_day['holiday'] = bike_day.holiday.astype('category')
bike_day['weekday'] = bike_day.weekday.astype('category')
bike_day['weather'] = bike_day.weather.astype('category')
bike_day['month'] = bike_day.month.astype('category')
bike_day['workingday'] = bike_day.workingday.astype('category')



# In[ ]:


bike_day.head()


# In[ ]:


# checking data set once more 

print(bike_day.dtypes)


# ### 2.1 Checking for missing values

# **1. Create a matrix with values = False and NaN = True**
# 
# **2. Find the true values**
# 
# **3. Substitute with average (imputing)**
# 

# In[ ]:


bike_day.isnull().any()


# There are no missing values

# ## 3. Visulaizing Data: Plotting Features vs. Daily Count

# ### 3.1 Checking for outliers 
# 
# - plotting boxplots

# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=bike_day,y="total_cnt",x="month",orient="v",ax=axes[0][0])
sns.boxplot(data=bike_day,y="total_cnt",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=bike_day,y="total_cnt",x="weekday",orient="v",ax=axes[1][0])
sns.boxplot(data=bike_day,y="total_cnt",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Total Count vs. Month")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Total Count vs. Season")
axes[1][0].set(xlabel='Weekday', ylabel='Count',title="Total Count vs. Weekday")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Total Count vs. Working Day")


# **No outliers found in the daily data set. In the case of outliers, see 3.2 Removing outliers**

# ### 3.2 Removing outliers 
# 
# If there were and any outliers this is how to remove them:

# In[ ]:


bike_day_WithoutOutliers = bike_day[np.abs(bike_day["total_cnt"]-bike_day["total_cnt"].mean())<=(3*bike_day["total_cnt"].std())] 

print ("Shape Of The Before Ouliers: ",bike_day.shape)
print ("Shape Of The After Ouliers: ",bike_day_WithoutOutliers.shape)


# **There aren't any outliers that were removed**

# ### 3.3 Checking for correlations between count and numerical features
# 
# A fast way of identifying correlations between features is with a correlation matrix:

# In[ ]:


corrMatt = bike_day[['temp','atemp','windspeed','humidity','registered','casual','total_cnt']].corr()

mask = np.array(corrMatt)
# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=0.7, 
            square=True,
            annot=True,
            cmap="YlGnBu")
fig.set_size_inches(8,10)


# Positive correlations are displayed in blue and negaive correlations in yellow. 
# To visually understand correlations, plotting each feature with respect to total count can be helpful for trend recognition. 

# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=3)
fig.set_size_inches(20, 13)
sns.scatterplot(data=bike_day,y="total_cnt",x="temp",ax=axes[0][0])
sns.scatterplot(data=bike_day,y="total_cnt",x="humidity",ax=axes[0][1])
sns.scatterplot(data=bike_day,y="total_cnt",x="windspeed",ax=axes[0][2])
sns.scatterplot(data=bike_day,y="total_cnt",x="month",ax=axes[1][0])
sns.barplot(data=bike_day,y="total_cnt",x="season",ax=axes[1][1])
sns.scatterplot(data=bike_day,y="total_cnt",x="weather_num",ax=axes[1][2])



axes[0][0].set(xlabel='Temp Norm.',ylabel='Count',title="Total Count vs. Temp")
axes[0][1].set(xlabel='Humidity', ylabel='Count',title="Total Count vs. Humidity")
axes[0][2].set(xlabel='Windspeed', ylabel='Count',title="Total Count vs. Windspeed")
axes[1][0].set(xlabel=' ', ylabel='Count',title="Total Count vs. Weekday")
axes[1][1].set(xlabel=' ', ylabel='Count',title="Total Count vs. Working Day")
axes[1][2].set(xlabel='Weather Condition', ylabel='Count',title="Total Count vs. Working Day")


# #### Is the data normaly distributed?

# In[ ]:


fig,axes = plt.subplots(ncols=2,nrows=1)
fig.set_size_inches(12, 8)
sns.distplot(bike_day["total_cnt"],ax=axes[0]) # was macht distplot?
stats.probplot(bike_day["total_cnt"], dist='norm', fit=True, plot=axes[1])
#sns.distplot(np.log(bike_day_WithoutOutliers["total_cnt"]),ax=axes[1][0])
#stats.probplot(np.log1p(bike_day_WithoutOutliers["total_cnt"]), dist='norm', fit=True, plot=axes[1][1])


# **This check is needed, as most of the machine learning models are tailor-made for working with Normally distributed data sets.**
# - Data seems to be following a normal distribution and is not skewed.

# ## 4. Linear mixed model for daily data

# ### 4.1 Linear Regression Model: total count vs. temp, atemp, windspeed, humidity
# 
# Is there a linear dependence between total count and the features temp, atemp, windspeed, humidity?

# In[ ]:


import statsmodels.api as sm

#fit the model
regressors = bike_day[['temp','atemp', 'windspeed', 'humidity']] 
reg_const = sm.add_constant(regressors)
mod = sm.OLS(bike_day['total_cnt'], reg_const)
res = mod.fit()
#print the summary
print(res.summary())


# - coefficient = beta (slope)
# - const. coeff = beta_0 (intercept)
# - std error = diviation from average
# - P-value (derived from t-value = how insiginificant is my value)
# 
# - R-squared << 1, this means either no linearity or random effects in data + huge error (residual not 0)
# - There is a significance in the const. value because P = 0, the same goes for humidity and windspeed. 

# ### 4.2 Mixed Linear Model: Total count vs. atemp, windspeed, humidity with random effect of month
# 
# A more sophisticated model to allow both fixed and random effects. Here random effects within and between months are considered, as the month group gives a finer data increment (no. of groups = 12), compared to season and weather (no. of groups = 4) and the groups are on average of the same sizes. This is not the case for instance when considering weather conditions, as less data is available on weather condition 4 for example. It would result in a higher std. error.
# 
# In addition, only atemp is taken into consideration. 

# In[ ]:


import statsmodels.formula.api as smf

#fit the model
mixed = smf.mixedlm("total_cnt ~ atemp+windspeed+humidity", bike_day, groups = bike_day['month_num'],re_formula='~windspeed+humidity')
mixed_fit = mixed.fit()
#print the summary
print(mixed_fit.summary())


# - with temp as a feature,  model doesnt converge. Therefore atemp is used here
# - All fixed values (coef.) strongly significant as seen from P>z = 0
# - Confidence intervals contain no 0 values (different from 0)
# - Group, windspeed and humidity var are variances. 
# - Random effects can be extracted from co-variances (Cov)
# - Correlations can be extracted from co-var/var matrix: this shows weak correlations (negative var) between group and windspeed, group and humidity, group and windspeed+humidity, where the group is month. This is further indicated by the difference to the coef. 
# 
# 
# 

# # Analysing Hourly Data

# ## 5. Loading and Visulaizing Data

# In[ ]:


# load data
bike_hour = pd.read_csv('../input/hour.csv')


# In[ ]:


# Renaming Columns

bike_hour.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'weathersit':'weather',
                        'mnth':'month',
                        'hr':'hour',
                        'hum':'humidity',
                        'cnt':'total_cnt'},inplace=True)

import calendar
from datetime import datetime

# Creating new Columns

bike_hour["weekday"] = bike_hour.datetime.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
bike_hour["month"] = bike_hour.datetime.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])
bike_hour["season_num"] = bike_hour.season
bike_hour["season"] = bike_hour.season.map({1: 'Winter', 2 : 'Spring', 3 : 'Summer', 4 : 'Fall' })
bike_hour["weather_num"] = bike_hour.weather
bike_hour["weather"] = bike_hour.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ",                                         3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",                                         4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })



bike_hour['weekday_num'] = bike_hour.weekday.map({'Monday': 1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday': 7})
bike_hour['month_num'] = bike_hour.month.map({'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June' : 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12})



# defining categorical variables

bike_hour['season'] = bike_hour.season.astype('category')
bike_hour['holiday'] = bike_hour.holiday.astype('category')
bike_hour['weekday'] = bike_hour.weekday.astype('category')
bike_hour['weather'] = bike_hour.weather.astype('category')
bike_hour['month'] = bike_hour.month.astype('category')
bike_hour['workingday'] = bike_hour.workingday.astype('category')




# In[ ]:


bike_hour.head()


# In[ ]:


bike_hour.isnull().any()


# There are no missing data points

# ### 5.1 Checking for Outliers

# In[ ]:


fig, axes = plt.subplots(nrows=2,ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=bike_hour,y="total_cnt",x="month",orient="v",ax=axes[0][0])
sns.boxplot(data=bike_hour,y="total_cnt",x="season",orient="v",ax=axes[0][1])
sns.boxplot(data=bike_hour,y="total_cnt",x="weekday",orient="v",ax=axes[1][0])
sns.boxplot(data=bike_hour,y="total_cnt",x="workingday",orient="v",ax=axes[1][1])

axes[0][0].set(ylabel='Count',title="Total Count vs. Month")
axes[0][1].set(xlabel='Season', ylabel='Count',title="Total Count vs. Season")
axes[1][0].set(xlabel='Weekday', ylabel='Count',title="Total Count vs. Weekday")
axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Total Count vs. Working Day")


# **There are several outliers, that will be removed.**

# In[ ]:


bike_hour_WithoutOutliers = bike_hour[np.abs(bike_hour["total_cnt"]-bike_hour["total_cnt"].mean())<=(3*bike_hour["total_cnt"].std())] 

# how many outliers are removed?
print ("Shape Of The Before Ouliers: ",bike_hour.shape)
print ("Shape Of The After Ouliers: ",bike_hour_WithoutOutliers.shape)


# **Removing the outliers here is ok, as it amount to only 1.4% of the total data set.
# This step needs to be carefully considered, because valuable information can be contained in outliers. Anomalies can be gained from outlier information.** 

# ### 5.2 Is the data normally dirstributed? 

# In[ ]:


fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(12, 10)
sns.distplot(np.log(bike_hour["total_cnt"]),ax=axes[0][0])
stats.probplot(bike_hour["total_cnt"], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(bike_hour_WithoutOutliers["total_cnt"]),ax=axes[1][0])
stats.probplot((bike_hour_WithoutOutliers["total_cnt"]), dist='norm', fit=True, plot=axes[1][1])

axes[0][0].set(xlabel='log(Count)',title="With Outliers")
axes[1][0].set(xlabel='log(Count)',title="Without Outliers")


# **Compared to the daily data, the hourly data set is not entirely normally distributed and is left skewd. Caution when using ML models for predictions.**

# In[ ]:


# renaming bike_hour_WithoutOutliers to bike_hour
bike_hour = bike_hour_WithoutOutliers
print ("Shape Of Data after Cleaning: ",bike_hour.shape)


# ### 5.3 Plotting rest of data
# 

# In[ ]:


fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(8,15)
sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# Plotting average count vs. month
month_h_Aggregated = pd.DataFrame(bike_hour.groupby("month")["total_cnt"].mean()).reset_index()
month_h_Sorted = month_h_Aggregated.sort_values(by="total_cnt",ascending=False)
sns.barplot(data=month_h_Sorted,x="month",y="total_cnt",ax=ax1,order=sortOrder)
ax1.set(xlabel='Month', ylabel='Count',title="Count vs. Month")

# Plotting average count vs. season
hour_h_Aggregated = pd.DataFrame(bike_hour.groupby(["hour","season"],sort=True)["total_cnt"].mean()).reset_index()
sns.pointplot(x=hour_h_Aggregated["hour"], y=hour_h_Aggregated["total_cnt"],hue=hour_h_Aggregated["season"], data=hour_h_Aggregated, join=True,ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Count',title="Hourly Count vs. Season",label='big')

# Plotting average count vs. weekday
weekday_h_Aggregated = pd.DataFrame(bike_hour.groupby(["hour","weekday"],sort=True)["total_cnt"].mean()).reset_index()
sns.pointplot(x=weekday_h_Aggregated["hour"], y=weekday_h_Aggregated["total_cnt"],hue=weekday_h_Aggregated["weekday"], data=weekday_h_Aggregated, join=True,ax=ax3)
ax3.set(xlabel='Hour', ylabel='Count',title="Count vs. Weekday",label='big')


# There is a clear monthly trend (upper graph): during colder months the count is lower compared to warmer months. This is also seen in the seasonal category (center graph). On average the rental behaviour during spring, summer and fall is the same. A significant decrease is observed in winter. The hourly trend shows that peak times are between 7-9 am, when people drive to work and 4-7 pm when people drive home from work. There is also a small increase durring lunch time (12-13 am). The lower plot shows the hourly distribution at different week days. 
# Working days mirror the trend seen in the central plot. Weekend days on the other hand show an interesting effect: there aren't clear peak times during the weekend. Instead the average usage is equal for a large hourly interval (approx. from 10 am - 6 pm) and decreases outside of this interval. 
# 
# Overall it can be concluded that bike usage is at its maximum during working days and peak hours. On weekends the usage on average is noramlly distrbuted. Seasonal effects generally play a role, however only in winter months. 
# 

# ## 6 Feature Engineering

# In[ ]:


# Create X by defining features 
features = ['month','weather','temp','windspeed','season','humidity']
X_hour = bike_hour[features]

# Define y
y_hour = bike_hour.total_cnt


# ### 6.1 Predications from hourly data using Random Forest

# In[ ]:


# Split into training, test and validaion set

X_hour_train, X_hour_test, y_hour_train, y_hour_test  = train_test_split(X_hour, y_hour, test_size=0.2, random_state=1)

X_hour_train, X_hour_val, y_hour_train, y_hour_val = train_test_split(X_hour_train, y_hour_train, test_size=0.25, random_state=1)


# In[ ]:


# checking for categorical variables
s = (X_hour_train.dtypes == 'category')
object_cols_h = list(s[s].index)

print("Categorical variables:")
print(object_cols_h)


# In[ ]:


# Encoding Categoricals using One Hot Encoding
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 
OH_cols_train_h = pd.DataFrame(OH_encoder.fit_transform(X_hour_train[object_cols_h])) #training set
OH_cols_valid_h = pd.DataFrame(OH_encoder.transform(X_hour_val[object_cols_h])) # validation set
OH_cols_test_h = pd.DataFrame(OH_encoder.transform(X_hour_test[object_cols_h])) # validation set
OH_cols_X_h = pd.DataFrame(OH_encoder.transform(X_hour[object_cols_h])) #complete set of X


# One-hot encoding removed index; put it back
OH_cols_train_h.index = X_hour_train.index
OH_cols_valid_h.index = X_hour_val.index
OH_cols_test_h.index = X_hour_test.index
OH_cols_X_h.index = X_hour.index

# Remove categorical columns (will replace with one-hot encoding)
num_train_X_h = X_hour_train.drop(object_cols_h, axis=1)
num_val_X_h = X_hour_val.drop(object_cols_h, axis=1)
num_test_X_h = X_hour_test.drop(object_cols_h, axis=1)
num_X_h = X_hour.drop(object_cols_h, axis=1)



# Add one-hot encoded columns to numerical features
OH_train_X_h = pd.concat([num_train_X_h, OH_cols_train_h], axis=1)
OH_val_X_h = pd.concat([num_val_X_h, OH_cols_valid_h], axis=1)
OH_test_X_h = pd.concat([num_test_X_h, OH_cols_test_h], axis=1)
OH_X_h = pd.concat([num_X_h, OH_cols_X_h], axis=1) 


# In[ ]:


OH_X_h.head()


# In[ ]:


#predictions using the Random Forest Regressor

#Define the model. Set random_state to 1
rf_model = RandomForestRegressor(n_estimators=100,random_state=1)
rf_model.fit(OH_train_X_h, y_hour_train)
rf_val_predictions = rf_model.predict(OH_val_X_h)
rf_val_mae = mean_absolute_error(rf_val_predictions, y_hour_val)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))


# In[ ]:


#apply model on test set:

rf_model.fit(OH_test_X_h, y_hour_test)

test_preds_h = rf_model.predict(OH_test_X_h)
errors_h = abs(y_hour_test-test_preds_h)

# Calculate mean absolute percentage error (MAPE)
MAPE_h = 100 * (errors_h / test_preds_h)


# Calculate and display accuracy
accuracy = 100 - np.mean(MAPE_h)
print("Accuracy: {:,.2f}%".format(accuracy))



# In[ ]:


# apply model on all data

rf_model.fit(OH_X_h, y_hour)

preds_all_h = rf_model.predict(OH_X_h)
errors_all_h = abs(y_hour-preds_all_h)

# Calculate mean absolute percentage error (MAPE)
MAPE_all_h = 100 * (errors_all_h / preds_all_h)


# Calculate and display accuracy
accuracy_all_h = round(100 - np.mean(MAPE_all_h),2)
print("Accuracy: {:,.2f}%".format(accuracy_all_h))



# ### 6.2 Predications from daily data using Random Forest

# In[ ]:


# Create X by defining features 
features_d = ['month','weather','temp','windspeed','season','humidity']
X_day = bike_day[features_d]

# Define y
y_day = bike_day.total_cnt


# Split into training, test and validaion set

X_day_train, X_day_test, y_day_train, y_day_test  = train_test_split(X_day, y_day, test_size=0.2, random_state=1)

X_day_train, X_day_val, y_day_train, y_day_val = train_test_split(X_day_train, y_day_train, test_size=0.25, random_state=1)


# In[ ]:


X_day_train.head()


# In[ ]:


X_day_train.isnull().any()


# In[ ]:


# checking for categorical variables
s = (X_day_train.dtypes == 'category')
object_cols_d = list(s[s].index)

print("Categorical variables:")
print(object_cols_d)


# In[ ]:


# Encoding Categoricals using One Hot Encoding


# Apply one-hot encoder to each column with categorical data
OH_encoder_d = OneHotEncoder(handle_unknown='ignore', sparse=False) 
OH_cols_train_d = pd.DataFrame(OH_encoder_d.fit_transform(X_day_train[object_cols_d])) #training set
OH_cols_valid_d = pd.DataFrame(OH_encoder_d.transform(X_day_val[object_cols_d])) # validation set
OH_cols_test_d = pd.DataFrame(OH_encoder_d.transform(X_day_test[object_cols_d])) # validation set
OH_cols_X_d = pd.DataFrame(OH_encoder_d.transform(X_day[object_cols_d])) #complete set of X


# One-hot encoding removed index; put it back
OH_cols_train_d.index = X_day_train.index
OH_cols_valid_d.index = X_day_val.index
OH_cols_test_d.index = X_day_test.index
OH_cols_X_d.index = X_day.index

# Remove categorical columns (will replace with one-hot encoding)
num_train_X_d = X_day_train.drop(object_cols_d, axis=1)
num_val_X_d = X_day_val.drop(object_cols_d, axis=1)
num_test_X_d = X_day_test.drop(object_cols_d, axis=1)
num_X_d = X_day.drop(object_cols_d, axis=1)



# Add one-hot encoded columns to numerical features
OH_train_X_d = pd.concat([num_train_X_d, OH_cols_train_d], axis=1)
OH_val_X_d = pd.concat([num_val_X_d, OH_cols_valid_d], axis=1)
OH_test_X_d = pd.concat([num_test_X_d, OH_cols_test_d], axis=1)
OH_X_d = pd.concat([num_X_d, OH_cols_X_d], axis=1) 


# In[ ]:


#predictions using the Random Forest Model without specifying leaf nodes

#Define the model. Set random_state to 1
rf_model_d = RandomForestRegressor(n_estimators=100, random_state=1)
rf_model_d.fit(OH_train_X_d, y_day_train)
rf_val_predictions_d = rf_model_d.predict(OH_val_X_d)
rf_val_mae_d = mean_absolute_error(rf_val_predictions_d, y_day_val)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae_d))


# In[ ]:


#apply model on test set:

rf_model_d.fit(OH_test_X_d, y_day_test)

test_preds_d = rf_model_d.predict(OH_test_X_d)
errors_d = abs(y_day_test-test_preds_d)

# Calculate mean absolute percentage error (MAPE)
MAPE_d = 100 * (errors_d / test_preds_d)


# Calculate and display accuracy
accuracy_d = 100 - np.mean(MAPE_d)
print("Accuracy: {:,.2f}%".format(accuracy_d))


# In[ ]:


#apply model on complete data set:

model = rf_model_d.fit(OH_X_d, y_day)

preds_all_d = rf_model_d.predict(OH_X_d)
errors_all_d = abs(y_day-preds_all_d)

# Calculate mean absolute percentage error (MAPE)
MAPE_all_d = 100 * (errors_all_d / preds_all_d)


# Calculate and display accuracy
accuracy_all_d = 100 - np.mean(MAPE_all_d)
print("Accuracy: {:,.2f}%".format(accuracy_all_d))


# ### 6.3 Feature Importance (only for daily model)

# In[ ]:


feat_imp = model.feature_importances_

df = pd.DataFrame(feat_imp)

df['ind'] = np.arange(len(feat_imp))

df['feat_imp'] = df[0]

df['labels'] = df.ind.map({0: 'temp', 1 : 'windspeed', 2 : 'humidity', 3 : 'Jan',  4 : 'Feb' , 5 : 'Mar', 6 : 'Apr' , 7 : 'May', 8 : 'Jun' , 9 : 'Jul', 10 : 'Aug', 11 : 'Sep', 12 : 'Oct', 13 : 'Nov', 14 : 'Dec' , 15 : 'Weat1', 16 : 'Weat2', 17 : 'Weat3', 18 : 'Sea1', 19 : 'Sea2', 20 : 'Sea3', 21 : 'Sea4',})
print(df.head())

width = 0.1
ind = np.arange(len(feat_imp))
ax = df['feat_imp'].plot(kind='bar', xticks=df.index)
ax.set_xticklabels(df['labels'])


plt.title('Feature Importance')
plt.xlabel('Relative importance')
plt.ylabel('feature')


# As seen from the correllation analysis, the features temp, windspeed and humidity are important features to be used for predication. Suprisingly, also the extreme seansons winter and summer turned out to be significant. Less significant, (however still important) are the months. 

# # Conclusion and Outlook

# 1. Predications based on hourly data were only to 65% accurate, wheares predications based on daily data were to 90% accurate. From the business perspective, an hourly perdication is not very useful, because the time frame is too short to react to, for instance, a drastic increase in the count, or defective bikes that need tp be replaced. Here daily data suffice. 
# 
# 
# 2. This data can be applied for anomaly detecion in the city: by analysing less common weather conditions (e.g. condition 4) combined with the information on the date, one can derive anomalies or uncommon events in the city. This essentially means, filtering the days where the count was significantly lower than average and filtering by less common weather conditions or windspeed > 0.6. Remove days that are weekends or holidays. Apply a search engine to find more about this date.
# 
# One can also analyse the outliers to understand what happend and why it happend in case its related to a rare event in the city.
# 
# 
# 3. In addition to weather information, location coordinates might help indentify other significant patterns, such as how long is the drive on average for different days, seasons etc. Which locations need more bikes than provided or accumalte more bikes than needed to be used? What are key locations to place bikes at and which ones are important for effective distribution?   
# 

# In[ ]:




