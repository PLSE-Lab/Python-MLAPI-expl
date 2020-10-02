#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

data=pd.read_csv('/kaggle/input/SolarEnergy/SolarPrediction.csv')
col_names=data.columns
print(col_names)
data=data.rename({'Data':'Date'}, axis=1)
data['date']=data['Date'].str.split(' ').str.get(0)
data['Datetime'] = pd.to_datetime(data['date'].apply(str)+' '+data['Time']) #Datetime column is of type datetime64
data['Hour_Minute'] = data['Datetime'].dt.strftime('%H:%M')
data['Hour'] = data['Datetime'].dt.strftime('%H')
data['Datetime']=sorted(data['Datetime']) ## date column was not sorted

data=data.drop(columns=['UNIXTime', 'Date', 'date', 'Time'])
data=data.set_index('Datetime')
#print(data.head())

data['TimeSunSet']=pd.to_datetime(data['TimeSunSet'], format = "%H:%M:%S")
data['TimeSunRise']=pd.to_datetime(data['TimeSunRise'], format = "%H:%M:%S")
data['DayLength'] = data['TimeSunSet'] - data['TimeSunRise']
data['DayLength'] = data['DayLength'].dt.total_seconds().floordiv(60)
data['month']=data.index.month


# Now the dataframe 'data' is ready for EDA which is done in next step.
# 

# In[ ]:


########################################## Exploratory Data Analysis ###################################################
plt.figure(figsize=(12,8))
sns.barplot(x=data['Temperature'],y=data['Radiation'])
plt.xlabel('Temperature (F)')
plt.ylabel('Radiation (W/m2)')
plt.title('Radiation versus Temperature')
plt.show()


# As it can be seen increasing temperature almost always results in increasing of the Radiation.
# 

# In[ ]:


plt.figure(figsize=(12,8))
sns.barplot(x=data['Humidity'],y=data['Radiation'])
plt.xlabel('Humidity')
plt.ylabel('Radiation (W/m2)')
plt.title('Radiation versus Humidity')
plt.show()


# From the figure above we can see that there is no linear relationship between Radiation and Humidity.

# Now we want to know how Radiation changes per month:

# In[ ]:


months = np.arange(9,13)
rad_vs_month=data.loc[:, ['Radiation', 'month']].groupby('month').sum()
rad_vs_month.plot()
plt.xticks(months,['September', 'October', 'November', 'December'], rotation=30)
plt.show()


# In October Radiation is highest and in December it i slowest which is logical!

# Now let's check the distribution of Radiation during the day time:

# In[ ]:


rad_vs_hour=data.loc[:, ['Radiation', 'Hour']].groupby('Hour').mean()
rad_vs_hour.plot(kind='bar')
plt.xlabel('Time of the day (hour)')
plt.ylabel('Radiation(W/m2)')
plt.title('Total Radiation per time of the day')
plt.show()


# As it can be seen, around noon is the maximum Radiation which makes sense.

# Since we have the dey length for everyday, we can see how Radiation varies by daylength:

# In[ ]:


rad_vs_time=data.loc[:, ['Radiation', 'DayLength']].groupby('DayLength').mean()
rad_vs_time.plot()
plt.title('Average Radiation versus day length')
plt.xlabel('Day length (minutes)')
plt.ylabel('Radiation(W/m2)')
plt.show()


# There is no strong correlation between daylength and Radiation, we can maybe ignore this feature while predicting Radiation. Now, to better understand the correlation between different features and Radiation we plot heatmap:

# In[ ]:


## correlation matrix and heatmap
cols_heatmap=['Temperature', 'Radiation', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayLength']
corr_mat = data.loc[:, cols_heatmap].corr(method='pearson')
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_mat, annot=True) # plot heatmap; annot=True shows the number, without it, there is no correlation value, just colors!
plt.xticks(rotation=30)
plt.show()


# It can be seen from heatmap that there is high correlation between temperature and Radiation, a low correlation between pressure and Radiation.

# Now we split our data to train and validation. We want to use all the data from September to 2016-12-29 for training our models and use the data from 30 and 31 December 2016 as validation. 
# After we predict Radiation values for 30 and 31 December, we can compare the result with the observed data (validation). 
# 

# In[ ]:


######################################### Specify Tain, test and validation data #######################################
# Validation data is unseen data by model
data_usefulcols=data.drop(columns=['month', 'Hour', 'Hour_Minute', 'TimeSunSet', 'TimeSunRise', 'WindDirection(Degrees)', 'Speed'])
train=data_usefulcols.loc[:'2016-12-29', data_usefulcols.columns]
print(train.tail())


# In[ ]:


validation=data_usefulcols.loc['2016-12-30':, data_usefulcols.columns]
print(validation.head())


# Now prepare X and y for our models:

# In[ ]:


X_validation=validation.drop(columns='Radiation')
y_validation=validation[['Radiation']]
Result=y_validation.copy()
X=train.drop(columns='Radiation')
y=train[['Radiation']] # used for XGBoost
y_lgb=train['Radiation'] # used for LGBM


# We use Random Forest, XGBoost and LightGBM for prediction: 

# In[ ]:


############################################ Building Random Forest model  #############################################
rf=RandomForestRegressor()

# finding the best parameters by RandomizedSearchCV
param_rf = {'bootstrap':['False'], 'n_estimators': [500], "max_depth": [20], "max_features": np.arange(3,5).tolist(), "min_samples_leaf": np.arange(80,100).tolist(), "criterion": ["mse"]}
rf_cv=RandomizedSearchCV(rf, param_rf, cv=2)

# Extract the best estimator
rf_cv.fit(X, np.ravel(y))

best_model = rf_cv.best_estimator_
print(best_model)
y_pred_rf=best_model.predict(X_validation)
Result['Predicted_Radiation_RF']=y_pred_rf
rmse = np.sqrt(mean_squared_error(y_validation, y_pred_rf))
print("Root Mean Squared Error RF: {}".format(rmse))


# In[ ]:


################################################ Build XGBoost model ###################################################

xgb = XGBRegressor(max_depth=35, random_state=42, n_estimators=1500, learning_rate=0.005, booster='gbtree', objective='reg:squarederror', min_child_weight=0.1, silent=1, n_jobs=10)

xgb.fit(X, y.values.ravel())
y_pred=xgb.predict(X_validation)

predicted=pd.DataFrame(y_pred)
Result['Predicted_Radiation_XGBoost']=y_pred

rmse = np.sqrt(mean_squared_error(y_validation, y_pred))
print("Root Mean Squared Error XGBoost: {}".format(rmse))

feat_imp = pd.Series(xgb.feature_importances_, index= X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances XGBoost') # note: there in no feature importance for lgbm
plt.ylabel('Feature Importance Score')
plt.show()


# We can see from the figure above that Temperature and the length of the day are the most important features for this problem.

# In[ ]:


######################################### Build lightGB model ####################################################
X_lgb = X.values

params = {
      'num_leaves': 700,
      'min_child_weight': 0.34,
      'feature_fraction': 0.979,
      'bagging_fraction': 0.818,
      'min_data_in_leaf': 700,
      'objective': 'regression',
      'max_depth': 40,
      'learning_rate': 0.1,
      "boosting_type": "gbdt",
      "bagging_seed": 11,
      "metric": 'rmse',
      "verbosity": -1,
      'reg_alpha': 0.0001,
      'reg_lambda': 2.9,
      'random_state': 666,
    }

lgb_train = lgb.Dataset(X.values, label=y_lgb.values)
# Train LightGBM model
m_lgb = lgb.train(params, lgb_train, 400)
y_pred_lgb=m_lgb.predict(X_validation)
#print(np.round(y_pred_lgb[0], 6))

Result['Predicted_Radiation_LGB']=y_pred_lgb
print(Result.loc[:, ['Predicted_Radiation_XGBoost', 'Radiation']].head())

rmse_lgb = np.sqrt(mean_squared_error(y_validation, y_pred_lgb))
print("Root Mean Squared Error LGBM: {}".format(rmse_lgb))


# Now we plot the results to get a better insight about the predictions:

# In[ ]:


######################################### Performance comparison #######################################################
Result=Result.rename({'Radiation':'Radiation_observed'}, axis=1)
Result.plot(figsize=(20,12))
plt.ylabel('Radiation(W/m2)')
plt.title('Performance of different models in prediction of Radiation for 30 and 31 December 2016')
plt.xlabel('Date and Time')
plt.show()


# Looking at the RMS error we conclude that LightGBM has the best prediction here, however by looking at the figure above, we can see that XGBoost predictions of 31 December are more match with the observed data. All the models have considerable error between noon 30 December and morning 31 December. 
# Although the train data is really small, the results are reasonable! 
# 

# 

# In[ ]:





# In[ ]:




