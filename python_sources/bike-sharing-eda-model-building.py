#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Demand

# Data Fields:
# date - hourly date 
# 
# season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
# 
# holiday - whether the day is considered a holiday
# 
# workingday - whether the day is neither a weekend nor holiday
# 
# weekday- Weekday number 0-sunday etc.
# 
# weather -
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 
# temp - temperature in Celsius
# 
# atemp - "feels like" temperature in Celsius
# 
# humidity - relative humidity
# 
# windspeed - wind speed
# 
# casual - number of non-registered user rentals initiated
# 
# registered - number of registered user rentals initiated
# 
# demand - number of total rentals (Dependent Variable)

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


# # Data Exploration

# In[ ]:


df=pd.read_csv('../input/bike-sharing/Bike_Sharing.csv') 


# In[ ]:


df.head()


# In[ ]:


df.info()


# Types of variables:
# 
# Categorical - Season, Holiday, Working day, Weather
# Timeseries - Datetime
# Numerical - Temp, aTemp, Humidity, Windspeed, Casual, Registered, Count

# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# # Missing Value Analysis

# In[ ]:


df.isnull().sum()


# There isn't any missing values

# # Feature Engineering

# Steps:
# 
# changing the datatype of "month","hour","weekday","season","holiday","workingday" and "weather" to category.
# 
# Drop the date,index column as we already have necessary details from other columns

# In[ ]:


#changing to category datatype
col_cat = ["hour","weekday","month","season","weather","holiday","workingday"]
for var in col_cat:
    df[var] = df[var].astype("category")


# In[ ]:


df.dtypes


# Dropping unncessary Columns

# In[ ]:


df  = df.drop(["index","date"],axis=1)


# # Exploratory Data Analysis

# # Visualise the continuous features Vs demand

# Temperature vs Demand

# In[ ]:


#plt.subplot(2,2,1)
plt.title('Temperature Vs Demand')
plt.scatter(df['temp'], df['demand'], c='b')


# In[ ]:


plt.title('atemp Vs Demand')
plt.scatter(df['atemp'], df['demand'], c='b')


# In[ ]:


sns.scatterplot(x="temp", y="atemp", data=df, hue="demand")
plt.show()


# As the temperature increases number of rides will also get increased.

# Humidity vs demand

# In[ ]:


plt.title('Humidity Vs Demand')
plt.scatter(df['humidity'], df['demand'], c='b')


# Very little change in demand for change or increase in humidity.

# Windspeed vs demand

# In[ ]:


sns.scatterplot(x="windspeed", y="demand", data=df, hue="demand")
plt.show()


# The wind speed plot also shows a pattern that as the wind speed increases up to the particular point,it does not show any variation but beyond a point the demand for rentals go down significantly.

# # Visualise the Categorical features Vs demand

# Demand vs Season

# In[ ]:


colors = ['g', 'r', 'm', 'b']
plt.title('Average Demand per Season')
cat_list = df['season'].unique()
cat_average = df.groupby('season').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# Demand varies depending upon the season.It's the highest during the fall while lowest during the spring which is Season 1

# Month vs Demand

# In[ ]:


colors = ['g', 'r', 'm', 'b']
plt.title('Average Demand per month')
cat_list = df['month'].unique()
cat_average = df.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# The months which fall during the summer shows higher demand.

# Holiday vs Demand

# In[ ]:


plt.title('Average Demand per Holiday')
cat_list = df['holiday'].unique()
cat_average = df.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# Weekday vs Demand

# In[ ]:


plt.title('Average Demand per Weekday')
cat_list = df['weekday'].unique()
cat_average = df.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# It  does not matter which day of the week it is,there is hardly any change in demand

# This is not so important and we will be better if we drop this feature.

# Year vs Demand

# In[ ]:


plt.title('Average Demand per Year')
cat_list = df['year'].unique()
cat_average = df.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# We have only two years of data and we will not know for sure how it will be for 5 or 6 years. So it's better we drop this feature too.

# Hour vs Demand

# In[ ]:


plt.title('Average Demand per hour')
cat_list = df['hour'].unique()
cat_average = df.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# you can see that demand is very low during the early morning or past midnight and it picks up during the certain hours

# Demand vs Working day

# In[ ]:


plt.title('Average Demand per Workingday')
cat_list = df['workingday'].unique()
cat_average = df.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# We can see there is no much change whether it is working day or not.So People use the bikes to travel to nearby workplaces during the weekdays and could possibly be using it for exercise and fun during the weekends.
# 
# It's better we drop this feature

# Demand vs Weather

# In[ ]:


plt.title('Average Demand per Weather')
cat_list = df['weather'].unique()
cat_average = df.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)


# During rainy days the demand is very low and when the cloud is clear the demand is very high

# # Analyzing Target Variable

# In[ ]:


sns.set_style('darkgrid')
sns.distplot(df['demand'], bins = 100, color = 'blue')


# Our target variable is right-skewed.

# In[ ]:


#Q-Q Plot
from scipy import stats
plt = stats.probplot(df['demand'], plot=sns.mpl.pyplot)


# Our target variable is not normally distributed.

# In[ ]:


sns.boxplot(x = 'demand', data = df, color = 'blue')


# There are multiple outliers in the variable.

# # Outlier Analysis

# In[ ]:


#Calculating the number of outliers
Q1 = df['demand'].quantile(0.25)
Q3 = df['demand'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['demand'] < (Q1 - 1.5 * IQR)) | (df['demand'] > (Q3 + 1.5 * IQR))]
print((len(outliers)/len(df))*100)


# 2.90% of the target values are above Q3 + 1.5IQR.

# Let's remove the outliers

# In[ ]:


df_final = df[np.abs(df["demand"]-df["demand"].mean())<=(3*df["demand"].std())]
print ("Shape Of The Before Ouliers: ",df.shape)
print ("Shape Of The After Ouliers: ",df_final.shape)


# # Testing Assumptions of Multiple Linear Regression

# Collinearity and Multi-Collinearity Check

# In[ ]:


tc = df.corr()
sns.heatmap(tc, annot = True, cmap = 'coolwarm')


# Demand is derived from Casual and Registered so it is highly correlated with these two features.We'll have to omit these variables. Temp and atemp are highly correlated. So we can remove any one of the variable.

# Dropping Irrelevant features

# In[ ]:


df_final = df_final.drop(['weekday', 'year', 'workingday', 'atemp','casual', 'registered'], axis=1)


# # Autocorrelation test

# In[ ]:


import matplotlib.pyplot as plt
# Autocorrelation of demand using acor
dff1 = pd.to_numeric(df_final['demand'], downcast='float')
plt.acorr(dff1, maxlags=12)


# There is high auto-correlation of target variable

# # Normality Check

# In[ ]:


fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(12, 10)
sns.distplot(df_final["demand"],ax=axes[0][0])
stats.probplot(df_final["demand"], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(df_final["demand"]),ax=axes[1][0])
stats.probplot(np.log1p(df_final["demand"]), dist='norm', fit=True, plot=axes[1][1])


# In[ ]:


df_final['demand'] = np.log(df_final['demand'])


# # Solving the problem of autocorrelation

# In[ ]:


# Solve the problem of Autocorrelation
# Shift the demand by 3 lags

t_1 = df_final['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

t_2 = df_final['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

t_3 = df_final['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

df_final_lag = pd.concat([df_final, t_1, t_2, t_3], axis=1)


# In[ ]:


df_final_lag.head()


# In[ ]:


df_final_lag = df_final_lag.dropna()


# Now we have same dataset with additinal three lag value columns.

# In[ ]:


df.columns


# Let's Analyze windspeed column

# In[ ]:


df_final_lag['windspeed'].value_counts()


# As you can see winspeed has many 0 values. 
# 
# Now i am filling windspeed's 0 value using random forest.
# 
# Random Forest is ensemble method gives better prediction compared to other models.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
df_Wind_0 = df_final_lag[df_final_lag["windspeed"]==0]
df_Wind_Not0 = df_final_lag[df_final_lag["windspeed"]!=0]
Columns = ["season","weather","humidity","month","temp"]
rf_model = RandomForestRegressor()
rf_model.fit(df_Wind_Not0[Columns],df_Wind_Not0["windspeed"])

wind0Values = rf_model.predict(X= df_Wind_0[Columns])
df_Wind_0["windspeed"] = wind0Values
data = df_Wind_Not0.append(df_Wind_0)
data.reset_index(inplace=True)
data.drop('index',inplace=True,axis=1)


# In[ ]:


data.dtypes


# # Dummy Variables

# Create Dummy Variables and drop first to avoid dummy variables trap

# Here we are not maintaining any order hence we can go with one-hot encoding

# In[ ]:


data = pd.get_dummies(data, drop_first=True)


# In[ ]:


data.columns


# In[ ]:


data.shape


# We successfully completed data processing stage .Next we go ahed and build models.

# # Model Building With Evaluation

# Splitting X and Y

# In[ ]:


X = np.array(data.loc[:,data.columns!='demand'])
Y = np.array(data.loc[:,data.columns=='demand'])


# In[ ]:


print(X.shape)
print(Y.shape)


# Random Search

# In[ ]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics


# Model Function

# In[ ]:


def regression(X, Y, reg, param_grid, test_size=0.20):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
      
    
    reg = RandomizedSearchCV(reg,parameters, cv = 10,refit = True)
    reg.fit(X_train, Y_train)     

    return X_train, X_test, Y_train, Y_test, reg


# Evaluation Function

# In[ ]:


def evaluation_metrics(X_train, X_test, Y_train, Y_test, reg):
    Y_pred_train = reg.best_estimator_.predict(X_train)
    Y_pred_test = reg.best_estimator_.predict(X_test)
    
    print("Best Parameters:",reg.best_params_)
    print('\n')
    print("Mean cross-validated score of the best_estimator : ", reg.best_score_) 
    print('\n')
    MAE_train = metrics.mean_absolute_error(Y_train, Y_pred_train)
    MAE_test = metrics.mean_absolute_error(Y_test, Y_pred_test)
    print('MAE for training set is {}'.format(MAE_train))
    print('MAE for test set is {}'.format(MAE_test))
    print('\n')
    MSE_train = metrics.mean_squared_error(Y_train, Y_pred_train)
    MSE_test = metrics.mean_squared_error(Y_test, Y_pred_test)
    print('MSE for training set is {}'.format(MSE_train))
    print('MSE for test set is {}'.format(MSE_test))
    print('\n')
    RMSE_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred_train))
    RMSE_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_test))
    print('RMSE for training set is {}'.format(RMSE_train))
    print('RMSE for test set is {}'.format(RMSE_test))
    print('\n')
    r2_train = metrics.r2_score(Y_train, Y_pred_train)
    r2_test = metrics.r2_score(Y_test, Y_pred_test)
    print("R2 value for train: ", r2_train)
    print("R2 value for test: ", r2_test)


# # Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg = LinearRegression()
parameters = {'fit_intercept':[True,False],'normalize':[True,False], 'copy_X':[True, False]}
X_train, X_test, Y_train, Y_test, linreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)
evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = linreg)


# # Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


reg = DecisionTreeRegressor()
parameters = {'max_depth':[5,6,7,8,9,10]}
X_train, X_test, Y_train, Y_test, DTreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)
evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = DTreg)


# # Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


reg = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators':[10,15,20,25],'max_depth':[5,6,7,8,9,10]}
X_train, X_test, Y_train, Y_test, RFreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)
evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = RFreg)


# # Gradient Boosting Regressor
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


reg = GradientBoostingRegressor()
parameters = {'alpha':[0.01,0.001,0.0001],'n_estimators':[100,150,200],'max_depth':[3,5,7]}
X_train, X_test, Y_train, Y_test, XGreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)
evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = XGreg)


# # Support Vector Regressor
# 

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


reg = SVR()
parameters = {'max_iter':[1000,5000,10000]}
X_train, X_test, Y_train, Y_test, SVRreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)
evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = SVRreg)


# # Multi Layer Perceptron Regressor
# 

# In[ ]:


from sklearn.neural_network import MLPRegressor


# In[ ]:


reg = MLPRegressor(activation='tanh',early_stopping=True)
parameters = {'solver':['sgd', 'adam'],'learning_rate_init':[0.01,0.001,0.0001],'hidden_layer_sizes':[10,25,50],'max_iter':[500,1000]}
X_train, X_test, Y_train, Y_test, MLPreg = regression(X, Y, reg, param_grid=parameters, test_size=0.20)
evaluation_metrics(X_train, X_test, Y_train, Y_test, reg = MLPreg)


# # Conclusion

# Multi Layer Perceptron Regressor perfoms well compared to other models. 

# # Best Estimator

# In[ ]:


MLPreg.best_estimator_


# # Actual Vs Predicted

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[ ]:


Y_Pred_test = MLPreg.best_estimator_.predict(X_test)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,7))
ax.scatter(Y_test, Y_Pred_test, edgecolors=(0, 0, 0))
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k-', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()


# As we can see there is a good relationship between actual and predicted Values

# # END 
