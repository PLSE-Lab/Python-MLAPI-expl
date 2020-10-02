#!/usr/bin/env python
# coding: utf-8

# ![](http://storage.googleapis.com/kaggle-competitions/kaggle/3948/media/bikes.png)
# 
# I am super excited to share my first kernel with the Kaggle community. This kernel is for all the aspiring data scientists who wants to learn and review their knowledge. As I go on in this journey and learn new topics, I will incorporate them with each new updates. Going back to the topics of this kernel, I will do visualizations to explain the data, and machine learning algorithms to forecast bike rental demand  in the Capital Bikeshare program in Washington, D.C.

# In[ ]:


#Let's import the usual suspects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Importing the dataset
train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
data = train.append(test, sort = False)
data.head()


# In[ ]:


data.info()


# Types of variables:
# * Categorical - Season, Holiday, Working day, Weather
# * Timeseries - Datetime
# * Numerical - Temp, aTemp, Humidity, Windspeed, Casual, Registered, Count

# In[ ]:


data.describe()


# **Exploratory Data Analysis**

# In[ ]:


#Histogram for count
sns.set_style('darkgrid')
sns.distplot(train['count'], bins = 100, color = 'green')
plt.show()


# In[ ]:


#Q-Q Plot
from scipy import stats
plt = stats.probplot(train['count'], plot=sns.mpl.pyplot)


# In[ ]:


#Boxplot for count
import matplotlib.pyplot as plt
sns.boxplot(x = 'count', data = train, color = 'mediumpurple')
plt.show()


# These three charts above can tell us a lot about our target variable.
# 
# * Our target variable, count is not normally distributed.
# * Our target variable is right-skewed.
# * There are multiple outliers in the variable.

# In[ ]:


#Calculating the number of outliers
Q1 = train['count'].quantile(0.25)
Q3 = train['count'].quantile(0.75)
IQR = Q3 - Q1
outliers = train[(train['count'] < (Q1 - 1.5 * IQR)) | (train['count'] > (Q3 + 1.5 * IQR))]
print((len(outliers)/len(data))*100)


# 1.72% of the target values are above Q3 + 1.5IQR. Let's get rid of this.

# In[ ]:


#Data without the outliers in count
data = data[~data.isin(outliers)]
data = data[data['datetime'].notnull()]


# In[ ]:


sns.barplot(x = 'season', y = 'count', data = train, estimator = np.average, palette='coolwarm')
plt.ylabel('Average Count')
plt.show()


# In[ ]:


sns.barplot(x = 'holiday', y = 'count', data = train, estimator = np.average, palette='deep')
plt.ylabel('Average Count')
plt.show()


# In[ ]:


sns.barplot(x = 'workingday', y = 'count', data = train, estimator = np.average, palette='colorblind')
plt.ylabel('Average Count')
plt.show()


# In[ ]:


sns.barplot(x = 'weather', y = 'count', data = train, estimator = np.average, palette='deep')
plt.ylabel('Average Count')
plt.show() 


# In[ ]:


plt.figure(figsize = (10,7))
tc = train.corr()
sns.heatmap(tc, annot = True, cmap = 'coolwarm', linecolor = 'white', linewidths=0.1)


# Count is hightly correlated with Casual and Registered. It's because Count is derived from Casual and Registered. We'll have to omit these variables. Temp and atemp are highly correlated.

# In[ ]:


#Convert to integer variables
columns=['season', 'holiday', 'workingday', 'weather']
for i in columns:
    data[i] = data[i].apply(lambda x : int(x))


# In[ ]:


#Convert string to datatime and create Hour, Month and Day of week
data['datetime'] = pd.to_datetime(data['datetime'])
data['Hour'] = data['datetime'].apply(lambda x:x.hour)
data['Month'] = data['datetime'].apply(lambda x:x.month)
data['Day of Week'] = data['datetime'].apply(lambda x:x.dayofweek)


# In[ ]:


plt.figure(figsize = (8,4))
sns.lineplot(x = 'Month', y = 'count', data = data, estimator = np.average, hue = 'weather', palette = 'coolwarm')
plt.ylabel('Average Count')
plt.show()


# In[ ]:


data[data['weather'] == 4]


# There is no line plot for weather = 4, because there is only three data point for weather = 4

# In[ ]:


fig, axes = plt.subplots(ncols = 2, figsize = (15,5), sharey = True)
sns.pointplot(x = 'Hour', y = 'count', data = data, estimator = np.average, hue = 'workingday', ax = axes[0], palette = 'muted')
sns.pointplot(x = 'Hour', y = 'count', data = data, estimator = np.average, hue = 'holiday', ax = axes[1], palette = 'muted')
ax = [0,1]
for i in ax:
    axes[i].set(ylabel='Average Count')


# * During working days there is a high demand around the 7th hour and 17th hour. There is a lower demand during 0 to 5th hour and 10 to 14th hour.
# * During non workin days there is a high demand during 10 to 14th hour. There is a lower demand around the 7th hour.

# In[ ]:


plt.figure(figsize = (10,4))
sns.pointplot(x = 'Hour', y = 'count', data = data, estimator=np.average, hue = 'Day of Week', palette='coolwarm')


# Clearly, weekend and weekdays follows a different pattern.

# In[ ]:


sns.jointplot(x = 'atemp', y = 'count', data = data, kind = 'kde', cmap = 'plasma')
plt.show()


# In[ ]:


plt.figure(figsize = (8,4))
sns.pointplot(x = 'Hour', y = 'casual', data = data, estimator = np.average, color = 'blue')
sns.pointplot(x = 'Hour', y = 'registered', data = data, estimator = np.average, color = 'red')
plt.ylabel('Registered')
plt.show()


# In[ ]:


#Histogram for Windspeed
sns.set_style('darkgrid')
sns.distplot(data['windspeed'], bins = 100, color = 'purple') #Windspeed cannot be 0.
plt.show()


# **Feature Engineering**

# In[ ]:


#Replacing 0s in windspeed with the mean value grouped by season
data['windspeed'] = data['windspeed'].replace(0, np.nan)
data['windspeed'] = data['windspeed'].fillna(data.groupby('weather')['season'].transform('mean'))
sns.distplot(data['windspeed'], bins = 100, color = 'red')
plt.show()


# In[ ]:


#Encoding cyclical features
data['Month_sin'] = data['Month'].apply(lambda x: np.sin((2*np.pi*x)/12))
data['Month_cos'] = data['Month'].apply(lambda x: np.cos((2*np.pi*x)/12))
data['Hour_sin'] = data['Hour'].apply(lambda x: np.sin((2*np.pi*(x+1))/24))
data['Hour_cos'] = data['Hour'].apply(lambda x: np.cos((2*np.pi*(x+1))/24))
data['DayOfWeek_sin'] = data['Day of Week'].apply(lambda x: np.sin((2*np.pi*(x+1))/7))
data['DayOfWeek_cos'] = data['Day of Week'].apply(lambda x: np.cos((2*np.pi*(x+1))/7))


# As the target variable is a highly skewed data, we will try to transform this data using either log, square-root or box-cox transformation. After trying out all three, log square gives the best result. Also as the evaluation metric is RMSLE, using log would help as it would allow to less penalize the large difference in final variable values.

# In[ ]:


#trainsforming target variable using log transformation
data['count'] = np.log(data['count'])


# In[ ]:


#Converting Categorical to numerical - Removing Co-Linearity
data_ = pd.get_dummies(data=data, columns=['season', 'holiday', 'workingday', 'weather'])
train_ = data_[pd.notnull(data_['count'])].sort_values(by=["datetime"])
test_ = data_[~pd.notnull(data_['count'])].sort_values(by=["datetime"])


# In[ ]:


#Standardizing numerical variables
from sklearn.preprocessing import StandardScaler
cols = ['temp','atemp','humidity', 'windspeed', 'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin','DayOfWeek_cos']
features = data[cols]

#Standard Scaler
scaler = StandardScaler().fit(features.values)
data[cols] = scaler.transform(features.values)


# In[ ]:


#Predictor columns names
cols = ['temp','atemp','humidity', 'windspeed', 'Month_sin', 'Month_cos', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin','DayOfWeek_cos', 'season_1','season_2', 'season_3',
        'holiday_0', 'workingday_0', 'weather_1', 'weather_2', 'weather_3']


# **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


#train test split
X = train_[cols]
y = train_['count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[ ]:


lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.intercept_)


# In[ ]:


plt.figure(figsize = (18,4))
coeff = pd.DataFrame(lm.coef_, index = X.columns, columns = ['Coefficient'])
sns.barplot(x = coeff.index, y = 'Coefficient', data = coeff, color = 'red')


# In[ ]:


plt.figure(figsize = (8,4))
pred = lm.predict(X_test)
sns.scatterplot(x = y_test, y = pred)
plt.xlabel('Count')
plt.ylabel('Predictions')
plt.show()


# The variability between the actual values and the predicted values is higher.

# In[ ]:


sns.distplot((y_test-pred),bins=100, color = 'gray')
plt.show()


# The residual distribution is normal.

# In[ ]:


print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(np.exp(y_test), np.exp(pred))))


# **Ridge Regression**

# In[ ]:


from sklearn.linear_model import Ridge
#Assiging different sets of alpha values to explore which can be the best fit for the model. 
temp_msle = {}
for i in np.linspace(0, 40, 20):
    ridge = Ridge(alpha= i, normalize=True)
    #fit the model. 
    ridge.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    pred = ridge.predict(X_test)

    msle = np.sqrt(metrics.mean_squared_log_error(np.exp(y_test), np.exp(pred)))
    temp_msle[i] = msle


# In[ ]:


temp_msle


# **Lasso Regression**

# In[ ]:


from sklearn.linear_model import Lasso
## Assiging different sets of alpha values to explore which can be the best fit for the model. 
temp_msle = {}
for i in np.logspace(-10, -1, 20):
    ## Assigin each model. 
    lasso = Lasso(alpha= i, normalize=True, tol = 0.1)
    ## fit the model. 
    lasso.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    pred = lasso.predict(X_test)

    msle = np.sqrt(metrics.mean_squared_log_error(np.exp(y_test), np.exp(pred)))
    temp_msle[i] = msle


# In[ ]:


temp_msle


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 500)
rfr.fit(X_train, y_train)


# In[ ]:


plt.figure(figsize = (8,4))
pred = rfr.predict(X_test)
sns.scatterplot(x = y_test, y = pred)
plt.xlabel('Count')
plt.ylabel('Predictions')
plt.show()


# The variability between the actual values and the predicted values is lesser than the linear regression.

# In[ ]:


sns.distplot((y_test-pred),bins=100, color = 'gray')


# In[ ]:


#RMSLE
print('RMSLE:', np.sqrt(metrics.mean_squared_log_error(np.exp(y_test), np.exp(pred))))


# In[ ]:


#submission
new = test_[cols]
pred = rfr.predict(new)
submission = pd.DataFrame({'datetime':test['datetime'],'count':np.exp(pred)})
submission['count'] = submission['count'].astype(int)
submission.to_csv('submission.csv',index=False)

