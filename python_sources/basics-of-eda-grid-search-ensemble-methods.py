#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import os
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import sklearn
from datetime import date


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv(f'/kaggle/input/train.csv')
train_label = pd.read_csv(f'/kaggle/input/train_label.csv',header=None)
test_data = pd.read_csv(f'/kaggle/input/test.csv')
test_label = pd.read_csv(f'/kaggle/input/test_label.csv', header=None)


# In[ ]:


train_data.head()


# In[ ]:


train_label.shape, train_data.shape, test_data.shape,test_label.shape


# In[ ]:


train_data['Total Booking'] = train_label


# In[ ]:


train_data.head()


# In[ ]:


date_data = pd.DataFrame(pd.to_datetime(train_data['datetime']))
date_data['Total Booking'] = train_label
date_data = date_data.set_index('datetime')
date_data.plot()
plt.ylabel('Total Bookings')


# In[ ]:


weekly = date_data.resample('W').sum()        
weekly.plot()        
plt.ylabel('Weekly total bookings'); 


# In[ ]:


by_weekday = date_data.groupby(date_data.index.dayofweek).mean()        
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']        
by_weekday.plot();


# In[ ]:


weekend = np.where(date_data.index.weekday < 5, 'Weekday', 'Weekend')        
by_time = date_data.groupby([weekend, date_data.index.time]).mean() 
hourly_ticks = 4 * 60 * 60 * np.arange(6)
fig, ax = plt.subplots(1, 2, figsize=(14, 6))        
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays', xticks=hourly_ticks, style=[':', '--', '-'])        
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends',  xticks=hourly_ticks,style=[':', '--', '-'])


# In[ ]:


g = sns.FacetGrid(train_data, col='workingday')
g.map(plt.hist, 'Total Booking', bins=20)


# In[ ]:


grid = sns.FacetGrid(train_data, col='workingday', row='season', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Total Booking', bins=20)
grid.add_legend();


# In[ ]:


g = sns.FacetGrid(train_data, col='holiday')
g.map(plt.hist, 'Total Booking', bins=20)


# In[ ]:


grid = sns.FacetGrid(train_data, col='holiday', row='season', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Total Booking', bins=20)
grid.add_legend();


# In[ ]:


train_data.isnull().sum()


# The info() method is useful to get a quick description of data, in particular the total number of rows, and each attribute's type and number of non-null values.

# In[ ]:


train_data.info()


# From the above info() method we can understand- 
# 1. 7 attributes are numerical.
# 2. No null values in the data.
# 3. 3 Columns are of object type datetime, season and weather.
# 
# Let's explore further:
# 

# In[ ]:


sns.countplot(x="season", data=train_data)


# In[ ]:


sns.countplot(y="weather", data=train_data)


# In[ ]:


train_data.describe()


# In[ ]:


attributes = ['temp',	'atemp',	'humidity',	'windspeed',	'Total Booking']


# In[ ]:


train_data[attributes].hist(bins=50, figsize=(20,15))


# Outlier Analysis using Boxplots:

# In[ ]:


#sns.boxplot(x='windspeed',data=train_data)
plt.figure(figsize = (10,5))
ax = sns.boxplot(data = train_data, orient = "h", color = "violet", palette = "Set1")
plt.show()


# In[ ]:


corr_matrix = train_data.corr()


# In[ ]:


corr_matrix["Total Booking"].sort_values(ascending=False)


# In[ ]:


sns.pairplot(train_data[attributes])


# In[ ]:


sns.jointplot(x="Total Booking", y="temp", data=train_data);


# **Preprocessing and Feature Engineering**
# 1. Convert the categorical variables in the train_data to one-hot encoding
# 2. Extract Date Features
# 3. Round decimal data
# 4. Normalize data for training

# Often when dealing with continuous numeric attributes like proportions or percentages, we may not need the raw values having a high amount of precision. Hence it often makes sense to round off these high precision percentages into numeric integers. 

# In[ ]:


def preprocessing(data):
  
  date_data = pd.DataFrame(pd.to_datetime(data['datetime']))

  #Extracting Year from Date
  data['Year'] = date_data['datetime'].dt.year

  #Extracting Month from Date
  data['Month'] = date_data['datetime'].dt.month

  #Extracting the weekday name of the date
  data['day_name'] = date_data['datetime'].dt.day_name()

  final_data = data.drop(columns=['datetime'])

  def truncate(n):
    return round(n)

  final_data['temp'] = final_data['temp'].apply(truncate)
  final_data['atemp'] = final_data['atemp'].apply(truncate)
  final_data['windspeed'] = final_data['windspeed'].apply(truncate)

  attributes = ['season','weather','day_name']
  one_hot_df = pd.get_dummies(final_data[attributes])

  final_data = pd.concat([final_data,one_hot_df],axis=1)
  final_data = final_data.drop(columns=attributes)
    
  return final_data


# In[ ]:


final_data = preprocessing(train_data)
final_data.columns


# In[ ]:


attrib = ['Total Booking','weather_ Heavy Rain + Thunderstorm ']
final_data = final_data.drop(columns=attrib)


# In[ ]:


final_data.head()


# Regression Models:

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data_final = scaler.fit_transform(final_data)
X_train, X_test, y_train, y_test = train_test_split(train_data_final, train_label, test_size=0.2, random_state=200)


# Linear Regression:

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression(normalize=True)


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


print('R2-Score',r2_score(y_test,predictions))


# In[ ]:


from sklearn.svm import LinearSVR


# In[ ]:


svm_reg = LinearSVR(epsilon=1.5,max_iter=1000,random_state=200)
svm_reg.fit(X_train,y_train.values.ravel())


# In[ ]:


preds = svm_reg.predict(X_test)


# In[ ]:


print('R2-Score',r2_score(y_test,preds))


# Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()


# In[ ]:


forest_reg.fit(X_train,y_train.values.ravel())


# In[ ]:


preds = forest_reg.predict(X_test)


# In[ ]:


print('R2-Score',r2_score(y_test,preds))


# Grid Search:

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = [{'n_estimators':[20,30], 'max_features': [2,4,6,8]},
              {'bootstrap':[False],'n_estimators':[3,10], 'max_features':[2,3,4]}]


# In[ ]:


forest_reg = RandomForestRegressor()


# In[ ]:


grid_search = GridSearchCV(forest_reg, param_grid,cv=5,scoring='r2',return_train_score=True)


# In[ ]:


grid_search.fit(X_train, y_train.values.ravel())


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# In[ ]:


forest_reg1 = RandomForestRegressor(max_features=6,n_estimators=30)


# In[ ]:


forest_reg1.fit(X_train,y_train.values.ravel())


# In[ ]:


preds = forest_reg1.predict(X_test)


# In[ ]:


print('R2-Score after Grid Search best Parameters',r2_score(y_test,preds))


# Ensemble Methods:

# In[ ]:


from sklearn.ensemble import VotingRegressor


# In[ ]:


voting_clf = VotingRegressor(estimators=[('lr', lm),('rf', forest_reg1),('svm', svm_reg)])
voting_clf.fit(X_train, y_train.values.ravel()).predict(X_test)


# In[ ]:


X_train.shape, y_train.shape, X_test.shape,y_test.shape


# In[ ]:


from sklearn.metrics import r2_score
for clf in (lm,forest_reg1,svm_reg,voting_clf):
  clf.fit(X_train,y_train.values.ravel())
  y_pred = clf.predict(X_test)
  print(clf.__class__.__name__,r2_score(y_test,y_pred))


# In[ ]:


test_data = preprocessing(test_data)


# In[ ]:


test_data.head()


# In[ ]:


test_data = scaler.fit_transform(test_data)


# Testing:

# In[ ]:


final_preds = forest_reg1.predict(test_data)


# In[ ]:


print('R2-Score of test data',r2_score(test_label, final_preds))

