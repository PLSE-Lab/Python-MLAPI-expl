#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for visualization
import seaborn as sns # for statistical visualization
plt.style.use('ggplot') # Set style for plotting


# ## Data Cleaning

# In[ ]:


# Read 10,000,000 rows so that the kernel won't died easily
train = pd.read_csv('../input/train.csv', nrows = 10_000_000)


# In[ ]:


# Look at the top 3 rows of data
train.head(3)


# In[ ]:


# Structure and data types
train.info()


# In[ ]:


# Statistical analysis overlook
pd.set_option('float_format', '{:f}'.format) # Print entire number instead of x + ye

train.describe()


# Well, some records are definitely wrong. I'll check those out now.
# 
# Obviously, a taxi can only have at most 5 passengers. ( that's think the 5th might be a baby)

# In[ ]:


train = train[train.fare_amount > 0]


# In[ ]:


train.shape


# In[ ]:


train = train.loc[train.fare_amount < 600]


# In[ ]:


train.shape


# **The latitude of New York City, NY, USA is 40.730610, and the longitude is -73.935242. **

# In[ ]:


# So I set up a longitude range for the ride
train = train.loc[train.pickup_longitude < -71]
train = train.loc[train.pickup_longitude > -74.5]


# In[ ]:


train.shape


# In[ ]:


# And a latitude range for the ride
train = train.loc[train.pickup_latitude < 42]
train = train.loc[train.pickup_latitude > 40]


# In[ ]:


train.shape


# In[ ]:


train = train.loc[train.dropoff_longitude < -71]
train = train.loc[train.dropoff_longitude > -74.5]


# In[ ]:


train = train.loc[train.dropoff_latitude < 42]
train = train.loc[train.dropoff_latitude > 40]


# In[ ]:


train.shape


# In[ ]:


train['longitude_diff'] = train['dropoff_longitude'] - train['pickup_longitude']

train['latitude_diff'] = train['dropoff_latitude'] - train['pickup_latitude']


# In[ ]:


train = train.loc[train.longitude_diff < 5]
train = train.loc[train.longitude_diff > -5]


# In[ ]:


train = train.loc[train.latitude_diff < 5]
train = train.loc[train.latitude_diff > -5]


# In[ ]:


train = train.loc[train.passenger_count > 0]
train = train.loc[train.passenger_count <= 7]


# In[ ]:


train.shape


# In[ ]:


train.head()


# ## Feature Engineering (Derivative Features)

# In[ ]:


target = train[['fare_amount']]
train_df = train.drop('fare_amount',axis=1)
test = pd.read_csv('../input/test.csv')

test['longitude_diff'] = test['dropoff_longitude'] - test['pickup_longitude']
test['latitude_diff'] = test['dropoff_latitude'] - test['pickup_latitude']

test_df = test
train_df['is_train'] = 1
test_df['is_train'] = 0
train_test = pd.concat([train_df,test_df],axis=0)


# In[ ]:


train_test['year'] = train_test.pickup_datetime.apply(lambda x: x[:4])


# In[ ]:


train_test['month'] = train_test.pickup_datetime.apply(lambda x: x[5:7])


# In[ ]:


train_test['hour'] = train_test.pickup_datetime.apply(lambda x: x[11:13])


# In[ ]:


import datetime

train_test['pickup_datetime'] = train_test.pickup_datetime.apply(
    lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d'))


# In[ ]:


train_test['day_of_week'] = train_test.pickup_datetime.apply(lambda x: x.weekday())


# In[ ]:


train_test['pickup_date'] = train_test.pickup_datetime.apply(lambda x: x.date())


# In[ ]:


from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start='2009-01-01', end='2015-12-31').to_pydatetime()

train_test['holidat_or_not'] = train_test.pickup_datetime.apply(lambda x: 1 if x in holidays else 0)


# In[ ]:


train_test = train_test.drop(['key','pickup_datetime','pickup_date'],axis=1)


# In[ ]:


train_test.info()


# In[ ]:


train_test['year'] = train_test['year'].astype('int')
train_test['hour'] = train_test['hour'].astype('int')


# In[ ]:


train_test.head()


# ## Clustering
# 
# I want to cluster the longtitude and latitude into 6 clusters, and make them dummy variables.

# In[ ]:


plt.scatter(train_test['pickup_longitude'],train_test['pickup_latitude'],alpha=0.2)


# In[ ]:


plt.scatter(train_test['dropoff_longitude'],train_test['dropoff_latitude'],alpha=0.2)


# In[ ]:


test = train_test[train_test['pickup_longitude'] < -73]
test = test[test['pickup_latitude'] < 41.5]

test.head()


# In[ ]:


plt.scatter(test['pickup_longitude'],test['dropoff_longitude'],alpha=0.2)


# In[ ]:


plt.scatter(test['pickup_latitude'],test['dropoff_latitude'],alpha=0.2)


# In[ ]:


from sklearn.cluster import KMeans

train_test_geo = train_test[['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']]

model = KMeans(n_clusters = 4)
model.fit(train_test_geo)
labels = model.predict(train_test_geo)


# In[ ]:


train_test['cluster'] = labels

clusters = pd.get_dummies(train_test['cluster'],prefix='Cluster',drop_first=False)

train_test = pd.concat([train_test,clusters],axis=1).drop('cluster',axis=1)


# In[ ]:


train = train_test[train_test.is_train == 1].drop(['is_train'],axis=1)
test = train_test[train_test.is_train == 0].drop(['is_train'],axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Simple Exploratory

# In[ ]:


train['fare_amount'] = target

matrix = train.corr()
sns.heatmap(matrix)


# In[ ]:


train_subset = train[:1000]
sns.pairplot(train_subset, vars=['fare_amount', 'passenger_count', 'longitude_diff','latitude_diff',
                          'year', 'hour', 'day_of_week','Cluster_0','Cluster_1','Cluster_2','Cluster_3'])


# ## Modeling

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


X = train.drop('fare_amount',axis=1)
y = train[['fare_amount']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


from sklearn.model_selection import GridSearchCV

scorer = metrics.make_scorer(metrics.mean_squared_error)

clf = RandomForestRegressor()

parameters = {'n_estimators': [25], 'max_features': [5,15], 'max_depth': [15,30],
              'min_samples_split':[3],'min_samples_leaf':[2], 'random_state':[0]}

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

best_predictions = best_clf.predict(X_test)

error = np.sqrt(metrics.mean_squared_error(y_test,best_predictions))
print(error)


# In[ ]:


features = X.columns[:X.shape[1]]
importances = best_clf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(best_clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# In[ ]:


predictions = best_clf.predict(test)

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = predictions
submission.to_csv('submission.csv',index=False)

