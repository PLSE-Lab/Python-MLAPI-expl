#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()


# In[ ]:


data.isnull().sum()


# > last_review & reviews_per_month has lots of missing data

# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.describe()


# ***data.describe gives the mean, max and other datas,  e.g. price column has mean of 152 dollars and max value is 10,000 , which means there are outliers in the data set***

# ***What kind of data are there in last_review?

# In[ ]:


data['last_review'].describe()


# In[ ]:


data['reviews_per_month'].fillna(method ='ffill', inplace = True) ## Forward fill menthod


# ***Instead of eliminating this data, I have decided to fill the missing values of both the review will forward fill method.***

# ***Above I have added forward fill in "reviews_per_month" column wherever there were Null values***

# In[ ]:


data['last_review'].fillna(method ='ffill', inplace = True) ## Forward fill menthod


# ***Above I have added forward fill in "last_review" column wherever there were Null values***

# In[ ]:


data['reviews_per_month'].isnull().sum()


# In[ ]:


data['last_review'].isnull().sum()


# In[ ]:


data['reviews_per_month'].describe()


# In[ ]:


data.isnull().sum()


# ***there are still null values in th data which we will eliminate using dropna method as below. Also I will eliminate the latitude and longitude as I will not be using them in this excersie***

# In[ ]:


data = data.dropna(axis = 0) # ***droping all rows conatining null values***#
#data.drop(['latitude', 'longitude'], axis = 1, inplace =True)
data.head()


# In[ ]:


data.drop(['latitude', 'longitude'], axis = 1, inplace =True)
data.head()


# In[ ]:


import seaborn as sns
sns.set(rc={'figure.figsize':(10,5)})
sns.set(font_scale=1)
sns.relplot(x="minimum_nights",y="price", data=data)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.boxplot(x=data['reviews_per_month'])


# ***As I mentioned earlier there are outliers in the data which can be seen also thro the box plot***

# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))
ax.bar(data['price'], data['neighbourhood_group'])
ax.set_ylabel('Area')
ax.set_xlabel('Price')
plt.show()


# In[ ]:


Q1 = data.quantile(0.15)
Q3 = data.quantile(0.85)
IQR = Q3 - Q1
print(IQR)

***To remove the outliers I have used the IQR method***
# In[ ]:


data_out = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
data_out.shape


# *** With the above code I am removing all the outliers from the dataset.
# data_out contains values without ouliers***

# In[ ]:


data_out.head()


# In[ ]:


sns.barplot(x='minimum_nights', y='price', data = data_out)


# *releationship between number of nights and price is shown above*

# In[ ]:


fig, ax = plt.subplots(figsize=(5,5))
ax.bar(data_out['price'], data_out['neighbourhood_group'])
ax.set_ylabel('Area')
ax.set_xlabel('Price')
ax.set_color = 'b'
plt.show()


# ***Relationship between price and area is shown above***

# In[ ]:


sns.barplot(x=data_out['room_type'], y=data_out['price'], data = data_out)


# ***Enitire home/Apt have the highest price as comapred to private and shared rooms***

# In[ ]:


sns.barplot(x=data_out['neighbourhood_group'], y=data_out['minimum_nights'], data = data_out)


# ***Brooklyn has average booking of 3 nights***

# In[ ]:


chart = sns.barplot(x=data_out['neighbourhood_group'], y=data_out['availability_365'], data = data_out)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)


# ***Saten Island has availablity of more than 175 days***

# In[ ]:


chart = sns.barplot(x=data_out['room_type'], y=data_out['number_of_reviews'], data = data_out)


# In[ ]:


neig_price = data_out.groupby(by = ['neighbourhood_group','neighbourhood'])[ 'price'].count()
neig_price


# ***Above code shows the neighbourhood_group and neighbourhood in relation to price count***

# In[ ]:


neig_min_nigths = data_out.groupby(by = ['neighbourhood_group','host_id'])[ 'minimum_nights'].count()
neig_min_nigths


# ***Above code shows the neighbourhood_group and host_id in relation to mini_nights***

# In[ ]:


neig_prop_type = data_out.groupby(by = ['room_type','neighbourhood_group','neighbourhood'])[ 'price'].count()
neig_prop_type


# ***Above code shows the room_type & neighbourhood_group in relation to price count***

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn import preprocessing


# In[ ]:


data_out.head()


# In[ ]:


df= data_out.drop(['name', 'host_id','last_review', 'reviews_per_month', 'host_name'], axis = 1)
df.head()


# In[ ]:


df.columns


# In[ ]:


le = preprocessing.LabelEncoder()

df['neighbourhood_group'] =le.fit_transform(df['neighbourhood_group'])
df['neighbourhood'] =le.fit_transform(df['neighbourhood'])
df['room_type'] =le.fit_transform(df['room_type'])


# In[ ]:


X = df.drop(['price'], axis=1)
y =df['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

print('Mean Absolute Error_lng:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_lng:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_lng:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_lng:', r2_score(y_test, y_pred).round(3))


# In[ ]:


ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train) #training the algorithm

y_pred = ridge.predict(X_test)

print('Mean Absolute Error_ridge:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_ridge:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_ridge:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_ridge:', r2_score(y_test, y_pred).round(3))


# In[ ]:


clf = Lasso(alpha=0.1)

clf.fit(X_train, y_train) #training the algorithm

y_pred = clf.predict(X_test)

print('Mean Absolute Error_lasso:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_lasso:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_lasso:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_lasso:', r2_score(y_test, y_pred).round(3))


# In[ ]:


logreg = LogisticRegression(solver = 'lbfgs')
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print('Mean Absolute Error_logreg:', metrics.mean_absolute_error(y_test, y_pred).round(3))  
print('Mean Squared Error_logreg:', metrics.mean_squared_error(y_test, y_pred).round(3))  
print('Root Mean Squared Error_logreg:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))
print('r2_score_logreg:', r2_score(y_test, y_pred).round(3))


# In[ ]:


# create regressor object 
rfe = RandomForestRegressor(n_estimators = 100, random_state = 42) 
 
# fit the regressor with x and y data 
rfe.fit(X, y)   
y_pred=rfe.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_score_RFE:', r2_score(y_test, y_pred).round(3))


# In[ ]:


ABR = AdaBoostRegressor(n_estimators = 100, random_state = 42) 
  
# fit the regressor with x and y data 
ABR.fit(X, y)   
y_pred=ABR.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('r2_score_ABR:', r2_score(y_test, y_pred).round(3))


# ***Conclusion: Based on RMSE value RandomForest Regressor is the best suited model***

# In[ ]:




