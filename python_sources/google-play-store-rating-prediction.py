#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ## Data Cleaning

# In[ ]:


data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


data.head(10)


# In[ ]:


# Removing irrelevant features
data = data.drop(['App','Last Updated','Current Ver','Android Ver'],axis='columns')


# In[ ]:


data.head(10)


# In[ ]:


# checking for null values
data.isna().sum()


# In[ ]:


# drop the entire record if null value is present in 'any' of the feature
data.dropna(how='any',inplace=True)


# In[ ]:


data.shape


# In[ ]:


data.isna().sum()


# In[ ]:


data.dtypes


# In[ ]:


# changing the datatype of Review column from integer from object
data = data.astype({'Reviews':'int'})


# In[ ]:


data.Size.value_counts().head()


# In[ ]:


data.Size.value_counts().tail()


# In[ ]:


# Replacing 'Varies with device' value with Nan values
data['Size'].replace('Varies with device', np.nan, inplace = True ) 


# In[ ]:


# Removing the suffixes (k and M) and representing all the data as bytes 
# (i.e)for k, value is multiplied by 100 and for M, the value is multiplied by 1000000 
data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) *              data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))


# In[ ]:


# filling "Varies with device" with mean of size in each category
data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)


# In[ ]:


# Removing comma(,) and plus(+) signs
data.Installs = data.Installs.apply(lambda x: x.replace(',',''))
data.Installs = data.Installs.apply(lambda x: x.replace('+',''))


# In[ ]:


# changing the datatype from object to integer
data = data.astype({'Installs':'int'})


# In[ ]:


data.Price.value_counts()


# In[ ]:


# Removing dollar($) sign and changing the type to float
data.Price = data.Price.apply(lambda x: x.replace('$',''))
data['Price'] = data['Price'].apply(lambda x: float(x))


# In[ ]:


data.Genres.value_counts().tail()


# Many genre contain only few record, it may make a bias.
# Then, I decide to group it to bigger genre by ignore sub-genre (after " ; " sign)

# In[ ]:


data['Genres'] = data.Genres.str.split(';').str[0]


# In[ ]:


data.Genres.value_counts()


# In[ ]:


# Group Music & Audio as Music
data['Genres'].replace('Music & Audio', 'Music',inplace = True)


# In[ ]:


data['Content Rating'].value_counts()


# In[ ]:


# Removing the entire row from the data where content rating is unrated as there is only one row
data = data[data['Content Rating'] != 'Unrated']


# In[ ]:


data.dtypes


# ## Data Preprocessing

# In[ ]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
column_trans = make_column_transformer(
                (OneHotEncoder(),['Category','Installs','Type','Content Rating','Genres']),
                (StandardScaler(),['Reviews','Size','Price']),
                remainder = 'passthrough')


# ## Train Test Split

# In[ ]:


# Choosing X and y value
X = data.drop('Rating',axis='columns')
y = data.Rating


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


# In[ ]:


column_trans.fit_transform(X_train)


# ## Regression Models

# ### 1. Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
linreg = LinearRegression()
pipe = make_pipeline(column_trans,linreg)


# In[ ]:


from sklearn.model_selection import cross_validate
linreg_score = cross_validate(pipe, X_train, y_train, cv=10, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)
print('Mean Absolute Error: {}'.format(linreg_score['test_neg_mean_absolute_error'].mean()))
print('Mean Squared Error: {}'.format(linreg_score['test_neg_mean_squared_error'].mean()))
print('Root Mean Squared Error: {}'.format(np.sqrt(-linreg_score['test_neg_mean_squared_error'].mean())))


# ### 2. Support Vector Regressor (SVR)

# In[ ]:


from sklearn.svm import SVR
svr = SVR()
pipe = make_pipeline(column_trans,svr)
svr_score = cross_validate(pipe, X_train, y_train, cv=10, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)
print('Mean Absolute Error: {}'.format(svr_score['test_neg_mean_absolute_error'].mean()))
print('Mean Squared Error: {}'.format(svr_score['test_neg_mean_squared_error'].mean()))
print('Root Mean Squared Error: {}'.format(np.sqrt(-svr_score['test_neg_mean_squared_error'].mean())))


# ### 3. Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(n_estimators=100, max_features=3, min_samples_leaf=10)
pipe = make_pipeline(column_trans,forest_model)
rfr_score = cross_validate(pipe, X_train, y_train, cv=10, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],return_train_score=False)
print('Mean Absolute Error: {}'.format(rfr_score['test_neg_mean_absolute_error'].mean()))
print('Mean Squared Error: {}'.format(rfr_score['test_neg_mean_squared_error'].mean()))
print('Root Mean Squared Error: {}'.format(np.sqrt(-rfr_score['test_neg_mean_squared_error'].mean())))


# ## Testing on Test Set

# ### 1. Linear Regression

# In[ ]:


pipe = make_pipeline(column_trans,linreg)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
print('Mean Absolute Error: {}'.format(mean_absolute_error(y_pred,y_test)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred,y_test)))
print('Root Mean Squared Error: {}'.format(np.sqrt(mean_absolute_error(y_pred,y_test))))


# ### 2. Support Vector Regressor

# In[ ]:


pipe = make_pipeline(column_trans,svr)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


# In[ ]:


print('Mean Absolute Error: {}'.format(mean_absolute_error(y_pred,y_test)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred,y_test)))
print('Root Mean Squared Error: {}'.format(np.sqrt(mean_absolute_error(y_pred,y_test))))


# ### 3. Random Forest Regressor

# In[ ]:


pipe = make_pipeline(column_trans,forest_model)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)


# In[ ]:


print('Mean Absolute Error: {}'.format(mean_absolute_error(y_pred,y_test)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred,y_test)))
print('Root Mean Squared Error: {}'.format(np.sqrt(mean_absolute_error(y_pred,y_test))))

