#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ## Feature Engineering

# In[ ]:


# import our dataset
data = pd.read_csv('../input/avocado-prices/avocado.csv')


# In[ ]:


# first 10 observations of our dataset
data.head(10)


# In[ ]:


# renaming column names into meaningful names (refer kaggle's avacado dataset description)
data = data.rename(columns={'4046':'PLU_4046','4225':'PLU_4225','4770':'PLU_4770'})


# In[ ]:


# removing unnecessary column
data = data.drop(['Unnamed: 0'],axis = 1)
data.head(10)


# In[ ]:


# convert the type of Date feature from obj to datetime type
data['Date'] = pd.to_datetime(data['Date'])


# In[ ]:


# categorizing into several seasons
def season_of_date(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'


# In[ ]:


# creating a new feature 'season' and assign the corresponding season for the Date using map function over our season_of_date function
data['season'] = data.Date.map(season_of_date)


# In[ ]:


# now, we can see the season feature appended at the last
data.head(10)


# In[ ]:


# no of observations for each seasons
data.season.value_counts()


# In[ ]:


# droping date feature
data = data.drop(['Date'],axis = 1)


# ## Data Preprocessing

# In[ ]:


# converting categorical features of text data into model-understandable numerical data
label_cols = ['type','region','season']
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data[label_cols] = data[label_cols].apply(lambda x : label.fit_transform(x))


# In[ ]:


# Scaling the features and 
# spliting the label encoded features into distinct features inorder to prevent our model to think that columns have data with some kind of order or hierarchy
# column_tranformer allows us to combine several feature extraction or transformation methods into a single transformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
scale_cols = data.drop(['AveragePrice','type','year','region','season'],axis=1)
col_trans = make_column_transformer(
            (OneHotEncoder(), data[label_cols].columns),
            (StandardScaler(), scale_cols.columns),
            remainder = 'passthrough')


# ## Train Test Split

# In[ ]:


# splitting our dataset into train and test set such that 20% of observations are considered as test set
X = data.drop(['AveragePrice'],axis=1)
y = data.AveragePrice
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Regression Models
# 
# ### 1. Linear Regression

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
pipe = make_pipeline(col_trans,linreg)
pipe.fit(X_train,y_train)


# In[ ]:


y_pred_test = pipe.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# ### 2. Support Vector Regressor (SVR)

# In[ ]:


from sklearn.svm import SVR
svr = SVR()
pipe = make_pipeline(col_trans,svr)
pipe.fit(X_train,y_train)


# In[ ]:


y_pred_test = pipe.predict(X_test)


# In[ ]:


print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# ### 3. Decision Tree Regressor

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dr=DecisionTreeRegressor()
pipe = make_pipeline(col_trans,dr)
pipe.fit(X_train,y_train)


# In[ ]:


y_pred_test = pipe.predict(X_test)


# In[ ]:


print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# ### 4. Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor()
pipe = make_pipeline(col_trans,forest_model)
pipe.fit(X_train,y_train)


# In[ ]:


y_pred_test = pipe.predict(X_test)


# In[ ]:


print('MAE for testing set: {}'.format(mean_absolute_error(y_pred_test,y_test)))
print('MSE for testing set: {}'.format(mean_squared_error(y_pred_test,y_test)))
print('RMSE for testing set: {}'.format(np.sqrt(mean_squared_error(y_pred_test,y_test))))


# In[ ]:


sns.distplot((y_test-y_pred_test),bins=50)


# Notice here that our residuals looked to be normally distributed and that's really a good sign which means that our model was a correct choice for the data.

# RandomForestRegressor outperfomed LinearRegression, SVR and DecisionTreeRegressor with an RMSE of 0.148.
# 
# We can increase the performance to some more extent by tweaking the parameters of the models (especially for the models like RandomForestRegressor and DecisionTreeRegressor) and can use hyperparameter tuning techniques such as GridSearchCV and RandomizedSearchCV to find out the best parameters for our models!
# 
# Thanks! Please do Upvote if you like my notebook. Any Suggestions are welcome!
