#!/usr/bin/env python
# coding: utf-8

# **In this notebook I'm testing for the first time XGBoost library, by applying it on the Boston Housing dataset, on which I'll predict the house prices considering 14 features of the houses in the database.
# Firstly, I'm testing the model without tuning any parameter, and then I'll take the parameters one by one and try to improve the model's mean square error.**

# Importing libraries here:

# In[ ]:



import numpy as np 
import pandas as pd 
from xgboost import XGBRegressor
from sklearn.datasets import load_boston

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# load dataset
house_price = load_boston()
df_labels = pd.DataFrame(house_price.target)
df = pd.DataFrame(house_price.data)
print(df_labels.head())
print(df.head())


# The database is splitted in two:  1. the target variable, the price - df_labels, and 2. the rest of the independent variables, df.

# Now giving the price column its rightful name:

# In[ ]:


df_labels.columns = ['PRICE']
df.columns = house_price.feature_names
print(df_labels.head())


# And putting them together to have a full database:

# In[ ]:


df_total = df.merge(df_labels, left_index = True, right_index = True)
df_total.head()


# Now let's standardize the data and make the train/test split:

# In[ ]:


df = preprocessing.scale(df)
X_train, X_test, y_train, y_test = train_test_split(
    df, df_labels, test_size=0.3, random_state=10)


# Now starts the XGBoost part, with the first test, non-tuned:

# In[ ]:


#XGBoost part here!

my_model = XGBRegressor()
my_model.fit(X_train, y_train, verbose=False)


# Now let's have a look at the MSE on the train database:

# In[ ]:


#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
rmse 


# I get a 0.012 MSE, ok, quite impressively and not necessarily good low value, so let's see how it performs on the test set:

# In[ ]:


#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
rmse 


# Got a 3.431 MSE (for comparison, using a non-tuned Linear Regression I got 5.41 so  XGBoost - even non-tuned, seems better that Linear Regression).
# 
# Now, let's start tuning the first parameter:

# 1. n_estimators

# In[ ]:


#n_estimators usually varies between 100 and 1000 so let's try it:

my_model = XGBRegressor(n_estimators = 100)
my_model.fit(X_train, y_train, verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# So, getting the same MSE tells us that in fact n_estimators is by default 100. 
# Let's try with more options:

# In[ ]:


#n_estimators 2nd option, 200:

my_model = XGBRegressor(n_estimators = 200)
my_model.fit(X_train, y_train, verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# The MSE improved, on both datasets, even if on the test one the difference is small.

# In[ ]:


#n_estimators 3rd option, 500:

my_model = XGBRegressor(n_estimators = 500)
my_model.fit(X_train, y_train, verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# Not a significant difference, so I'll test the highest value:

# In[ ]:


#n_estimators 4th option, 1000:

my_model = XGBRegressor(n_estimators = 1000)
my_model.fit(X_train, y_train, verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# I'll stop the tests on this and try to have a look at the next parameter:

# 2. early_stopping_rounds

# In[ ]:


# starting with a minimum of 5 early stopping rounds:
my_model = XGBRegressor(n_estimators = 1000)
my_model.fit(X_train, y_train,early_stopping_rounds=5,eval_set=[(X_test, y_test)], verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# The MSE is not better than when we didn't set this parameter, so let's try with another value for it:

# In[ ]:


# continuing with 15 early stopping rounds:
my_model = XGBRegressor(n_estimators = 1000)
my_model.fit(X_train, y_train,early_stopping_rounds=15,eval_set=[(X_test, y_test)], verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# 3.4303 MSE, slightly better.

# In[ ]:


# continuing with 15 early stopping rounds:
my_model = XGBRegressor(n_estimators = 1000)
my_model.fit(X_train, y_train,early_stopping_rounds=50,eval_set=[(X_test, y_test)], verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# In[ ]:


# continuing with ( too much) 100 early stopping rounds:
my_model = XGBRegressor(n_estimators = 1000)
my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# I'll stop here with the early stopping rounds parameter and move on to the next one:

# 3. learning_rate

# In[ ]:


# starting with a small learning rate:
my_model = XGBRegressor(n_estimators = 1000,learning_rate=0.05)
my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# Smallest MSE until now, 3.42; let's increase the learning rate:

# In[ ]:


# increasing the learning rate:
my_model = XGBRegressor(n_estimators = 1000,learning_rate=0.1)
my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)

#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# MSE even lower, nice, should I try even more?

# In[ ]:


my_model = XGBRegressor(n_estimators = 1000,learning_rate=0.15)
my_model.fit(X_train, y_train,early_stopping_rounds=100,eval_set=[(X_test, y_test)], verbose=False)
#early_stopping_rounds can very well be kept at a much lower value, as it doesn't make a big difference
#on train set
from sklearn.metrics import mean_squared_error
y_train_predicted = my_model.predict(X_train)
mse = mean_squared_error(y_train_predicted, y_train)
rmse = np.sqrt(mse)
print(rmse)

#on test set
y_test_predicted = my_model.predict(X_test)
mse = mean_squared_error(y_test_predicted, y_test)
rmse = np.sqrt(mse)
print(rmse)


# No. It's already starting to increase again.

# **Conclusion**

# **For a variable that looks as follows, the best option when tuning XGBoost parameters was one that produced a MSE of 3.37 on the test set; this isn't, by no means, the perfect solution. The next step is to automatize this testing part so I could get a better match for the three parameters I worked on.**

# In[ ]:


df_labels.describe()

