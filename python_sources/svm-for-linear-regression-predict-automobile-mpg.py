#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Problem Statement: Given 8 pieces of information (features) about a vehicle, predict its mileage
# 
# 
# ##### Note:
# 
# * The file does not come with headers, so we specify them explicitly

# In[ ]:


import pandas as pd

print(pd.__version__)


# In[ ]:


auto_data = pd.read_csv('/kaggle/input/auto-mpg.data', delim_whitespace = True, header = None,
                       names = [
                                'mpg',
                                'cylinders',
                                'displacement',
                                'horsepower',
                                'weight',
                                'aceeleration',
                                'model',
                                'origin',
                                'car_name'
    ])


# In[ ]:


auto_data.head()


# In[ ]:


auto_data.info()


# In[ ]:


auto_data.describe()


# #### Convert horsepower feature to numeric

# In[ ]:


auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')


# In[ ]:


auto_data.info()


# #### Check if car_name feature add any value for modeling or not

# In[ ]:


auto_data['car_name'].nunique()


# #### Since out of 398 rows, there are 305 distinct car names so there is no value of having the feature. Drop the feature from data set

# In[ ]:


auto_data = auto_data.drop(['car_name'], axis=1)


# In[ ]:


auto_data.head()


# #### Check if there is any NaN value or not

# In[ ]:


auto_data_nan = auto_data[auto_data.isnull().any(axis=1)]
auto_data_nan.head(10)


# In[ ]:


auto_data_final = auto_data.dropna(axis=0)
auto_data_final[auto_data_final.isnull().any(axis=1)]


# In[ ]:


from sklearn.model_selection import train_test_split

X = auto_data_final.drop('mpg', axis=1)
y = auto_data_final['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state =0)


# In[ ]:


from sklearn.svm import SVR

model = SVR(kernel='linear', C=1.0)
model.fit(X_train, y_train)


# In[ ]:


model.coef_


# In[ ]:


y_predict = model.predict(X_test)


# > #### Calculate Mean Sqaured Error

# In[ ]:


from sklearn.metrics import mean_squared_error

model_mse = mean_squared_error(y_predict, y_test)
print(model_mse)


# #### You can either drop the feature or impute NaN with some meaningful value. I am going for data imputation.

# In[ ]:


# Check the correlation matrix to derive horsepower feature by help of other feature
corr = auto_data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(4)


# #### As you can see horsepower is strongly correleated with feature cylinders, displacement and weight.
# 
# #### For simplicity sake, I am considering cylinder feature

# In[ ]:


auto_data_4_cylinders = auto_data[auto_data['cylinders'] ==4]
print(len(auto_data_4_cylinders))
auto_data_4_cylinders.head()


# #### Draw histogram to understand the data distribution for feature horsepower

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

auto_data_4_cylinders['horsepower'].plot.hist(bins=10, alpha=0.5)


# #### Since histogram seems normal distribution, we can pick mean as our imputation stratergy

# In[ ]:


import numpy as np
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[ ]:


auto_data_4_cylinders['horsepower'] = imp_mean.fit_transform(auto_data_4_cylinders[['horsepower']])


# #### Check if NaN has been removed or not

# In[ ]:


auto_data_4_cylinders[auto_data_4_cylinders.isnull().any(axis=1)].head()


# #### Repeat the same process with 6 cylinder vehicles

# In[ ]:


auto_data_6_cylinders = auto_data[auto_data['cylinders']==6]
auto_data_6_cylinders.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
auto_data_6_cylinders['horsepower'].plot.hist(bins=10, alpha=0.5)


# #### It seems 160 is an outlier, so plot the histogram excluding 160

# In[ ]:


auto_data_6_cylinders[auto_data_6_cylinders['horsepower']< 160]['horsepower'].plot.hist(bins=10, alpha=0.5)


# #### This looks like normal distribution, so we can go for data imputation with mean strategy
# 
# #### Printing the target rows for imputation

# In[ ]:


auto_data_6_cylinders[auto_data_6_cylinders.isnull().any(axis=1)].head()


# In[ ]:


import numpy as np
from sklearn.impute import SimpleImputer

mean_imp = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[ ]:


mean_imp.fit(auto_data_6_cylinders[auto_data_6_cylinders['horsepower'] < 160][['horsepower']])

auto_data_6_cylinders['horsepower'] = mean_imp.transform(auto_data_6_cylinders[['horsepower']])


# #### Check if NaN has been removed from dataset or not

# In[ ]:


auto_data_6_cylinders[auto_data_6_cylinders.isnull().any(axis=1)]


#  #### Since we removed all NaN, so now it is time to merge back all dataset together

# In[ ]:


auto_data_others = auto_data[~auto_data['cylinders'].isin((4,6))]
print(len(auto_data_others))


# In[ ]:


auto_data_final = pd.concat([auto_data_others, auto_data_4_cylinders, auto_data_6_cylinders], axis=0)
print(len(auto_data_final))


# In[ ]:


# Uncomment below if you want to drop the rows rather than data imputation
# auto_data_final = auto_data.dropna(axis=0)


# In[ ]:


auto_data_final[auto_data_final.isnull().any(axis=1)]


# In[ ]:


print(len(auto_data_final))
auto_data_final.head()


# #### Start with model training
# 
# #### split the data into train/test

# In[ ]:


from sklearn.model_selection import train_test_split

X = auto_data_final.drop('mpg', axis=1)
y = auto_data_final['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state =0)


# In[ ]:


from sklearn.svm import SVR

model = SVR(kernel='linear', C=1.0)
model.fit(X_train, y_train)


# #### Check the coefficients for each of feature

# In[ ]:


model.coef_


# #### Get R-squared value for training data

# In[ ]:


model.score(X_train, y_train)


# #### Get Predictions on test data

# In[ ]:


y_predict = model.predict(X_test)


# #### Compare between predicted and actual value of mpg

# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (15, 6)

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')

plt.legend()
plt.show()


# #### Get R-square score on test data

# In[ ]:


model.score(X_test, y_test)


# #### Calculate Mean Square Error

# In[ ]:


from sklearn.metrics import mean_squared_error

model_mse = mean_squared_error(y_predict, y_test)
print(model_mse)

