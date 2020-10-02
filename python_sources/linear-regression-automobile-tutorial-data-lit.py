#!/usr/bin/env python
# coding: utf-8

# In[14]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[15]:


# Load the dataset
data = pd.read_csv('../input/Automobile_data.csv')
# List the available columns
list(data)


# In[16]:


# Preprocess the dataset by coercing the important columns to numeric values
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')
# And removing any rows which contain missing data
data.dropna(subset=['price', 'horsepower'], inplace=True)


# In[17]:


from scipy.stats.stats import pearsonr
pearsonr(data['horsepower'], data['price'])


# In[18]:


from bokeh.io import output_notebook
from bokeh.plotting import ColumnDataSource, figure, show

# enable notebook output
output_notebook()

source = ColumnDataSource(data=dict(
    x=data['horsepower'],
    y=data['price'],
    make=data['make'],
))

tooltips = [
    ('make', '@make'),
    ('horsepower', '$x'),
    ('price', '$y{$0}')
]

p = figure(plot_width=600, plot_height=400, tooltips=tooltips)
p.xaxis.axis_label = 'Horsepower'
p.yaxis.axis_label = 'Price'

# add a square renderer with a size, color, and alpha
p.circle('x', 'y', source=source, size=8, color='blue', alpha=0.5)

# show the results
show(p)


# In[19]:


# split our data into train (75%) and test (25%) sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)


# In[20]:


from sklearn import linear_model
model = linear_model.LinearRegression()
# the linear regression model expects a 2d array, so we add an extra dimension with reshape
# input: [1, 2, 3], output: [ [1], [2], [3] ]
# this allows us to regress on multiple independent variables later
training_x = np.array(train['horsepower']).reshape(-1, 1)
training_y = np.array(train['price'])
# perform linear regression
model.fit(training_x, training_y)
# output is a nested array in the form of [ [1] ]
# squeeze removes all zero dimensions -> [1]
# asscalar turns a single number array into a number -> 1
slope = np.asscalar(np.squeeze(model.coef_))
intercept = model.intercept_
print('slope:', slope, 'intercept:', intercept)


# In[21]:


# Now let's add the line to our graph
from bokeh.models import Slope
best_line = Slope(gradient=slope, y_intercept=intercept, line_color='red', line_width=3)
p.add_layout(best_line)
show(p)


# In[22]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# define a function to generate a prediction and then compute the desired metrics
def predict_metrics(lr, x, y):
    pred = lr.predict(x)
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    return mae, mse, r2

training_mae, training_mse, training_r2 = predict_metrics(model, training_x, training_y)

# calculate with test data so we can compare
test_x = np.array(test['horsepower']).reshape(-1, 1)
test_y = np.array(test['price'])
test_mae, test_mse, test_r2 = predict_metrics(model, test_x, test_y)

print('training mean error:', training_mae, 'training mse:', training_mse, 'training r2:', training_r2)
print('test mean error:', test_mae, 'test mse:', test_mse, 'test r2:', test_r2)


# In[23]:


cols = ['horsepower', 'engine-size', 'peak-rpm', 'length', 'width', 'height']
# preprocess the data as before (coerce to number)
for col in cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
# And removing any rows which contain missing data
data.dropna(subset=['price', 'horsepower'], inplace=True)

# Let's see how strongly each column is correlated to price
for col in cols:
    print(col, pearsonr(data[col], data['price']))


# In[24]:


# split train and test data as before
model_cols = ['horsepower', 'engine-size', 'length', 'width']
multi_x = np.column_stack(tuple(data[col] for col in model_cols))
multi_train_x, multi_test_x, multi_train_y, multi_test_y =     train_test_split(multi_x, data['price'], test_size=0.25)


# In[25]:


# fit the model as before
multi_model = linear_model.LinearRegression()
multi_model.fit(multi_train_x, multi_train_y)
multi_intercept = multi_model.intercept_
multi_coeffs = dict(zip(model_cols, multi_model.coef_))
print('intercept:', multi_intercept)
print('coefficients:', multi_coeffs)


# In[26]:


# calculate error metrics
multi_train_mae, multi_train_mse, multi_train_r2 = predict_metrics(multi_model, multi_train_x, multi_train_y)
multi_test_mae, multi_test_mse, multi_test_r2 = predict_metrics(multi_model, multi_test_x, multi_test_y)
print('training mean error:', multi_train_mae, 'training mse:', multi_train_mse, 'training r2:', multi_train_r2)
print('test mean error:', multi_test_mae, 'test mse:', multi_test_mse, 'test r2:', multi_test_r2)


# In[ ]:




