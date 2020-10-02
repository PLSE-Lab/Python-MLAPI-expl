#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:



data = pd.read_csv('../input/Automobile_data.csv')
list(data)


# In[5]:


data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')
data.dropna(subset=['price','horsepower'], inplace=True)
from scipy.stats import pearsonr
pearsonr(data['horsepower'],data['price'])


# In[6]:


from bokeh.io import output_notebook
from bokeh.plotting import ColumnDataSource, figure, show
output_notebook()
source = ColumnDataSource(data=dict(
x=data['horsepower'],
y=data['price'],
make=data['make'],
))
tooltips = [
    ('make','@make'),
    ('horsepower','$x'),
    ('price','$y{$0}')
]
p= figure(plot_width=600, plot_height=400, tooltips=tooltips)
p.xaxis.axis_label = 'horsepower'
p.yaxis.axis_label = 'price'

p.circle('x','y',source=source,size=8,color='blue',alpha=0.5)
show(p)


# In[8]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)
from sklearn import linear_model
model = linear_model.LinearRegression()
training_x = np.array(train['horsepower']).reshape(-1,1)
training_y = np.array(train['price'])
model.fit(training_x, training_y)
slope = np.asscalar(np.squeeze(model.coef_))
intercept = model.intercept_
print('slope:',slope,'intercept:',intercept)


# In[9]:


from bokeh.models import Slope
best_line = Slope(gradient=slope, y_intercept=intercept, line_color='red', line_width=3)
p.add_layout(best_line)
show(p)


# In[13]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def predict_metrics(lr,x,y):
    pred = lr.predict(x)
    mae=mean_absolute_error(y, pred)
    mse=mean_squared_error(y,pred)
    r2=r2_score(y,pred)
    return  mae,mse,r2
training_mae,training_mse,training_r2 = predict_metrics(model, training_x, training_y)
test_x=np.array(test['horsepower']).reshape(-1,1)
test_y=np.array(test['price'])
test_mae,test_mse,test_r2 = predict_metrics(model,test_x,test_y)
print('training mean error:',training_mae,'training_mse:',training_mse,'training_r2:',training_r2)
print('test mean arror:',test_mae,'teat_mse:',test_mse,'test_r2:',test_r2)


# In[18]:


cols=['horsepower','engine-size','peak-rpm','length','width','height']
for col in cols:
    data[col]=pd.to_numeric(data[col], errors='coerce')
    
data.dropna(subset=['price','horsepower'],inplace=True)
for col in cols:
    print(col, pearsonr(data[col],data['price']))


# In[21]:


model_cols=['horsepower','engine-size','peak-rpm','length','width','height']
multi_x = np.column_stack(tuple(data[col] for col in model_cols))
multi_train_x,multi_test_x,multi_train_y,multi_test_y=    train_test_split(multi_x,data['price'],test_size=0.25)


# In[23]:


multi_model = linear_model.LinearRegression()
multi_model.fit(multi_train_x,multi_train_y)
multi_intercept=multi_model.intercept_
multi_coeffs=dict(zip(model_cols,multi_model.coef_))
print('intercepr:',multi_intercept)
print('coefficients:',multi_coeffs)


# In[24]:


multi_train_mae,multi_train_mse,multi_train_r2 = predict_metrics(multi_model, multi_train_x, multi_train_y)
multi_test_mae,multi_test_mse,multi_test_r2 = predict_metrics(multi_model,multi_test_x,multi_test_y)
print('multi training mean error:',multi_train_mae,'multi_training_mse:',multi_train_mse,'multi training_r2:',multi_train_r2)
print('test mean arror:',multi_test_mae,'teat_mse:',multi_test_mse,'test_r2:',multi_test_r2)

