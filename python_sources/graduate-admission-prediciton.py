#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression #Linear Regression model
from sklearn.metrics import mean_squared_error #Function to calculate RMSE
from sklearn.model_selection import train_test_split
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Loading the Data**

# In[ ]:


raw_data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
raw_data.head()


# In[ ]:


raw_data.describe()


# * Data has no missing or null values.
# * There's low standard deviation so there's no need to remove outliers.
# * All the data is quantifiable so the raw data can be used as it is to make predictions.

# In[ ]:


raw_data = raw_data.drop(['Serial No.'],axis = 1)
data = raw_data.copy()
data


# In[ ]:


target = data['Chance of Admit ']
inputs = data.drop(['Chance of Admit '],axis = 1)
inputs


# In[ ]:


x_train = inputs[0:399]
y_train = target[0:399]
x_test = inputs[400:399]
y_test = target[400:399]


# In[ ]:


reg = LinearRegression()
reg.fit(inputs,target)


# In[ ]:


true_value = target.loc[400:499]


# In[ ]:


predictions = pd.DataFrame({'Chance of Admit ':reg.predict(inputs.loc[400:499])})
predictions = predictions.round(2)
predictions


# In[ ]:


reg.coef_


# In[ ]:


plt.scatter(true_value,predictions,alpha = 0.8)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.show()


# Weight of Each Class

# In[ ]:


reg_summary = pd.DataFrame(inputs.columns.values,columns = ['Classes'])
reg_summary['Weights']= reg.coef_
reg_summary.round(3)


# In[ ]:


print('This model has an accuracy rate of',(reg.score(inputs,target)*100).round(3),'%')


# In[ ]:


output = pd.DataFrame({'Chance of Admit ':reg.predict(inputs)})
output.to_csv('my_submission.csv')


# # RMSE

# In[ ]:


mean_squared_error(true_value,predictions)

