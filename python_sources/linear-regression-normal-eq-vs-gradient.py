#!/usr/bin/env python
# coding: utf-8

# #### Tags
# - Linear Regression
# - Normal Form Equation
# - SKlearn Linear Regression
# - Data Visualization

# ##### Web Page Link
# [Linear Regression]( https://sites.google.com/view/horizon-ml/machine-learning/supervised-learning/algorithms/linear-regression)

# In[ ]:


import numpy as np
import pandas as pd
import pandas_profiling as pp

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[ ]:


data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


pp.ProfileReport(data)


# #### Data Analysis - based on initial analysis
# - No null values.
# - all fields are numerical.
# - Seriel no. is just a unique identifier like index , not required for prediction
# - University ranking and Research are sort of categorical data.
# - Gre Score  can be bucketized.
# - Ration of data in Research field is almost equal.

# In[ ]:


data.columns


# In[ ]:


## Analyse the chance of Admit - based on Gre Score for different ranked University
get_ipython().run_line_magic('matplotlib', 'inline')
sns.relplot(x="GRE_Score", y="Chance_of_Admit_", hue="University_Rating", data=data);


# In[ ]:


## Analyse the chance of Admit - based on Gre Score for research and non research
sns.relplot(x="GRE_Score", y="Chance_of_Admit_", hue="Research", data=data);


# In[ ]:


### How Gre And Toffle Score are related 
sns.relplot(x="GRE_Score", y="TOEFL_Score",  data=data);
## seems to be linear relation


# In[ ]:


sns.catplot(x="Research", y="Chance_of_Admit_", data=data);


# In[ ]:


sns.catplot(x="University_Rating", y="Chance_of_Admit_", data=data);


# #### Data Preprocessing
# - shuffle data
# - divide train and test data
# - have normalized and non normalized data.
# - feature selection and label value

# In[ ]:


data = shuffle(data)


# In[ ]:


features = ['GRE_Score', 'TOEFL_Score', 'University_Rating', 'SOP','LOR_', 'CGPA', 'Research']
label = ['Chance_of_Admit_']
X = data[features]
y = data[label]


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


scaler = StandardScaler()
X_train_scal = scaler.fit_transform(X_train)
X_val_scal = scaler.transform(X_val)


# In[ ]:


print('Shape of X train', X_train.shape)
print('Shape of y train', y_train.shape)

print('Shape of X val', X_val.shape)
print('Shape of y val', y_val.shape)


# #### Prediction using Normal Equation

# In[ ]:


## Normal Form using un scaled data
get_ipython().run_line_magic('time', '')
step1 = np.dot(X_train.T, X_train)
step2 = np.linalg.pinv(step1)
step3 = np.dot(step2, X_train.T)
theta = np.dot(step3, y_train)


# In[ ]:


print('Shape of weights', theta.shape)


# In[ ]:


y_pred = np.dot(X_val, theta)


# In[ ]:


print('MSE', mean_squared_error(y_pred, y_val))
print('MAE', mean_absolute_error(y_pred, y_val))


# In[ ]:


## Normal Form using Scaler data
get_ipython().run_line_magic('time', '')
step1 = np.dot(X_train_scal.T, X_train_scal)
step2 = np.linalg.pinv(step1)
step3 = np.dot(step2, X_train_scal.T)
theta = np.dot(step3, y_train)


# In[ ]:


y_pred = np.dot(X_val_scal, theta)


# In[ ]:


print('MSE', mean_squared_error(y_pred, y_val))
print('MAE', mean_absolute_error(y_pred, y_val))


# #### Prediction using Regressor

# In[ ]:


get_ipython().run_line_magic('time', '')
fitter = SGDRegressor(loss="squared_loss", penalty=None)
fitter.fit(X_train, y_train)


# In[ ]:


y_pred = fitter.predict(X_val)


# In[ ]:


print('MSE', mean_squared_error(y_pred, y_val))
print('MAE', mean_absolute_error(y_pred, y_val))


# In[ ]:


get_ipython().run_line_magic('time', '')
fitter = SGDRegressor(loss="squared_loss", penalty=None)
fitter.fit(X_train_scal, y_train)


# In[ ]:


y_pred = fitter.predict(X_val_scal)


# In[ ]:


print('MSE', mean_squared_error(y_pred, y_val))
print('MAE', mean_absolute_error(y_pred, y_val))

