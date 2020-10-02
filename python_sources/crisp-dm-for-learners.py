#!/usr/bin/env python
# coding: utf-8

# Let us understand CRISP-DM for begineers.
# 
# 
# 
# 
# https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining

# Business Understanding = We are trying to predict the rent of the houses in Brazil. The output is the predicted rent amount. 

# In[ ]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


input_directory = '/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv'


# In[ ]:


input_dataset = pd.read_csv(input_directory)


# Data Understanding - we have to understand what kind of data we have.
# Data Preparation - we will preprocess the dataset that we have.

# In[ ]:


input_dataset.shape


# There are 10,692 samples and 13 features in the dataset. 13 features include the target feature

# In[ ]:


input_dataset.head()


# In[ ]:


input_dataset.dtypes


# The feature 'floor' is supposed to be a numeric feature. We have to change it to number.

# In[ ]:


input_dataset['floor'].value_counts()


# The feature floor is having 2461 values as '-'. Which means no floors present. We have to change '-' to 0 and convert them into numbers. 

# In[ ]:


input_dataset['floor'] = input_dataset['floor'].replace('-',0)


# In[ ]:


input_dataset['floor'].value_counts()


# In[ ]:


input_dataset['floor'] = pd.to_numeric(input_dataset['floor'])


# Now we have changed the feature to numeric feature. Let us check the missing values now. 

# In[ ]:


input_dataset.isnull().sum()


# In[ ]:


input_dataset['animal'].value_counts()


# In[ ]:


input_dataset['furniture'].value_counts()


# Let us change these three features to numeric using label encoders.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[ ]:


input_dataset['city'] = encoder.fit_transform(input_dataset['city'])
input_dataset['animal'] = encoder.fit_transform(input_dataset['animal'])
input_dataset['furniture'] = encoder.fit_transform(input_dataset['furniture'])


# In[ ]:


input_dataset.dtypes


# Now all the datatypes are numeric.

# In[ ]:


sns.heatmap(input_dataset.corr())


# The 'total' is highly correlated with 'hoa', The 'rent amount' is highly correlated with 'fire insurance' .......INTERESTING !!!

# In[ ]:


plt.plot(input_dataset['total (R$)'], input_dataset['hoa (R$)'], 'o')
plt.suptitle('total VS hoa')


# Looks like we got some outliers in the dataset.

# In[ ]:


# Let us remove the outliers
input_dataset.sort_values('hoa (R$)', ascending = False).head(10)
# the Indexes - [255, 6979, 6230, 2859, 2928, 1444] are outliers. Let us remove them


# In[ ]:


input_dataset = input_dataset.drop([255, 6979, 6230, 2859, 2928, 1444])


# In[ ]:


input_dataset.sort_values('hoa (R$)', ascending = False).head()


# In[ ]:


plt.plot(input_dataset['total (R$)'], input_dataset['hoa (R$)'], 'o')
plt.suptitle('total VS hoa (after removing outliers)')


# We still have some outliers

# In[ ]:


input_dataset.sort_values('total (R$)', ascending = False).head()


# In[ ]:


input_dataset = input_dataset.drop([6645, 2182])


# In[ ]:


plt.plot(input_dataset['total (R$)'], input_dataset['hoa (R$)'], 'o')
plt.suptitle('total VS hoa (after removing outliers)')


# Now let us check 'rent amount' and 'fire insurance'

# In[ ]:


plt.plot(input_dataset['rent amount (R$)'], input_dataset['fire insurance (R$)'], 'o')
plt.suptitle('rent amount VS fire insurance')


# In[ ]:


input_dataset.sort_values('rent amount (R$)',ascending = False).head()


# In[ ]:


input_dataset.sort_values('fire insurance (R$)',ascending = False).head()


# In[ ]:


input_dataset = input_dataset.drop([7748])


# In[ ]:


plt.plot(input_dataset['rent amount (R$)'], input_dataset['fire insurance (R$)'], 'o')
plt.suptitle('rent amount VS fire insurance')


# I think we can proceed with the modelling now.

# Now the dataset has been preprocessed. Let us go to the Modelling and Evaluation stages.

# In[ ]:


# separating training and testing data
from sklearn.model_selection import train_test_split
inputs = input_dataset.drop(columns = 'rent amount (R$)')
target = input_dataset['rent amount (R$)']
train_input, test_input, train_target, test_target = train_test_split(inputs, target, test_size = 0.2, random_state = 101)


# In[ ]:


print(train_input.shape)
print(train_target.shape)
print(test_input.shape)
print(test_target.shape)


# In[ ]:


# importing machine learning algorithms
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor


# In[ ]:


# importing evaluation metrics for regression algorithms
from sklearn.metrics import mean_squared_error


# In[ ]:


# KNN Algorithm
knn = KNeighborsRegressor(n_neighbors = 2)
model = knn.fit(train_input, train_target)
print('model = ', model)
prediction = knn.predict(test_input)
print(np.sqrt(mean_squared_error(test_target, prediction)))


# In[ ]:


# Logistic Algorithm
logit = LogisticRegression()
model = logit.fit(train_input, train_target)
print('model = ',model)
prediction = model.predict(test_input)
print(np.sqrt(mean_squared_error(test_target, prediction)))


# In[ ]:


# Linear Regression Algorithm
linear = LinearRegression()
model = linear.fit(train_input, train_target)
print('model = ',model)
prediction = model.predict(test_input)
print(np.sqrt(mean_squared_error(test_target, prediction)))


# In[ ]:


# Support Vector Machine
svm = SVR()
model = svm.fit(train_input, train_target)
print('model = ', model)
prediction = model.predict(test_input)
print(np.sqrt(mean_squared_error(test_target, prediction)))


# In[ ]:


# Multilayered Perceptron
mlp = MLPRegressor()
model = mlp.fit(train_input, train_target)
print('model = ', model)
prediction = model.predict(test_input)
print(np.sqrt(mean_squared_error(test_target, prediction)))


# Linear Regression is a good algorithm for this requirement.

# Finally, the trained model will be deployed in the production.
# 
# 
# Please comment if any improvements can be done :). Thanks.
