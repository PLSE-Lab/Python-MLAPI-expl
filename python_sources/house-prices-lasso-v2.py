#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# ** Data loading **

# In[2]:


# Load train data
train = pd.read_csv('../input/train.csv')


# ** Data cleansing **

# In[3]:


# Check original train shape
train.shape


# In[4]:


# Check train data
train.head()


# In[5]:


# Check fields summary
train.describe()


# In[6]:


# Removing outliers
train = train[train['GrLivArea']<4000]
train.shape


# In[7]:


# Select target values
y = train['SalePrice']
y.head()


# In[8]:


y = np.log(y)
y.head()


# In[9]:


# Drop ID column
train_ids = train['Id']
train.drop(['Id'], axis=1, inplace=True)


# In[10]:


# Check correlation to SalePrice
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
corr.iloc[:,-1:]


# In[11]:


# Get categorical features
categorical_features = train.select_dtypes(include = ['object']).columns
categorical_features


# In[12]:


# Get numerical features and drop target column
numerical_features = train.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
numerical_features


# In[13]:


# Split numerical and categorical features
train_num = train[numerical_features]
train_cat = train[categorical_features]
(train.shape, train_num.shape, train_cat.shape)


# In[14]:


# Fill null values with median for numerical features
from sklearn.impute import SimpleImputer
print("Nulls in numerical features: " + str(train_num.isnull().values.sum()))
num_imputer = SimpleImputer(strategy = 'median')
train_num[numerical_features] = num_imputer.fit_transform(train_num)
train_num.shape


# In[15]:


# Fill null values with most frequent for categorical features
print("Nulls in categorical features: ", str(train_cat.isnull().values.sum()))
cat_imputer = SimpleImputer(strategy = 'most_frequent')
train_cat[categorical_features] = cat_imputer.fit_transform(train_cat)
train_cat.shape


# In[16]:


# Ordinal encoding
#from sklearn.preprocessing import OrdinalEncoder
#encoder = OrdinalEncoder()
#train_cat = pd.DataFrame(encoder.fit_transform(train_cat), columns = categorical_features, index = train.index)


# In[17]:


# One hot encoding (better than Ordinal for Lasso)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
train_cat = pd.DataFrame(encoder.fit_transform(train_cat), index = train.index)
train_cat.shape


# In[18]:


# Merge back categorical and numerical features
train = pd.concat([train_cat, train_num], axis=1)
train.shape


# In[19]:


# Scale data
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
train = scaler.fit_transform(train)


# ** Training **

# In[144]:


from sklearn.model_selection import train_test_split
# Split train and test data
X = train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=47)


# In[151]:


# Prepare Lasso model
#from sklearn.linear_model import Lasso
#model = Lasso(alpha=0.001, random_state=47)
from sklearn.linear_model import LassoCV
model = LassoCV(cv=20, random_state=47, verbose=True, alphas=[1e-4, 5e-4, 1e-3, 5e-3])


# In[152]:


# Execute training
model.fit(X_train, y_train)


# ** Evaluation **

# In[153]:


# Execute prediction with train and test data
y_predict = model.predict(X_test)
y_predict_train = model.predict(X_train)


# In[154]:


# Check-up real versus predicted data
true_x_pred = pd.DataFrame({'SalePrice': np.exp(y_test), 'Predicted': np.exp(y_predict)})
true_x_pred_train = pd.DataFrame({'SalePrice': np.exp(y_train), 'Predicted': np.exp(y_predict_train)})
true_x_pred.head()


# In[155]:


# Calculates root mean squared log error
from sklearn.metrics import mean_squared_log_error
from math import sqrt
rmsle = sqrt(mean_squared_log_error(true_x_pred['SalePrice'], true_x_pred['Predicted']))
rmsle_train = sqrt(mean_squared_log_error(true_x_pred_train['SalePrice'], true_x_pred_train['Predicted']))
print("RMSLE (test):", rmsle)
print("RMSLE (train):", rmsle_train)


# In[156]:


# Plot real x predicted
import matplotlib.pyplot as plt
plt.scatter(true_x_pred_train['Predicted'], true_x_pred_train['SalePrice'], c = "silver", label = "Training")
plt.scatter(true_x_pred['Predicted'], true_x_pred['SalePrice'], c = "blue", label = "Test")
plt.title("Lasso")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10, 660000], [10, 660000], c = "red")
plt.show()


# ** Load and prepare test data **

# In[62]:


test = pd.read_csv('../input/test.csv')
test_ids = test['Id']
test.shape


# In[63]:


# Drop Id column
test.drop(['Id'], axis=1, inplace=True)


# In[64]:


# Split numerical and categorical features
test_num = test[numerical_features]
test_cat = test[categorical_features]


# In[65]:


# Fill numerical nulls with median
test_num[numerical_features] = num_imputer.transform(test_num)
test_num.shape


# In[66]:


# Fill categorical nulls with most frequent
test_cat[categorical_features] = cat_imputer.transform(test_cat)
test_cat.shape


# In[67]:


# One hot encoding
test_cat = pd.DataFrame(encoder.transform(test_cat), index = test.index)
test_cat.shape


# In[68]:


# Merge back categorical and numerical
test = pd.concat([test_cat, test_num], axis=1)
(test.shape)


# In[69]:


# Scale data
test = scaler.transform(test)


# ** Predict test data **

# In[70]:


# Execute prediction
test_prediction = np.exp(model.predict(test))
output = pd.DataFrame({'Id': test_ids, 'SalePrice': test_prediction})
output.head()


# In[71]:


# Generate file for submission
output.to_csv('submission.csv', index=False)


# In[72]:


# Generate binaries for models
from sklearn.externals import joblib
joblib.dump(num_imputer, 'num_imputer.joblib')
joblib.dump(cat_imputer, 'cat_imputer.joblib')
joblib.dump(encoder, 'encoder.joblib')
joblib.dump(scaler, "scaler.joblib")
joblib.dump(model, 'model.joblib')


# ** Download links **
# 
# [submission.csv](./submission.csv)

# [num_imputer.joblib](./num_imputer.joblib)

# [cat_imputer.joblib](./cat_imputer.joblib)

# [encoder.joblib](./encoder.joblib)

# [scaler.joblib](./scaler.joblib)

# [model.joblib](./model.joblib)

# In[ ]:




