#!/usr/bin/env python
# coding: utf-8

# # Read Data to Pandas dataframes

# In[ ]:


import pandas as pd

train_file_path = '../input/house-prices-advanced-regression-techniques/train.csv' # this is the path to the Iowa data that you will use
pd_train = pd.read_csv(train_file_path)

test_file_path = '../input/house-prices-advanced-regression-techniques/test.csv' # this is the path to the Iowa data that you will use
pd_test = pd.read_csv(test_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Input files has been read !!!')


# # Analyze provided data

# In[ ]:


pd_train.head()


# In[ ]:


pd_fields = pd_train.columns
pd_fields


# In[ ]:


#['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType',
#'HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual',
#'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
#'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']
categorical_fields = pd_train.columns[pd_train.dtypes == 'object'].values
categorical_fields


# # Analyze data quality (Is there NA values ?)

# In[ ]:


na_values = pd_train.isna().sum()
na_values[na_values != 0]


# In[ ]:


na_test_values = pd_test.isna().sum()
na_test_values[na_test_values != 0]


# # Transform categorical fields to numerical values

# In[ ]:


import numpy as np


# In stead of analyzing correlation between variables, let's use a brute force Algorithm obtaining information from Numerical fields and from Categorical fields (after converting them to Numerical values to be able to use its data)

# In[ ]:


# Before transforming Categorical to Numerical values, assign a value for NA values
for field in categorical_fields:
    pd_train[field].fillna('---',inplace=True)
    pd_test[field].fillna('---',inplace=True)


# In[ ]:


all_possible_values ={}
def convertToNumeric(field):
    inverse_map = {}
    
    all_possible_values[field] = np.unique(np.concatenate([pd_train[field].unique(), pd_test[field].unique()]))
    for value in enumerate(all_possible_values[field]):
        inverse_map[value[1]] = value[0]    
    
    pd_train[field] = pd_train[field].map(inverse_map)
    pd_test[field]  = pd_test[field].map(inverse_map)
    
# Now, transform to Numerical values
for field in categorical_fields:
    convertToNumeric(field)
    
#all_possible_values    


# # Fill NA for Numerical fields

# In[ ]:


numerical_fields = set(pd_fields) - set(categorical_fields)

# Assign mean value to NA provided at Numerical fields
for field in numerical_fields:
    # print(field)
    pd_train[field].fillna(pd_train[field].mean(),inplace=True)
    if field != 'SalePrice':
        pd_test [field].fillna(pd_train[field].mean(),inplace=True)


# # Check all NA values has been converted correctly

# In[ ]:


na_values = pd_train.isna().sum()
na_values[na_values != 0]


# In[ ]:


na_test_values = pd_test.isna().sum()
na_test_values[na_test_values != 0]


# # Create Training and Validation sets (using sklearn)

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# # Id is useless for Price predictions (drop it!)

# In[ ]:


# Train
x_values = pd_train.drop(['Id','SalePrice'], axis=1).values
y_values = pd_train['SalePrice'].values

print ("x_values.shape = ", x_values.shape)
print ("y_values.shape = ", y_values.shape)

# Test
x_test = pd_test.drop('Id',axis=1).values

print ("x_test.shape = ", x_test.shape)


# # Data scaling

# In[ ]:


x_scaler = StandardScaler()
x_scaler.fit(x_values)
x_values_scaled = x_scaler.transform(x_values)
x_test_scaled   = x_scaler.transform(x_test)

#mean_sale_price = y_values.mean()

y_values_scaled = y_values

#y_scaler = StandardScaler()
#y_scaler.fit(y_values)
#y_values_scaled = y_scaler.transform(x_values)
#y_test_scaled   = y_scaler.transform(y_test)


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_values_scaled, y_values_scaled, test_size = 0.1)

print ("x_train.shape = ", x_train.shape)
print ("y_train.shape = ", y_train.shape)
print ("x_val.shape = ", x_val.shape)
print ("y_val.shape = ", y_val.shape)


# # Tensor Flow Regression

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt


# In[ ]:


# Define Dense Neuronal Network in TensorFlow
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(50,activation='sigmoid'))
model.add(tf.keras.layers.Dense(50,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='relu'))
model.add(tf.keras.layers.Dense(1))


# Use a custom defined loss function to avoid high dependecny on outliers

# In[ ]:


def tensorflow_loss_function_test():
    # TensorFlow tests to learn how to implement 'custom_loss_function'
    y_true = tf.constant([1.,2.,3,40000,5,6,7,8,9,100000000])
    y_pred = y_true * 2

    y_diff = tf.abs(tf.math.log(1 + y_true) - tf.math.log(1 + y_pred))  
    print("y_diff_count = ", y_diff.shape)
    print("")
    print("val = ", tf.keras.backend.get_value(y_diff))
    print("")
    sorted_value, sorted_indexes = tf.nn.top_k(tf.negative(y_diff), k = tf.cast(0.7 * tf.cast(tf.size(y_diff),tf.float32),tf.int32), sorted=True)
    sorted_value = tf.negative(sorted_value)
    print("sorted_diff_count = ", sorted_value.shape)
    print("sorted_val = ", tf.keras.backend.get_value(sorted_value))


# In[ ]:


def custom_loss_function(y_true, y_pred):    
    #print("y_true = ", tf.keras.backend.get_value(y_true))
    y_diff = tf.abs(tf.math.log(1+y_true) - tf.math.log(1+y_pred))    
    return tf.reduce_sum(tf.square(y_diff)) / tf.cast(tf.size(y_diff), tf.float32)

    # START : Test 
    
    #num_values = tf.size(y_diff)
    #print("y_diff_size = ", y_diff.shape)
    
    # Do not use the 5% percentile values with greater Error to avoid Outliers !!!
    # Use negative to work with minimum values, avoid maximum values for Outlayers ! 
    # num_values_for_loss = tf.cast(0.95 * tf.cast(tf.size(y_diff),tf.float32),tf.int32)
    #num_values_for_loss = 10
    #y_diff_sorted_trunc, indexes_sorted = tf.nn.top_k(tf.negative(y_diff), k = num_values_for_loss, sorted=True)
    # Non necessary to change sign, because later is 'squared'
    #y_diff_sorted_trunc = tf.negative(y_diff_sorted_trunc)
    
    # END : Test 
        
    
    


# In[ ]:


model.compile(optimizer='adam',
              loss=custom_loss_function)
history = model.fit(x_train,y_train,
                    batch_size=32,
                    epochs=250,
                    validation_data=(x_val,y_val))


# # Plot convergence graph

# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.yscale('log')
plt.show()


# This Neuronal Network is **overfitting** the input data !!! 

# # Now that the ANN has been defined - Calibrate to All train values

# In[ ]:


history = model.fit(x_values_scaled,y_values_scaled,
                    batch_size=32,
                    epochs=100)


# # Avoid Outliers from Calibration & Re-calibrate

# In[ ]:


y_pred = model.predict(x_values)
y_pred = np.ravel(y_pred)

y_diff = np.abs(np.log(1+y_values) - np.log(1+y_pred))

sort_index = np.argsort(y_diff)

max_index = int(0.95 * np.size(y_values))
index_2_consider = sort_index[:max_index]


# In[ ]:


print("min_diff_value =", y_diff[sort_index[0]])
print("max_diff_value =", y_diff[sort_index[sort_index.size-1]])
print("")
print("max_diff_value_after_trunc =", y_diff[sort_index[max_index-1]])


# In[ ]:


x_values_scaled_no_outliers = x_values_scaled[index_2_consider]
y_values_scaled_no_outliers = y_values_scaled[index_2_consider]
x_values_scaled_no_outliers.shape


# In[ ]:


history = model.fit(x_values_scaled_no_outliers,y_values_scaled_no_outliers,
                    batch_size=32,
                    epochs=100)


# # Evatuate Test set & Kaggle Submission

# In[ ]:


pd_result_test = pd.DataFrame()
pd_result_test['Id'] = pd_test['Id']
pd_result_test['SalePrice'] = model.predict(x_test_scaled)


# In[ ]:


pd_result_test.to_csv('Submission.csv',index = False)


# 
# **If you have any questions or hit any problems, come to the [Learn Discussion](https://www.kaggle.com/learn-forum) for help. **
# 
# **Return to [ML Course Index](https://www.kaggle.com/learn/machine-learning)**

# 
