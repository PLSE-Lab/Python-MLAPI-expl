#!/usr/bin/env python
# coding: utf-8

# Importing the essential libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset

# In[ ]:


dataset = pd.read_csv('../input/Training_Dataset_v2.csv')


# A look at the first 5 values of the Dataset

# In[ ]:


dataset.head()


# A look at the last 5 values of the Dataset

# In[ ]:


dataset.tail()


# lead_time has missing values as Nan

# In order to replace them,Simple Imputer is used

# In[ ]:


from sklearn.impute import SimpleImputer
dataset["lead_time"] = SimpleImputer(strategy = "median").fit_transform(dataset["lead_time"].values.reshape(-1,1))


# NaN values have been removed from lead_time

# In[ ]:


dataset.head()


# In[ ]:


dataset = dataset.dropna()


# Negeative values exist in perf_6_month_avg and perf_12_month_avg

# In[ ]:


for col in ['perf_6_month_avg','perf_12_month_avg']:
    dataset[col] = SimpleImputer(missing_values=-99,strategy='mean').fit_transform(dataset[col].values.reshape(-1,1))


# Negative values have been replaced

# In[ ]:


dataset.head()


# Encoding the categorical features having Yes and No

# In[ ]:


for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
               'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
        dataset[col] = (dataset[col] == 'Yes').astype(int)


# In[ ]:


dataset.head()


# Since ANN is going to be used for prediction,Feature Scaling is used 

# In[ ]:


from sklearn.preprocessing import normalize
qty_related = ['national_inv', 'in_transit_qty', 'forecast_3_month', 
                   'forecast_6_month', 'forecast_9_month', 'min_bank',
                   'local_bo_qty', 'pieces_past_due', 'sales_1_month', 
                   'sales_3_month', 'sales_6_month', 'sales_9_month',]
dataset[qty_related] = normalize(dataset[qty_related], axis=1)


# In[ ]:


dataset.head()


# Separating the Dependent and Independent Variables

# In[ ]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# Splitting them into Training and Test Sets

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Importing TensorFlow for Deep Learning

# In[ ]:


import tensorflow as tf


# In[ ]:


import keras


# In[ ]:


from keras.models import Sequential


# In[ ]:


from keras.layers import Dense


# Initializing the ANN

# In[ ]:



ann = tf.keras.models.Sequential()


# Adding the first input layer and first hidden layer

# In[ ]:



ann.add(tf.keras.layers.Dense(units=12, activation='relu'))


# Adding the second hidden layer

# In[ ]:



ann.add(tf.keras.layers.Dense(units=15, activation='relu'))


# Adding the output layer

# In[ ]:



ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Compiling the ANN

# In[ ]:



ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training Set

# In[ ]:



ann.fit(X_train,y_train,batch_size=32,epochs=5)


# Prediction on Test Set

# In[ ]:


y_pred = ann.predict(X_test)


# Since Sigmoid Activation Function is used,a probability of 0.5 is taken as a dividing point

# In[ ]:


y_pred = (y_pred > 0.5)


# In[ ]:


print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Importing Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Checking Accuracy Score

# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_pred,y_test))

