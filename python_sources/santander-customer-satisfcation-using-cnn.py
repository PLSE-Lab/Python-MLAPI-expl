#!/usr/bin/env python
# coding: utf-8

# In this file we will perform CNN on the Santander File.
# 
# Feature Selection , inversing and transposing the file to achieve a better accuracy

# In[ ]:


# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


# In[ ]:


# import the data
train = pd.read_csv('/kaggle/input/santander-customer-satisfaction/train.csv')
test = pd.read_csv('/kaggle/input/santander-customer-satisfaction/test.csv')

# check the shape
train.shape, test.shape


# In[ ]:


train.head()
# test.head()


# Our training data contains 'Target' Column extra ,which is our dependent variable and testing data contains only the independent variables.
# 
# Also the Column ID is of no relevance so will drop it.

# In[ ]:


# Dividing the dataframe into dependent and independent variable
X=train.drop(['ID','TARGET'], axis=1)
y=train['TARGET']
test=test.drop(['ID'], axis=1)

# check the shape again
X.shape, y.shape, test.shape


# In[ ]:


# Now we will seperate the data into training and testing dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=99)

X_train.shape,X_test.shape,y_train.shape,y_test.shape


# We will perform Feature Selection Methods hers.
# 
# Removing Constant, Quasi Constant and Duplicate Features

# In[ ]:


# Constant and Quasi Constant
filter=VarianceThreshold(0.01)
# this will be used to remove the data which has low variance of 1% or below


# In[ ]:


X_train=filter.fit_transform(X_train)
X_test=filter.transform(X_test)
test=filter.transform(test)
# now check the shape again
X_train.shape, X_test.shape, test.shape


# We can see that earlier we had 369 features, now our features are 268.
# 
# Now we will remove the duplicate features as well. We will remove the duplicates by transposing the data , converting the columns into rows and rows into columns

# In[ ]:


X_train_T=X_train.T
X_test_T=X_test.T
test_T=test.T


# In[ ]:


# converting the transposed value into a dataframe
X_train_T=pd.DataFrame(X_train_T)
X_test_T=pd.DataFrame(X_test_T)
test_T=pd.DataFrame(test_T)


# In[ ]:


X_train_T
X_test_T
test_T


# Now we will check how many features are duplicate

# In[ ]:


X_train_T.duplicated().sum()


# In[ ]:


test_T.duplicated().sum()


# Store the duplicates into a variable

# In[ ]:


duplicated=X_train_T.duplicated()
duplicated


# Here all the values which are True are the duplicate ones,
# We will perform inversion on the values by changing True to False, and False to True.

# In[ ]:


# perform inversion
features_to_keep=[not index for index in duplicated]
features_to_keep


# Now we will remove the duplicates and keep only the unique features and also we will transpose the dataframe again into its original shape

# In[ ]:


X_train=X_train_T[features_to_keep].T
print(X_train.shape)

X_test=X_test_T[features_to_keep].T
print(X_test.shape)

test=test_T[features_to_keep].T
print(test.shape)


# This preprocessing will help into getting a better accuracy for the model.

# In[ ]:


X_train


# There is variance in the dataset so we will scale the data

# In[ ]:


# scaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
test=scaler.transform(test)

X_train


# In[ ]:


# check the shape again
X_train.shape,X_test.shape,test.shape,y_train,y_test,


# we will convert the shape acceptable to our neural network and y_test and y_train into numpy format

# In[ ]:


# reshaping
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
test=test.reshape(test.shape[0],test.shape[1],1)
# check the shape again
X_train.shape, X_test.shape, test.shape


# In[ ]:


# series to numpy
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()


# Now our data is completely preprocessed , we will Build a model for it

# In[ ]:


# import libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv1D,MaxPool1D,BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam


# In[ ]:


# model
model=Sequential()
# layers
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(252,1)))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


# compiling the model
model.compile(optimizer=Adam(learning_rate=0.00005), loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', '# fittting the model\nhistory=model.fit(X_train,y_train, epochs=10,batch_size=128 ,validation_data=(X_test,y_test))')


# In[ ]:


# plotting the data
pd.DataFrame(history.history).plot(figsize=(10,8))
plt.grid(True)
plt.show()


# In[ ]:


# prediction
y_pred=model.predict_classes(test)
y_pred


# In[ ]:


# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# In[ ]:


get_ipython().run_line_magic('pinfo', 'plot_confusion_matrix')


# In[ ]:


mat=confusion_matrix(y_test, y_pred[:15204,])
plot_confusion_matrix(conf_mat=mat, figsize=(7,7))


# In[ ]:


y_pred.ndim


# As the data in y_pred is 2 dimensional, we will convert the same into 1 dim

# In[ ]:


y_pred=np.ravel(y_pred)
y_pred


# In[ ]:


results = pd.Series(y_pred,name="TARGET")
results


# In[ ]:


submission = pd.concat([pd.Series(range(1,75819),name = "ID"),results],axis = 1)
submission


# In[ ]:


submission.to_csv("submission.csv",index=False)


# In[ ]:




