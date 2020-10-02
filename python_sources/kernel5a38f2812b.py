#!/usr/bin/env python
# coding: utf-8

# ## Tags
# 
# - Multi Classclasification
# - Dimension Reduction [PCA]
# - Accuracy, Precision, Recall, F1 Score
# - Neural Network - Deep Learning using Tensorflow

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn



from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


tf.compat.v2.test.is_gpu_available(cuda_only=False,min_cuda_compute_capability=None)


# In[ ]:


## load data
data = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')
data_test = pd.read_csv('/kaggle/input/mnist-in-csv/mnist_train.csv')


# In[ ]:


## understand data - first overview
data.head()


# In[ ]:


## Shuffle the data
data = shuffle(data)


# In[ ]:


## Shape of the data
print('Shape of the complete dataset: ',data.shape)


# In[ ]:


## Seperating Features and labels
y = data[['label']]
X = data.drop(['label'], axis = 1) 


# In[ ]:


## For Test Data
y_test = data_test[['label']]
X_test = data_test.drop(['label'], axis = 1) 


# In[ ]:


print('Shape of the Features: ', X.shape)
print('Shape of the Label: ', y.shape)


# In[ ]:


y.label.unique()


# ### Data Analysis - First Level
# 
# - number of examples are arounf 60,000 which is good
# - number of features seems to be on the higher side, we have to check how to work with that.
# - we have in total 10 classes , one for each number for 0 to 9

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

some_digit = X.values[10].reshape(28,28)
plt.imshow(some_digit, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.show()

some_digit = X.values[110].reshape(28,28)
plt.imshow(some_digit, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.show()


# ### Data Analysis - Second Level
# 
# - feature represent the 784 coloums which is binary of 28 * 28 image cell. 
# - current image is the gray scale image.
# - most the pixels are empty always , as small area is used for the representation of the number.

# In[ ]:


## to uderstand the data distribution of each class in the dataset
sn.countplot(x="label", data=y)


# In[ ]:


desc = X.describe()


# In[ ]:


desc.head(10)


# In[ ]:


## As we can see , plotting the max value for each pixel , edges and boundaries never had any value.
plt.imshow(desc.values[7].reshape(28,28), cmap=matplotlib.cm.Blues, interpolation='nearest')


# In[ ]:


## As we can see , plotting the mean value for each pixel , edges and boundaries never had any value.
## mostly values are concentrated in the middle part of the iage only.
plt.imshow(desc.values[1].reshape(28,28), cmap=matplotlib.cm.Blues, interpolation='nearest')


# ### Data Analysis
# 
# - We have almost equal number of examples for each class.
# - By seeing the graph of max and mean value at each pixel , we can confirm edges are almost never used , mostly data is concentrated in the middle only.

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


print('Shape of training features ', X_train.shape)
print('Shape of validation features ', X_val.shape)

print('Shape of training label', y_train.shape)
print('Shape of validation label', y_val.shape)


# In[ ]:


## scale the data, same scaler should be used for test data also
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# In[ ]:


X_test = scaler.transform(X_test)


# In[ ]:


input_dimension = X_train.shape[1] # this represent number of features

### hyper parameters
epochs = 20
batch_size = 256

### model
model = Sequential()
model.add(Dense(256, input_shape=(input_dimension,), activation='relu', kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(Dense(126, activation='relu'))
model.add(Dense(54, activation='relu',kernel_regularizer= tf.keras.regularizers.l1(0.01)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X_train, y_train.values, epochs=epochs, batch_size=batch_size,
          validation_data=(X_val, y_val.values))


# In[ ]:


plt.plot( history.history['accuracy'], color='skyblue', linewidth=2, label='training acc')
plt.plot( history.history['val_accuracy'], color='green', linewidth=2, label='val acc')

plt.plot( history.history['loss'], color='skyblue', linewidth=2, linestyle='dashed', label="training loss")
plt.plot( history.history['val_loss'], color='green', linewidth=2, linestyle='dashed', label="val loss")
plt.legend()


# ### Understand the affect of Change in learning rate
# - We will try to change the batch size , and understand the affect on model.

# In[ ]:


input_dimension = X_train.shape[1] # this represent number of features

### hyper parameters
epochs = 20
batch_size = 256

### model
model = Sequential()
model.add(Dense(256, input_shape=(input_dimension,), activation='relu', kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(Dense(126, activation='relu'))
model.add(Dense(54, activation='relu',kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X_train, y_train.values, epochs=epochs, batch_size=batch_size,
          validation_data=(X_val, y_val.values))


# In[ ]:


plt.plot( history.history['accuracy'], color='skyblue', linewidth=2, label='training acc')
plt.plot( history.history['val_accuracy'], color='green', linewidth=2, label='val acc')

plt.plot( history.history['loss'], color='skyblue', linewidth=2, linestyle='dashed', label="training loss")
plt.plot( history.history['val_loss'], color='green', linewidth=2, linestyle='dashed', label="val loss")
plt.legend()


# As can be seen in the first examples the loss curve is not smooth, so it make sense to ower the learning rate a bit to have a better result , As we can see lowering the learning rate not only help us to get the smooth loss curve , but the owerall accuracy was also improved.

# In[ ]:


y_test_pred = model.predict_classes(X_test)


# In[ ]:


print('Shape of predicction ', y_test_pred.shape)
print('Shape of test ', y_test.shape)


# In[ ]:


conf_mat = confusion_matrix(y_test, y_test_pred)
print(conf_mat)


# In[ ]:


print(classification_report(y_test, y_test_pred))


# In[ ]:


## Dark color represent the correctness of classification
plt.matshow(conf_mat , cmap = plt.cm.Blues)


# In[ ]:


## TO ca;culate the error rate , sum instance of each class and than divide it
row_sum = conf_mat.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mat/ row_sum


# In[ ]:


## Dark color represent the higher error
np.fill_diagonal(norm_conf_mx,0)
plt.matshow(norm_conf_mx, cmap = plt.cm.Blues)


# ### Understand the feature reduction 
# 

# In[ ]:


pca = PCA(n_components=512)
X_train = pca.fit_transform(X_train)


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))


# In[ ]:


print('Shape of input after PCA', X_train.shape)


# In[ ]:


X_val = pca.transform(X_val)
X_test = pca.transform(X_test)


# In[ ]:



### hyper parameters
input_dimension = X_train.shape[1] # this represent number of features
epochs = 20
batch_size = 256

### model
model = Sequential()
model.add(Dense(256, input_shape=(input_dimension,), activation='relu', kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(Dense(126, activation='relu'))
model.add(Dense(54, activation='relu',kernel_regularizer= tf.keras.regularizers.l1(0.001)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X_train, y_train.values, epochs=epochs, batch_size=batch_size,
          validation_data=(X_val, y_val.values))


# In[ ]:


plt.plot( history.history['accuracy'], color='skyblue', linewidth=2, label='training acc')
plt.plot( history.history['val_accuracy'], color='green', linewidth=2, label='val acc')

plt.plot( history.history['loss'], color='skyblue', linewidth=2, linestyle='dashed', label="training loss")
plt.plot( history.history['val_loss'], color='green', linewidth=2, linestyle='dashed', label="val loss")
plt.legend()


# In[ ]:


y_test_pred = model.predict_classes(X_test)


# In[ ]:


conf_mat = confusion_matrix(y_test, y_test_pred)
print(conf_mat)


# In[ ]:


print(classification_report(y_test, y_test_pred))


# #### Note : Model with PCA executed comparatively faster as without it. With PCA we may even chose smaller model since the number if inputs have decreased. Now lets use the data augmentation to increase the data set and achive better numbers. Since CNN perform better with image , next step should be to perform Classification on images using CNN acheive more than 97 % accuracy.
