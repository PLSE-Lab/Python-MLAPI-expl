#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
housing = pd.read_csv('../input/housing.csv')
housing.head()
##Reshuffle the data for our train validate test split.
housing = housing.sample(frac=1, random_state=1212942)
## split the data 80 20 into train test
X = housing.iloc[:,:-1]
y = housing.iloc[:,-1]



## fill in the missing values with the onehotencoder for categorical values and median on simpleimputer for other values
from sklearn.compose import ColumnTransformer

num_attrs = list(X)
num_attrs.remove("ocean_proximity")
cat_attrs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
        ("num", SimpleImputer(strategy='median'),num_attrs),
        ("cat", OneHotEncoder(), cat_attrs)
    ])
X_transform = full_pipeline.fit_transform(X)


# In[3]:


#Split the data 65 15 20 
X_transform = np.array(X_transform)
X_train, X_validate, X_test = np.split(X_transform, [int(.65*len(X)), int(.8*len(X))])

y = np.array(y)
y_train, y_validate, y_test = np.split(y, [int(.65*len(y)), int(.8*len(y))])



# In[4]:


#Use standard scalar to scale the data 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)
X_test = scaler.transform(X_test)


# In[5]:


from keras.models import Sequential
from keras.layers import Dense

##Grid Search
# layers = [1,2,4,8]
# size = [10, 30, 100, 300]
# best_layers = 0
# best_size = 0


# for s in size:
#     for l in layers:
#         calHousingNN = Sequential()
#         calHousingNN.add(Dense(units=s, activation='relu', input_dim=13))
#         for i in range(l):
#             calHousingNN.add(Dense(units=s, activation='relu'))
        
#         calHousingNN.add(Dense(units=1))
#         calHousingNN.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
#         history =  calHousingNN.fit(X_train, y_train, epochs=30, batch_size = 4092, validation_data = (X_validate, y_validate), verbose=0)
#         print("layers = " + str(l) + " - size = " + str(s))
#         print("Validation MAE: " + str(history.history['val_mean_absolute_error'][-1]))
        
    


# ##declare the model
calHousingNN = Sequential()

calHousingNN.add(Dense(units=300, activation='relu', input_dim=13))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=300, activation='relu'))
calHousingNN.add(Dense(units=1))

calHousingNN.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mae'])

history =  calHousingNN.fit(X_train, y_train, epochs=30, verbose=0, validation_data = (X_validate, y_validate))


# In[6]:


calHousingNN.summary()


# In[7]:


#plot validation error and training error

import matplotlib.pyplot as plt

history.history.keys()

# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model Accuracy')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[8]:


from sklearn.metrics import accuracy_score, explained_variance_score
test_preds = calHousingNN.predict(X_test)
loss_and_metrics = calHousingNN.evaluate(X_test, y_test)
print("Test set MAE: " + str(loss_and_metrics[-1]))
print("Explained Variance: " + str(explained_variance_score(y_test, test_preds)))


# In[9]:


count_below = 0
count_above = 0
for i in test_preds:
        if(i<15000):
            count_below +=1
        elif(i>500000):
            count_above +=1
            

print("Predictions below $15,000: " +str(count_below) + " . Predictions above $500,000: " + str(count_above))
#How much better is the test MAE than your best result from Lab3


# The test MAE was 37,389, compared to 41155 in the model from Lab 3

# ## Part2 ##

# In[10]:


from keras.datasets import cifar10
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[11]:


X_train = X_train/255
X_test = X_test/255


# In[12]:


from functools import partial
import keras

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding='SAME')

cifar_CNN = Sequential([
    DefaultConv2D(filters=64,kernel_size=7, input_shape=[32,32,3]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    Dense(units=128,activation="relu"),
    keras.layers.Dropout(0.5),
    Dense(units=64,activation="relu"),
    keras.layers.Dropout(0.5),
    Dense(units=10, activation="softmax")
])


# In[13]:


from keras import optimizers

sgd = optimizers.SGD(lr=0.023)

cifar_CNN.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = cifar_CNN.fit(X_train, y_train, epochs=30, verbose=1, validation_split=0.15)


# learning rate 0.01 loss: 0.4258 - acc: 0.8551 - val_loss: 1.0377 - val_acc: 0.7069
# 
# learning rate 0.02 loss: 0.2384 - acc: 0.9255 - val_loss: 1.1425 - val_acc: 0.7385
# 
# learning rate 0.023 loss: 0.2063 - acc: 0.9347 - val_loss: 1.1865 - val_acc: 0.7460
# 
# learning rate 0.024 loss: 0.2125 - acc: 0.9345 - val_loss: 1.3941 - val_acc: 0.7180
# 
# learning rate 0.025 loss: 0.2236 - acc: 0.9312 - val_loss: 1.1262 - val_acc: 0.7420
# 
# learning rate 0.0275 loss: 0.2034 - acc: 0.9372 - val_loss: 1.1797 - val_acc: 0.7373
# 
# learning rate 0.05 loss: 0.2989 - acc: 0.9069 - val_loss: 1.9075 - val_acc: 0.6368
# 
# learning rate 0.1 loss: 1.5297 - acc: 0.5281 - val_loss: 1.4259 - val_acc: 0.5360
# 

# In[14]:


#predict on the test set and get test accuracy
test_preds = cifar_CNN.predict(X_test)
loss_and_metrics = cifar_CNN.evaluate(X_test, y_test)
print("Test Set Accuraccy Score: " + str(loss_and_metrics[-1]))


# In[15]:


len(y_test)


# In[16]:


# iterate through the y test examples for each class, and find the first misclassified example. store the index of this example in misclassified
misclassified_index = []
misclassified_proba_max = []
for i in range(0,10):
    misclassified_index.append(0)
    misclassified_proba_max.append(0)


for i in range(0,10):
    #find the misclassified examples and their probabilites
    for j in range(len(y_test)):

        if(np.argmax(y_test[j], axis=0) == i and np.argmax(test_preds[j], axis=0) != i):
            prob = test_preds[j,np.argmax(test_preds[j], axis=0)] - test_preds[j,np.argmax(y_test[j], axis=0)]
            if(misclassified_proba_max[i] < prob):
                misclassified_proba_max[i] = prob
                misclassified_index[i] = j
            

            
print(len(misclassified_proba_max))


# In[27]:


labels = ["airplane"
,"automobile"
,"bird"
,"cat"
,"deer"
,"dog"
,"frog"
,"horse"
,"ship"
,"truck"]



for i in range(0,10):
    plt.imshow(X_test[misclassified_index[i]])
    plt.title("Original class:  " + labels[i] + " - Misclassified as:  " + labels[np.argmax(test_preds[misclassified_index[i]], axis=0)] + " - Misclassified Value:  " + str(misclassified_proba_max[i]))
    plt.show()
    

    


# In[ ]:




