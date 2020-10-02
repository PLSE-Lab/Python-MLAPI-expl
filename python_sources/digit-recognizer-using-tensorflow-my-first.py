#!/usr/bin/env python
# coding: utf-8

# # This notebook is a revision to what I have studied about image classification using Tensorflow and keras.
# 
# As a revision, and well to check how the same code applies to a new dataset, I will follow the code from the tutorial and maybe then do some experiments with changes in optimizer or epochs.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Step 1: Load the data set

# In[ ]:


df_train= pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# In[ ]:


df_train.head()


# ## Step 2: Preparing a labels dataset for training 
# 
# The dataset has been set-up as:
# 1. Label 
# 2. Pixel distribution, ranging from 0 to 783 i.e. 784 pixels (28 *28)
# 
# We need to extract the 'label' column into another dataframe.

# In[ ]:


y_train=df_train['label']
x_train= df_train.drop(labels=['label'], axis=1) # axis=1 means a column
print('Shape of the y_train is:', y_train.shape)
print('Shape of the x_train is:', x_train.shape)
print('Shape of the x_train is:', df_test.shape)


# In[ ]:


y_train.value_counts()


# ## Step 3: Use One-hot encoding using to_categorical

# In[ ]:


from tensorflow.keras.utils import to_categorical
y_train_encoded= to_categorical(y_train)
print(' New shape for y_train is:', y_train_encoded.shape)


# ## Step 4: Normalizing
# When the test and train data are already split, we should apply same preprocessing to both the datasets to avoid any anomoly

# In[ ]:


x_train_mean= x_train.mean()
x_train_std= x_train.std()
epsilon= 1e-10

x_train_norm= (x_train-x_train_mean)/ (x_train_std+ epsilon)
X_test= (df_test-x_train_mean)/ (x_train_std+ epsilon)


# ## Step 5: Create the model
# ### Trial 1: 
# Using Sequential model and splitting the model into 2 hidden layers

# In[ ]:


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model= Sequential([
    Dense(128, activation= 'relu', input_shape=(784,)),
    Dense(128, activation= 'relu'),
    Dense(10, activation='softmax')
])


# ## Step 6: Compile the model
# Any model compiler needs
# 1. Loss: difference between predicted and real values 
# 2. Optimizer : Define the algorithm used to minimize the loss
# 3. Metric: To monitor the performance of the network

# In[ ]:


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics= ['acc'])


# ## Step 7: Training the model
# To check the accuracy of the model, and whether the model simply learned the
# weights and biases, we will first split the training data.

# In[ ]:


from sklearn.model_selection import train_test_split
X=x_train_norm
Y= y_train_encoded ## keeping a copy of our last worked datasets

X_train, X_val, Y_train, Y_val= train_test_split(x_train_norm, y_train_encoded,test_size= 0.2, random_state=0)


# In[ ]:


print (X_train.shape, X_val.shape)


# In[ ]:


h=model.fit(X_train,Y_train, epochs=10)


# ## Step 8: Validation and Predictions:
# This is the most crucial step, to check if our model has simply learned the weights and biases, or our network is successful

# In[ ]:


loss, accuracy= model.evaluate(X_val, Y_val)
print('Accuracy of our model is: ', (accuracy*100))


# ## Trial 2: Lets add a dense layer and see if there is any variations

# In[ ]:


model_1= Sequential([
   Dense(128, activation= 'relu', input_shape=(784,)),
   Dense(128, activation= 'relu'),
   Dense(128, activation= 'relu'),
   Dense(10, activation='softmax')
])
model_1.compile(optimizer='sgd', loss='categorical_crossentropy', metrics= ['acc'])
h=model_1.fit(X_train,Y_train, epochs=10)


# In[ ]:


loss, accuracy= model_1.evaluate(X_val, Y_val)
print('Accuracy of our model is: ', (accuracy*100))


# ## Trial 3: Using RMSProp
# 
# We see similar results. 
# 
# Would changing the optimizer from a common Stochastic Gradient descent to a iterative RMSProp have a difference?
# 
# To reduce overfitting, lets add a Dropout step of the factor 0.15 to our train set.

# In[ ]:


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dropout
model_1= Sequential([
    Dense(128, activation= 'relu', input_shape=(784,)),
    Dense(128, activation= 'relu'),
    Dense(128, activation= 'relu'),
])
model_1.add(Dropout(0.15))
model_1.add (Dense(10, activation='softmax'))

model_1.compile(optimizer=RMSprop (lr=0.001), loss='categorical_crossentropy', metrics= ['acc'])
h=model_1.fit(X_train,Y_train, epochs=15)


# In[ ]:


loss, accuracy= model_1.evaluate(X_val, Y_val)
print('Accuracy of our model is: ', (accuracy*100))


# Minor leap! 

# ## Trial 4: Using Adam
# 
# As a last trial, lets use Adam, a similar optimizer to RMS Prop.
# 

# In[ ]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout # gave worst results
model_2= Sequential([
    Dense(128, activation= 'relu', input_shape=(784,)),
    Dense(128, activation= 'relu'),
    Dense(128, activation= 'relu'),
])

model_2.add (Dense(10, activation='softmax'))

model_2.compile(optimizer=Adam (lr=0.001), loss='categorical_crossentropy', metrics= ['acc'])
h=model_2.fit(X_train,Y_train, epochs=10)


# In[ ]:


loss, accuracy= model_2.evaluate(X_val, Y_val)
print('Accuracy of our model is: ', (accuracy*100))


# Since Model 2 gave the best accuracy, lets use that.
# 
# 
# *PS: We have to take a run with the full dataset before (X and Y saved earlier) before we make a submission.*

# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model_2= Sequential([
    Dense(128, activation= 'relu', input_shape=(784,)),
    Dense(128, activation= 'relu'),
    Dense(128, activation= 'relu'),
])

model_2.add (Dense(10, activation='softmax'))

model_2.compile(optimizer=Adam (lr=0.001), loss='categorical_crossentropy', metrics= ['acc'])
h=model_2.fit(X,Y, epochs=10)


# In[ ]:


predictions = model_2.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR_submissions.csv", index=False, header=True)

