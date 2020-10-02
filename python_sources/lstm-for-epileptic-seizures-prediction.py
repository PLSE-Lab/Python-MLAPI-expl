#!/usr/bin/env python
# coding: utf-8

# **Link to the dataset:**:[https://www.kaggle.com/harunshimanto/epileptic-seizure-recognition](https://www.kaggle.com/harunshimanto/epileptic-seizure-recognition)

# # Importing the libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


# # Load the Dataset

# In[ ]:


ESR = pd.read_csv('../input/Epileptic Seizure Recognition.csv')
ESR.head()


# # Read and Show Dataset
# * The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds.
# 
# * The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.
# 
# * We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time.
# 
# * So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.
# 
# * The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178

# In[ ]:


ESR.head()


# In[ ]:


ESR.head()


# In[ ]:


cols = ESR.columns
tgt = ESR.y
tgt.unique()
tgt[tgt>1]=0
ax = sn.countplot(tgt,label="Count")
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)


# As we can see, there are 178 EEG features and 5 possible classes. The main goal of the dataset it's to be able to correctly identify epileptic seizures from EEG data, so a binary classification between classes of label 1 and the rest (2,3,4,5). In order to train our model, let's define our independent variables (X) and our dependent variable (y).

# # &#128205; Data Pre-processing
# 
# ## What is Data Pre-pocessing?
# Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues. Data preprocessing prepares raw data for further processing.
# > 

# # &#128205; 1. Checking Missing Data[](http://)

# In[ ]:


ESR.isnull().sum()


# In[ ]:


ESR.info()


# In[ ]:


ESR.describe()


# In[ ]:


X = ESR.iloc[:,1:179].values
X.shape


# In[ ]:


plt.subplot(511)
plt.plot(X[1,:])
plt.title('Classes')
plt.ylabel('uV')
plt.subplot(512)
plt.plot(X[7,:])
plt.subplot(513)
plt.plot(X[12,:])
plt.subplot(514)
plt.plot(X[0,:])
plt.subplot(515)
plt.plot(X[2,:])
plt.xlabel('Samples')


# In[ ]:


y = ESR.iloc[:,179].values
y


# To make this a binary problem, let's turn the non-seizure classes 0 while maintaining the seizure as 1.

# In[ ]:


y[y>1]=0
y


# # &#128295; Building Machine Learning Models

# ##  Splitting the Dataset into the Training set and Test set
# 

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)
from keras.utils import to_categorical
y = to_categorical(y)
y


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# ## Feature Scaling

# In[ ]:


x_train = np.reshape(x_train, (x_train.shape[0],1,X.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0],1,X.shape[1]))


# In[ ]:


import tensorflow as tf
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
tf.keras.backend.clear_session()

model = Sequential()
model.add(LSTM(64, input_shape=(1,178),activation="relu",return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,activation="sigmoid"))
model.add(Dropout(0.5))
#model.add(LSTM(100,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(50))
#model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
from keras.optimizers import SGD
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
model.summary()


# In[ ]:


history = model.fit(x_train, y_train, epochs = 100, validation_data= (x_test, y_test))
score, acc = model.evaluate(x_test, y_test)


# In[ ]:


from sklearn.metrics import accuracy_score
pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)
print(expected_classes.shape)
print(predict_classes.shape)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")

