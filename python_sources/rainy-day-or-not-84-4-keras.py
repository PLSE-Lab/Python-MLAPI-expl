#!/usr/bin/env python
# coding: utf-8

# ## Initial step
# Import required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ## Exploring data
# Load the data and explore it.

# In[ ]:


df = pd.read_csv('../input/weatherAUS.csv')
df.head()


# Looks like we have "NaN"s.
# Let's find out how many nans df have.

# In[ ]:


df.count().sort_values()


# First 4 columns have a lot of nans, let's drop them and drop RISK_MM

# In[ ]:


df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am', 'RISK_MM'], axis=1, inplace=True)


# And finally drop NaNs

# In[ ]:


df.dropna(inplace=True)
df.head()


# Much cleaner.
# 
# Convert date column in DateTime format, set date as index and sort in.
# 
# For machine learning it's not necessary, it's just fun.

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)
df.head()


# Let's plot MaxTemp.

# In[ ]:


df['MaxTemp'].rolling(365).mean().plot()


# Back to machine learning.
# 
# We have a lot of "strings".
# 
# Convert them to numbers.

# In[ ]:


df['Location'].unique()


# In[ ]:


df['Location'] = df['Location'].astype('category').cat.codes
df['WindGustDir'] = df['WindGustDir'].astype('category').cat.codes
df['WindDir9am'] = df['WindDir9am'].astype('category').cat.codes
df['WindDir3pm'] = df['WindDir3pm'].astype('category').cat.codes
df['RainToday'] = df['RainToday'].astype('category').cat.codes
df['RainTomorrow'] = df['RainTomorrow'].astype('category').cat.codes


# Explore data

# In[ ]:


df.head()


# Drop the Date columm. We don't need it any more .

# In[ ]:


df.reset_index(drop=True, inplace=True)
df.head()


# Since we sort the data we need to shuffle it.
# 
# Anyway shuffling data it's always good practice. 

# In[ ]:


df = shuffle(df)
df.head()


# Create train data and labels.

# In[ ]:


X = df.drop('RainTomorrow', axis=1)
y = df['RainTomorrow']


# In[ ]:


X.describe()


# We definitely need to normalize the data.
# 
# Convert it to np array and use normalize utils from keras.

# In[ ]:


X = X.values
X = normalize(X)


# Split the dataset into three sets.
# 
# 
# train - 80% valid - 10% test - 10%

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)


# In[ ]:


X_train.shape


# ## Create the model.
# Our playground. Feel free to try a different variation

# In[ ]:


model = tf.keras.models.Sequential([
       
    tf.keras.layers.Dense(128, input_shape=(17,), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

            
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy',
              optimizer=Adam(0.00001),
              metrics=['acc'])


# Explore the model.

# In[ ]:


model.summary()


# ## Training
# Finally, train the model.

# In[ ]:


history = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val),
                    verbose=1,
                   )


# ## Evaluate the model
# Plot our accuracy and loss for understanding problems: "high bias" and "high variance".

# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# After finishing playing with model and we are happy with achieved accuracy, evaluate your model on the test set.

# In[ ]:


loss, accuracy = model.evaluate(X_test, y_test)


# In[ ]:


acc = accuracy * 100
plt.bar(1, acc)
plt.text(0.92,45,f'{acc:.2f}%', fontsize=20)
plt.title('Accuracy')
plt.xticks([])
plt.ylabel('Percent')
plt.show()


# <center>Thanks for reading.</center>
# 
# <center>Vote if you like it.</center>
