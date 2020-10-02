#!/usr/bin/env python
# coding: utf-8

# ## Glass Classification | Deep Learning | Keras
# *   Initial step
# *   Exploring data
# *   Create the model
# *   Evaluate the model

# ## **Initial step**
# 
# Basic initial step import required libraries
# 
# *   pandas - reading and manipulating data
# *   numpy - linear algebra
# *   matplotlib - plot graphs
# *   tensorflow - self-explanatory
# *   sklearn - only for data splitting and confusion matrix

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ## Exploring data
# 
# Load the training data and explore it

# In[ ]:


df_train = pd.read_csv('../input/glass.csv')
print(f'The train set contain {df_train.shape[0]} examples')
print(f'The train set contain {df_train.shape[1]} features')
df_train.head()


# Looks like we have a pretty small data set.
# 
# The last column contains labels.
# 
# Extract Labels to a different data frame.

# In[ ]:


X_train = df_train.drop('Type', axis = 1)
y_train = df_train['Type']


# Let's plot the training set.
# 
# Seem like dataset is not very good balanced.

# In[ ]:


glass_classes = y_train.unique()
values = y_train.value_counts()

plt.bar(glass_classes, values)
plt.title('Train set')
plt.xlabel('Glass Classes')
plt.ylabel('Examples count')
plt.show()


# Explore the values.
# 
# Max value is 78.41 and min is 0.29.
# 
# This data need to normalize.

# In[ ]:


X_train.describe()


# keras.utils.normalize don't work with a pandas data frame. 
# 
# Convert it to array and then normalize.
# 
# Print the first item of the new array.

# In[ ]:


X_train = df_train.values
X_train = normalize(X_train)
print(X_train[0])


# Now we need to convert our label to "one hot vector". Print shape of the new data frame to make sure everything is right.

# In[ ]:


y_train = to_categorical(y_train)
y_train.shape


# Split the dataset into three sets.
# 
# Since data it significantly small we will split to:
# 
# train - 80%
# valid - 10%
# test - 10%

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)


# ## Create the model.
# 
# Our playground. Feel free to try a different variation

# In[ ]:


model = tf.keras.models.Sequential([
       
    tf.keras.layers.Dense(256, input_shape=(10,), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
            
    tf.keras.layers.Dense(8, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['acc'])

model


# Explore the model.

# In[ ]:


model.summary()


# ### Training
# 
# 
# Finally, train the model.

# In[ ]:


history = model.fit(X_train, y_train,
                    epochs=400,
                    validation_data=(X_val, y_val),
                    verbose=2,
                   )


# ## Evaluate the model
# 
# Plot our accuracy and loss for understanding problems: "high bias" and "high variance".
# 

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


model.evaluate(X_test, y_test)


# Good job.
# 
# Let's look at the confusion matrix.

# In[ ]:


y_pred = model.predict(X_test)
y_pred_cl = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_test, axis = 1)

confusion_matrix(y_true, y_pred_cl)

