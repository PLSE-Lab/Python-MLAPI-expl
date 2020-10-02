#!/usr/bin/env python
# coding: utf-8

# ## Digit Recognizer CNN | Keras
# *   Initial step
# *   Data preparation
# *   Create the model.
# *   Evaluate the model
# *   Make a submission.

# ## **Initial step**
# 
# Basic initial step import required libraries
# 
# 
# *   matplotlib - plot images and results
# *   pandas - reading and manipulating data
# *   numpy - linear algebra
# *   tensorflow - self-explanatory
# *   sklearn - only for data splitting and confusion matrix
# *   randrange - for selecting random images

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from random import randrange


# ## Data preparation
# 
# Load the training data with pandas and explore it

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
print(f'The train set contain {df_train.shape[0]} examples')
df_train.head(3)


# Looks like we have in first column labels and everything else it's an unrolled 28x28 image.
# Extract Labels to different data.

# In[ ]:


X_train = df_train.drop('label', axis = 1)
y_train = df_train['label']


# Let's plot the training set. Looks like we have a balanced data set.

# In[ ]:


digits = y_train.unique()
values = y_train.value_counts()

plt.bar(digits, values)
plt.title('Train set')
plt.xlabel('Digit')
plt.ylabel('Examples count')
plt.xticks(np.arange(len(digits)))
plt.show()


# Normalize training examples.
# 
# Normalization makes training more efficient. Why this happening? I will write a post later.

# In[ ]:


X_train = X_train / 255


# Reshape training set.
# 
# Now it will be 28x28 image.

# In[ ]:


X_train = X_train.values.reshape(-1,28,28,1)


# Now we can plot random images with the label. Just rerun the cell and explore different training examples.

# In[ ]:


rnd_digit = randrange(X_train.shape[0])
img = X_train[rnd_digit][:,:,0]
label = y_train[rnd_digit]
plt.title(f'This is number {label}')
plt.axis('off')
plt.imshow(img, cmap=plt.cm.binary)


# Now we need convert our label to "one hot vector". Print the result to make sure everything is right.

# In[ ]:


y_train = to_categorical(y_train)
y_train.shape


# Split the data into three sets.
# 
# Since data it significantly small we will split to:
# 
# train - 80%
# valid - 10%
# test - 10%

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5)


# ### Data augmentation. 
# 
# When working with images is always not enough data. With Keras we can augment it in the fly using ImageDataGenerator

# In[ ]:


train_datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        )

train_datagen.fit(X_train)


# ## Create the model.
# 
# Our playground. Feel free to try a different variation

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['acc'])


# Explore the model.

# In[ ]:


model.summary()


# ### Training
# 
# Batch size is another hyperparameter to tune.
# Finally, train the model.

# In[ ]:


batch_size = 32
history = model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        epochs=20,
        validation_data=(X_val, y_val),
        )


# ## Evaluate the model
# 
# Plot our accuracy and loss for understanding problems: "high bias" and "high variance".
# 
# Interesting, in this dataset we don't need BatchNorm or Dropout.

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


# ## Make a submission.
# 
# Load test data, normalize and reshape it.

# In[ ]:


test = pd.read_csv('../input/test.csv')
test = test / 255
test = test.values.reshape(-1,28,28,1)
print(f'The test set contain {test.shape[0]} examples')


# Make prediction on test.csv

# In[ ]:


pred = model.predict(test)
pred = np.argmax(pred, axis = 1)


# Create dataframe

# In[ ]:


pred_csv = pd.DataFrame(pred, columns= ['Label'])
pred_csv.index += 1
pred_csv.head()


# Save dataframe as a csv file

# In[ ]:


pred_csv.to_csv('submission.csv', index_label='ImageId' )

