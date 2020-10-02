#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center"><h2>Hey, my dear friend!</h2></div>  
# <div style="text-align: center"><h6>I created this kernel for the people who take first steps in the Deep Learning</h6></div>  
# <div style="text-align: center"><h6>Don't forget upvote if you like the kernel</h6></div>  

# ![](https://mathematicaforprediction.files.wordpress.com/2013/08/digitimageswithzenbrush-testset.jpg)

# # Let's begin

# ## Import all neccesary libraries

# In[ ]:


# Linear algebra
import numpy as np

# Data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

# Deep Learning
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import np_utils

# Visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# ## Convert pandas DataFrame to NumPy array

# In[ ]:


x_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

x_test = test_df.values


# ## Convert data to 2D representation

# In[ ]:


img_height, img_width = 28, 28
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)


# In[ ]:


num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)


# In[ ]:


y_train[:5]


# ## Data Normalization
# Data Normalization will help increase accuracy of the model  
# *255* - highest value

# In[ ]:


x_test = x_test / 255.0
x_train = x_train / 255.0


# ## Create deep learning model

# In[ ]:


model = Sequential()

model.add(Conv2D(
    filters=32,
    kernel_size=(5,5),
    input_shape=(img_height, img_width, 1), 
    padding="Same",
    activation="relu"
))
model.add(Conv2D(
    filters=32,
    kernel_size=(5,5),
    input_shape=(img_height, img_width, 1), 
    padding="Same",
    activation="relu"
))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    input_shape=(img_height, img_width, 1), 
    padding="Same",
    activation="relu"
))
model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    input_shape=(img_height, img_width, 1), 
    padding="Same",
    activation="relu"
))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation="softmax"))


# ## Compile model

# In[ ]:


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())


# ## Callback for saving best model

# In[ ]:


checkpoint = ModelCheckpoint(
    'best_model.hdf5',
    monitor='val_acc',
    save_best_only=True,
    verbose=1
)


# ## Reduce learning rate when a metric has stopped improving
# Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.

# In[ ]:


lr_reduction = ReduceLROnPlateau(
    monitor='val_acc',
    patience=3,
    factor=0.5,
    min_lr=0.00001,
    verbose=1
)


# In[ ]:


batch_size = 96
epochs = 40
validation_size = 0.3


# In[ ]:


history = model.fit(
    x=x_train, 
    y=y_train, 
    batch_size=batch_size, 
    epochs=epochs,
    verbose=1,
    validation_split=validation_size,
    callbacks=[checkpoint, lr_reduction]
)


# ## Train/test plot

# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(history.history['acc'], label='Train')
plt.plot(history.history['val_acc'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ## Load weights of the best model

# In[ ]:


model.load_weights('best_model.hdf5')


# ## Predict classes
# Method *predict* returns the probability that the image belongs to one of the classes (0,1,2,...,8,9).  

# In[ ]:


probabilities = model.predict(x_test)


# ## Convert probabilities to class
# In the competition we should predict class.  
# Hence, we should convert probabilities to classes

# In[ ]:


predictions = np.argmax(probabilities, axis=1)


# ## Create submission

# In[ ]:


submission = pd.DataFrame(data={
    'ImageId': list(range(1, predictions.shape[0]+1)),
    'Label': predictions
})


# In[ ]:


submission.head()


# ## Save submission to file

# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




