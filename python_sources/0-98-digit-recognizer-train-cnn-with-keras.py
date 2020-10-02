#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np


# # Preprocessing Datasets

# ## Get Data Path

# In[ ]:


train_data_path = "../input/digit-recognizer/train.csv"
test_data_path = "../input/digit-recognizer/test.csv"
sample_submission_path = "../input/digit-recognizer/sample_submission.csv"


# # Train Data

# ### Load train data

# In[ ]:


train_data = pd.read_csv(train_data_path)


# In[ ]:


train_data.head()


# ### split the data into training and testing

# In[ ]:


# get label data
train_labels = train_data.iloc[:, 0]

# get image data
train_images = train_data.iloc[:, 1:]


# ### OnehotEncoding

# In[ ]:


train_labels_onehot = pd.get_dummies(train_labels)


# #### before onehot encoding

# In[ ]:


train_labels.head()


# #### after onehot encoding

# In[ ]:


train_labels_onehot.head()


# ### Scaling Data

# In[ ]:


train_images /= 255.


# ### Convert to numpy arrray

# In[ ]:


train_images = train_images.values
train_labels_onehot = train_labels_onehot.values


# ### Create Train Datasets

# In[ ]:


x_train = []
for image in train_images:
    image_data = image.reshape(28, 28, 1)
    x_train.append(image_data)


# ### Convert List to Array

# In[ ]:


x_train = np.array(x_train)
y_train = train_labels_onehot


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# ## Train CNN Model with Keras

# ### Import Keras

# In[ ]:


from keras import models
from keras import layers


# ### Define Model

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(2, 2), activation="relu", input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(2, 2), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(2, 2), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation="softmax"))


# ### Model Summary

# In[ ]:


model.summary()


# ### Set optimizer and loss

# In[ ]:


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# ### Train Model

# In[ ]:


model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=0)


# ### Save Model

# In[ ]:


model.save("digit_cnn.h5")


# ## Predict Test Data

# ### Prepare Test Data

# #### Load Test Data

# In[ ]:


test_images = pd.read_csv(test_data_path)


# #### Convert to numpy array

# In[ ]:


test_images = test_images.values.astype(np.float32)


# #### Scaling Test Data

# In[ ]:


test_images /= 255.


# #### Create Test Data

# In[ ]:


x_test = []
for image_data in test_images:
    image = image_data.reshape(28, 28, 1)
    x_test.append(image)


# #### Convert List to Array

# In[ ]:


x_test = np.array(x_test)


# ### Predict Test Data

# #### Predict

# In[ ]:


predictions = model.predict(x_test)


# #### Get Max Index

# In[ ]:


predictions[0]


# In[ ]:


labels = np.argmax(predictions, axis=1)


# ## Create Submission CSV

# ### Load Sample Submission

# In[ ]:


sample = pd.read_csv(sample_submission_path)


# ### Replace Sample to Predict Data

# In[ ]:


sample["Label"] = labels
submission = sample


# ### Convert Data to CSV

# In[ ]:


submission.to_csv("submission.csv", index=False)

