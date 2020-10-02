#!/usr/bin/env python
# coding: utf-8

# This Notebook is basically for beginners who just want to get started with CNN.
# > If you find this Notebook Helpful then please upvote...

# ## Importing Important Libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import tensorflow as tf


# Reading data from the dataset.

# In[ ]:


train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')


# Lets try looking at the training data.

# In[ ]:


train_data.head()


# Only the First column is label which is our target and all other columns are the pixels of *28X28* images.

# In[ ]:


test_data.head()


# As obvious test data does not have labels and these are to be predicted.

# ## Kind of Preprocessing and Visualizing Images

# In the next cell, I have removed the label column from training data and take it in a target variable.

# In[ ]:


target = train_data['label']
train_data.drop(['label'], axis = 1, inplace = True)


# In[ ]:


print(train_data.shape)
print(test_data.shape)


# So now, we have both training and testing dataset in same shape.

# In[ ]:


target.head()


# Lets visualize first few targets with the help of their pixels after reshaping a row into *28X28*.

# In[ ]:


plt.figure(figsize = (15,6))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(np.array(train_data.iloc[i]).reshape(28,28), cmap = 'gray')
plt.tight_layout()


# Now, lets visualize last few targets with the help of their pixels after reshaping a row into *28X28*.

# In[ ]:


plt.figure(figsize = (15,6))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(np.array(train_data.iloc[-i]).reshape(28,28), cmap = 'gray')
plt.tight_layout()


# Well, I don't think there will be any null values in the data but still, lets confirm it with their info().

# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# Ok, so for now, we have a very decent dataset to work on. But, in order to apply CNN we have to make our data in to 4D as CNN takes only 4D as input.

# Lets reshape our data with the simplest command possible.

# In[ ]:


train_data = np.array(train_data).reshape(len(train_data),28,28,1)
train_data = train_data/255
test_data = np.array(test_data).reshape(len(test_data),28,28,1)
test_data = test_data/255


# Now, lets see the shape of our data.

# In[ ]:


print(train_data.shape)
print(test_data.shape)


# This is the prefect shape for our convolution model.

# ## CNN Models

# Here, I have tried different models with some changes.

# Lets try the first model as the simplest one (I think so).

# In[ ]:


model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape = (28,28,1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])


# Optimizing with 'sgd'

# In[ ]:


model_1.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In the next cell, I am training the complete trainable data with 20% as validation data.

# In[ ]:


history_1 = model_1.fit(train_data, target, epochs=10, validation_split = 0.2)


# I think the accuracies on both training and validation set are very nice and decent.
# > But I am not stopping here and going to try few more.

# Before that, lets look at the growth of accuracy with epoch on a plot.

# In[ ]:


plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()


# Now, lets add another Convolution layer in our model.
# > But this time with 32 filters.

# In[ ]:


model_2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])


# Optimizing with 'sgd' again.

# In[ ]:


model_2.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Again, making a validation set of 20% from training set.

# In[ ]:


history_2 = model_2.fit(train_data, target, epochs=10, validation_split = 0.2)


# Well, using another Convolution layer improved our result a bit.

# Plotting the Graph again.

# In[ ]:


plt.plot(history_2.history['accuracy'])
plt.plot(history_2.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()


# Lets try with Dropout now.

# In[ ]:


model_3 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])


# With all the things remain same.

# In[ ]:


model_3.compile(optimizer ='sgd', loss ='sparse_categorical_crossentropy', metrics =['accuracy'])


# Same code in the following cell as well.

# In[ ]:


history_3 = model_3.fit(train_data, target, epochs = 10, validation_split = 0.2)


# Using Dropout, our results didn't changed much.

# Plotting also remains the same.

# In[ ]:


plt.plot(history_3.history['accuracy'])
plt.plot(history_3.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()


# Now, lets try to optimize the above model with 'rmsprop'.

# In[ ]:


model_4 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), input_shape = (28,28,1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])


# Here is the change, note that I have changed the optimizer as 'rmsprop'.

# In[ ]:


model_4.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# Again fitting

# In[ ]:


history_4 = model_4.fit(train_data, target, epochs = 10, validation_split = 0.2)


# Wow, 'rmsprop' and the simplest model does the trick.

# And, again plotting.

# In[ ]:


plt.plot(history_4.history['accuracy'])
plt.plot(history_4.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()


# ##### What could be the reason for this curve according to you?

# ## Prediction

# I am using model_4 as my predictor.

# In[ ]:


results = model_4.predict(test_data)


# Predict function gives output as the probability of all 10 digits.

# In[ ]:


results


# Now from these probabilities, we will select the digit with highest probability store in the same variable.

# In[ ]:


results = np.argmax(results, axis = 1)


# We got the results as we wanted.

# In[ ]:


results


# ## Submission

# In[ ]:


submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')


# In[ ]:


submission.shape


# In[ ]:


submission.head()


# In[ ]:


submission['Label'] = results


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('my_submission.csv', index = False)


# ## Wrap Up
# 
# So, as we have come to the end of this notebook, I want you to notice that I have just give you the glimse of what and how are things get done with CNN in a very simple lines of code. Although there are so many possibilities of making your model and try it out with different optimizers and loss. I have just applied very simple code above to make things clear on how to make a start.
# 
# Also note that I have trained my model only with 10 epochs. So, that can also be tune by you according to the loss and optimizer you use.
# 
# Thanks for making till the end...

# #### Upvote if u like

# In[ ]:




