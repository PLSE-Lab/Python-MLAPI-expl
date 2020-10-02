#!/usr/bin/env python
# coding: utf-8

# ## Determine baseline

# Before investing time in building a neural net, we should understand what would constitute an improvement over random guessing. 
# 
# Since each image is one of ten digits, random guessing has a 10% chance of getting the digit right. **So if our model achieves of accuracy of >10% against the test dataset, we win.**

# ## Set up imports and magics

# We need **keras** for the neural net, **matplotlib** for charting our loss and accuracy across epochs, **numpy** for decoding one-hot predictions, and **pandas** for loading the data.

# In[ ]:


from keras import callbacks
from keras import layers
from keras import models
from keras import regularizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Make sure loss and accuracy charts appear directly in this notebook.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Preprocess data

# We *could* read the Kaggle-provided data file directly into a Numpy array, but Pandas has easier-to-understand methods for handling CSVs. So instead, we'll read the data file into a Pandas DataFrame and then convert that into a Numpy array.

# In[ ]:


train_data_df = pd.read_csv('../input/train.csv')
train_data = train_data_df.values


# Per naming standards, store our training data in *X_train*. Exclude the labels, since we'll store that in a separate *y_train* variable. 

# In[ ]:


X_train = train_data[:, 1:]


# Cast the training data as 16-bit floats, which most GPUs find faster to deal with than 32-bit floats. Also, scale the training data so it ranges from 0 to 1.0. Neural nets usually operate better with this range of input data.

# In[ ]:


X_train = X_train.astype('float16') / 255.0


# Convert the training data (which consists of images) to 28x28 matrices, representing the 28 pixel height, 28 pixel width of the images.

# In[ ]:


X_train = X_train.reshape((42000, 28, 28, 1))


# Store the labels in *y_train* (per naming conventions), and convert them to one-hot encoding.

# In[ ]:


y_train = train_data[:, 0]
y_train = to_categorical(y_train)


# Do most of the same preprocessing as described above (minus processing the targets), but this time on our test data.

# In[ ]:


test_data_df = pd.read_csv('../input/test.csv')
test_data = test_data_df.values

X_test = test_data
X_test = X_test.astype('float16') / 255.0
X_test = X_test.reshape((28000, 28, 28, 1))


# ## Establish Keras callbacks

# Use a callback to halt training if it hasn't seen any improvement in validation accuracy (during exploratory training runs) or training accuracy (during "production" runs, when when we train on the entire dataset with nothing held aside for validation) over the last three training epochs. It also realoads the weights from whichever epoch produced the highest validation accuracy. Sweet!

# In[ ]:


# use this callback during exploratory training
# stop_early_callback = callbacks.EarlyStopping(monitor='val_acc', 
#                                               patience=3, 
#                                               restore_best_weights=True,
#                                               verbose=1)

# use this callback during "production" training, when we don't set aside a validation dataset
stop_early_callback = callbacks.EarlyStopping(monitor='acc', 
                                              patience=5, 
                                              restore_best_weights=True,
                                              verbose=1)

callbacks_list = [stop_early_callback]


# ## Build and train a convolutional neural net

# Our model has three convolution layers in the base and one dense layer in the head. Experimentation with this dataset revealed that L1 and L2 regularizers have no effect. BachNormalization layers have no effect. The dropout layer in the head does marginally reduce overfitting.
# 
# All hyperparameters were set through long trial and error. It was easy to mess up validation accuracy, but very hard to get any higher than 99.0%.

# In[ ]:


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['acc'])

validation_split = 0.0 # we'll need this variable later, when deciding whether to graph our validation figures

history = model.fit(X_train, 
                    y_train, 
                    callbacks=callbacks_list, 
                    epochs=30, 
                    batch_size=256, 
                    validation_split=validation_split, 
                    verbose=2)


# ## Graph our progress

# When experimenting with hyperparamter values, it's helpful to see graphs of four values after each epoch:
# * training loss
# * validation loss
# * training accuracy
# * validation accuracy 

# In[ ]:


epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], 'ro', label='training loss')
if validation_split > 0:
    plt.plot(epochs, history.history['val_loss'], 'r', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(epochs, history.history['acc'], 'bo', label='training acc')
if validation_split > 0:
    plt.plot(epochs, history.history['val_acc'], 'b', label='val acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()


# ## Make predictions for Kaggle's test data

# In[ ]:


predictions_one_hot = model.predict(X_test)


# The CNN's output is one-hot encoded, so we need to convert it from (for example) **[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]** to **2** since the latter is the form that Kaggle expects in submissions.

# In[ ]:


predictions_ints = [np.argmax(prediction_one_hot) for prediction_one_hot in predictions_one_hot]


# Make a CSV containing our predictions, using Kaggle's required format.

# In[ ]:


with open('predictions.csv', 'w') as predictions_file:
    predictions_file.write('ImageId,Label' + '\n')
    for i in range(len(test_data)):
        predictions_file.write(f'{i + 1},{predictions_ints[i]}' + '\n')

