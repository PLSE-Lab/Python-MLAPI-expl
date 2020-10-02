#!/usr/bin/env python
# coding: utf-8

# ## A simple Neural Net model

# Classification of hand-written digits using a simple Neural Network Model on the MNIST dataset.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


x_train = train.drop(columns=['label'])
y_train = train['label']
g = sns.countplot(y_train)


# We have similar counts for each digit. The data is not skewed towards any single digit.

# #### Normalisation
# Normalisation has a very big impact on the overall performance of the network. Non-normalised nets do not perform well.
# Since the pixel values range from 0-255, this forms a very large variation in pixel intensity values. By dividing it with 255, the values are normalised in the range 0-1.

# In[ ]:


x_train = x_train / 255.0
test = test / 255.0
x_train.describe()


# > #### Creating the Network

# I have used Keras to create the network. Keras is a high-level API which is capable of running on tensorflow, theano, CNTK etc backend.
# Here I have used tensorflow backend.
# 

# I will be using a sequential feed-forward model i.e the data flows from left to right, there are no feedback connections.<br>
# For activation function, i will be using the Rectified Linear Unit (ReLU) for the hidden layers and SoftMax function for the output layer.
# The softmax function creates a probability distribution from 0 to 1 hence makes it convinient to predict classes.<br>
# Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored (setting their wieghts to zero) for each training sample. This drops randomly a propotion of the network and forces the network to learn features in a distributed way. This technique also improves generalization and reduces the overfitting.

# In[ ]:


# Creating the network

model = keras.models.Sequential() # Using the Sequentioal feed-forward model
model.add(keras.layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],))) # 1st hidden layer with 128 neurons
#model.add(keras.layers.Dense(128, activation='relu')) # 2nd hidden layer
model.add(keras.layers.Dropout(0.5)) # Randomly deactivates some neurons. (for 0.5, deactivates 50% neurons) Prevents overfitting.
model.add(keras.layers.Dense(128, activation='relu')) # 3rd hidden layer
model.add(keras.layers.Dense(10, activation='softmax')) # Output layer (using softmax activation function as we require categorical values)


# #### Initiailising parameters

# In[ ]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# Since the loss function used is *sparse_categorical_crossentropy* we need not convert our labels to one-hot encodings.

# ### Splitting in training and validation set

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=np.random.seed(2))


# ### Training

# In[ ]:


x_train = X_train
y_train = Y_train
history = model.fit(x_train, y_train, epochs=30, validation_data = (X_val,Y_val)) # 30 epochs gave around 98.05% accuracy


# ### Training and Validation plots

# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# Observing the above plots, we can see that model is performing pretty well, although there is a slight problem of overfitting after 15 epochs.
# Validation loss increases after around 15 epochs.

# ### Confusion Matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_train)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = y_train
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


# Confusion matrix is a great way of visualising the performance or rather the shortcomings of our model.
# Here we can see that the moedel is having problems classifying the digits 9,8,6. On further observation 9 is most confused with 7, 8 and 6 are most confused as 0. These mistakes are common and though can be improved upon by using a CNN, these mistakes can are also frequent by humans.

# In[ ]:


predictions = model.predict([test])


# In[ ]:


predictions


# #### Checking validation loss and acc

# In[ ]:


val_loss, val_acc = model.evaluate(X_val, Y_val)
print(val_loss, val_acc)


# * We have an accuracy of 98.21%

# In[ ]:


results = np.argmax(predictions,axis = 1)  # Since each prediction is a 1-hot array

results = pd.Series(results,name="Label")
#results


# In[ ]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
#print(submission)


# Writing predictions to csv file

# In[ ]:


submission.to_csv('submission.csv', index=False)


# More accurate results can be achieved using:
# *  a CNN model for training
# *  Data-augmentation to enhance the quality of the dataset
# *  Using a modified and tuned optimiser instead of default 'adam'
# *  Using Annealers for tweaking learning rate during training.

# 
