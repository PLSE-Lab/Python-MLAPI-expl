#!/usr/bin/env python
# coding: utf-8

# This notebook applys typical CNN to classify MNIST data using Keras.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(514) # setting the seed so each time the result is the same

from sklearn.metrics import confusion_matrix

# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)
Y_train.value_counts()


# From the result of value_counts(), we see that the dataset is quite balanced so there is no need to enable stratify=True in train-test splitting.
# 
# Before building a model, let's prepare the data first. First check if there is any missing data and perform standardization...

# In[ ]:


# See if there is any missing values
print("Number of missing value in training set: {}".format(train.isnull().sum().sum()))
print("Number of missing value in test set: {}".format(test.isnull().sum().sum()))


# There is no missing data in any set. Let's proceed to standard procedures (standization, label-encoding, train-test split etc).

# In[ ]:


# Standization of pixel values
X_train = X_train / 255.0
test = test / 255.0

# Reshape to (num_images, row, col, color_channel)
X_train = X_train.values.reshape(X_train.shape[0],28,28,1)
test = test.values.reshape(test.shape[0],28,28,1)

# Encode labels to one-hot vectors
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train)

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=514)


# We can show one of the image by using imshow from matplotlib.

# In[ ]:


# visualization of image
plt.imshow(X_train[0][:,:,0], cmap=plt.get_cmap('gray'))


# Next we proceed to set up a convolutional neural network using Keras. Follow the example "VGG-like convnet" as given in [Keras' guide](https://keras.io/getting-started/sequential-model-guide/) with additional of setting padding equal to same. Also a dropout is applied as a mean of regularization. Here the dropout rates are well-known hyperparameters which has to be chosen by trial-and-error...

# In[ ]:


# setup of CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='Same',
                 activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='Same',
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='Same',
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


# RMSprop is chosen to be the optimizer instead of Stochastic Gradient Descent (SGD) as given in the example. The method "ReduceLROnPlateau" is chosen to be an annealer of the learning rate so as to reach the global minimum more efficiently. The number of epochs and batch size are also hyperparameters...

# In[ ]:


from keras.optimizers import RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, 
                              verbose=1, factor=0.2, min_lr=0.0001)
epochs = 30 #hyperparameter
batch_size = 64 #hyperparameter


# Apply the ImageDataGenerator class to produce augmented data again to prevent overfitting. The parameters are chosen to resemble roughtly the process of hand writting.
# Fit the augmented data to the CNN model. Record the history so as to plot the loss and accuracy curves later...

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=10, 
            width_shift_range=0.1, height_shift_range=0.1, 
            zoom_range = 0.1)
datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs=epochs, validation_data=(X_val,Y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[reduce_lr])


# In[ ]:


# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# We can see from the plot that validation loss goes up slightly after 25th epoch. It is a sign of overfitting. Tune around the hyperparameters to achieve a better result.
# 
# Anyway we proceed to predict values from the validation dataset and prepare the confusion matrix plot...

# In[ ]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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

Y_pred = model.predict(X_val)
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
Y_true = np.argmax(Y_val, axis = 1) 

# compute and plot the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_confusion_matrix(confusion_mtx, classes = range(10))


# In[ ]:


# finally compute the result using the test data
y_test = model.predict(test)
sub = pd.DataFrame()
sub['ImageId'] = range(1,28001)
sub['Label'] = np.argmax(y_test,axis = 1)
sub.to_csv('submission.csv',index=False)

