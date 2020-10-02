#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


# ## Data loading and overview

# In[ ]:


raw_train = pd.read_csv('../input/fashion-mnist_train.csv')
raw_test = pd.read_csv('../input/fashion-mnist_test.csv')


# In[ ]:


raw_train.head()


# In[ ]:


raw_test.head()


# ## Split labels and image pixels from raw data file

# In[ ]:


x_train = raw_train.iloc[:, 1:]
x_test = raw_test.iloc[:, 1:]
y_train = raw_train['label']
y_test = raw_test['label']


# In[ ]:


type(x_train), type(x_test), type(y_train), type(y_test)


# Because the data loaded by pandas is not numpy array as default, so we must to convert it into numpy array, and also normalize data if nessesary.

# In[ ]:


x_train = np.asarray(x_train) / 255.0
x_test = np.asarray(x_test) / 255.0

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


# In[ ]:


type(x_train), type(x_test), type(y_train), type(y_test)


# In[ ]:


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# transform data before seed into model

# In[ ]:


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[ ]:


x_train.shape, x_test.shape


# # Here we go to define our model by Keras

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


model = Sequential()

model.add(Conv2D(32, 5, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, 5, padding='same', activation='relu'))
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))

model.add(Conv2D(64, 3, padding='same', activation ='relu'))
model.add(Conv2D(64, 3, padding='same', activation ='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# In[ ]:


model.summary()


# we use Adam optimizer for better performance and accuracy

# In[ ]:


model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


# After several experimented, i found that model seems going to overfitting after nearly 9 epochs and reach best score at that point.
# 
# While training, we split the training data set into 2 part: train (90%) and validation (10%). In keras, it simply be done by param *validation_split=0.1*

# In[ ]:


epochs = 15
batch_size = 150


# In[ ]:


# best_model_ckpt = ModelCheckpoint(filepath='models/fashion_mnist_cnn_keras.hdf5', verbose=1, save_best_only=True)
# history = model.fit(x_train, y_train,
#                     validation_split=0.1,
#                     epochs=epochs, 
#                     batch_size=batch_size,
#                     callbacks=[best_model_ckpt])


# In[ ]:


history = model.fit(x_train, y_train,
                    validation_split=0.15,
                    epochs=epochs, 
                    batch_size=batch_size)


# ## Time for test

# In[ ]:


# from keras.models import load_model
# model = load_model('models/fashion_mnist_cnn_keras.hdf5')
score = model.evaluate(x_test, y_test)
print('Test Loss: {}'.format(score[0]))
print('Test Acc: {}'.format(score[1]))


# Here we visualize the loss and accuracy while training into graph to see our performance

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)


# We also viasulize which classes were often classfied wrong. It's 6 and 2 

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



y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis=1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_test, axis=1) 
# Compute the confusion matrix
confusion_matrix = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_matrix, classes = range(10)) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




