#!/usr/bin/env python
# coding: utf-8

# * [2. Show image](#show_image)
# * [3. Set up CNN model](#setup_cnn)
# * [4. Compile CNN model](#compile_cnn)
# * [5. Fit CNN model](#fit_cnn)
# * [6. Show CNN train history](#cnn_train_history)
# * [7. Show top 6 prediction errors](#show_pred_error)
# * [8. Final score and position](#score)
# * [9. Referred kernel](#reference)

# In[ ]:


from keras.utils import to_categorical
import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np 


# ## 1. Prepare data

# ### 1.1 Load CSV

# In[ ]:


data = pd.read_csv("../input/train.csv")
y = data.label
x = data.drop('label', axis=1)


# ### 1.2 split data to train and validation set

# In[ ]:


x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size = 0.1, random_state=30)


# ### 1.3 Reshape and normalize: x_train

# In[ ]:


x_train = np.array(x_train).reshape(len(x_train), 28, 28, 1).astype('float32') / 255
x_validation = np.array(x_validation).reshape(len(x_validation), 28, 28, 1).astype('float32') / 255


# ### 1.4 One-Hot Encoding: y_train 

# In[ ]:


y_train = to_categorical(y_train, 10)  
y_validation = to_categorical(y_validation, 10)  


# ### 1.5 Display data shape

# In[ ]:


print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_validation.shape: ', x_validation.shape)
print('y_train.shape: ', y_validation.shape)


# <a id='show_image'></a>
# ## 2. Show image

# In[ ]:


plt.imshow(x_train[0][:,:,0])
plt.title('label: {}'.format(np.argmax(y_train[0])))


# <a id='setup_cnn'></a>
# ## 3. Set up CNN model

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28, 28, 1),
                 activation='relu'))
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=36,
                 kernel_size=(3, 3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))


# <a id='compile_cnn'></a>
# ## 4. Compile CNN model

# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# <a id='fit_cnn'></a>
# ## 5. Fit CNN model

# ### 5.1 Data augmentation
# We can transform images by shifting, rotating, zooming, flipping etc. to generate more data to train model.

# In[ ]:


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images
datagen.fit(x_train)


# ### 5.2 Decay learning rate
# Reduce the learning rate by half if the accuracy is not improved after 3 epoch.

# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


# ### 5.3 Fit by batched train data

# In[ ]:


batch_size = 32
epochs = 30
train_history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                    epochs=epochs, validation_data=(x_validation, y_validation),
                                    verbose=2, steps_per_epoch=x_train.shape[0] // batch_size
                                    , callbacks=[learning_rate_reduction])


# <a id='cnn_train_history'></a>
# ## 6. Show CNN train history: 30 epochs

# In[ ]:


plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
epoch_num = len(train_history.epoch)
final_epoch_train_acc = train_history.history['acc'][epoch_num - 1]
final_epoch_validation_acc = train_history.history['val_acc'][epoch_num - 1]
plt.text(epoch_num, final_epoch_train_acc, 'train = {:.3f}'.format(final_epoch_train_acc))
plt.text(epoch_num, final_epoch_validation_acc-0.01, 'valid = {:.3f}'.format(final_epoch_validation_acc))
plt.title('Train History')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.xlim(xmax=epoch_num+1)
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ## 7. Show top 6 prediction errors

# In[ ]:


def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
            n += 1
    plt.show()


# In[ ]:


y_pred = model.predict(x_validation)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_validation, axis=1)

errors = (y_pred_classes - y_true != 0)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_prob_errors = y_pred[errors]
y_true_classes_errors = y_true[errors]
x_validation_errors = x_validation[errors]

y_pred_maxProb_errors = np.max(y_pred_prob_errors, axis=1)
y_true_prob_errors = np.diagonal(np.take(y_pred_prob_errors, y_true_classes_errors, axis=1))
deltaProb_pred_true_errors = y_pred_maxProb_errors - y_true_prob_errors
sorted_delaProb_errors = np.argsort(deltaProb_pred_true_errors)

# Top 6 errors
top6_errors = sorted_delaProb_errors[-6:]

# Show the top 6 errors
display_errors(top6_errors, x_validation_errors, y_pred_classes_errors, y_true_classes_errors)


# ## 8. Submit my prediction
# 

# In[ ]:


data = pd.read_csv("../input/test.csv")
x_test = data.values.reshape(len(data.values), 28, 28, 1).astype('float32') / 255
print('x_test.shape: ', x_test.shape)


# In[ ]:


prediction = model.predict_classes(x_test)
df = pd.DataFrame(prediction)
df.index += 1
df.index.name = 'ImageId'
df.columns = ['Label']
df.to_csv('submission.csv', header=True)


# <a id='score'></a>
# ## 8. Final score and position (2018/06/29)
# * Score = 0.99642
# * Position = Top 8%

# <a id='reference'></a>
# ## 9. Referred kernel
# [Yassine Ghouzam: Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)

# In[ ]:




