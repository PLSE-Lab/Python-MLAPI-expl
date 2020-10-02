#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split


# In[ ]:


train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
X = train.drop(['label'],1).values
Y = train['label'].values
x_test = test.values

X = X/255.
x_test = x_test/255.

X = X.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

Y = to_categorical(Y)


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(X,Y, test_size=0.1)


# In[ ]:


def get_model():
    In = Input(shape=(28,28,1))
    x = Conv2D(32, (4,4), activation="relu",padding="same")(In)
    x = Conv2D(32, (4,4), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(64, (4,4), activation="relu", padding="same")(x)
    x = Conv2D(64, (4,4), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (4,4), activation="relu",padding="same")(x)
    x = Conv2D(128, (4,4), activation="relu")(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    Out = Dense(10, activation="softmax")(x)
    model = Model(In, Out)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
    
model = get_model()
model.summary()


# In[ ]:


data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


epochs = 50
batch_size = 128

train_generator = data_generator.flow(x_train, y_train, batch_size=batch_size)
valid_generator = data_generator.flow(x_valid, y_valid, batch_size=batch_size)


# In[ ]:


model.fit_generator(train_generator, epochs=epochs, steps_per_epoch = x_train.shape[0]//batch_size,
                    validation_data = valid_generator, validation_steps = x_valid.shape[0]//batch_size)


# In[ ]:


preds = model.predict(x_test, verbose=1)
preds = np.array([np.argmax(i) for i in preds])
preds


# In[ ]:


submission['Label'] = preds
submission.to_csv("submission.csv", index=False)
submission.head()


# In[ ]:





# In[ ]:





# In[ ]:




