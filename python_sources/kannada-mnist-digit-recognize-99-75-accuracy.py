#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Importing all the required libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# ## Read train data set

# In[ ]:


train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
print(train.shape)


# ## Separating dependent and independent variables

# In[ ]:


y = train["label"]
X = train.drop("label", axis = 1)
print(y.value_counts().to_dict())
y = to_categorical(y, num_classes = 10)
del train


# ## Reshape the independent variables 

# In[ ]:


X = X / 255.0
X = X.values.reshape(-1,28,28,1)


# ## Shuffle Split Train and Test from original dataset**

# In[ ]:


seed=2
train_index, valid_index = ShuffleSplit(n_splits=1,
                                        train_size=0.9,
                                        test_size=None,
                                        random_state=seed).split(X).__next__()


# In[ ]:


x_train = X[train_index]
Y_train = y[train_index]
x_test = X[valid_index]
Y_test = y[valid_index]


# ## Model definition

# In[ ]:


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Valid', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


# In[ ]:


model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='Valid', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))


# In[ ]:


model.add(Flatten())
model.add(Dense(519, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=["accuracy"])


# In[ ]:


annealer = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=2, factor=0.5, min_lr=0.0000001)


# In[ ]:


datagen = ImageDataGenerator(featurewise_center=False,samplewise_center=False,featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False)


# ## Parameters

# In[ ]:


epochs = 30
batch_size = 64
validation_steps = 10000


# ## Start training

# In[ ]:


train_generator = datagen.flow(x_train, Y_train, batch_size=batch_size)


# In[ ]:


test_generator = datagen.flow(x_test, Y_test, batch_size=batch_size)


# In[ ]:


history = model.fit_generator(train_generator,
                    steps_per_epoch=x_train.shape[0]//batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=validation_steps//batch_size,
                    callbacks=[annealer])


# In[ ]:


score = model.evaluate(x_test, Y_test)


# In[ ]:


score


# ## Saving Model for future API

# In[ ]:


model.save('Digits-Kannada-1.3.0.h5')


# ## summarize history for accuracy

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


# ## summarize history for loss

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


# ## Model predict

# In[ ]:


test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
print(test.shape)


# In[ ]:


test = test.drop(['id'],axis=1)


# In[ ]:


test = np.array(test)
test = test.reshape(5000,784)
print(test.shape)
test = test/255
test = test.reshape(-1,28,28,1)
test.shape


# In[ ]:


p = np.argmax(model.predict(test), axis=1)


# In[ ]:


len(p)


# In[ ]:


p


# In[ ]:


valid_loss, valid_acc = model.evaluate(x_test, Y_test, verbose=0)
valid_p = np.argmax(model.predict(x_test), axis=1)
target = np.argmax(Y_test, axis=1)
cm = confusion_matrix(target, valid_p)


# In[ ]:


cm


# ## Preparing file for submission

# In[ ]:


sample_file = pd.DataFrame({"Id": list(range(1,len(p)+1)),"Label": p})
sample_file.to_csv("kannada_didgit_kera_cnn.csv", index=False, header=True)


# In[ ]:


sample = pd.read_csv("kannada_didgit_kera_cnn.csv")


# In[ ]:


sample.head(10)

