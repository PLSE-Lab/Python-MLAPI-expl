#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import Image
Image("../input/amer_sign2.png")


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/sign_mnist_train.csv')
test = pd.read_csv('../input/sign_mnist_test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


Image("../input/american_sign_language.PNG")


# In[ ]:


labels = train['label'].values


# In[ ]:


unique_val = np.array(labels)
np.unique(unique_val)


# In[ ]:


plt.figure(figsize = (18,8))
sns.countplot(x =labels)


# In[ ]:


train.drop('label', axis = 1, inplace = True)


# In[ ]:


images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])
images = np.array([i.flatten() for i in images])


# In[ ]:


from sklearn.preprocessing import LabelBinarizer
label_binrizer = LabelBinarizer()
labels = label_binrizer.fit_transform(labels)


# In[ ]:


labels


# In[ ]:


plt.imshow(images[0].reshape(28,28))


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.3, random_state = 101)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout


# In[ ]:


batch_size = 128
num_classes = 24
epochs = 50


# In[ ]:


x_train = x_train / 255
x_test = x_test / 255


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)


# In[ ]:


x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


# In[ ]:


plt.imshow(x_train[0].reshape(28,28))


# In[ ]:


model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation = 'relu', input_shape=(28, 28 ,1) ))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.20))
model.add(Dense(num_classes, activation = 'softmax'))


# In[ ]:


model.compile(loss = keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=epochs, batch_size=batch_size)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','test'])
plt.show()


# In[ ]:


test_labels = test['label']


# In[ ]:


test.drop('label', axis = 1, inplace = True)


# In[ ]:


test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])


# In[ ]:


test_labels = label_binrizer.fit_transform(test_labels)


# In[ ]:


test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)


# In[ ]:


test_images.shape


# In[ ]:


y_pred = model.predict(test_images)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(test_labels, y_pred.round())


# In[ ]:


from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:




