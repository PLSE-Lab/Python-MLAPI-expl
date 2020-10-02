#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train_norm = "../input/chest-xray-pneumonia/chest_xray/train/NORMAL"
train_pneu = "../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA"
val_norm = "../input/chest-xray-pneumonia/chest_xray/val/NORMAL"
val_pneu = "../input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA"
test_norm = "../input/chest-xray-pneumonia/chest_xray/test/NORMAL"
test_pneu = "../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA"


# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:


infected_images = []
for file in os.listdir(train_pneu):
    img = Image.open(os.path.join(train_pneu, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    infected_images.append(img)


# In[ ]:


for file in os.listdir(val_pneu):
    img = Image.open(os.path.join(val_pneu, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    infected_images.append(img)


# In[ ]:


print(len(infected_images))


# In[ ]:


normal_images = []
for file in os.listdir(train_norm):
    img = Image.open(os.path.join(train_norm, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    normal_images.append(img)


# In[ ]:


for file in os.listdir(val_norm):
    img = Image.open(os.path.join(val_norm, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    normal_images.append(img)


# In[ ]:


print(len(normal_images))


# In[ ]:


X_train = np.asarray(infected_images + normal_images)
y_train = np.asarray([1 for _ in range(len(infected_images))] + [0 for _ in range(len(normal_images))])


# In[ ]:


print(X_train.shape)
print(y_train.shape)

X_train = X_train.reshape((5232, 36, 36,1))
print(X_train.shape)
print(y_train.shape)


# In[ ]:


test_infected_images = []
for file in os.listdir(test_pneu):
    img = Image.open(os.path.join(test_pneu, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    test_infected_images.append(img)


# In[ ]:


test_normal_images = []
for file in os.listdir(test_norm):
    img = Image.open(os.path.join(test_norm, file)).convert('L')
    img = img.resize((36,36))
    img = np.asarray(img)
    test_normal_images.append(img)


# In[ ]:


X_test = np.asarray(test_infected_images + test_normal_images)
y_test = np.asarray([1 for _ in range(len(test_infected_images))] + [0 for _ in range(len(test_normal_images))])


# In[ ]:


print(X_test.shape)
print(y_test.shape)

X_test = X_test.reshape((624, 36, 36,1))
print(X_test.shape)
print(y_test.shape)


# In[ ]:


X = np.asarray(infected_images + test_infected_images + normal_images + test_normal_images)
y = np.asarray([1 for _ in range(len(infected_images)+len(test_infected_images))] + [0 for _ in range(len(test_normal_images)+len(normal_images))])


# In[ ]:


print(X.shape)
print(y.shape)
X = X.reshape((5856, 36, 36, 1))
print(X.shape)
print(y.shape)


# In[ ]:


from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes = 2)


# In[ ]:


from sklearn.utils import shuffle


# In[ ]:


X, y = shuffle(X, y)
X = X / 255.0


# In[ ]:


for i in range(10):
    print(y[i])
    plt.imshow(X[i].reshape((36, 36)))
    plt.show()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[ ]:


print("Train size:", X_train.shape, y_train.shape)
print("Test size:", X_test.shape, y_test.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D


# In[ ]:


model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape = X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


history = model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose = 1)


# In[ ]:


pred = [np.argmax(i) for i in model.predict(X_test)]
pred[:5]


# In[ ]:


tru = [np.argmax(i) for i in y_test]
from sklearn.metrics import confusion_matrix
confusion_matrix(tru, pred)


# In[ ]:


model.save('pneumonia2.h5')

