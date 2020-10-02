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
        
"""
UNIMPORTANT DATA IMPORT CODE
"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
batch1 = unpickle("../input/cifar10-python/cifar-10-batches-py/data_batch_1")
batch2 = unpickle("../input/cifar10-python/cifar-10-batches-py/data_batch_2")
batch3 = unpickle("../input/cifar10-python/cifar-10-batches-py/data_batch_3")
batch4 = unpickle("../input/cifar10-python/cifar-10-batches-py/data_batch_4")
batch5 = unpickle("../input/cifar10-python/cifar-10-batches-py/data_batch_5")
test_batch = unpickle("../input/cifar10-python/cifar-10-batches-py/test_batch")
def load_data0(btch):
    labels = btch[b'labels']
    imgs = btch[b'data'].reshape((-1, 32, 32, 3))
    
    res = []
    for ii in range(imgs.shape[0]):
        img = imgs[ii].copy()
        #img = np.transpose(img.flatten().reshape(3,32,32))
        img = np.fliplr(np.rot90(np.transpose(img.flatten().reshape(3,32,32)), k=-1))
        res.append(img)
    imgs = np.stack(res)
    return labels, imgs

labels, imgs = load_data0(batch1)
imgs.shape

def load_data():
    x_train_l = []
    y_train_l = []
    for ibatch in [batch1, batch2, batch3, batch4, batch5]:
        labels, imgs = load_data0(ibatch)
        x_train_l.append(imgs)
        y_train_l.extend(labels)
    x_train = np.vstack(x_train_l)
    y_train = np.vstack(y_train_l)
    
    x_test_l = []
    y_test_l = []
    labels, imgs = load_data0(test_batch)
    x_test_l.append(imgs)
    y_test_l.extend(labels)
    x_test = np.vstack(x_test_l)
    y_test = np.vstack(y_test_l)
    
    
    return (x_train, y_train), (x_test, y_test)

(X_train_val, y_train_val), (X_test, y_test) = load_data()

del batch1, batch2, batch3, batch4, batch5, test_batch


"""
SPLIT DATA INTO TRAIN/VALIDATION
"""

from tensorflow import keras
import matplotlib.pyplot as plt

print("Loaded data")
print('\n\n\n')

X_train_val_gray = np.dot(X_train_val, [0.299, 0.587, 0.114])
X_train_gray, X_val_gray = X_train_val_gray[:40000,:], X_train_val_gray[40000:,:]
y_train, y_val = y_train_val[:40000,:], y_train_val[40000:,:]

X_train_gray_scaled = X_train_gray / 255
X_val_gray_scaled = X_val_gray / 255

"""
DEFINE MODEL
"""

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32,32]))

model.add(keras.layers.Dense(500, kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(50, kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("elu"))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Dense(10, activation="softmax", kernel_initializer="he_normal"))

"""
Compile Model
"""
rmsprop = keras.optimizers.RMSprop(lr=3e-4, decay=1e-4)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=rmsprop,
              metrics=["accuracy"])

"""
TRAIN MODEL
"""

early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train_gray_scaled, y_train, epochs=200,
                    validation_data=(X_val_gray_scaled, y_val),
                   callbacks=[early_stopping])