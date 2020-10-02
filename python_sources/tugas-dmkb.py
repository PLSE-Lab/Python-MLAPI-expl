#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
np.random.seed(666)
import cv2
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Langkah awal pada penggunaan python adalah mendeklarasikan library yang akan digunakan pada proses
# seperti kode di atas, pada kode ini menggunakan numpy untuk penghitungan algebra, cv2 merupakan OpenCV yang digunakan untuk pemrosesan gambar, pandas digunakan untuk processing data seperti membuka file dll.

# In[ ]:


#Load the data.
train = pd.read_json("../input/train.json")
test = pd.read_json("../input/test.json")


# Kode diatas adalah digunakan untuk membuka/input data yang akan digunakan.

# In[ ]:


#Generate the training data
#Create 3 bands having HH, HV and avg of both
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
Y_train=train['is_iceberg']


# Proses diatas merupakan mengolah data yang sudah di inputkan sehingga dapat digunakan untuk proses selanjutnya yaitu proses cnn. untuk proses selanjutnya memerlukan 3 parameter input dan 1 parameter output. untuk data ini input menggunakan x_band_1, x_band_2, x_training ( yang merupakan hasil dari (x_band_1 + x_band_2)/2 ), dengan aksis = -1. dan untuk output menggunakan is_iceberg dengan 1 merupakan iceberg dan 0 merupakan kapal.

# In[ ]:


from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau


# Kode diatas merupakan deklarasi library yang akan digunakan untuk proses CNN.

# In[ ]:


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]


# In[ ]:


def getModel():
    #Build keras model
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(8, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = getModel()
model.summary()


# Kode diatas merupakan fungsi fungsi yang akan digunakan pada proses CNN.
# Arsitektur dan Hyperparameter
# * 4 Layer konvolusi dengan fungsi aktivasi ReLu
# * 1 Input layer 
# * 2 Hidden layer (512 Neuron dan 256 Neuron) dengan fungsi aktivasi ReLu
# * 1 Output layer (1 Neuron) dengan fungsi aktivasi sigmoid
# * Learning rate = 0.001 (optimizer Adamax)
# * Untuk perhitungan loss, menggunakan binary_crossentropy
# * 100 epoch
# * Menggunakan dropout = 0.2
# 

# In[ ]:


batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


# In[ ]:


X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, Y_train, random_state=1, train_size=0.75)

model=getModel()
model.fit(X_train_cv, y_train_cv, batch_size=batch_size, epochs=100, verbose=1, validation_data=(X_valid, y_valid), callbacks=[earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)


# kode diatas merupakan proses training pada data. proses training ini dilakukan dengan epoch sebanyak 100 dan ukuran batch adalah 32. untuk setiap epoch, data akan dibagi menjadi 75% data train dan 25% data validasi.

# In[ ]:


model.load_weights(filepath = '.mdl_wts.hdf5')

score = model.evaluate(X_train, Y_train, verbose=1)
print('Train score:', score[0])
print('Train accuracy:', score[1])


# kode diatas adalah untuk mengecek skor dan akurasi dari data train.

# In[ ]:


X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([X_band_test_1[:, :, :, np.newaxis]
                          , X_band_test_2[:, :, :, np.newaxis]
                         , ((X_band_test_1+X_band_test_2)/2)[:, :, :, np.newaxis]], axis=-1)
pred_test = model.predict(X_test)


# In[ ]:


submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
submission.head(10)


# proses selanjutnya adalah memprediksi hasil dengan data test. kemudian simpan hasil pada submission.csv.

# In[ ]:


submission.to_csv("./submission.csv", index=False)


# In[ ]:




