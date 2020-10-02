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
        pass
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import librosa
import librosa.display as libdisplay
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


max_pad_len = 388
def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccs


# In[ ]:


ls  /kaggle/input/dataset-bagus


# In[ ]:


dataset_path = '/kaggle/input/heartbeat-sounds/set_a'
metadata = pd.read_csv('/kaggle/input/dataset-bagus/DATASET_A_NEW.csv')

features = []


# In[ ]:


import glob, os
os.chdir("/kaggle/input/heartbeat-sounds/set_a")


MAX_LEN_MFCC = []

for file in glob.glob("*.wav"):
    #print(file)
    #extract_features(file)

    audio,sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    #print(mfccs.shape[1])
    MAX_LEN_MFCC.append(mfccs.shape[1])

print(max(MAX_LEN_MFCC)   )

# dari data dapet 388 len nya 


# In[ ]:


for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(dataset_path),str(row['fname']))
    
    class_label = row['fname']
    data = extract_features(file_name)
    
    features.append([data,class_label])


# In[ ]:


DATAX = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(DATAX), ' files')


# In[ ]:


str(DATAX['class_label'][0]).split('__')[0]

LABEL = []

for i in range(len(DATAX)):
  str(DATAX['class_label'][i]).split('__')[0]
  LABEL.append(str(DATAX['class_label'][i]).split('__')[0])    


# In[ ]:


DATAX['LABEL'] = LABEL

NEW_DATAX = DATAX.drop(columns=['class_label'])

#NEW_DATAX.to_csv(r'NEW_DATAX.csv')

NEW_DATAX.head(10)


# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, utils
import matplotlib.pyplot as plt
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder

X = np.array(NEW_DATAX.feature.tolist())
y = np.array(NEW_DATAX.LABEL.tolist())


# In[ ]:


le = LabelEncoder()
yy = tf.keras.utils.to_categorical(le.fit_transform(y))

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# In[ ]:


#Untuk menentukan column nanti di model

print('NEW_DATAX shape >> {}'.format(NEW_DATAX.shape))
print('len X_train >> {}'.format(len(x_train)))
print('len X_test >> {}'.format(len(x_test)))

# untuk dapetin num_rows, num_clumns d
print('X_train shape >> {}'.format(x_train.shape))

# len(mfccs) --> nanti kita gunakan untuk num_rows
# max(MAX_LEN_MFCC) --> dibunakan untuk num_columns & max_pad_len pada fungsi extract_features
print('Num_Rows >> {}'.format(len(mfccs)))
print('Num_Columns >> {}'.format(max(MAX_LEN_MFCC)))


# **Mempersiapkan nilai untuk Train & Test**

# In[ ]:


num_rows = 40
num_columns = 388
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2


# > **CNN MODEL**

# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GlobalAveragePooling2D())

model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))


# In[ ]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[ ]:


model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


# In[ ]:


#tf.keras.callbacks.ModelCheckpoint

from datetime import datetime 

#num_epochs = 500
#num_batch_size = 128

#num_epochs = 500
#num_batch_size = 256

num_epochs = 1000
num_batch_size = 256

#num_epochs = 1000
#num_batch_size = 256

#checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='sound_classification_Bagus.hdf5', 
#                               verbose=1, save_best_only=True)
start = datetime.now()

#model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# In[ ]:


score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


# In[ ]:


#model.save('my_model_BAGUS_SOUND.h5')


# In[ ]:


def buat_prediction(file_name):
    prediction_feature = extract_features(file_name) 
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )


# Prediction

# In[ ]:


filename = '/kaggle/input/heartbeat-sounds/set_a/Aunlabelledtest__201012172010.wav' 
buat_prediction(filename)


# In[ ]:


filename = '/kaggle/input/heartbeat-sounds/set_a/Aunlabelledtest__201108222257.wav' 
buat_prediction(filename)

