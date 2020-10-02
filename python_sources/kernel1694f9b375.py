# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# importing the libraries

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

from keras.preprocessing.image import ImageDataGenerator

# Loading the data
df  = pd.read_csv('../input/aerial-cactus-identification/train.csv')

df['has_cactus'] = df['has_cactus'].apply(str)

gen = ImageDataGenerator(rescale=1./255., 
                             validation_split=0.1, horizontal_flip=True, vertical_flip=True,rotation_range = 3,
                            width_shift_range = 0.1,height_shift_range = 0.1,shear_range = 0.1)

train_gen = gen.flow_from_dataframe(df, directory='../input/aerial-cactus-identification/train/train', x_col='id', y_col='has_cactus', 
                                    target_size=(32, 32),class_mode='categorical', 
                                    batch_size=1024, shuffle=True, subset='training')

test_gen = gen.flow_from_dataframe(df, directory='../input/aerial-cactus-identification/train/train', x_col='id', y_col='has_cactus', 
                                    target_size=(32, 32),class_mode='categorical', 
                                    batch_size=1024, shuffle=True, subset='validation')

# Importin keras

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten,BatchNormalization, Activation, GlobalAveragePooling2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

def basline_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size= (3,3), strides=(1,1),padding='same',use_bias=False,
                     input_shape = (32,32,3), kernel_regularizer=l2(1e-4) ))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=32, kernel_size= (3,3), strides=(1,1),padding='same',use_bias=False,
                     input_shape = (32,32,3), kernel_regularizer=l2(1e-4) ))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    
    model.add(Conv2D(filters=32, kernel_size= (3,3), strides=(1,1),padding='same',use_bias=False,
                     input_shape = (32,32,3), kernel_regularizer=l2(1e-4) ))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPool2D())
    
    
    
    model.add(Conv2D(filters=64, kernel_size= (3,3), strides=(1,1),padding='same',use_bias=False,
                     input_shape = (32,32,3), kernel_regularizer=l2(1e-4) ))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters=64, kernel_size= (3,3), strides=(1,1),padding='same',use_bias=False,
                     input_shape = (32,32,3), kernel_regularizer=l2(1e-4) ))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    
    model.add(Conv2D(filters=64, kernel_size= (3,3), strides=(1,1),padding='same',use_bias=False,
                     input_shape = (32,32,3), kernel_regularizer=l2(1e-4) ))
    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPool2D())
    
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation='softmax'))

    return model

model = basline_model()

model.compile(optimizer= 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()

model.fit_generator(train_gen, steps_per_epoch=500, epochs=10, validation_data= train_gen,
                               validation_steps=100)

model.save_weights('cactus_weights.h5')
model.save('5_epoch_training.h5')

# Prediction the results

test_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

pred = np.empty(test_df.shape[0])

for n in tqdm(range(test_df.shape[0])):
  data = np.array(Image.open("../input/aerial-cactus-identification/test/test/" + test_df.id[n]))
  data = data.astype(np.float32) / 255.0
  pred[n] = model.predict(data.reshape(1,32,32,3))[0][1]

test_df['has_cactus'] = pred
test_df.to_csv('../input/aerial-cactus-identification/sample_submission.csv', index=False)
