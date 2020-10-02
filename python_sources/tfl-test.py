# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf 
import keras
import pandas
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from tensorflow.python.keras.layers import Lambda
from sklearn.model_selection import train_test_split
from glob import glob 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 128
num_train_images = 1000

CLASSES = os.listdir('../input/food41/images')
TRAIN_DIR = '../input/food41/images'

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)

base_model = ResNet50 (weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(HEIGHT,WIDTH), batch_size=BATCH_SIZE)

def build_finetune_model (base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    return finetune_model
    
dropout = .5

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=[1024, 1024], 
                                      num_classes=101)

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

history = finetune_model.fit_generator(train_generator, epochs=1, workers=8, steps_per_epoch=num_train_images // BATCH_SIZE,
shuffle=True)

img = image.load_img('../input/food41/images/apple_pie/1005649.jpg', target_size=(300,300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = finetune_model.predict(x)
print(preds)

# print(os.listdir("../input/food41"))

# Any results you write to the current directory are saved as output.