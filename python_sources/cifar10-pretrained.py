import numpy as np
import pandas as pd 
import os
import keras
import os
import pandas as pd
import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical
from tensorflow.keras import optimizers
model = InceptionV3(weights='imagenet',include_top=False,input_shape=(128,128,3))
model.summary()
from keras.layers import Dense, Activation, Dropout, Flatten,Input
from keras.models import Model
x = model.output
x = Flatten()(x)
x = Dense(4096, activation='tanh')(x) 
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.7)(x)
x = Dense(10, activation='softmax')(x)
model1 = Model(inputs=model.input,outputs=x,name='model1')
model1.summary()
from keras import optimizers
model1.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/kaggle/input/cifar10/cifar10/CIFAR10/TRAINSET',
                                                    target_size=(128, 128),
                                                    batch_size=128,
                                                    class_mode='categorical')
model1.fit_generator(train_generator,
                     steps_per_epoch=52,
                     epochs=10)
                     
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('/kaggle/input/cifar10/cifar10/CIFAR10/TESTSET',
                                                        target_size=(128, 128),
                                                        batch_size=128,
                                                        class_mode='categorical')
score = model1.evaluate_generator(test_generator, steps=69, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])