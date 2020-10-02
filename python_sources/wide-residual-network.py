#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
A pratice to build ResNet unit, in order to stack thess unit to form Residual Network

Using CIFAR10 as testing dataset
'''

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Add, Input, AveragePooling2D
from keras.layers import Flatten
from keras import optimizers, regularizers
from keras.callbacks import LearningRateScheduler

import math


# In[ ]:


num_classes        = 10

batch_size         = 128
epochs             = 300
iterations         = math.ceil(50000/ batch_size)
weight_decay       = 1e-4


# In[ ]:


def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test

def scheduler(epoch):
    if epoch < 60:
        return 1e-1
    if epoch < 120:
        return 2e-2
    if epoch < 160:
        return 4e-3
    return 8e-4


# In[ ]:


def conv3x3(img, filters):
    #x = BatchNormalization(momentum=0.9, epsilon=1e-5)(img)
    x = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(img))
    x = Conv2D(filters, kernel_size=(3,3), padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    return x

def identity(img, filters, kfold=2):
    shortcut = img

    x = conv3x3(img, kfold*filters)
    x = conv3x3(x, kfold*filters)
    x = Add()( [x, shortcut])

    return x

def projection_block(img, filters, kfold=2, stride=(2,2)):
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(img)
    x = Activation('relu')(x)
    
    shortcut = x
    
    x = Conv2D(kfold*filters, kernel_size=(3,3), strides=stride, padding='same',
                kernel_initializer='he_normal', 
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = conv3x3(x, kfold*filters)

    shortcut = Conv2D(kfold*filters, kernel_size=(1,1), strides=stride, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay))(shortcut)
    x = Add()( [x, shortcut])

    return x

def wrn(img, filters, stacks):
    x = Conv2D(filters[0], kernel_size=(3,3), strides=(1,1), padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(img)

    x = projection_block(x, filters[0], kfold=8, stride=(1,1))
    for i in range(1, stacks):
        x = identity(x, filters[0], kfold=8)
    
    x = projection_block(x, filters[1], kfold=8, stride=(2,2))
    for i in range(1, stacks):
        x = identity(x, filters[1], kfold=8)

    x = projection_block(x, filters[2], kfold=8, stride=(2,2))
    for i in range(1, stacks):
        x = identity(x, filters[2], kfold=8)

    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(num_classes,activation='softmax',
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)

    return x


# In[ ]:


if __name__ == '__main__':
    '''filters = [64, 128, 256, 512]
    stacks  = [2,3,5,2]'''
    
    filters = [16,32,64]
    #stacks  = [9,8,8]
    stacks  = 2
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, x_test = color_preprocessing(x_train, x_test)
    
    img_input   = Input(shape=(32,32,3))
    output      = wrn(img_input, filters=filters, stacks=stacks)
    resnet      = Model(img_input, output)

    resnet.summary()


# In[ ]:


# set optimizer
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# set callback
cbks = [LearningRateScheduler(scheduler)]

# dump checkpoint if you need.(add it to cbks)
# ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

# set data augmentation
print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant',cval=0.)

datagen.fit(x_train)

# start training
resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                     steps_per_epoch=iterations,
                     epochs=epochs,
                     callbacks=cbks,
                     validation_data=(x_test, y_test))
#resnet.save('resnet_{:d}_{}.h5'.format(layers,args.dataset))

scores = resnet.evaluate(x_test, y_test, batch_size=256)
print('Accy: %06.5f' % scores[1])

