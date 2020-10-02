# -*- coding: utf-8 -*-
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import LearningRateScheduler as LRS
from keras.preprocessing.image import ImageDataGenerator



batch_size = 32
num_classes = 20

#!wget https://www.dropbox.com/s/sakfqp6o8pbgasm/data.tgz
#!tar xvzf data.tgz

path = '../input/'

# Load 
x_train = np.load(path+'x_train.npy')
x_test = np.load(path+'x_test.npy')

y_train = np.load(path+'y_train.npy')
y_test = np.load(path+'y_test.npy')

# Stats
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

## View some images
#plt.imshow(x_train[500,:,:,: ] )
#plt.show()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


x_train /= 255
x_test /= 255


## Labels
y_train=y_train-1
y_test=y_test-1

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def CBGN(model,filters,lname,ishape=0):
    if (ishape!=0):
        model.add(Conv2D(filters, (3, 3), padding='same',
                 input_shape=ishape))
    else:
        model.add(Conv2D(filters, (3, 3), padding='same'))


    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))

    model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(BN())
    model.add(GN(0.3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name=lname))

    return model

##############################################
## DEFINE A DATA AUGMENTATION GENERATOR
## WITH MULTIPLE INPUTS
##############################################
datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=20,
    zoom_range=[1.0,1.2],
    horizontal_flip=True)

testdatagen = ImageDataGenerator()

def multiple_data_generator(generator, X,Y,bs):
    genX = generator.flow(X, Y,batch_size=bs)
    while True:
      [Xi,Yi] = genX.next()
      yield [Xi,Xi],Yi

##############################################
## LOAD IMAGENET PRETRAINED MODEL
## FREEZE ALL FEATURE LAYERS AND RENAME IT
##############################################
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import ZeroPadding2D

pretrained1 = VGG16(include_top=False, input_shape=x_train.shape[1:])
pretrained2 = InceptionResNetV2(include_top=False, input_shape=x_train.shape[1:])
print(pretrained1.input.shape,pretrained1.output.shape)
print(pretrained2.input.shape,pretrained2.output.shape)
for i in pretrained1.layers:
    i.trainable=False
    i.name='model1_'+i.name
for i in pretrained2.layers:
    i.trainable=False
    i.name='model2_'+i.name

d1=Dropout(0.5)(pretrained1.output)
pad=ZeroPadding2D(((1,0),(0,1)))(pretrained2.output)
d2=Dropout(0.5)(pad)

def outer_product(x):
    depth = (pretrained1.output.shape[3],pretrained2.output.shape[3])
    size = (int(pretrained1.output.shape[1]),int(pretrained2.output.shape[2]))
    print(size,depth,x[0].shape,x[1].shape)
    
    phi_I = tf.einsum('ijkm,ijkn->imn',x[0],x[1])# Einstein Notation  [batch,1,1,depth] x [batch,1,1,depth] -> [batch,depth,depth]
    print(phi_I.shape)
    phi_I = tf.reshape(phi_I,[-1,depth[0]*depth[1]])	        # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
    print(phi_I.shape)
    phi_I = tf.divide(phi_I,size[0]*size[1])								  # Divide by feature map size [sizexsize]
    print(phi_I.shape)

    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))		# Take signed square root of phi_I
    z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)								              # Apply l2 normalization
    return z_l2
x = Lambda(outer_product, name='outer_product')([d1,d2])
predictions=Dense(num_classes, activation='softmax', name='predictions')(x)
model = Model(inputs=[pretrained1.input,pretrained2.input], outputs=predictions)

## OPTIM AND COMPILE
opt = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()

## FIRST STEP TRAINING, TRAIN ONLY PREDICTION LAYER
epochs1 = 15
epochs2 = 75
history1=model.fit_generator(multiple_data_generator(datagen, x_train, y_train, batch_size),
                            steps_per_epoch=len(x_train) / batch_size, 
                            epochs=epochs1,
                            validation_steps=len(x_test) / batch_size,
                            validation_data=multiple_data_generator(testdatagen, x_test, y_test, batch_size),
                            #callbacks=[set_lr],
                            verbose=2)
model.save('bilinear-2models-vgg16-2step.h5')

## FREEZE PREDICTION LAYER AND UNFREEZE FEATURE LAYERS
for l in model.layers:
    l.trainable=True
model.get_layer('predictions').trainable=False
# DEFINE A LEARNING RATE SCHEDULER
def scheduler(epoch):
    if epoch < 40:
        return 1e-3
    elif epoch < 60:
        return 1e-4
    return 1e-5
set_lr = LRS(scheduler)
opt = SGD(1e-3, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

## PERFORM FINE-TUNING
history2=model.fit_generator(multiple_data_generator(datagen, x_train, y_train, batch_size),
                            steps_per_epoch=len(x_train) / batch_size, 
                            epochs=epochs2,
                            initial_epoch=epochs1,
                            validation_steps=len(x_test) / batch_size,
                            validation_data=multiple_data_generator(testdatagen, x_test, y_test, batch_size),
                            callbacks=[set_lr],
                            verbose=2)
model.save('bilinear-shared-vgg16-2step.h5')
import matplotlib.pyplot as plt
plt.plot(history1.history['acc']+history2.history['acc'])
plt.plot(history1.history['val_acc']+history2.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plot.png')


