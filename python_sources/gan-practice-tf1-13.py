# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
images = os.listdir("../input/img_align_celeba/img_align_celeba")
trainData = []
# print(type(images))
for i in range(300):
    trainData.append(cv2.resize(cv2.imread("../input/img_align_celeba/img_align_celeba/"+images[i]),(64,64)))
#(training examples. height, width, channels)
trainData = np.float32((np.array(trainData) - 127.5)/127.5)
#(218,178,3)
# print(trainData.shape)
# trainDataset = tf.data.Dataset.from_tensor_slices(trainData).shuffle(256).batch(10)

def generatorModel():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Reshape((4,4,1024)))
    
    model.add(layers.Conv2DTranspose(512, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    
    model.add(layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    
    return model

# noise = tf.random.normal([1,100])
# generated = gen(noise, training=False)
# print(generated)
# tensor = generated[0,:,:,0]
# print(tf.keras.backend.get_value(tensor))
# tensorInArr = tf.keras.backend.get_value(tensor)
# plt.imshow(tensorInArr, cmap='gray')

def discriminatorModel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128,(4,4), strides=(2,2), padding='same', input_shape=[64,64,3]))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256,(4,4), strides=(2,2), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(512,(4,4), strides=(2,2), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(1024,(4,4), strides=(2,2), padding = 'same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(1,(4,4), strides=(1,1), padding = 'same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminatorLoss (real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)
    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    total = real_loss + fake_loss
    return total

def generatorLoss (output):
    return cross_entropy(tf.ones_like(output), output)

generator = generatorModel()
discriminator = discriminatorModel()
genOpt = tf.train.AdamOptimizer()
disOpt = tf.train.AdamOptimizer()
# genOpt = tf.keras.optimizers.Adam(0.0002)
# disOpt = tf.keras.optimizers.Adam(0.0002)

print(type(genOpt))
# @tf.function
def trainStep(images):
    noise = tf.random.normal([10,100])
    with tf.GradientTape() as genTape, tf.GradientTape() as disTape:
        gen_img = generator(noise, training = True)
        fakeOut = discriminator(gen_img, training = True)
        realOut = discriminator(images, training=True)
        genLoss = generatorLoss(fakeOut)
        disLoss = discriminatorLoss(realOut, fakeOut)
    genGrads = genTape.gradient(genLoss, generator.trainable_variables)
    disGrads = disTape.gradient(disLoss, discriminator.trainable_variables)
    
    genOpt.apply_gradients(zip(genGrads, generator.trainable_variables))
    disOpt.apply_gradients(zip(disGrads, discriminator.trainable_variables))
    # for i in xrange(100)

# print(trainData[0:30].shape)
cp = tf.train.Checkpoint(genOpt=genOpt, disOpt=disOpt, generator=generator, discriminator=discriminator)
def train(dataset):
    for epoch in range(20):
        for i in range(30):
            trainStep(dataset[i:i+10])
        print("epoch {}".format(epoch))
    noise = tf.random.normal([10,100])
    generator(noise, training=False)
    generated = generator(noise, training=False)
    tensor = generated[0,:,:,0]
    tensorInArr = tf.keras.backend.get_value(tensor)
    plt.imshow(tensorInArr, cmap='gray')
train(trainData)


# Any results you write to the current directory are saved as output.