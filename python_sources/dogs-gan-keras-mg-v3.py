#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xml.etree.ElementTree as ET 
import matplotlib.pyplot as plt, zipfile 
from PIL import Image 

from tqdm import tqdm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.initializers import RandomNormal, Constant
from keras import initializers
from keras.layers import Input, Dense, Dropout, Flatten, Reshape , LeakyReLU , Lambda, PReLU, Concatenate
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, UpSampling2D, Conv2DTranspose,BatchNormalization
from keras import regularizers
from keras.optimizers import SGD, Adam

from keras import backend as K
from keras.engine import *
from keras.utils import conv_utils

from IPython.display import FileLink, FileLinks

import time
start_time = time.time()

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


PATH = '../input/all-dogs/all-dogs/'
imageNames = os.listdir(PATH)
#print(images)
img = plt.imread(PATH + imageNames[np.random.randint(0,len(imageNames))])
plt.imshow(img)
print(img.size)


# In[ ]:


PATH_ANNO = '../input/annotation/Annotation/'
breeds = os.listdir(PATH_ANNO)
imagesInput = np.zeros((len(imageNames)*2,64,64,3))
images_breed = []
i = 0
print(imagesInput.shape)
for breed in breeds:
    for dog in os.listdir(PATH_ANNO+breed):
        tree = ET.parse(PATH_ANNO + breed + '/' + dog)
        root = tree.getroot()
        try: img = Image.open(PATH + root.find('filename').text +'.jpg')
        except: continue
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            img_crop = img.crop((xmin, ymin, xmax, ymax))
            w = img_crop.size[0]; h = img_crop.size[1];
            a=0; b=0
            if w<h:
                w2 = 64; h2 = int((64/w)*h)
                #b = np.random.randint(0,(h2-64)) if (h2-64 > 0) else 0
            else:
                h2 = 64; w2 = int((64/h)*w)
                #a = np.random.randint(0,(w2-64)) if (w2-64 > 0) else 0
            img_crop = img_crop.resize((w2,h2), Image.ANTIALIAS)
            img_crop = img_crop.crop((0+a, 0+b, 64+a, 64+b))
            imagesInput[i,:,:,:] = np.asarray(img_crop)
            images_breed.append(obj.find('name').text)
            i += 1
imagesInput = imagesInput[:i,:,:,:]        
flip_imagesInput = np.flip(imagesInput,2)
imagesInput = np.vstack((imagesInput,flip_imagesInput))
flip_imagesInput = None
images_breed = images_breed + images_breed
imagesInput = imagesInput / (255 / 2) - 1
print(imagesInput.shape,len(images_breed))


# In[ ]:


rnd = np.random.randint(0,imagesInput.shape[0])
plt.imshow(imagesInput[rnd]/2 + .5)
print(imagesInput[0].shape)


# In[ ]:


def load_img_full():    
    images = np.zeros((len(imageNames),64,64,3))
    i = 0
    for i, dog in enumerate(imageNames):
        img = Image.open(PATH + dog)
        w = img.size[0]; h = img.size[1];
        a=0; b=0
        if w<h:
            w2 = 64; h2 = int((64/w)*h)
            b = np.random.randint(0,(h2-64)) if (h2-64 > 0) else 0
        else:
            h2 = 64; w2 = int((64/h)*w)
            a = np.random.randint(0,(w2-64)) if (w2-64 > 0) else 0
        img = img.resize((w2,h2), Image.ANTIALIAS)
        img = img.crop((0+a, 0+b, 64+a, 64+b))
        images[i,:,:,:] = np.asarray(img)
    images = images / (255 / 2) - 1
    print(images.shape)
    return images
imagesInput = np.vstack((imagesInput,load_img_full()))
print(imagesInput.shape)


# **[Spectral Normalization](https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py)**

# In[ ]:


class ConvSN2D(Conv2D):

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        #Spectral Normalization
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
                
        outputs = K.conv2d(
                inputs,
                W_bar,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ConvSN2DTranspose(Conv2DTranspose):

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                         initializer=initializers.RandomNormal(0, 1),
                         name='sn',
                         trainable=False)
        
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True  
    
    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = input_shape[h_axis], input_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_length(height,
                                              stride_h, kernel_h,
                                              self.padding,
                                              out_pad_h)
        out_width = conv_utils.deconv_length(width,
                                             stride_w, kernel_w,
                                             self.padding,
                                             out_pad_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)
            
        #Spectral Normalization    
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            #Accroding the paper, we only need to do power iteration one time.
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)
        self.kernel = W_bar
        
        outputs = K.conv2d_transpose(
            inputs,
            self.kernel,
            output_shape,
            self.strides,
            padding=self.padding,
            data_format=self.data_format)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


# [**Self Attention**](https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras)   
# [More](https://sthalles.github.io/advanced_gans/)

# In[ ]:


class Attention(Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True


    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[-1]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]  // Why matmul vs K.batch_dot

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


# In[ ]:


def PixelwiseNorm(x):
    alpha=1e-8
    y = x ** 2
    y = K.mean(y, axis=-1, keepdims=True)
    y = y + alpha
    y = K.sqrt(y)
    y = x / y  # normalize the input x volume
    return y


# # **Generator**

# In[ ]:


drop = 0.1
init = RandomNormal(mean=0.0, stddev=0.02) #'glorot_uniform'#
gen_input = Input(shape=(100,))

x = Dense(1024, activation='relu',)(gen_input)
x = BatchNormalization(momentum=0.8)(x)
x = Reshape((4,4,64))(x)

x3 = ConvSN2DTranspose(256, (3, 3),strides=(1, 1),padding='same',kernel_initializer=init)(x)
x5 = ConvSN2DTranspose(128, (5, 5),strides=(1, 1),padding='same',kernel_initializer=init)(x)
x = Concatenate(axis=-1)([x3,x5])
x = Lambda(PixelwiseNorm)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x, training=True)


x3 = ConvSN2DTranspose(128, (3, 3),strides=(2, 2),padding='same',kernel_initializer=init)(x)
x5 = ConvSN2DTranspose(64, (5, 5),strides=(2, 2),padding='same',kernel_initializer=init)(x)
x = Concatenate(axis=-1)([x3,x5])
x = Lambda(PixelwiseNorm)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x, training=True)

x3 = ConvSN2DTranspose(64, (3, 3),strides=(2, 2),padding='same',kernel_initializer=init)(x)
x5 = ConvSN2DTranspose(32, (5, 5),strides=(2, 2),padding='same',kernel_initializer=init)(x)
x = Concatenate(axis=-1)([x3,x5])
x = Lambda(PixelwiseNorm)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x, training=True)

x3 = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same',kernel_initializer=init)(x)
x5 = ConvSN2DTranspose(16, (5, 5),strides=(2, 2),padding='same',kernel_initializer=init)(x)
x = Concatenate(axis=-1)([x3,x5])
x = Lambda(PixelwiseNorm)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x, training=True)

x = Attention(32+16)(x)

x3 = ConvSN2DTranspose(32, (3, 3), strides=(2, 2), padding='same',kernel_initializer=init)(x)
x5 = ConvSN2DTranspose(16, (5, 5),strides=(2, 2),padding='same',kernel_initializer=init)(x)
x = Concatenate(axis=-1)([x3,x5])
x = Lambda(PixelwiseNorm)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x, training=True)


#x = AveragePooling2D(pool_size=(2, 2),strides=(1, 1),padding='same')(x)

x = ConvSN2D(3, (5, 5), activation='tanh', padding='same',kernel_initializer=init)(x)

generator = Model(gen_input, x)
generator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')
generator.summary()


# In[ ]:


inpt = np.random.random(100*2).reshape(2,100)
img = generator.predict(inpt)
print(img.shape)
plt.imshow(img[0]/2+.5)
plt.imshow(img[1]/2+.5)
#print(img[1]/2+.5)


# # **Discriminator**

# In[ ]:


init = RandomNormal(mean=0.0, stddev=0.02)
drop = 0.0
dis_input = Input(shape=(64,64,3,))
x = ConvSN2D(32, (5, 5), padding='same',kernel_initializer=init)(dis_input)
x = BatchNormalization(momentum=0.8)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x)

#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
x = ConvSN2D(64, (5, 5), padding='same',strides=(2, 2),kernel_initializer=init)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = BatchNormalization(momentum=0.8)(x)
x = Dropout(drop)(x)

#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
x = ConvSN2D(64, (3, 3), padding='same',strides=(2, 2),kernel_initializer=init)(x)
x = BatchNormalization(momentum=0.8)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x)

#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
x = ConvSN2D(128, (3, 3), padding='same',strides=(2, 2),kernel_initializer=init)(x)
x = BatchNormalization(momentum=0.8)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x)

#x = MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
x = ConvSN2D(256, (3, 3), padding='same',strides=(2, 2),kernel_initializer=init)(x)
x = BatchNormalization(momentum=0.8)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x)

#x = Conv2D(64, (3, 3), padding='valid',strides=(1, 1),kernel_initializer=init)(x)
#x = LeakyReLU()(x)
#x = BatchNormalization()(x)
#x = Dropout(drop)(x)

x = ConvSN2D(512, (3, 3), padding='same',strides=(1, 1),kernel_initializer=init)(x)
x = BatchNormalization(momentum=0.8)(x)
x = PReLU(alpha_initializer=Constant(value=0.3))(x)
x = Dropout(drop)(x)

#x = AveragePooling2D(pool_size=(2, 2),strides=(1, 1),padding='same')(x)

x = Flatten()(x)
x = Dense(1 , activation = "sigmoid")(x)
discriminator = Model(dis_input, x)
discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')
discriminator.summary()


# In[ ]:


gan_input = Input(shape=(100,))
discriminator.trainable=False
x = generator(gan_input)
x = discriminator(x)
gan = Model(gan_input, x)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
gan.summary()


# In[ ]:


def gen_noise(batch_size=128):
    noise = np.random.randn(batch_size,100)
    #noise = np.random.normal(size = 100 * batch_size).reshape(batch_size,100)
    # noise =  np.random.random(100 * batch_size).reshape(batch_size,100)
    return noise 


# In[ ]:


print("--- %s seconds ---" % (time.time() - start_time))


# # **Training**

# In[ ]:


def training(epochs=3, batch_size=128):
    imagesProgress = np.zeros((epochs,64,64,3))
    progress_noise = gen_noise(1)
        
    min_learning_rate = 0.00005
    max_learning_rate = K.eval(gan.optimizer.lr)
    cycle_length = batch_size * 1200
    mult_factor=2
    batch_since_restart = 0
    global learning_rate_log
    learning_rate_log = []
    
    for e in range(1,epochs+1):
        #print(imagesInput.shape[0]//batch_size)
        #print('Epoch:', e)
        np.random.shuffle(imagesInput)
        for b in tqdm(range(imagesInput.shape[0]//batch_size)):
            noise = gen_noise(batch_size)
            gen_imgs = generator.predict(noise)
            real_imgs = imagesInput[b*batch_size:(b+1)*batch_size]
            #X = np.concatenate([real_imgs, gen_imgs])
            #y_dis = np.zeros(2*batch_size) 
            #y_dis[:batch_size] = .8 +  np.random.normal(loc=0, scale=.050)
            
            y_dis = np.ones(batch_size) - np.abs(np.random.normal(loc=.2, scale=.1))
            #y_dis = 1 - y_dis if np.random.random() <= .05 else y_dis
            discriminator.trainable=True
            discriminator.train_on_batch(real_imgs,y_dis)
            
            y_dis.fill(0.0)
            y_dis += np.abs(np.random.normal(loc=.2, scale=.1))
            #y_dis = 1 - y_dis if np.random.random() <= .05 else y_dis
            discriminator.trainable=True
            discriminator.train_on_batch(gen_imgs,y_dis)
                        
            noise = gen_noise(batch_size//2)
            #noise[:,np.random.randint(100)] += 1
            y_gen = np.ones(batch_size//2) - .1
            
            discriminator.trainable=False
            gan.train_on_batch(noise, y_gen)
            #inpt = np.random.random(100).reshape(1,100)
                        
            learning_rate = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + np.cos(np.pi * batch_since_restart/cycle_length))
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            if batch_since_restart >= cycle_length:
                batch_since_restart = 0
                learning_rate = max_learning_rate
                cycle_length = cycle_length * mult_factor
            batch_since_restart += 1
            K.set_value(gan.optimizer.lr, learning_rate)
            learning_rate_log.append(learning_rate)
          
        inpt = gen_noise(1)
        i = np.random.randint(0,real_imgs.shape[0])
        print('Epoch: ', e,
        ' || Gen: ' , discriminator.predict(generator.predict(inpt))[0,0],
        ' || Dog: ' , discriminator.predict(real_imgs[i:i+1])[0,0])
        #print(np.average(y_dis))
        imagesProgress[e-1,:,:,:] = generator.predict(progress_noise)[0]
        os.system(f'echo \"{e}\"')
        if (time.time() - start_time) > 31800:
            imagesProgress = imagesProgress[:e-1,:,:,:]
            break
    return imagesProgress
try: imagesProgress
except NameError: imagesProgress = np.zeros((0,64,64,3))
imagesProgress = np.vstack((imagesProgress,training(epochs=600,batch_size=128)))


# In[ ]:


plt.plot(learning_rate_log)


# In[ ]:


print('Epochs: ' , imagesProgress.shape[0])
columns = 6 ; rows = min(6,(imagesProgress.shape[0] // columns) + 1);
fig=plt.figure(figsize=(32, 5 * rows))
j=0
for i in range(0 , min(36,imagesProgress.shape[0])):
    fig.add_subplot(rows,columns,i+1)
    plt.imshow(imagesProgress[int(j)]/2+.5)
    j += max(1,imagesProgress.shape[0] / 36)
plt.show()


# In[ ]:


columns = 6 ; rows = 4;
inpt = gen_noise(columns * rows)
#inpt[:,np.random.randint(100)] += 1
img = generator.predict(inpt)
#print(discriminator.predict(imagesInput[:10]))
#print(discriminator.predict(img))
#print(img[0,0,:10])
fig=plt.figure(figsize=(32, 5 * rows))
for i in range(0 , columns * rows):
    fig.add_subplot(rows,columns,i+1)
    plt.imshow(img[i]/2+.5)
plt.show()


# In[ ]:


z = zipfile.PyZipFile('images.zip', mode='w')
inpt = gen_noise(10000)
imgs = generator.predict(inpt) 
imgs = imgs + 1
imgs = imgs / 2
print(imgs.shape)
for k in range(10000):
    f = str(k)+'.png'
    img = imgs[k,:,:,:]#.numpy()
    tf.keras.preprocessing.image.save_img(
        f,
        img,
        scale=True
    )
    z.write(f); os.remove(f)
z.close()
FileLinks('.')
#!ls


# In[ ]:


#inpt = gen_noise(10000)
#img = generator.predict(inpt) 
#img = img + 1
#img = img * (255 / 2)
#np.array(img).min(), np.array(img).max()
#if not os.path.exists('../tmp'):
#    os.mkdir('../tmp')
#for i in range(0,img.shape[0]):
#    plt.imsave('../tmp/dog_'+ str (i)+'.png' , img[i].astype(np.uint8))
#import shutil
#shutil.make_archive('images', 'zip', '../tmp')


# In[ ]:


#print(img[9].astype(np.uint8))
#im = Image.fromarray(img[0].astype(np.uint8))
#plt.imshow(im)


# In[ ]:


print("--- %s seconds ---" % (time.time() - start_time))

