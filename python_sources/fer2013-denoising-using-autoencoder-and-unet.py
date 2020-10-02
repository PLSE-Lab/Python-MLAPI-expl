#!/usr/bin/env python
# coding: utf-8

# * Image denoising is to remove noise from a noisy image, so as to restore the true image
# * In this notebook FER2013 dataset is used which contains approx 35 thousand images of 7 different emotions
# * Image is grayscale of size 48*48

# # Importing libraries

# In[ ]:


from keras.datasets import fashion_mnist, mnist
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.models import Model

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd# Any results you write to the current directory are saved as output.
from IPython.display import display, Image

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

# Any results you write to the current directory are saved as output.
from IPython.display import display, Image


# ## Extract data from CSV

# In[ ]:


# get the data
filname = '../input/facial-expression/fer2013/fer2013.csv'

#different labels of images(not useful known about for current problem)
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

#different features names
names=['emotion','pixels','usage']

#Reading data in dataframe
df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)
im=df['pixels']
df.head(10)


# ## Adding labels and images(pixel values) in respective array

# In[ ]:


#reading data and labels from dataset and appending in list

def getData(filname):
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X), np.array(Y)
    return X, Y


# In[ ]:


#extracting data from dataset
X, Y = getData(filname)
num_class = len(set(Y))
#print(num_class)


# ## Reshaping images

# In[ ]:


# keras with tensorflow backend
N, D = X.shape

#reshaping the dataset
X = X.reshape(N, 48, 48, 1)


# # Extracting Data and splitting train and test 

# In[ ]:


#splitting data in train, test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)


# In[ ]:


#Taking 5000 images 

x_train = x_train[:5000]
x_test = x_test[:5000]


# In[ ]:


x_train.shape


# # Data Preprocessing

# In[ ]:


#NOrmalizing the images
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

#reshaping the images
x_train = np.reshape(x_train, (len(x_train), 48, 48, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 48, 48, 1))  # adapt this if using `channels_first` image data format


#adding noise in data
noise_factor = 0.1
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

#clipping put data near to 0--->0 aand data near to 1-->1(eg=0.3-->0 or 0.7-->1)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# # Visualization of 10 Data

# In[ ]:


n = 10


# In[ ]:


plt.figure(figsize=(48, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_train_noisy[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # One of the way we can achieve our goal of removing noise is AutoEncoder
# 
# **Copied from Keras Blog(https://blog.keras.io/building-autoencoders-in-keras.html):
# **
# * What are autoencoders good for?
# * Today two interesting practical applications of autoencoders are data denoising (which we feature later in this post), and dimensionality reduction for data visualization. With appropriate dimensionality and sparsity constraints, autoencoders can learn data projections that are more interesting than PCA or other basic techniques.

# **Refer to Keras Blog for better idea : https://blog.keras.io/building-autoencoders-in-keras.html**

# ## AutoEncoder Architecture

# In[ ]:


display(Image(filename="/kaggle/input/images-architecture/images_architecture/autoencoder.png"))


# # Construction of Model

# In[ ]:


input_img = Input(shape=(48, 48, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)


# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='MSE')


# # AutoEncoder Summary

# In[ ]:


autoencoder.summary()


# # Training Model

# In[ ]:


autoencoder.fit(x_train_noisy, x_train,
                epochs=35,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))


# ## Making Prediction

# In[ ]:


predict = autoencoder.predict(x_test_noisy)


# # Visualizing the prediction

# ## Original Test images

# In[ ]:


n=10


# In[ ]:


plt.figure(figsize=(40, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# ## Noised Test images

# In[ ]:


plt.figure(figsize=(40, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# ## Generated Test images

# In[ ]:


plt.figure(figsize=(40, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(predict[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # AutoEncoder: Train Loss VS validation loss

# In[ ]:


epochs = range(len(autoencoder.history.history['loss']))

plt.plot(epochs,autoencoder.history.history['loss'],'r', label='train_loss')
plt.plot(epochs,autoencoder.history.history['val_loss'],'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()


# # Structural Similarity Index
# 
# * When comparing images, the mean squared error (MSE)--while simple to implement--is not highly indicative of perceived similarity. Structural similarity aims to address this shortcoming by taking texture into account
# 
# 

# In[ ]:


from skimage.measure import compare_ssim, compare_psnr
from skimage import data, img_as_float


# In[ ]:


compare_ssim(x_test, predict, multichannel=True)


# # PSNR
# 
# Source: https://www.mathworks.com/help/vision/ref/psnr.html
# 
# * The PSNR block computes the peak signal-to-noise ratio, in decibels, between two images. This ratio is used as a quality measurement between the original and a compressed image. The higher the PSNR, the better the quality of the compressed, or reconstructed image.
# 
# * The mean-square error (MSE) and the peak signal-to-noise ratio (PSNR) are used to compare image compression quality. The MSE represents the cumulative squared error between the compressed and the original image, whereas PSNR represents a measure of the peak error. The lower the value of MSE, the lower the error.

# In[ ]:


compare_psnr(x_test, predict)


# # Not quite good. Lets try UNET

# **Anwser taken from quora: https://www.quora.com/Why-is-U-Net-considered-as-an-autoencoder**
# 
# The classical auto-encoder architecture has the following property:
# - First, it takes an input and reduces the receptive field of the input as it goes through the layers of its encoder units. Finally at the end of the encoder part of the architecture, the input is reduced to a linear feature representation.
# - Next, the linear feature representation is upsampled (or its receptive field increased) by the decoder portion of the auto-encoder. So that at the other end of the autoencoder the result is of the same dimension as the input it received.
# Such an architecture is ideal for preserving the dimensionality of input->output. But, the linear compression of the input leads to a bottleneck that does not transmit all features.
# 
# The U-Net has both the properties listed above, but it uses deconv units and overcomes the bottleneck limitation by adding skip connections that allow feature representations to pass through the bottleneck.

# * Refer to original paper for better idea: https://arxiv.org/abs/1505.04597
# * Implementation : https://towardsdatascience.com/u-net-b229b32b4a71

# # UNET Structure

# In[ ]:


display(Image(filename="/kaggle/input/images-architecture/images_architecture/unet.png"))


# # Construction of Model

# In[ ]:


def build_model(input_layer, start_neurons):
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(start_neurons*1,(3,3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(start_neurons*2,(3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(start_neurons*4,(3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
    
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(start_neurons*8,(3,3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    #Middle
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same')(pool4)
    convm = Conv2D(start_neurons * 16, (3,3), activation='relu', padding='same')(convm)
    
    #upconv part
    deconv4 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    uconv4 = Conv2D(start_neurons*8, (3,3), activation='relu', padding='same')(uconv4)
    
    deconv3 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    uconv3 = Conv2D(start_neurons*4, (3,3), activation='relu', padding='same')(uconv3)
    
    deconv2 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    uconv2 = Conv2D(start_neurons*2, (3,3), activation='relu', padding='same')(uconv2)
    
    deconv1 = Conv2DTranspose(start_neurons*8,(3,3), strides=(2,2), padding='same')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(start_neurons*1, (3,3), activation='relu', padding='same')(uconv1)
    
    output_layer = Conv2D(1, (1,1), padding='same', activation='sigmoid')(uconv1)
    return output_layer


# In[ ]:


#Defining target image size and Input size and building model
img_size_target = 48
input_layer = Input((img_size_target, img_size_target,1))
output_layer = build_model(input_layer,64)


# In[ ]:


#Initializing and compiling model
model_unet = Model(input_layer, output_layer)
model_unet.compile(optimizer='adam', loss='MSE')


# # UNET Summary

# In[ ]:


model_unet.summary()


# # Training Model

# In[ ]:


model_unet.fit(x_train_noisy, x_train,
                epochs=15,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))


# # Making Prediction

# In[ ]:


predict = model_unet.predict(x_test_noisy)


# # Visualizing the prediction

# In[ ]:


n=10


# In[ ]:


plt.figure(figsize=(40, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # Noised Test images

# In[ ]:


plt.figure(figsize=(48, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # Generated Test images

# In[ ]:


plt.figure(figsize=(40, 48))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(predict[i].reshape(48, 48))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# # UNET: Train Loss VS validation loss

# In[ ]:


epochs = range(len(model_unet.history.history['loss']))

plt.plot(epochs,model_unet.history.history['loss'],'r', label='train_loss')
plt.plot(epochs,model_unet.history.history['val_loss'],'b', label='val_loss')
plt.title('train_loss vs val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()


# # Structural Similarity Index
# 
# * When comparing images, the mean squared error (MSE)--while simple to implement--is not highly indicative of perceived similarity. Structural similarity aims to address this shortcoming by taking texture into account

# In[ ]:


compare_ssim(x_test, predict, multichannel=True)


# # PSNR

# In[ ]:


compare_psnr(x_test, predict)

