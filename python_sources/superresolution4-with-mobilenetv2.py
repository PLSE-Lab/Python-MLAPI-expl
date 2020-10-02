#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import os
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
os.listdir("../input")


# **The image loading part**

# In[ ]:


def toArray(k):
    return np.array(list(k.getdata())).reshape(k.size[1], k.size[0], 3)


# In[ ]:


from sklearn.metrics import mean_absolute_error as mae 
from skimage.measure import compare_psnr as psnr

train_data = []
for img_path in os.listdir("../input/images"):
    train_data += [Image.open('../input/images/'+img_path)]
for img_path in os.listdir("../input/general100"):
    train_data += [Image.open('../input/general100/'+img_path)]


# In[ ]:


for img_path in os.listdir("../input/image-classification/images/images/art and culture")[:110]:
    train_data += [Image.open('../input/image-classification/images/images/art and culture/'+img_path)]


# I'll generate some training samples:

# In[ ]:


def imageListToNiceSamples(images, downscale_factor = 2, img_size = 40, n_convolutions = 4): 
    X = []
    Y = []
    for image in tqdm(images):
        cutoff = n_convolutions
        size = np.array(image.size)
        samples_from_image = size//img_size
        newimage = image.resize(size//downscale_factor, resample = Image.BICUBIC).resize(size, resample = Image.BICUBIC)
        try:
            image_array = toArray(image)
            newimage_array = toArray(newimage)
            if(image_array.shape[2]==1):
                continue
            X_temp = []
            Y_temp = []
          #  print(size, image.size, samples_from_image)
            for j in range(samples_from_image[0]):
                for i in range(samples_from_image[1]):
                    x = newimage_array[i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size,:]/130-0.99
                    y = image_array[i*img_size:(i+1)*img_size,j*img_size:(j+1)*img_size,:]/130-0.99 # this for preserving image size (as now images are large padding boundaries won't change much)

                    X_temp+=[x.reshape(1,img_size,img_size,3)]
                    Y_temp+=[y.reshape(1,img_size,img_size,3)]
                    
            #X_temp_2 = np.concatenate([np.array(X_temp)[:,:,:,:,0],np.array(X_temp)[:,:,:,:,1],np.array(X_temp)[:,:,:,:,2]], axis=0) # Channel separation doesn't work just for mobilenet because it uses depthise separable convolutions
            #Y_temp_2 = np.concatenate([np.array(Y_temp)[:,:,:,:,0],np.array(Y_temp)[:,:,:,:,1],np.array(Y_temp)[:,:,:,:,2]], axis=0)
            X+=[np.concatenate(X_temp, axis=0)] 
            del X_temp
            
            Y+=[np.concatenate(Y_temp, axis=0)]
            del Y_temp
        except:
            continue # There may be black and white images in the data or else, and I don't need to care for them.
            
    return(np.concatenate(X, axis=0), np.concatenate(Y, axis=0))


# In[ ]:


image_size = 224
n_convolutions = 3
downscale_factor = 2
X, y = imageListToNiceSamples(train_data, img_size = image_size, downscale_factor = downscale_factor, n_convolutions=n_convolutions)


# In[ ]:


val_data = []
for img_path in os.listdir("../input/image-classification/images/images/art and culture")[200:230]:
    val_data += [Image.open('../input/image-classification/images/images/art and culture/'+img_path)]
X_val, y_val = imageListToNiceSamples(val_data, img_size = image_size, downscale_factor = downscale_factor, n_convolutions=n_convolutions)
# Here I gave up as using ShuffleSplit made memory blow out.


# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Activation, Dropout, Lambda, MaxPooling2D, BatchNormalization, Reshape, Flatten, Input, Concatenate, Add, Conv2DTranspose
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.applications.mobilenet_v2 import MobileNetV2


# In[ ]:


def getModel(lr = 0.002, dropout_rate = .2, input_dropout = .2, conv_layer_size = 64, image_size =224, downscale_factor =4, n_conv = 2): # encapsulation to facilitate skopt usage
    opt = Nadam(lr)

    inpt = Input((image_size,image_size, 3))
    
    mn2 = MobileNetV2(include_top=False, input_tensor = inpt)
    #mn2.summary()
    x = mn2.get_layer("Conv1_pad")(inpt)
    x = mn2.get_layer("Conv1")(x)
    x = mn2.get_layer("bn_Conv1")(x)
    x = mn2.get_layer("Conv1_relu")(x)
    x = mn2.get_layer("expanded_conv_depthwise")(x)
    x = mn2.get_layer("expanded_conv_depthwise_BN")(x)
    x = mn2.get_layer("expanded_conv_depthwise_relu")(x)
    x = mn2.get_layer("expanded_conv_project")(x)
    x = mn2.get_layer("expanded_conv_project_BN")(x)
    x = mn2.get_layer("block_1_expand")(x)
    x = mn2.get_layer("block_1_expand_BN")(x)
    x = mn2.get_layer("block_1_expand_relu")(x)
    x = mn2.get_layer("block_1_pad")(x)
    x = mn2.get_layer("block_1_depthwise")(x)
    x = mn2.get_layer("block_1_depthwise_BN")(x)
    x = mn2.get_layer("block_1_depthwise_relu")(x)
    x = mn2.get_layer("block_1_project")(x)
    y = mn2.get_layer("block_1_project_BN")(x)
    y = mn2.get_layer("block_2_expand")(y)
    y = mn2.get_layer("block_2_expand_BN")(y)
    y = mn2.get_layer("block_2_expand_relu")(y)
    y = mn2.get_layer("block_2_depthwise")(y)
    y = mn2.get_layer("block_2_depthwise_BN")(y)
    y = mn2.get_layer("block_2_depthwise_relu")(y)
    y = mn2.get_layer("block_2_project")(y)
    y = mn2.get_layer("block_2_project_BN")(y)
    z = mn2.get_layer("block_2_add")([y,x])
    
    aux_model = Model(inpt,z)
    for layer in aux_model.layers: #-6 overfit
        layer.trainable = False

    conv2dT = Conv2DTranspose(conv_layer_size,4, strides = 4)(z)
    conv2dT = BatchNormalization()(conv2dT)
    conv2dT = Activation('relu')(conv2dT)
    concat = Concatenate(axis = 3)([inpt, conv2dT])
    mid = concat
    for i in range(n_conv):
        mid = Conv2D(conv_layer_size,3, padding='same')(mid)
        mid = BatchNormalization()(mid)
        mid = Activation('tanh')(mid)
    output = Activation('tanh')(Conv2D(3,3, padding='same')(mid))
    
    model = Model(inpt,output)
    model.compile(loss = 'mae', optimizer = opt) 
    return(model)
    


# In[ ]:


benchmark = psnr(y_val, X_val)
print(benchmark)
ss = StandardScaler()
target = ss.fit_transform(y.reshape(-1,150528)-X.reshape(-1,150528)).reshape(-1,224,224,3)/2
target_val = ss.transform(y_val.reshape(-1,150528)-X_val.reshape(-1,150528)).reshape(-1,224,224,3)/2


# In[ ]:


psnrs=[]
for i in range(10):
    model = getModel(dropout_rate = .35, input_dropout = 0.0, image_size = image_size, conv_layer_size = 16, n_conv=3, downscale_factor = downscale_factor)
    #model.summary() #to give overview of number of params
    stop = EarlyStopping(patience=3, restore_best_weights = True) #patience derived empirically
    model.fit(X[:125*(i+1)], y[:125*(i+1)], validation_data = [X_val,y_val], batch_size = 8, epochs = 20, callbacks = [stop], verbose = False)
    target_pred = model.predict(X_val)
    y_pred = X_val + ss.inverse_transform(target_pred.reshape(-1,150528)*2).reshape(-1,224,224,3)*2
    psnrs+=[psnr(y_val, y_pred)]


# In[ ]:


plt.plot(psnrs)
plt.title('model psnr')
plt.ylabel('psnr')
plt.xlabel('samples used/50')
plt.show()

