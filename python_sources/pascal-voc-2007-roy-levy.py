#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Workshop - Assignment 3 - Pascal VOC 2007
# 
# ### *Roy Levy - 313577611*
# ---
# 

# ### In this assignment we will try to build a fully-convolutional neural network in order to adress the technique of Super-Resolution<br>Our main goal is trying to improve details of images with bad resolution, the process in order to achieve this goal will be presented in the following article.

# ### We will use the Pascal VOC 2007 dataset. This dataset is comprise of 5011 with 288x288 images, which we will consider as high resolution.<br> Our goal is to transform these images to low resolution and try to predict with these pictures the high resolution ones.

# In[ ]:


from tensorflow import keras
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from tensorflow.keras.preprocessing.image import load_img
import cv2
import gc
import random
import math
from keras import backend as K

from tensorflow.keras.layers import Conv2D,UpSampling2D,Input,LeakyReLU,Activation, add, Concatenate,Lambda
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16


# # Loading the images
# ---
# ### We will start by loading the images, notice that we will use small amount of images first in order to check if our process works.<br>Once we will get a better understanding of the data we will load all of it and try to use it in our models.

# In[ ]:


def load_imgs(how_many):
    """
        returns images in range of 'how_many'
    """
    path = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    imgs_path = os.listdir(path)
    imgs = []
    
    for i,img in enumerate(imgs_path):
        if(i >= how_many):
            return imgs
        imgs.append(load_img(path+img))
    return imgs        


# In[ ]:


def to_np_array(images):
    new_images = []
    for image in images:
        new_images.append(np.array(image))
    return (np.array(new_images)/255.0)


# In[ ]:


def get_resized_images(images,size):
    """
        returns 3 np.arrays of the resized images
        the sizes are 72x72, 144x144, 288x288
    """
    imgs = []
    
    for img in images:
        img = img.resize((size,size))
        imgs.append(img)
        
    return to_np_array(imgs)


# In[ ]:


# loading only the first 100 pictures
imgs = load_imgs(100)

# getting the images in 72X72, 144X144, 288X288 pixels
imgs_72 = get_resized_images(imgs,72)
imgs_144 = get_resized_images(imgs,144)
imgs_288 = get_resized_images(imgs,288)

# printing the images shape
imgs_72.shape,imgs_144.shape,imgs_288.shape


# In[ ]:


def plot_imgs(rows,cols,imgs,titles):
    """
        plots images side by side
    """
    fix,ax = plt.subplots(rows,cols,figsize=(20,12))
    for i in range(len(imgs)):
        if len(imgs)//cols > 1:
            ax[i//cols,i%cols].set_title(titles[i])
            ax[i//cols,i%cols].imshow(imgs[i])
        else:
            ax[i].set_title(titles[i])
            ax[i].imshow(imgs[i])


# In[ ]:


titles_3 = ["72x72_train","144x144_train","288x288_train"]
curr_img = 5
plot_imgs(1,3,[imgs_72[curr_img],imgs_144[curr_img],imgs_288[curr_img]],titles_3)


# ### In order to save memory we will define the image generators below

# In[ ]:


def get_gen_one(lower,upper,size,epochs=1):
    for e in range(epochs):
        path = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
        imgs_path = os.listdir(path)
    
        for i,img in enumerate(imgs_path):
            if(i in range(lower,upper)):
                curr_img = load_img(path+img)
                curr_img = curr_img.resize((size,size))
                curr_img = np.array(curr_img)/255.0
                yield np.expand_dims(curr_img,axis=0)


# In[ ]:


def get_gen_two(lower,upper,size0,size1,epochs=1):
    for e in range(epochs):
        path = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
        imgs_path = os.listdir(path)
    
        for i,img in enumerate(imgs_path):
            if(i in range(lower,upper)):
                curr_img0 = load_img(path+img)
                curr_img0 = curr_img0.resize((size0,size0))
                curr_img0 = np.array(curr_img0)/255.0
            
                curr_img1 = load_img(path+img)
                curr_img1 = curr_img1.resize((size1,size1))
                curr_img1 = np.array(curr_img1)/255.0
            
                yield np.expand_dims(curr_img0,axis=0), np.expand_dims(curr_img1,axis=0)


# In[ ]:


def get_gen_three(lower,upper,size0,size1,size2,epochs=1):
    for e in range(epochs):
        path = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
        imgs_path = os.listdir(path)
    
        for i,img in enumerate(imgs_path):
            if(i in range(lower,upper)):
                curr_img = load_img(path+img)
                curr_img0 = curr_img.resize((size0,size0))
                curr_img0 = np.array(curr_img0)/255.0
            
                curr_img1 = curr_img.resize((size1,size1))
                curr_img1 = np.array(curr_img1)/255.0
            
                curr_img2 = curr_img.resize((size2,size2))
                curr_img2 = np.array(curr_img2)/255.0
            
                yield np.expand_dims(curr_img0,axis=0), [np.expand_dims(curr_img1,axis=0), np.expand_dims(curr_img2,axis=0)]


# ## Building Our First Model - ConvNN on 72x72 images
# ---
# ### We will start by building a simple convolutional model in the following architechture.<br>We will predict on the 144x144 images.
# 
# ![image.png](attachment:image.png)

# In[ ]:


def get_simple_sr_model():
    inp = Input(shape=(None,None,3))
    x = Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(inp)
    x = Conv2D(64,3,activation=LeakyReLU(0.2),padding='same')(x)    
    x = UpSampling2D()(x)
    x = Conv2D(3,1,activation='sigmoid',padding='same',name="output_144")(x)
    
    return Model(inp,x)


# ### We will use the following callback:<br> on each end of epoch our model will predict on all the images given to the callback.<br>The predictions will be later be save as GIF's.

# In[ ]:


import tensorflow
from PIL import Image

class Gif_Callback(tensorflow.keras.callbacks.Callback):
    
    def __init__(self,imgs,names,folder,two=True):
        super().__init__()
        self.imgs = imgs
        self.preds = []
        for i in range(len(self.imgs)):
            self.preds.append([])
        self.names = names
        self.folder = folder
        os.makedirs('./'+self.folder)
        self.two = two
        
    def on_epoch_end(self, epoch, logs=None):
        for i in range(len(self.imgs)):
            pred = self.model.predict(self.imgs[i])
            self.preds[i].append(pred)
            
    def save_gif(self,p,path,size):
        mult = 30
        imgs = []
        for pi in p:
            pi = np.array(pi)
            pi = np.squeeze(pi)
            img = Image.fromarray(np.uint8(pi*255))
            img = img.resize((size,size))
        
            for i in range(mult):
                imgs.append(img)
                
        imgs[0].save(path,
               save_all=True, append_images=imgs[1:], optimize=False, duration=40, loop=0)
        
        
    def on_train_end(self,logs=None):

        for i in range(len(self.preds)):
            p = self.preds[i]
            path_0 = './'+self.folder+'/'+str(self.names[i])+'_144.gif'
            path_1 = './'+self.folder+'/'+str(self.names[i])+'_288.gif'
            if(self.two):
                self.save_gif(p[0],path_0,144)
                self.save_gif(p[1],path_1,288)
            else:
                self.save_gif(p,path_0,144)


# In[ ]:


names = np.arange(10)
imgs_gif = imgs_72[:10]
imgs_gif = list(map(lambda img : np.expand_dims(img,axis=0),imgs_gif))


# In[ ]:


first_sr_model = get_simple_sr_model()
first_sr_model.compile(loss='mse',optimizer='adam')

imgs_72_144_gen_train = get_gen_two(0,80,72,144,20)
imgs_72_144_gen_val = get_gen_two(80,100,72,144,20)

history_first = first_sr_model.fit_generator(imgs_72_144_gen_train,
                              steps_per_epoch=80,
                              validation_data=imgs_72_144_gen_val,
                              validation_steps = 20,
                              epochs=20,
                              callbacks=[Gif_Callback(imgs_gif,names,"model1",two=False)])


# In[ ]:


def plot_psnr(history,titles):
    plt.figure(figsize=(10,5))
    for title in titles:
        plt.plot(history.history[title])
    
    plt.title('Model loss PSNR')
    plt.ylabel('PSNR Loss')
    plt.xlabel('Epoch')
    plt.legend(titles, loc='lower right')
    plt.show()


# In[ ]:


def plot_losses(history,psnr=True):
    """
        plot losses based on a given model history
    """
    all_titles = [*history.history.keys()]
    psnr_titles = list(filter(lambda t: "PSNR" in t,all_titles))
    titles = list(filter(lambda t: "PSNR" not in t,all_titles))
    
    plt.figure(figsize=(10,5))
    for title in titles:
        plt.plot(history.history[title])
    
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(titles,loc="upper right")
    plt.show()
    if(psnr):
        plot_psnr(history,psnr_titles)


# In[ ]:


plot_losses(history_first,psnr=False)


# ### as we can see the loss based on 'mse' is around 0.01. we will try to improve this.<br>still we need to rememeber that this is just a small chunk of our data.<br><br> Now let's look on our predictions so far...

# In[ ]:


def plot_predict_imgs(rows,cols,preds,stop,offset,titles):
    for i in range(stop):
        preds_lst = [preds[j][i] for j in range(len(preds))]
        imgs_lst = [imgs_72[i+offset],imgs_144[i+offset],imgs_288[i+offset]]
        plot_imgs(rows,cols,imgs_lst+preds_lst,titles)


# In[ ]:


# predicting on the last 20 images - validation_data
titles_4 = ["72x72_train","144x144_train","288x288_train","144x144_predict"]
preds_first_sr = first_sr_model.predict(imgs_72[80:])
plot_predict_imgs(1,4,[preds_first_sr],10,80,titles_4)


# ### We can see that our images looks pixelized. The improvment is noticeable but is minor.

# ## Building Our Second Model
# ---
# ### We will try to get a better understanding where our model is more effecient.<br> In order to do so we will add another output layer of the 288x288 images and see the results.
# 
# ![image.png](attachment:image.png)
# 
# ### This model will be later used when we will train on the whole data.<br> We will continue to build modifications based on this model.

# In[ ]:


def get_second_sr_model():
    inp = Input(shape=(None,None,3))
    x = Conv2D(64,3,activation='relu',padding='same')(inp)
    x = Conv2D(64,3,activation='relu',padding='same')(x)    
    x = UpSampling2D()(x)
    y = UpSampling2D()(x)
    y = Conv2D(3,1,activation='sigmoid',padding='same',name="output_288")(y)   
    x = Conv2D(3,1,activation='sigmoid',padding='same',name="output_144")(x)
    
    return Model(inputs=inp,outputs=[x,y])


# In[ ]:


second_sr_model = get_second_sr_model()
second_sr_model.compile(loss='mse',optimizer='adam',loss_weights=[0.2,0.8])

imgs_72_144_288_gen_train = get_gen_three(0,80,72,144,288,20)
imgs_72_144_288_gen_val = get_gen_three(80,100,72,144,288,20)

history_second = second_sr_model.fit_generator(imgs_72_144_288_gen_train,
                              steps_per_epoch=80,
                              validation_data=imgs_72_144_288_gen_val,
                              validation_steps = 20,
                              epochs=20,
                              callbacks=[Gif_Callback(imgs_gif,names,"model2")])


# In[ ]:


plot_losses(history_second,psnr=False)


# In[ ]:


titles_5 = ["72x72_train","144x144_train","288x288_train","144x144_predict","288x288_predict"]
preds_second_sr = second_sr_model.predict(imgs_72[80:])
plot_predict_imgs(1,5,preds_second_sr,10,80,titles_5)


# In[ ]:


del preds_second_sr
gc.collect()


# ### We can see that the transition between the 72x72 train images to the other images are quite the same, in the case of 288x288 the colors are a bit off.

# ## Building Our Third Model
# ---
# ### After we got a better understanding on how to use our second model it's time to test it on the whole data.<br>First we will load the images and resize them as before, then test it on the second model.

# ### First we will define the following PSNR(peak signal to noise ratio) metric, which is relevant to our super-resolution task.

# In[ ]:


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303


# In[ ]:


imgs_72_144_gen_train = get_gen_three(1000,5012,72,144,288,4)
imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288,4)


# In[ ]:


third_sr_model = get_second_sr_model()
third_sr_model.compile(loss='mse',optimizer='adam',loss_weights=[0.2,0.8],metrics=[PSNR])

# validating on the first 1000 images, training on the rest
history_third = third_sr_model.fit_generator(imgs_72_144_gen_train,
                                             steps_per_epoch=4011,
                                             validation_data=imgs_72_144_288_gen_val,
                                             validation_steps = 1000,
                                             epochs=4,
                                             callbacks=[Gif_Callback(imgs_gif,names,"model3")])


# In[ ]:


plot_losses(history_third)


# In[ ]:


imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288)
preds_third_sr = third_sr_model.predict_generator(imgs_72_144_288_gen_val,1000)
plot_predict_imgs(1,5,preds_third_sr,10,0,titles_5)


# ### We can see a littile improvement from our previous models, the predictions are still quite pixelized.<br><br>Let's try another method

# In[ ]:


del preds_third_sr
gc.collect()


# ## Building Our Fourth Model
# ---
# ### We will try to improve our model by adding a residual block.<br>We will construct our model based on the following architecture.
# ![image.png](attachment:image.png)
# 

# In[ ]:


def residual_block(h,w,num_cnl):
    inp = Input(shape=(h,w,num_cnl))
    x = Conv2D(num_cnl,3,padding='same',activation=LeakyReLU(0.2))(inp)
    x = Conv2D(num_cnl,3,padding='same',activation=LeakyReLU(0.2))(x)
    x = add([inp,x])
    return Model(inp,Activation(LeakyReLU(0.2))(x))


# In[ ]:


def residual_sr_model(h=None,w=None):
    inp = Input(shape=(h,w,3))
    x = Conv2D(32,3,padding='same',activation=LeakyReLU(0.2))(inp)
    x = residual_block(h,w,32)(x)
    x = residual_block(h,w,32)(x)
    x = UpSampling2D(size=(2,2))(x)
    y = residual_block(h,w,32)(x)
    y = UpSampling2D(size=(2,2))(y)
    y = Conv2D(3,1,activation='sigmoid',padding='same',name="output_288")(y)  
    x = Conv2D(3,1,activation='sigmoid',padding='same',name="output_144")(x)
    return Model(inputs=inp,outputs=[x,y])


# In[ ]:


imgs_72_144_gen_train = get_gen_three(1000,5012,72,144,288,4)
imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288,4)


# In[ ]:


fourth_sr_model = residual_sr_model()
fourth_sr_model.compile(loss='mse',optimizer='adam',metrics=[PSNR])

# validating on the first 1000 images, training on the rest
history_fourth = fourth_sr_model.fit_generator(imgs_72_144_gen_train,
                                             steps_per_epoch=4011,
                                             validation_data=imgs_72_144_288_gen_val,
                                             validation_steps = 1000,
                                             epochs=4,
                                             callbacks=[Gif_Callback(imgs_gif,names,"model4")])


# In[ ]:


plot_losses(history_fourth)


# In[ ]:


imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288)
preds_fourth_sr = fourth_sr_model.predict_generator(imgs_72_144_288_gen_val,1000)
plot_predict_imgs(1,5,preds_fourth_sr,10,0,titles_5)


# ### We can see a major improvment from the previous model. Although the predictions are relativly blurry, they are much less pixelized especially in the 288x288 predictions. 

# In[ ]:


del preds_fourth_sr
gc.collect()


# ## Building Our Fifth Model
# ---
# ### We will try another approach. Instead on the residual block we will add a dialated block, this notion is because we wan't learn also big areas of the input.<br>We will construct our model based on the following architecture.
# ![image.png](attachment:image.png)

# In[ ]:


def dilated_block(h,w,num_cnl):
    inp = Input(shape=(h,w,num_cnl), name='input')
    d1 = Conv2D(32, 3, padding='same', dilation_rate=(1,1), activation= LeakyReLU(0.2))(inp)
    d2 = Conv2D(32, 3, padding='same', dilation_rate=(2,2), activation=LeakyReLU(0.2))(inp)
    d3 = Conv2D(32, 3, padding='same', dilation_rate=(4,4), activation=LeakyReLU(0.2))(inp)
    x = Concatenate()([d1,d2,d3])
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)
    return Model(inp,x)


# In[ ]:


def dilated_sr_model(h=None,w=None):
    inp = Input(shape=(h,w,3))
    x = Conv2D(32,3,padding='same',activation=LeakyReLU(0.2))(inp)
    x = dilated_block(h,w,32)(x)
    x = dilated_block(h,w,32)(x)
    x = UpSampling2D(size=(2,2))(x)
    y = dilated_block(h,w,32)(x)
    y = UpSampling2D(size=(2,2))(y)
    y = Conv2D(3,1,activation='sigmoid',padding='same',name="output_288")(y)  
    x = Conv2D(3,1,activation='sigmoid',padding='same',name="output_144")(x)
    return Model(inputs=inp,outputs=[x,y])


# In[ ]:


imgs_72_144_gen_train = get_gen_three(1000,5012,72,144,288,4)
imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288,4)


# In[ ]:


fifth_sr_model = dilated_sr_model()
fifth_sr_model.compile(loss='mse',optimizer='adam',metrics=[PSNR])

# validating on the first 1000 images, training on the rest
history_fifth = fifth_sr_model.fit_generator(imgs_72_144_gen_train,
                                             steps_per_epoch=4011,
                                             validation_data=imgs_72_144_288_gen_val,
                                             validation_steps = 1000,
                                             epochs=4,
                                             callbacks=[Gif_Callback(imgs_gif,names,"model5")])


# In[ ]:


plot_losses(history_fifth)


# In[ ]:


imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288)
preds_fifth_sr = fifth_sr_model.predict_generator(imgs_72_144_288_gen_val,1000)
plot_predict_imgs(1,5,preds_fifth_sr,10,0,titles_5)


# In[ ]:


del preds_fifth_sr
gc.collect()


# ## Building Our Sixth Model
# ---
# ### Another method that we will try is to use feature extractor, for instance VGG16.<br>The goal is to use a pretrained network with our model in order to learn better the transition from bad to better resolution of the images.<br>We will use the following architecture:
# ![image.png](attachment:image.png)

# In[ ]:


def get_vgg_model(h=None,w=None):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(None,None,3))
    vgg_layer = Model(inputs=vgg_model.input,outputs=vgg_model.get_layer("block1_conv2").output)
    
    inp = Input(shape=(h,w,3))
    x = Conv2D(64,3,padding='same',activation=LeakyReLU(0.2))(inp)
    vgg = vgg_layer(inp)
    x = Conv2D(64,3,padding='same',activation=LeakyReLU(0.2))(x)
    x = Concatenate()([x,vgg])
    x = UpSampling2D()(x)
    z = Conv2D(3,1,activation='sigmoid',padding='same',name="output_144")(x)
    y = UpSampling2D()(z)
    y = Conv2D(3,1,activation='sigmoid',padding='same',name="output_288")(y)   
    return Model(inputs=inp,outputs=[z,y])  


# In[ ]:


imgs_72_144_gen_train = get_gen_three(1000,5012,72,144,288,4)
imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288,4)


# In[ ]:


sixth_sr_model = get_vgg_model()
sixth_sr_model.compile(loss='mse',optimizer='adam',metrics=[PSNR])

# validating on the first 1000 images, training on the rest
history_sixth = sixth_sr_model.fit_generator(imgs_72_144_gen_train,
                                             steps_per_epoch=4011,
                                             validation_data=imgs_72_144_288_gen_val,
                                             validation_steps = 1000,
                                             epochs=4,
                                             callbacks=[Gif_Callback(imgs_gif,names,"model6")])


# In[ ]:


plot_losses(history_sixth)


# In[ ]:


imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288)
preds_sixth_sr = sixth_sr_model.predict_generator(imgs_72_144_288_gen_val,1000)
plot_predict_imgs(1,5,preds_sixth_sr,10,0,titles_5)


# In[ ]:


del preds_sixth_sr
gc.collect()


# ## Building Our Seventh Model
# ---
# ### Our final method before will be to replace the upsampling block with depth_to_space blocks. This will allow us to learn pixels that are relativly close to the actual ones, we are performing something like pixel shuffling.
# ![image.png](attachment:image.png)

# ### We will use the following callback in order to present a GIF of this model predictions on each epoch

# In[ ]:


def get_depth_to_space_model(h=None,w=None):
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(None,None,3))
    vgg_layer = Model(inputs=vgg_model.input,outputs=vgg_model.get_layer("block1_conv2").output)
    
    inp = Input(shape=(h,w,3))
    x = Conv2D(64,3,padding='same',activation=LeakyReLU(0.2))(inp)
    vgg = vgg_layer(inp)
    x = Conv2D(64,3,padding='same',activation=LeakyReLU(0.2))(x)
    x = Concatenate(name="cnt")([x,vgg])
    x = Lambda(lambda x:tf.nn.depth_to_space(x,2),name="lamb")(x)
    x = Conv2D(3,1,activation='sigmoid',padding='same',name="output_144")(x)
    z = Conv2D(64,3,activation='sigmoid',padding='same')(x)
    y = Lambda(lambda x:tf.nn.depth_to_space(x,2),name="lamb2")(z)
    y = Conv2D(3,1,activation='sigmoid',padding='same',name="output_288")(y)
    return Model(inp,outputs=[x,y])
    


# In[ ]:


imgs_72_144_gen_train = get_gen_three(1000,5012,72,144,288,4)
imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288,4)


# In[ ]:


seventh_sr_model = get_depth_to_space_model()
seventh_sr_model.compile(loss='mse',optimizer='adam',metrics=[PSNR])

# validating on the first 1000 images, training on the rest
history_seventh = seventh_sr_model.fit_generator(imgs_72_144_gen_train,
                                             steps_per_epoch=4011,
                                             validation_data=imgs_72_144_288_gen_val,
                                             validation_steps = 1000,
                                             epochs=4,
                                             callbacks=[Gif_Callback(imgs_gif,names,"model7")])


# In[ ]:


plot_losses(history_seventh)


# In[ ]:


titles_5 = ["72x72_train","144x144_train","288x288_train","144x144_predict","288x288_predict"]
imgs_72_144_288_gen_val = get_gen_three(0,1000,72,144,288)
preds_seventh_sr = seventh_sr_model.predict_generator(imgs_72_144_288_gen_val,1000)
plot_predict_imgs(1,5,preds_seventh_sr,10,0,titles_5)


# In[ ]:


del preds_seventh_sr
gc.collect()


# ## Conclusions
# ---
# ### We saw that different types of architecture yields different results in our quest to perform the task of Super-Resolutions.<br> We defined the metric of PSNR to test how much we lose in the 'compression' process, i.e. the task.<br>We saw that the although the PSNR stays very much the same, there are still significant results between model.<br> So far our seventh model was the best, we can take the task even further by trying to combine our different models in order to yield better results.
