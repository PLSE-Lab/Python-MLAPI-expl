#!/usr/bin/env python
# coding: utf-8

# Written by: Yaniv Rotaru <br>
# Deep Learning workshop - Class Assignment (assignment 3 - Image super-resolution)
# 
# 

# In[ ]:


# import necessary libraries
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from tensorflow.keras import layers, models
from keras.layers import UpSampling2D,Input,LeakyReLU,Lambda,add,Activation,Concatenate
from keras.utils import plot_model
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import tensorflow as tf
from keras.applications import VGG16
import math
from keras import backend as K


# # Define evaluation metric PSNR

# In[ ]:


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1))))


# # Using Generators to save RAM and load all the data
# first 1000 images for validation <br>
# The rest for training (4011)

# In[ ]:


#using generator to load images
def train_gen():
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)[1000:]
    for p in arr:
        img = cv2.imread(img_directory+p)
        img = np.asarray(img)/255  
        img72 = (cv2.resize(img, (72,72), interpolation = cv2.INTER_AREA))
        img144 = (cv2.resize(img, (144,144), interpolation = cv2.INTER_AREA))
        img72 = np.expand_dims(img72, axis=0)
        img144 = np.expand_dims(img144, axis=0)
        yield [img72,img144]
def val_gen_144():
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)[:100]
    for p in arr:
        img = cv2.imread(img_directory+p)
        img = np.asarray(img)/255 
        img72 = (cv2.resize(img, (72,72), interpolation = cv2.INTER_AREA))
        img144 = (cv2.resize(img, (144,144), interpolation = cv2.INTER_AREA))
        img72 = np.expand_dims(img72, axis=0)
        img144 = np.expand_dims(img144, axis=0)
        yield [img72,img144]
def val_gen_144_288():
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)[:100]
    for p in arr:
        img = cv2.imread(img_directory+p)
        img = np.asarray(img)/255 
        img72 = (cv2.resize(img, (72,72), interpolation = cv2.INTER_AREA))
        img144 = (cv2.resize(img, (144,144), interpolation = cv2.INTER_AREA))
        img72 = np.expand_dims(img72, axis=0)
        img144 = np.expand_dims(img144, axis=0)
        img288 = (cv2.resize(img, (288,288), interpolation = cv2.INTER_AREA))
        img288 = np.expand_dims(img288, axis=0)
        yield [img72,img144,img288]
def data_generator_train(arr_len ,batch = 32):
    while True:
        a = train_gen()
        batchsize = batch
        for i,j in enumerate(a):
            if i == 0:
                r = np.asarray(j[0])
                t = np.asarray(j[1])
            else:
                r = np.concatenate((r,np.asarray(j[0])))
                t = np.concatenate((t,np.asarray(j[1])))
            if i % (batchsize-1) == 0 and i != 0 or (i == arr_len-1):
                yield r[i-(batchsize-1):i],t[i-(batchsize-1):i]
def data_generator_val_144(arr_len ,batch = 32):
    while True:
        a = val_gen_144()
        batchsize = batch
        for i,j in enumerate(a):
            if i == 0:
                r = np.asarray(j[0])
                t = np.asarray(j[1])
            else:
                r = np.concatenate((r,np.asarray(j[0])))
                t = np.concatenate((t,np.asarray(j[1])))
            if i % (batchsize-1) == 0 and i != 0 or (i == arr_len-1):
                yield r[i-(batchsize-1):i],t[i-(batchsize-1):i]
def data_generator_val_144_288(arr_len ,batch = 32):
    while True:
        a = val_gen_144_288()
        batchsize = batch
        for i,j in enumerate(a):
            if i == 0:
                r = np.asarray(j[0])
                t = np.asarray(j[1])
                y = np.asarray(j[2])
            else:
                r = np.concatenate((r,np.asarray(j[0])))
                t = np.concatenate((t,np.asarray(j[1])))
                y = np.concatenate((y,np.asarray(j[2])))
                
            if i % (batchsize-1) == 0 and i != 0 or (i == arr_len-1):
                yield r[i-(batchsize-1):i],[t[i-(batchsize-1):i],y[i-(batchsize-1):i]]


# In[ ]:


def load_images(height=72,width=72):
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)[0:100]
    img_array = []
    for p in arr:
        img = cv2.imread(img_directory+p)
        img = np.asarray(img)/255
        img_array.append(cv2.resize(img, (height,width), interpolation = cv2.INTER_AREA))
    return img_array
img72 = load_images(72,72)
img144 = load_images(144,144)
img288 = load_images(288,288)
img72 = np.asarray(img72)  
img144 = np.asarray(img144)  
img288 = np.asarray(img288) 


# # Plot some Examples

# In[ ]:


# plot func with more then 1 row or col, expect images to be a list
def plot_images(images,titles = ['original','original','pred144'],figsize=(15,15)):
    nrows = len(images[0])
    ncols = len(images)
    fig, ax = plt.subplots(nrows = nrows,ncols = ncols,figsize=figsize)
    fig.tight_layout()

#     fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for i in range(nrows):
        for j in range(ncols):
            ax[i][j].imshow(images[j][i])
            if i == 0:
                ax[i][j].set_title(titles[j] + ' size = ' + str(images[j][i].shape))
plot_images([img72[0:4],img144[0:4],img288[0:4]],['original,','original,','original,'])


# # First model (step 2)

# In[ ]:


def build_model_step2():
    inp = Input(shape=(72, 72, 3))
    x = Conv2D(64, (3, 3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3, 1,activation = 'sigmoid',padding='same',name = 'output')(x)
    return Model(inputs=inp, outputs=x)


# In[ ]:


model1 = build_model_step2()
model1.compile(optimizer='adam', loss='mse',metrics=[PSNR])
model1.summary()
plot_model(model1, to_file='multiple_outputs.png')


# In[ ]:


def set_callbacks( name = 'best_model_weights',patience=8,tb_base_logdir='./logs/'):
#     from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
    cp = ModelCheckpoint(name +'.h5',save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=patience)
    return [rlop, es, cp]


# In[ ]:


# work on sample of the data
# hist1 = model1.fit(img72[0:80], img144[0:80], validation_data=(img72[80:100], img144[80:100]), shuffle=True, epochs=100,batch_size = 64,callbacks = set_callbacks(name = 'model1_weights'))
batch = 64
hist1 = model1.fit_generator(data_generator_val_144(4011,batch),validation_data=data_generator_val_144(1000,batch),
                             validation_steps = 1000//batch, steps_per_epoch= 4011//batch,epochs=10,callbacks = set_callbacks(name = 'best_model1_weights'))


# In[ ]:


def print_loss_psnr(hist,title_name):
    if len(hist.history.keys()) <= 5:
        plt.plot(hist.history['loss'], 'r', hist.history['val_loss'], 'b')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.title(title_name + ' loss')
        plt.legend(['loss','val_loss'])
        plt.figure()
        plt.plot(hist.history['PSNR'], 'r', hist.history['val_PSNR'], 'b')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.title(title_name + ' PSNR')
        plt.legend(['PSNR','val_PSNR'])
    else:
        plt.plot(hist.history['output1_loss'], 'r', hist.history['output2_loss'], 'b',
                hist.history['val_output1_loss'], 'g', hist.history['val_output2_loss'], 'y')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.title(title_name + ' loss')
        plt.legend(['output1_loss','output2_loss','val_output1_loss','val_output2_loss'])
        plt.figure()
        plt.plot(hist.history['output1_PSNR'], 'r', hist.history['output2_PSNR'], 'b',
                hist.history['val_output1_PSNR'], 'g', hist.history['val_output2_PSNR'], 'y')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.title(title_name + ' PSNR')
        plt.legend(['output1_PSNR','output2_PSNR','val_output1_PSNR','val_output2_PSNR'])


# In[ ]:


print_loss_psnr(hist = hist1,title_name = 'step 2 - model1')


# In[ ]:


pred144 = model1.predict(img72[80:100])
plot_images([img72[80:85],img144[80:85],pred144[0:5]],titles = ['original,','original,','pred,'])


# # Second model (step 3)

# In[ ]:


def build_model_step3():
    inp = Input(shape=(72, 72, 3))
    x = Conv2D(64, (3, 3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D((2,2))(x)
    out2 = UpSampling2D((2,2))(x)
    out2 = Conv2D(3, 1,activation = 'sigmoid', padding='same',name = 'output2')(out2)
    out1 = Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output1')(x)
    model = Model(inputs=inp, outputs=[out1, out2])
#     model.compile(optimizer='adam', loss='mse',loss_weights = [0.1,0.9])
    return model


# In[ ]:


model2 = build_model_step3()
model2.compile(optimizer='adam', loss='mse',metrics=[PSNR])
model2.summary()
plot_model(model2, to_file='multiple_outputs.png')


# In[ ]:


# hist2 = model2.fit(img72[0:80], [img144[0:80],img288[0:80]] ,validation_data=(img72[80:100],[img144[80:100],img288[80:100]]), shuffle=True, epochs=100,batch_size = 64,callbacks = set_callbacks())
batch = 64
hist2 = model2.fit_generator(data_generator_val_144_288(4011,batch),validation_data=data_generator_val_144_288(1000,batch),
                             validation_steps = 1000//batch, steps_per_epoch= 4011//batch,epochs=10,callbacks = set_callbacks(name = 'best_model2_weights'))


# In[ ]:


print_loss_psnr(hist = hist2,title_name = 'step 3 model')


# In[ ]:


pred2 = model2.predict(img72[80:100])
plot_images([img72[80:85],img144[80:85],img288[80:85],pred2[0][0:5],pred2[1][0:5]],titles = ['original,','original,','original,','pred1,','pred2,'])


# # model 3 (step 4)

# In[ ]:


def res(Channels_num = 32):
    inp = Input(shape=(None,None, Channels_num ))
    x = (Conv2D(Channels_num, (3, 3), padding='same'))(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = (Conv2D(Channels_num, (3, 3), padding='same'))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = add([x, inp])
    x = LeakyReLU(alpha=0.2)(x)
    return Model(inputs=inp, outputs=x)


# In[ ]:


model_res = res()
model_res.summary()
plot_model(model_res, to_file='multiple_outputs.png')


# In[ ]:


def build_model_step4():
    inp = Input(shape=(72, 72, 3))
    x = Conv2D(32, (3, 3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = res(32)(x)
    x = res(32)(x)
    x = UpSampling2D((2,2))(x)
    out1 = Conv2D(3, 1,activation = 'sigmoid', padding='same',name = 'output1')(x)
    x = res()(x)
    x = (UpSampling2D((2,2)))(x)
    out2 = (Conv2D(3, 1, activation = 'sigmoid',padding='same',name = 'output2'))(x)
    model = Model(inputs=inp, outputs=[out1, out2])
    return model


# In[ ]:


model3 = build_model_step4()
model3.compile(optimizer='adam', loss='mse',metrics=[PSNR])
model3.summary()
plot_model(model3, to_file='multiple_outputs.png')


# In[ ]:


# hist3 = model3.fit(img72[0:80], [img144[0:80],img288[0:80]] ,validation_data=(img72[80:100],[img144[80:100],img288[80:100]]), shuffle=True, epochs=100,batch_size = 64,callbacks = set_callbacks())
batch = 64
hist3 = model3.fit_generator(data_generator_val_144_288(4011,batch),validation_data=data_generator_val_144_288(1000,batch),
                             validation_steps = 1000//batch, steps_per_epoch= 4011//batch,epochs=10,callbacks = set_callbacks(name = 'best_model3_weights'))


# In[ ]:


print_loss_psnr(hist = hist3,title_name = 'step 4 model')


# In[ ]:


pred3 = model3.predict(img72[80:100])
plot_images([img72[80:85],img144[80:85],img288[80:85],pred3[0][0:5],pred3[1][0:5]],titles = ['original,','original,','original,','pred1,','pred2,'])


# # Model 4 (step 5)

# In[ ]:


def dil(Channels_num = 32):
    inp = Input(shape=(None,None, Channels_num))
    dil1 = (Conv2D(Channels_num, (3, 3),dilation_rate=1, padding='same'))(inp)
    dil1 = LeakyReLU(alpha=0.2)(dil1)
    dil2 = (Conv2D(Channels_num, (3, 3),dilation_rate=2, padding='same'))(inp)
    dil2 = LeakyReLU(alpha=0.2)(dil2)
    dil4 = (Conv2D(Channels_num, (3, 3),dilation_rate=4, padding='same'))(inp)
    dil4 = LeakyReLU(alpha=0.2)(dil4)
    
    x = Concatenate()([dil1, dil2,dil4])
    x = LeakyReLU(alpha=0.2)(x)
#     x = Activation('relu')(x)
    out = Conv2D(Channels_num, 3, padding='same')(x)
    out = LeakyReLU(alpha=0.2)(out)
    model = Model(inputs=inp, outputs=out)
    return model


# In[ ]:


model_dil = dil()
model_dil.summary()
plot_model(model_dil, to_file='multiple_outputs.png')


# In[ ]:


def step5_model():
    inp = Input(shape=(72, 72, 3))
    x = Conv2D(32, (3, 3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = dil()(x)
    x = dil()(x)
    x = UpSampling2D((2,2))(x)
    out1 = (Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output1'))(x)
    x = dil()(x)
    x = UpSampling2D((2,2))(x)
    out2 = (Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output2'))(x)
    model = Model(inputs=inp, outputs=[out1, out2])
    return model
# m2 = step5_model()


# In[ ]:


model4 = step5_model()
model4.compile(optimizer='adam', loss='mse',metrics=[PSNR])
model4.summary()
plot_model(model4, to_file='multiple_outputs.png')


# In[ ]:


# hist4 = model4.fit(img72[0:80], [img144[0:80],img288[0:80]] ,validation_data=(img72[80:100],[img144[80:100],img288[80:100]]), shuffle=True, epochs=100,batch_size = 64,callbacks = set_callbacks(patience = 10))
batch = 64
hist4 = model4.fit_generator(data_generator_val_144_288(4011,batch),validation_data=data_generator_val_144_288(1000,batch),
                             validation_steps = 1000//batch, steps_per_epoch= 4011//batch,epochs=10,callbacks = set_callbacks(name = 'best_model4_weights'))


# In[ ]:


print_loss_psnr(hist = hist4,title_name = 'step 5 model')


# In[ ]:


pred4 = model4.predict(img72[80:100])
plot_images([img72[80:85],img144[80:85],img288[80:85],pred4[0][0:5],pred4[1][0:5]],titles = ['original,','original,','original,','pred1,','pred2,'])


# # Model 5 (step 6)

# In[ ]:


def step6_model():
    vgg_model = VGG16(weights='imagenet',include_top=False,input_shape=(72, 72, 3))
    block1_conv2 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block1_conv2').output)

    inp = Input(shape=(72, 72, 3))
    x = Conv2D(64, (3, 3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    FeatureEx = block1_conv2(inp)
        

    x = Concatenate()([x, FeatureEx])

    x = UpSampling2D((2,2))(x)
    out1 = (Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output1'))(x)
    
    x = UpSampling2D((2,2))(x)
    out2 = (Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output2'))(x)
    model = Model(inputs=inp, outputs=[out1, out2])
    return model


# In[ ]:


model5 = step6_model()
model5.compile(optimizer='adam', loss='mse',metrics=[PSNR])
model5.summary()
plot_model(model5, to_file='multiple_outputs.png')


# In[ ]:


# hist5 = model5.fit(img72[0:80], [img144[0:80],img288[0:80]] ,validation_data=(img72[80:100],[img144[80:100],img288[80:100]]), shuffle=True, epochs=100,batch_size = 64,callbacks = set_callbacks())
hist5 = model5.fit_generator(data_generator_val_144_288(4011,batch),validation_data=data_generator_val_144_288(1000,batch),
                             validation_steps = 1000//batch, steps_per_epoch= 4011//batch,epochs=10,callbacks = set_callbacks(name = 'best_model5_weights'))


# In[ ]:


print_loss_psnr(hist = hist5,title_name = 'step 6 model')


# In[ ]:


pred5 = model5.predict(img72[80:100])
plot_images([img72[80:85],img144[80:85],img288[80:85],pred5[0][0:5],pred5[1][0:5]],titles = ['original,','original,','original,','pred1,','pred2,'])


# # Model 6 (step 7)

# In[ ]:


def step7_model():
    vgg_model = VGG16(weights='imagenet',include_top=False,input_shape=(72, 72, 3))
    block1_conv2 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block1_conv2').output)

    inp = Input(shape=(72, 72, 3))
    x = Conv2D(64, (3, 3), padding='same')(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    FeatureEx = block1_conv2(inp)
        

    x = Concatenate()([x, FeatureEx])

    sub_layer = Lambda(lambda x:tf.nn.depth_to_space(x,2))
    x = sub_layer(inputs=x)
    out1 = (Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output1'))(x)
    
#     x = Conv2D(64, (3, 3), padding='same')(x)
#     x = LeakyReLU(alpha=0.2)(x)
    
    x = sub_layer(inputs=x)
    out2 = (Conv2D(3, 1, activation='sigmoid', padding='same',name = 'output2'))(x)
    model = Model(inputs=inp, outputs=[out1, out2])
    return model


# In[ ]:


model6 = step7_model()
model6.compile(optimizer='adam', loss='mse',metrics=[PSNR])
model6.summary()
plot_model(model6, to_file='multiple_outputs.png')


# In[ ]:


# hist6 = model6.fit(img72[0:80], [img144[0:80],img288[0:80]] ,validation_data=(img72[80:100],[img144[80:100],img288[80:100]]), shuffle=True, epochs=100,batch_size = 64,callbacks = set_callbacks())
hist6 = model6.fit_generator(data_generator_val_144_288(4011,batch),validation_data=data_generator_val_144_288(1000,batch),
                             validation_steps = 1000//batch, steps_per_epoch= 4011//batch,epochs=10,callbacks = set_callbacks(name = 'best_model1_weights'))


# In[ ]:


print_loss_psnr(hist = hist6,title_name = 'step 7 model')


# In[ ]:


pred6 = model6.predict(img72[80:100])
plot_images([img72[80:85],img144[80:85],img288[80:85],pred6[0][0:5],pred6[1][0:5]],titles = ['original,','original,','original,','pred1,','pred2,'])


# # Conclusion:
# * We can see that there are variety of methods in the field of super-resolution (and i assume that there are much more) <br>
# * our evaluation metric was PSNR which computes the signal - to - noise ratio, ideally we want infinite value - the higher the better. <br>
# * All of our models PSNR values was more or less around 73 for the 144 resolution prediction and 70 for the 288 resolution prediction. <br>
# * models 3 and 5 achieved the highest scores, model 3 uses residual blocks (inspired by ResNet), and model 5 uses feature extraction from the pretrained vgg model. <br>
# * If we look carefully, we can notice that the 144 resolution prediction benefits better picture comparing to the 72 resolution, but still not as good as the original 144 resolution. <br> 
# 

# 
