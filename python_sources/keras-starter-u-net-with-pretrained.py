#!/usr/bin/env python
# coding: utf-8

# Simple example of U-net for segmentation in Keras

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm_notebook
import cv2

import keras
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import UpSampling2D, Conv2D, Activation, Input, Dropout, MaxPooling2D
from keras import layers
from keras import Model
from keras import backend as K
from keras.layers.core import Lambda


# In[ ]:


model = VGG19(weights='imagenet', include_top=False)
 
for layer in model.layers:
    print(layer)

model.save("severstal_vgg19.h5")


# In[ ]:


def Load_vgg19(img_input):
    #block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    #block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    
    x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
   
    x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
  
    x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    model = Model(inputs=[img_input], outputs=x)

    # Load pretrained
    pretrained_model = VGG19(include_top=False)

    for layer, pretrained_layer in zip(
            model.layers[2:], pretrained_model.layers[2:]):
        layer.set_weights(pretrained_layer.get_weights())
        
    imagenet_weights = pretrained_model.layers[1].get_weights()
    init_bias = imagenet_weights[1]
    init_kernel = np.average(imagenet_weights[0], axis=2)
    #print(init_kernel.shape)
    init_kernel = np.reshape(
        init_kernel,
        (init_kernel.shape[0],
            init_kernel.shape[1],
            1,
            init_kernel.shape[2]))
    init_kernel = np.dstack([init_kernel] * img_input.shape.as_list()[-1])  # input image is grayscale
    model.layers[1].set_weights([init_kernel, init_bias])
    return model


if 1:
    img = layers.Input(shape=(128, 128, 4))
    pretrained = Load_vgg19(img)
    for layer in pretrained.layers:
        print(layer.name, pretrained.get_layer(layer.name))


# In[ ]:


tr = pd.read_csv('../input/train.csv')
print(len(tr))
tr.head()


# In[ ]:


#df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)
#print(len(df_train))
df_train = tr
df_train['ImageId'], df_train['ClassId'] = zip(*df_train['ImageId_ClassId'].str.split('_')) #split imageId and classId
df_train['ClassId'] = df_train['ClassId'].astype(int)
df_train = df_train.pivot(index='ImageId',columns='ClassId',values='EncodedPixels') #remap
df_train['defects'] = df_train.count(axis=1) #count on defect type
df_train = df_train[df_train['defects'] > 0]
print(len(df_train))
df_train.head()
#print(df_train.iloc[6666-1].name)


# In[ ]:


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    if rle is not np.nan:
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


# In[ ]:


img_scale = 2
img_size = (1600 // img_scale,256 // img_scale)
classes_num = 4


# In[ ]:


#contrast enhancing

do_enhance = True

gamma = 1.2
inverse_gamma = 1.0 / gamma
look_up_table = np.array([((i/255.0) ** inverse_gamma) * 255.0 for i in np.arange(0,256,1)]).astype("uint8")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def contrast_enhancement(img):
    if not do_enhance:
        return img
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img[:,:,0] = clahe.apply(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
    return img

def gamma_correction(img):
    if not do_enhance:
        return img
    return cv2.LUT(img.astype('uint8'), look_up_table)


# In[ ]:


import random
def keras_generator(batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size): 
            flip = int(100)
            if random.uniform(0,1) > 0.5:
                flip = random.randint(-1,1)
            #print(flip)
                
            
            fn = df_train.iloc[i].name
            img = cv2.imread( '../input/train_images/'+fn )
            #plt.subplot(3,1,1)
            #plt.imshow(img)
            img = gamma_correction(img)
            #plt.subplot(3,1,2)
            #plt.imshow(img)
            img = contrast_enhancement(img)
            #plt.subplot(3,1,3)
            #plt.imshow(img)
            #break
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
            #mask_gt = np.zeros( shape = img.shape[0:2] ).astype(np.uint8) + 1
            #mask_gt = cv2.resize(mask_gt, img_size,cv2.INTER_NEAREST)
            #if flip != 100:
            #    mask_gt = cv2.flip(mask_gt,flip)
            #mask_gt = np.expand_dims(mask_gt,-1) #background
            #masks = [mask_gt]
            masks = []
            for cls in range(0,classes_num):
                mask = rle2mask(df_train[cls+1].iloc[i], img.shape)
                mask = np.squeeze(mask)
                mask = cv2.resize(mask, img_size,cv2.INTER_NEAREST)
                if flip != 100:
                    mask = cv2.flip(mask,flip)
                mask = np.expand_dims(mask,-1)
               # mask_gt[mask != 0] = 0 #move pixel from gt
                masks.append(mask)

            mask = np.concatenate(masks,axis=-1)
            img = cv2.resize(img, img_size,cv2.INTER_AREA)
            if flip != 100:
                img = cv2.flip(img,flip)
            x_batch += [img]
            y_batch += [mask]
                                    
        x_batch = np.array(x_batch) / 255.0
        y_batch = np.array(y_batch)

        #yield x_batch, np.expand_dims(y_batch, -1)
        yield x_batch, y_batch


# In[ ]:


for x, y in keras_generator(4):
    break
    
print(x.shape, y.shape)


# In[ ]:


test_image_id = 3
plt.subplot(classes_num+1,1,1)
plt.imshow(x[test_image_id])
for k in range(classes_num):
    plt.subplot(classes_num+1,1,k+2)
    plt.imshow(np.squeeze(y[test_image_id,:,:,k]))


# In[ ]:


def segment_cross_entropy(features, labels):
    features = K.softmax(features, axis=1)
    features = K.reshape(features,[-1])
    labels = K.reshape(labels,[-1])
    return K.categorical_crossentropy(labels, features, axis=-1)
        
#true_dist, coding_dist = np.ones((4,5,6,6))/2, np.ones((4,5,6,6))/2
#loss = segment_cross_entropy(coding_dist, coding_dist)


# In[ ]:


#Model
def get_net_raw(img_size,classes_num):
    inputs = Input((img_size[1], img_size[0], 3))
    #s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    #outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)
    outputs = Conv2D(classes_num,(1,1),activation = 'sigmoid')(c9)  #todos: try softmax

    model = Model(inputs=[inputs], outputs=[outputs])
    #model.compile(optimizer='adam', loss='binary_crossentropy')
    #model.compile(optimizer='adam',loss='categorical_crossentropy')
    return model


# In[ ]:


#Model
from keras import optimizers
def get_net_pretrained(img_size,classes_num):
    layers_feat = 'block1_conv2,block2_conv2,block3_conv4,block4_conv4,block5_conv4'.split(',')
    
    inputs = Input((img_size[1], img_size[0], 3))
    #s = Lambda(lambda x: x / 255) (inputs)
    pretrained = Load_vgg19(inputs)
    for layer in pretrained.layers:
        layer.trainable=False #freeze
    c1 = pretrained.get_layer(layers_feat[0]).output #1x
    c2 = pretrained.get_layer(layers_feat[1]).output #2x
    c3 = pretrained.get_layer(layers_feat[2]).output #4x
    c4 = pretrained.get_layer(layers_feat[3]).output #8x
    c5 = pretrained.get_layer(layers_feat[4]).output #16x
    c5 = Dropout(0.3) (c5)
    
    #decoder
    u6 = Conv2DTranspose(128*2, (2, 2), strides=(2, 2), padding='same') (c5) #8x
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64*2, (2, 2), strides=(2, 2), padding='same') (c6) #4x
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same') (c7) #2x
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16*2, (2, 2), strides=(2, 2), padding='same') (c8) #1x
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16*2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    #outputs = Conv2D(4, (1, 1), activation='sigmoid') (c9)
    outputs = Conv2D(classes_num,(1,1),activation = 'sigmoid')(c9)  #todos: try softmax

    model = Model(inputs=[inputs], outputs=[outputs])

    #model.compile(optimizer='adam', loss='binary_crossentropy')
    #model.compile(optimizer='adam',loss='categorical_crossentropy')
    return model


# In[ ]:


model = get_net_pretrained(img_size,classes_num)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')
for layer in model.layers:
    print(layer.name)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Fit model\nbatch_size = 32\nresults = model.fit_generator(keras_generator(batch_size), \n                              steps_per_epoch=100,\n                              epochs=1) \n\nmodel.save("severstal_s4.h5")')


# In[ ]:


pred = model.predict(x)
print(pred.shape,)
plt.imshow(np.squeeze(pred[3,:,:,3]))


# In[ ]:


testfiles=os.listdir("../input/test_images/")
len(testfiles)


# In[ ]:


get_ipython().run_cell_magic('time', '', "if 0:\n    import gc\n    #del pred\n    #del x\n    gc.collect()\n\n    test_img = []\n    for fn in tqdm_notebook(testfiles):\n            img = cv2.imread( '../input/test_images/'+fn )\n            img = cv2.resize(img,img_size)       \n            test_img.append(img)\n\n    print(len(test_img))")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if 0:\n    predict = model.predict(np.asarray(test_img))\n    print(len(predict)) ')


# In[ ]:




def mask2rle(img):
    tmp = np.rot90( np.flipud( img ), k=3 )
    rle = []
    lastColor = 0;
    startpos = 0
    endpos = 0

    tmp = tmp.reshape(-1,)   
    inds = np.argwhere(tmp == 1)
    if len(inds) == 0:
        return ' '.join([])
    inds = list(map(lambda x: x[0], inds))
    last = inds[0]
   # pdb.set_trace()
    for k in range(1,len(inds)):
        if inds[k] == inds[k-1] + 1:
            continue
        rle.append( str(last)+' '+str(inds[k-1]-last+1) )
        last = inds[k]
    return " ".join(rle)


# In[ ]:


get_ipython().run_cell_magic('time', '', "import pdb\nthresh_score = 0.8\nthresh_num = 3500\npred_rle = []\n\ntest_img = []\nfor fn in tqdm_notebook(testfiles):\n        img = cv2.imread( '../input/test_images/'+fn )\n        img = gamma_correction(img)\n        img = contrast_enhancement(img)\n        img = cv2.resize(img,img_size)       \n        #test_img.append(img)   \n        scores = model.predict(np.asarray([img]))\n        scores = np.squeeze(scores)\n        pred = np.argmax(scores,axis=-1)\n        #print(scores.shape)\n        for cls in range(0, classes_num):\n            mask = np.squeeze(pred == cls).astype(np.uint8)\n            score = scores[:,:,cls]\n            #print(mask.shape,'---')\n            mask[score < thresh_score] = 0\n            if np.sum(mask) < thresh_num:\n                mask = mask * 0\n            mask = cv2.resize(mask, (1600, 256), cv2.INTER_NEAREST)\n            rle = mask2rle(mask)\n            #print(rle)\n            pred_rle.append(rle)\n        ")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if 0:\n    import pdb\n    thresh_score = 0.8\n    thresh_num = 100\n    pred_rle = []\n    for fn, scores in enumerate(tqdm_notebook(predict)):  \n        pred = np.argmax(scores,axis=-1)\n        for cls in range(0, classes_num):\n            mask = np.squeeze(pred == cls).astype(np.uint8)\n            score = scores[:,:,cls]\n            mask[score < thresh_score] = 0\n            if np.sum(mask) < thresh_num:\n                mask = mask * 0\n            mask = cv2.resize(mask, (1600, 256), cv2.INTER_NEAREST)\n            rle = mask2rle(mask)\n            #print(rle)\n            pred_rle.append(rle)')


# In[ ]:


test_image_ind = 5
img_t = cv2.imread( '../input/test_images/'+ testfiles[test_image_ind])
plt.subplot(classes_num+1,1,1)
plt.imshow(img_t)

for cls in range(classes_num):
    mask_t = rle2mask(pred_rle[test_image_ind*4 + cls], img_t.shape)
    plt.subplot(classes_num+1,1,cls + 2)
    plt.imshow(mask_t)


# In[ ]:


sub = pd.read_csv( '../input/sample_submission.csv' )
sub.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'print(len(pred_rle)/4)\nfor fn_ind,fn in enumerate(testfiles):\n    for cls in range(0, classes_num):\n       # if fn_ind == 4 and cls == 4:\n       #     idx = sub[\'ImageId_ClassId\'] == "{}_{}".format(fn,cls)\n       #     print(sub[\'EncodedPixels\'][idx])\n       #     print(fn)\n        sub[\'EncodedPixels\'][sub[\'ImageId_ClassId\'] == "{}_{}".format(fn,cls+1)] = pred_rle[fn_ind * 4 + cls]')


# In[ ]:


img_s = cv2.imread( '../input/test_images/'+ sub['ImageId_ClassId'][4].split('_')[0])
plt.imshow(img_s)


# In[ ]:


sub.to_csv('submission.csv', index=False)

