#!/usr/bin/env python
# coding: utf-8

# Osteoporosis Detection using CNN and Keras by Mohsin
# 1. Importing the libraries
# 2. Training the set
# 3. Creating a model
# 4. Testing set by model COnv2D
# 5. Validating
# 6. Saving results to .csv
# 7. Dataset CT Scan of Bones Joints
# 8. Referring to the last work done
# 8. Improving the Estimated TIme Arrival and Lessening the Loss
# 9. Defining the region of interest of Vitamin D defficiency by preprocessing of the images
# 10. Concluded the Results

# In[ ]:


import os
print(os.listdir("../input"))
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import random as rnd
import math


# In[ ]:


IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 1
TRAIN_SIZE = 4000 
TRAIN_PATH = '../input/trainset/'
TEST_PATH = '../input/testset/'
ROI_PATH = "../input/roi/"


# In[ ]:


# Create useful helpers

def get_centers(dir_roi, resol=(2048, 2048)):
    ''' This function scans through the directory of the ROI files, reads the
        coordinates (in pixel) of the centers of all circular ROIs, then
        transforms them into relative coordinates (assumming image size 1x1) '''  

    img_id = []
    cx_pxl = []
    cy_pxl = []

    HEIGHT, WIDTH = resol
    
    # Get center coordinates of all sample images
    for dirpath, dirnames, files in os.walk(dir_roi):
        sample_id = os.path.split(dirpath)[1].split('.')[0]
        for file in files:
            img_id.append(sample_id + file[:4])
            cx_pxl.append(int(file[10:14]))
            cy_pxl.append(int(file[5:9]))

    df_roi = pd.DataFrame({'img_id': img_id,
                           'cx_pxl': cx_pxl,
                           'cy_pxl': cy_pxl},
                           columns = ['img_id', 'cx_pxl', 'cy_pxl'])
    df_roi['cx'] = df_roi.cx_pxl / float(WIDTH)
    df_roi['cy'] = df_roi.cy_pxl / float(HEIGHT)
            
    return df_roi


def coord_transfm(df_roi_org, shifts=(359, 359), cropped_resol=(1330, 1330)):
    shift_x, shift_y = shifts
    HEIGHT, WIDTH = cropped_resol
    df_roi = df_roi_org.copy()
    df_roi.cx_pxl = df_roi.cx_pxl - shift_x
    df_roi.cy_pxl = df_roi.cy_pxl - shift_y
    df_roi.cx = df_roi.cx_pxl / float(WIDTH)
    df_roi.cy = df_roi.cy_pxl / float(HEIGHT)
    
    return df_roi 


def draw_roi(img_path, img_list, true_centers, pred_centers=None, 
             actual_R=1.5, actual_imgsize=16.38, current_resol=1330, 
             org_resol=2048, rows=1, cols=3, model_name="Model"):
    
    resize_ratio = current_resol / org_resol
    R = math.floor(current_resol * actual_R/(actual_imgsize*resize_ratio))
    
    plt.figure(figsize=(8*cols, 8*rows))
    
    for i, img_id in enumerate(img_list):
        plt.subplot(rows, cols, i+1)
        img = cv2.imread(TRAIN_PATH + img_id + '.png')
        cx = math.floor(true_centers[i, 0] * current_resol)
        cy = math.floor(true_centers[i, 1] * current_resol)        
        
        if pred_centers is not None:
            cx_pred = math.floor(pred_centers[i, 0] * current_resol) 
            cy_pred = math.floor(pred_centers[i, 1] * current_resol)
            img_mod = cv2.circle(img, (cx_pred, cy_pred), R, (0,0,255), 3)  # Mark the predicted center in blue
            img_mod = cv2.circle(img_mod, (cx_pred, cy_pred), round(R*0.05), (0,0,255), -1)  
            img_mod = cv2.circle(img_mod, (cx, cy), R, (255,0,0), 2)  # Mark the true center in red
            img_mod = cv2.circle(img_mod, (cx, cy), round(R*0.05), (255,0,0), -1)  

        else:
            img_mod = cv2.circle(img, (cx, cy), R, (255,0,0), 3)  # Mark the true center in red
            img_mod = cv2.circle(img_mod, (cx, cy), round(R*0.05), (255,0,0), -1)  

        plt.imshow(img_mod)
    
        if pred_centers is not None:
            plt.title("Vitamin D Deficiency Point  \n" + model_name + ": {}\nHuman: {}"
                      .format(str((cx_pred, cy_pred)),
                              str((cx, cy))))
        else:
            plt.title("Vitamin D Deficiency Point: {}".format(str((cx, cy))))   
            

def mirror(df_imgs, flip_axis, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    
    pxl_cols = [col for col in df_imgs.columns if 'pxl' in col]
    num_imgs = df_imgs.shape[0]
    img_size = IMG_HEIGHT * IMG_WIDTH
    
    mat_images = df_imgs[pxl_cols].values
    mat_images = mat_images.reshape((num_imgs, IMG_HEIGHT, IMG_WIDTH))
    df_imgs_flip = df_imgs.copy()
    
    assert 'h' in flip_axis or 'v' in flip_axis,            "Flipping axis is not defined. Must be either 'horizontal' or 'vertical'."
    
    if 'h' in flip_axis:
        flipping = 1
        df_imgs_flip.img_id = df_imgs.img_id + 'hf'
        df_imgs_flip.cy = 1.0 - df_imgs.cy  
    elif 'v' in flip_axis:
        flipping = 2
        df_imgs_flip.img_id = df_imgs.img_id + 'vf'
        df_imgs_flip.cx = 1.0 - df_imgs.cx    
    
    # Flip images around the specified axis    
    mat_images_flip = np.flip(mat_images, flipping)                         .reshape((num_imgs, img_size))
    df_imgs_flip[pxl_cols] = mat_images_flip
    
    return df_imgs_flip


# ### Data Loading

# In[ ]:


df_centers_org = get_centers(ROI_PATH).sort_values(by='img_id')                                      .reset_index(drop=True)
print(df_centers_org.head())
df_centers = coord_transfm(df_centers_org)
print(df_centers.head())


# In[ ]:


images = pd.Series(sorted(os.listdir(TRAIN_PATH)))
img_ids = images.str.split('.').str[0]
assert df_centers.img_id.equals(img_ids), "Image lists don't match"


# Preprocessing of Images

# In[ ]:



img_size = IMG_HEIGHT * IMG_WIDTH
mat_images = np.zeros((TRAIN_SIZE, img_size))
train_images = images.sample(TRAIN_SIZE, random_state=10)
train_img_ids = train_images.str.split('.').str[0]
df_train = pd.DataFrame(mat_images, 
                        columns=['pxl' + str(i) for i in range(img_size)])
df_train.insert(0, 'img_id', train_img_ids.values)
df_train = pd.merge(df_train, df_centers[['img_id', 'cx', 'cy']], 
                    on='img_id', validate="1:1")
print(df_train.head())
print(df_train.shape)


# ### Manual image preprocessing

# In[ ]:


df_train_hflip = mirror(df_train, 'h')
df_train_vflip = mirror(df_train, 'v')


# In[ ]:


df_train = pd.concat([df_train, df_train_hflip, df_train_vflip],
                     ignore_index=True)
print("New dataframe's shape: {}".format(df_train.shape))
df_centers = df_train[['img_id', 'cx', 'cy']]


# In[ ]:


X = df_train.drop(columns=['img_id', 'cx', 'cy'])             .values.reshape((-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y = df_train[['cx', 'cy']].values
IDs = df_train.img_id.values
X /= 255.0


# In[ ]:


# Free RAM space
del df_train, df_train_hflip, df_train_vflip, mat_images


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val, IDs_train, IDs_val = train_test_split(X, Y, IDs, test_size=0.05, random_state=1)
print("Trainset shape: {}".format(X_train.shape))
print("Validateset shape: {}".format(X_val.shape))


# In[ ]:


import math 
import cv2
select_disp = df_centers.iloc[:TRAIN_SIZE].sample(n=3)
img_list = select_disp['img_id'].values
true_centers = select_disp[['cx', 'cy']].values
draw_roi(TRAIN_PATH, img_list, true_centers)


# **Simple Convolutional Neural Network Model**

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
cnn_model = Sequential()

cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5), 
                     padding = 'Same', activation ='relu', 
                     input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))

cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),
                     padding = 'Same', activation ='relu'))

cnn_model.add(MaxPool2D(pool_size=(2,2)))
# cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3),
                 padding = 'Same', activation ='relu'))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3),
                     padding = 'Same', activation ='relu'))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
# cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(512, activation = "relu"))
cnn_model.add(Dense(64, activation = "relu"))
# cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(2))

cnn_model.summary()


# In[ ]:


def mean_dist(y_pred, y_true):
    d = y_pred - y_true
    return tf.reduce_mean(tf.norm(d, axis=1))


# In[ ]:


import tensorflow as tf
cnn_model.compile(loss=mean_dist, optimizer='adam')
cnn_model.fit(X_train, Y_train, batch_size=64, epochs=8)


# In[ ]:


cnn_scores_train = cnn_model.evaluate(X_train, Y_train)
print("Score on trainset: {}".format(cnn_scores_train))
cnn_scores_val = cnn_model.evaluate(X_val, Y_val)
print("Score on validate set: {}".format(cnn_scores_val))


# In[ ]:


cnn_pred = cnn_model.predict(X_val, verbose=True)
cnn_results = pd.DataFrame(np.concatenate([cnn_pred, Y_val], axis=1), 
                          columns = ['cx_pred', 'cy_pred', 'cx', 'cy'])
cnn_results.insert(0, 'img_id', IDs_val)
print(cnn_results.head())


# In[ ]:


orig = [img_id for img_id in cnn_results.img_id 
        if 'f' not in img_id]
orig_imgs = cnn_results[cnn_results.img_id.isin(orig)]
selt_imgs = orig_imgs.sample(n=6)
img_list = selt_imgs['img_id'].values
true_centers = selt_imgs[['cx', 'cy']].values
pred_centers = selt_imgs[['cx_pred', 'cy_pred']].values

draw_roi(TRAIN_PATH, img_list, true_centers, 
         pred_centers, rows=2, cols=3, model_name="Simple CNN")


# In[ ]:


selt_dots = orig_imgs.sample(n=20)
img_list = selt_dots['img_id'].values
true_centers = (selt_dots[['cx', 'cy']].values * 1330).astype(int)
pred_centers = (selt_dots[['cx_pred', 'cy_pred']].values * 1330).astype(int)

plt.figure(figsize = (5, 5))
plt.scatter(true_centers[:, 0], true_centers[:, 1], c='r', alpha=0.5, label='Human')
plt.scatter(pred_centers[:, 0], pred_centers[:, 1], c='b', alpha=0.5, label='CNN model')
plt.axis('equal')
plt.legend()
plt.title("20 Random Selected Images of Bones")
plt.show() 


# In[ ]:


cnn_results.to_csv("CNN1.csv")


# **Convolutional Neural Network with Activation Functions**

# In[ ]:


from keras import layers
from keras.layers import Input, Add, Activation, ZeroPadding2D, BatchNormalization,                          AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform


# In[ ]:


def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), 
               padding = 'valid', name = conv_name_base + '2a', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1),
               padding = 'same', name = conv_name_base + '2b',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), 
               padding = 'valid', name = conv_name_base + '2c', 
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2c')(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


# In[ ]:


def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1,1), strides = (s,s), name = conv_name_base + '2a',
               padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(F2, (f,f), strides = (1,1), name = conv_name_base + '2b',
               padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(F3, (1,1), strides = (1,1), name = conv_name_base + '2c',
               padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X) 
    X_shortcut = Conv2D(F3, (1,1), strides = (s,s), padding = 'valid',
                        name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


# In[ ]:


def ResNet50(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((1, 1))(X_input) # mod (3,3) -> (1,1)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv1', # mod (7,7) -> (3,3); (2,2) -> (1,1)
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f=3, filters=[64, 64, 256], 
                            stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f=3, filters=[128,128,512],
                            stage=3, block='a', s=2) 
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block='b')
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block='c')
    X = identity_block(X, f=3, filters=[128,128,512], stage=3, block='d')
    X = convolutional_block(X, f=3, filters=[256,256,1024],
                            stage=4, block='a', s=2) 
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256,256,1024], stage=4, block='f')
    X = convolutional_block(X, f=3, filters=[512,512,2048],
                            stage=5, block='a', s=2) 
    X = identity_block(X, f=3, filters=[512,512,2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512,512,2048], stage=5, block='c')
    X = AveragePooling2D((2,2), name='avg_pool')(X)
    X = Flatten()(X)
    # X = Dense(1024, name='fc_1024', kernel_initializer=glorot_uniform(seed=0))(X)  # add an extra dense layer
    # X = Dense(64, name='fc_64', kernel_initializer=glorot_uniform(seed=0))(X)  # add an extra dense layer
    X = Dense(2, name='fc_outputs', kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs = X_input, outputs = X, name = 'Conv2')
    return model


# In[ ]:


ResNet_model = ResNet50(input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
ResNet_model.summary()


# In[ ]:


ResNet_model.compile(loss=mean_dist, optimizer='adam')
ResNet_model.fit(X_train, Y_train, batch_size=128, epochs=15)


# In[ ]:


ResNet_scores_train = ResNet_model.evaluate(X_train, Y_train, batch_size=128)
print("Score on trainset: {}".format(ResNet_scores_train))
ResNet_scores_val = ResNet_model.evaluate(X_val, Y_val)
print("Score on validate set: {}".format(ResNet_scores_val))


# In[ ]:


ResNet_pred = ResNet_model.predict(X_val, verbose=True)
ResNet_results = pd.DataFrame(np.concatenate([ResNet_pred, Y_val], axis=1), 
                              columns = ['cx_pred', 'cy_pred', 'cx', 'cy'])
ResNet_results.insert(0, 'img_id', IDs_val)
print(ResNet_results.head())


# In[ ]:


orig = [img_id for img_id in ResNet_results.img_id 
        if 'f' not in img_id]
orig_imgs = ResNet_results[ResNet_results.img_id.isin(orig)]
selt_imgs = orig_imgs.sample(n=6)
img_list = selt_imgs['img_id'].values
true_centers = selt_imgs[['cx', 'cy']].values
pred_centers = selt_imgs[['cx_pred', 'cy_pred']].values

draw_roi(TRAIN_PATH, img_list, true_centers, pred_centers, 
         rows=2, cols=3, model_name="ResNet50")


# In[ ]:


selt_dots = orig_imgs.sample(n=20)
img_list = selt_dots['img_id'].values
true_centers = (selt_dots[['cx', 'cy']].values * 1330).astype(int)
pred_centers = (selt_dots[['cx_pred', 'cy_pred']].values * 1330).astype(int)

plt.figure(figsize = (5, 5))
plt.scatter(true_centers[:, 0], true_centers[:, 1], c='r', alpha=0.5, label='Human')
plt.scatter(pred_centers[:, 0], pred_centers[:, 1], c='b', alpha=0.5, label='CNN with Activations Func')
plt.axis('equal')
plt.legend()
plt.title("20 randomly selected images")
plt.show() 


# In[ ]:


# Save results
ResNet_results.to_csv("CNN2_results.csv")

