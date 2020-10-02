#!/usr/bin/env python
# coding: utf-8

# > > ## Diabetic Retinopath detection with CNN
# ### ---
# ### Diabetic retinopathy affects blood vessels in the light-sensitive tissue called the retina that lines the back of the eye. It is the most common cause of vision loss among people with diabetes and the leading cause of vision impairment and blindness among working-age adults. It don't have any earaly symtoms. As of now, Retena photography is a way to detect the stage of Blindness. Automating it with ml, will help a lot in health domain. 
#  
# ### ---------------------------------------
# ### 1. [Import Required Libraries](#1)
# ### 2. [Loading Data ](#2)
# ### 3. [Data Visualization](#3)
# ### 4. [Train and Test dataset](#4)
# ### 5. [Data Pre-Processing](#6)
# ### 6. [Image Data Generator](#7)
# ### 7. [Model Architecture Design](#8)
# ### 8. [Keras Callback Funcations](#9)
# ### 9. [Transfer Learning](#10)
# ### 10. [Validation Accuracy & Loss](#11)
# ### 11. [Validation Accuracy](#12)
# ### 12. [Test-Time Augmentation](#13)
# ### 13. [Visualization Test Result](#14)
# ### ------------------------------------
# 
# 
# ## Stages Of Diabetic Retinopathy
# ### - NO DR
# ### - Non-Proliferative DR (NPDR)
# ### - Proliferative DR (PDR)

# <a id="1"></a> 
# # Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import os
import cv2
import PIL
import gc
import psutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import set_random_seed
from tqdm import tqdm
from math import ceil
import math
import sys
import gc

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

from keras.activations import softmax
from keras.activations import elu
from keras.activations import relu
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

gc.enable()

print(os.listdir("../input/"))


# <a id="2"></a>
# ## Exploratory Data Analysis
# #### - Loading Data 
# #### - Data Disribution
# #### - Data Visualization
# 

# In[ ]:


SEED = 7
np.random.seed(SEED)
set_random_seed(SEED)
dir_path = "../input/drdataset2/DR/"
IMG_DIM = 256  # 224 399 #
BATCH_SIZE = 12
CHANNEL_SIZE = 3
NUM_EPOCHS = 17
TRAIN_DIR = 'train_image'
TEST_DIR = 'test_image'
FREEZE_LAYERS = 2  # freeze the first this many layers for training
CLASSS = {0: "No DR", 1: "Non-Proliferative DR", 2: "Proliferative DR", 3: "Severe", 4: "Proliferative DR"}


# <a id="2"></a>
# ### Loading Data

# In[ ]:


df_train = pd.read_csv(os.path.join(dir_path, "train.csv"))
df_test = pd.read_csv(os.path.join(dir_path, "test.csv"))
NUM_CLASSES = df_train['diagnosis'].nunique()


# In[ ]:


print("Training set has {} samples and {} classes.".format(df_train.shape[0], df_train.shape[1]))
print("Testing set has {} samples and {} classes.".format(df_test.shape[0], df_test.shape[1]))


# <a id="6"></a>
# ### Split DataSet

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(df_train.id_code, df_train.diagnosis, test_size=0.15,
                                                    random_state=SEED, stratify=df_train.diagnosis)


# <a id="3"></a>
# ## Data Visualization
# 

# In[ ]:


chat_data = df_train.diagnosis.value_counts()
chat_data.plot(kind='bar');
plt.title('Sample Per Class');
plt.show()
plt.pie(chat_data, autopct='%1.1f%%', shadow=True, labels=["No DR", "Non-Proliferative DR (NPDR)", "Proliferative DR (PDR)"])
plt.title('Per class sample Percentage');
plt.show()


# ### Train and Test dataset 
# 

# In[ ]:


# Train & Test samples ratio
# Plot Data
labels = 'Train', 'Test'
sizes = df_train.shape[0], df_test.shape[0]
colors = 'lightskyblue', 'lightcoral'
# Plot
plt.figure(figsize=(7, 5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
plt.show()


# In[ ]:


def draw_img(imgs, target_dir, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(16, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = os.path.join(dir_path, f"{target_dir}/{row['id_code']}.png")
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axis[row, col].imshow(img)
    plt.suptitle(class_label)
    plt.show()


# In[ ]:


CLASS_ID = 0
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_image', CLASSS[CLASS_ID])


# In[ ]:


CLASS_ID = 1
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_image', CLASSS[CLASS_ID])


# In[ ]:


CLASS_ID = 2
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_image', CLASSS[CLASS_ID])


# In[ ]:


CLASS_ID = 'Test DataSet'
draw_img(df_test.sample(12, random_state=SEED), 'test_image', CLASS_ID)


# <a id="6"></a>
# ## Max Min Height and Width

# In[ ]:


def check_max_min_img_height_width(df, img_dir):
    max_Height , max_Width =0 ,0
    min_Height , min_Width =sys.maxsize ,sys.maxsize 
    for idx, row in df.iterrows():
        imgPath=os.path.join(dir_path,f"{img_dir}/{row['id_code']}.png") 
        img=cv2.imread(imgPath)
        H,W=img.shape[:2]
        max_Height=max(H,max_Height)
        max_Width =max(W,max_Width)
        min_Height=min(H,min_Height)
        min_Width =min(W,min_Width)
    return max_Height, max_Width, min_Height, min_Width


# In[ ]:


check_max_min_img_height_width(df_train, TRAIN_DIR)


# In[ ]:


check_max_min_img_height_width(df_test, TEST_DIR)


# <a id="7"></a>
# ## GrayScale Images
# #### Converting the Ratina Images into Grayscale. So, we can understand the regin or intest .

# In[ ]:


# Display some random images from Data Set with class categories ing gray
figure = plt.figure(figsize=(20, 16))
for target_class in (y_train.unique()):
    for i, (idx, row) in enumerate(
            df_train.loc[df_train.diagnosis == target_class].sample(5, random_state=SEED).iterrows()):
        ax = figure.add_subplot(5, 5, target_class * 5 + i + 1)
        imagefile = f"../input/drdataset2/DR/train_image/{row['id_code']}.png"
        img = cv2.imread(imagefile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_DIM, IMG_DIM))
        plt.imshow(img, cmap='gray')
        ax.set_title(CLASSS[target_class])


# ## Gaussian Blur

# In[ ]:




def draw_img_light(imgs, target_dir, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = os.path.join(dir_path, f"{target_dir}/{row['id_code']}.png")
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_DIM, IMG_DIM))
        img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , IMG_DIM/10) ,-4 ,128) # the trick is to add this line
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axis[row, col].imshow(img, cmap='gray')
    plt.suptitle(class_label)
    plt.show()


# In[ ]:


CLASS_ID = 0
draw_img_light(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_image', CLASSS[CLASS_ID])


# In[ ]:


CLASS_ID = 1
draw_img_light(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_image', CLASSS[CLASS_ID])


# In[ ]:


CLASS_ID = 2
draw_img_light(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_image', CLASSS[CLASS_ID])


# 
# <a id="2"></a>
# # Pre-Processing
# #### - Padding (removal) 
# #### - Gaussian Blur
# #### - Auto Cropping

# In[ ]:


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #       print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #       print(img.shape)
            return img


# In[ ]:


def circle_crop(img):   
    """
    Create circular crop around eye centre    
    """    
    
    #img = cv2.imread(img)
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))  
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    
    return img 


# In[ ]:


def load_ben_color(image, sigmaX=25):
    #image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    #image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    image = circle_crop(image)  
    return image


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNUM_SAMP=7\nfig = plt.figure(figsize=(25, 16))\nfor class_id in sorted(y_train.unique()):\n    for i, (idx, row) in enumerate(df_train.loc[df_train[\'diagnosis\'] == class_id].sample(NUM_SAMP, random_state=SEED).iterrows()):\n        ax = fig.add_subplot(5, NUM_SAMP, class_id * NUM_SAMP + i + 1, xticks=[], yticks=[])\n        path=f"../input/drdataset2/DR/train_image/{row[\'id_code\']}.png"\n        image = load_ben_color(cv2.imread(path))\n        plt.imshow(image)\n        ax.set_title(\'%d-%d-%s\' % (class_id, idx, row[\'id_code\']) )')


# <a id="8"></a>
# # Data Augmentation
# 

# In[ ]:


# print("available RAM:", psutil.virtual_memory())
gc.collect()
# print("available RAM:", psutil.virtual_memory())

df_train.id_code = df_train.id_code.apply(lambda x: x + ".png")
df_test.id_code = df_test.id_code.apply(lambda x: x + ".png")
#df_train['diagnosis'] = df_train['diagnosis'].astype('str')
#df_test['diagnosis'] = df_test['diagnosis'].astype('str')


# In[ ]:


df_train = pd.concat([df_train,pd.get_dummies(df_train['diagnosis'], prefix='diagnosis')],axis=1)
df_train.drop(['diagnosis'],axis=1, inplace=True)
df_train['diagnosis_0'] = df_train['diagnosis_0'].astype('str')
df_train['diagnosis_1'] = df_train['diagnosis_1'].astype('str')
df_train['diagnosis_2'] = df_train['diagnosis_2'].astype('str')


# In[ ]:


df_train


# In[ ]:


df_test = pd.concat([df_test,pd.get_dummies(df_test['diagnosis'], prefix='diagnosis')],axis=1)
df_test.drop(['diagnosis'],axis=1, inplace=True)
df_test['diagnosis_0'] = df_test['diagnosis_0'].astype('str')
df_test['diagnosis_1'] = df_test['diagnosis_1'].astype('str')
df_test['diagnosis_2'] = df_test['diagnosis_2'].astype('str')


# In[ ]:


# Creating the imageDatagenerator Instance 
datagenerator=ImageDataGenerator(#rescale=1./255,
                                        validation_split=0.15, 
                                        horizontal_flip=True,
                                        vertical_flip=True, 
                                        #rotation_range=40, 
                                        #zoom_range=0.2, 
                                        preprocessing_function=load_ben_color,
                                        shear_range=0.1,
                                        fill_mode='nearest')


# In[ ]:


#imgPath = f"../input/aptos2019-blindness-detection/train_images/cd54d022e37d.png"
imgPath = f"../input/drdataset2/DR/train_image/02685f13cefd.png"
# Loading image
img = cv2.imread(imgPath)
#img = x_train[0]
img = cv2.resize(img, (IMG_DIM, IMG_DIM))
data = img_to_array(img)
samples =np.expand_dims(data, 0)
i=5
it=datagenerator.flow(samples , batch_size=1)
for i in range(5):
    plt.subplot(230 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()


# 
# ## Image Data Generator

# In[ ]:


train_datagen=ImageDataGenerator(rescale=1./255,
                                        validation_split=0.15, 
                                        horizontal_flip=True,
                                        vertical_flip=True, 
                                        #rotation_range=40, 
                                        #zoom_range=0.2, 
                                        preprocessing_function=load_ben_color,
                                        shear_range=0.1,
                                        #fill_mode='nearest'
                                )
test_datagen=ImageDataGenerator(rescale=1./255, preprocessing_function=load_ben_color)


# In[ ]:


train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    #directory="../input/aptos2019-blindness-detection/train_images/",
                                                    directory="../input/drdataset2/DR/train_image/",
                                                    x_col="id_code",
                                                    y_col=["diagnosis_0","diagnosis_1","diagnosis_2"],
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="raw",
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    subset='training',
                                                    shuffle=True,
                                                    seed=SEED,
                                                    )
valid_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    #directory="../input/aptos2019-blindness-detection/train_images/",
                                                    directory="../input/drdataset2/DR/train_image/",                                                    
                                                    x_col="id_code",
                                                    y_col=["diagnosis_0","diagnosis_1","diagnosis_2"],
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="raw",
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    subset='validation',
                                                    shuffle=True,
                                                    seed=SEED
                                                    )

#del x_train
#del x_test
#del y_train
#del y_test
gc.collect()
#  color_mode= "grayscale",


# In[ ]:


test_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
                                                #directory="../input/aptos2019-blindness-detection/train_images/",
                                                directory="../input/drdataset2/DR/test_image/",                                                    
                                                x_col="id_code",
                                                y_col=["diagnosis_0","diagnosis_1","diagnosis_2"],
                                                batch_size=1,
                                                class_mode="raw",
                                                target_size=(IMG_DIM, IMG_DIM),
                                                shuffle=False,
                                                )


# # Transfer Learning on ResNet50

# In[ ]:


def create_resnet(img_dim, CHANNEL, n_class):
    input_tensor = Input(shape=(img_dim, img_dim, CHANNEL))
    base_model = ResNet50(weights=None, include_top= False, input_tensor=input_tensor)
    base_model.load_weights('../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(1024, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation=elu)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(n_class, activation='softmax', name="Output_Layer")(x)
    model_resnet = Model(input_tensor, output_layer)

    return model_resnet

gc.collect()
model_resnet = create_resnet(IMG_DIM, CHANNEL_SIZE, NUM_CLASSES)
model_resnet.summary()


# ### EarlyStopping and Learning Rate

# In[ ]:


early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
# Reducing the Learning Rate if result is not improving. 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=3, factor=0.2, min_lr=1e-8, mode='auto',
                              verbose=1)
NUB_TRAIN_STEPS = train_generator.n // train_generator.batch_size
NUB_VALID_STEPS = valid_generator.n // valid_generator.batch_size

NUB_TRAIN_STEPS, NUB_VALID_STEPS


# In[ ]:


model_resnet.layers[0].trainable =False
lr = 1e-3
optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr=lr, decay=0.01) 
model_resnet.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# model.summary()
gc.collect()

model_resnet.fit_generator(generator=train_generator,
                                     steps_per_epoch=NUB_TRAIN_STEPS,
                                     validation_data=valid_generator,
                                     validation_steps=NUB_VALID_STEPS,
                                     epochs=5,
                                     #use_multiprocessing=True,
                                     #workers=3,
                                     shuffle=True, 
                                     #callbacks=[early_stop, reduce_lr],
                                     verbose=1)
#gc.collect()


# In[ ]:


# # Layers 
# for i, lay in enumerate(model_resnet.layers):
#     print(i,lay.name)
# Training All Layers

for layers in model_resnet.layers[10:-1]:
    layers.trainable = True
    
    
lr = 1e-3
optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr=lr, decay=0.01) 
model_resnet.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# model.summary()
gc.collect()


# In[ ]:


history1 = model_resnet.fit_generator(generator=train_generator,
                                     steps_per_epoch=NUB_TRAIN_STEPS,
                                     validation_data=valid_generator,
                                     validation_steps=NUB_VALID_STEPS,
                                     epochs=NUM_EPOCHS,
                                     #use_multiprocessing=True,
                                     #workers=3,
                                     shuffle=True, 
                                     callbacks=[early_stop, reduce_lr],
                                     verbose=1)
#gc.collect()


# In[ ]:


model_resnet.save("resnet_600.h5")
model_resnet.save_weights("resnet_600_weights.h5")


#from keras.models import model_from_json
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)

#model_yaml=model.to_yaml()
#with open("model.yaml","w") as yaml_file:
#    yaml_file.write(model_yaml)
#model.save_weights("model.h5")


# In[ ]:


score, acc = model_resnet.evaluate_generator(train_generator, steps=150, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
print(score, acc)


# <a id="8"></a>
# # CNN-Model Architecture Design

# In[ ]:


def design_model():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(2, 2), input_shape=[IMG_DIM, IMG_DIM, CHANNEL_SIZE], activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(filters=64, kernel_size=(2, 2), activation=relu))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=256, activation=relu))
    #model.add(Dropout(rate=0.2))
    model.add(Dense(units=512, activation=relu))
    #model.add(Dropout(rate=0.2))
    model.add(Dense(3, activation='softmax'))
    return model

gc.collect()

model = design_model()
model.summary()


# ### Compile model

# In[ ]:


model.compile(optimizer=Adam(lr = 0.001) , loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history2 = model.fit_generator(generator=train_generator,
                    validation_data=valid_generator,
                    steps_per_epoch=NUB_TRAIN_STEPS,
                    validation_steps=NUB_VALID_STEPS,
                    verbose=1,
                    use_multiprocessing=True,
                    workers=3,
                    callbacks=[early_stop, reduce_lr],
                    shuffle=True,
                    max_queue_size=10,
                    epochs=NUM_EPOCHS)


# In[ ]:


model.save("CNN_new.h5")
model.save_weights("CNN_new_weights.h5")


# In[ ]:


score, acc = model.evaluate_generator(test_generator, steps=1136, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
print(score, acc)


# # Display Validation Accuracy & Loss
# 

# ## Accuracy ResNet50 vs CNN

# In[ ]:


val_acc1 = history1.history['val_accuracy']
val_acc2 = history2.history['val_accuracy']

plt.plot(val_acc1, label="accuracy")
plt.plot(val_acc2)
plt.xlabel("Epoch")
plt.ylabel("Val Accuracy")
plt.legend(['ResNet50', 'CNN'])
plt.plot(np.argmax(history1.history["val_accuracy"]), np.max(history1.history["val_accuracy"]), marker="x", color="r",
         label="best model")
plt.show()


# ### Resnet

# In[ ]:


accu = history1.history['accuracy']
val_acc = history1.history['val_accuracy']

plt.plot(accu, label="accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['accuracy', 'val_accuracy'])
plt.plot(np.argmax(history1.history["val_accuracy"]), np.max(history1.history["val_accuracy"]), marker="x", color="r",
         label="best model")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(history1.history["loss"], label="loss")
plt.plot(history1.history["val_loss"], label="val_loss")
plt.plot(np.argmin(history1.history["val_loss"]), np.min(history1.history["val_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


# ### CNN

# In[ ]:


accu = history2.history['accuracy']
val_acc = history2.history['val_accuracy']

plt.plot(accu, label="accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['accuracy', 'val_accuracy'])
plt.plot(np.argmax(history2.history["val_accuracy"]), np.max(history2.history["val_accuracy"]), marker="x", color="r",
         label="best model")
plt.show()


# In[ ]:


plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(history2.history["loss"], label="loss")
plt.plot(history2.history["val_loss"], label="val_loss")
plt.plot(np.argmin(history2.history["val_loss"]), np.min(history2.history["val_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # User interface

# In[ ]:


from keras.models import load_model
classifier= load_model("../input/drmodels/model_saves/resnet_2048.h5")

Patient101 = "../input/drdataset3/DR/train_images/4e54ccfd49b2.png"
Patient102 = "../input/drdataset3/DR/train_images/7a06ea127e02.png"
Patient103 = "../input/drdataset3/DR/train_images/1d3e9b939732.png"

Patient201 = "../input/drdataset3/DR/train_images/059bc89df7f4.png"
Patient202 = "../input/drdataset3/DR/train_images/435d900fa7b2.png"
Patient203 = "../input/drdataset3/DR/train_images/1006345f70b7.png"

Patient301 = "../input/drdataset3/DR/train_images/5b3e7197ac1c.png"
Patient302 = "../input/drdataset3/DR/train_images/7b211d8bd249.png"
Patient303 = "../input/drdataset3/DR/train_images/dad71ba27a9b.png"


# In[ ]:


#del df_class
df_class = pd.read_csv("../input/drdataset5/DR_categorical/0.csv")


# In[ ]:


df_class.id_code = df_class.id_code.apply(lambda x: x + ".png")
df_class['id_code'] = df_class['id_code'].astype('str')


# In[ ]:


#df_class = pd.concat([df_class,pd.get_dummies(df_class['diagnosis'], prefix='diagnosis')],axis=1)
#df_class['diagnosis_0']=list(df_class['diagnosis']-1)
#df_class['diagnosis_1']=list(df_class['diagnosis'])
#df_class['diagnosis_2']=list(df_class['diagnosis']-1)
#df_class.drop(['diagnosis'],axis=1, inplace=True)


# In[ ]:


#df_class['diagnosis'] = df_class['diagnosis'].astype('str')
#df_class = pd.concat([df_class,pd.get_dummies(df_class['diagnosis'], prefix='diagnosis')],axis=1)
#df_class.drop(['diagnosis'],axis=1, inplace=True)
#df_class['diagnosis_0'] = df_class['diagnosis_0'].astype('str')
#df_class['diagnosis_1'] = df_class['diagnosis_1'].astype('str')
#df_class['diagnosis_2'] = df_class['diagnosis_2'].astype('str')


# In[ ]:


dummy_datagen=ImageDataGenerator(rescale=1./255, preprocessing_function=load_ben_color)
dummy_generator = dummy_datagen.flow_from_dataframe(dataframe=df_class,
                                                    #directory="../input/aptos2019-blindness-detection/train_images/",
                                                    directory="../input/drdataset5/DR_categorical/0/",                                                    
                                                    x_col="id_code",
                                                    #y_col="diagnosis",
                                                    #y_col=["diagnosis_0","diagnosis_1","diagnosis_2"],
                                                    batch_size=1,
                                                    class_mode=None,
                                                    target_size=(IMG_DIM, IMG_DIM),
                                                    #shuffle=False,
                                                   seed = 7
                                                  )


# ## Generator Pred

# In[ ]:


tta_steps = 1
preds_tta = []
for i in tqdm(range(tta_steps)):
    dummy_generator.reset()
    preds = classifier.predict_generator(generator=dummy_generator, steps=ceil(df_class.shape[0]))
    #     print('Before ', preds.shape)
    preds_tta.append(preds)
#     print(i,  len(preds_tta))


# In[ ]:


final_pred = np.mean(preds_tta, axis=0)
predicted_class_indices = np.argmax(final_pred, axis=1)
    len(predicted_class_indices)


# In[ ]:


#Label Dictionary
label_maps = {0: 'No DR', 1: 'Non-Proliferative DR', 2: 'Proliferative DR'}
label =[label_maps[k] for k in predicted_class_indices]

print(label)


# ## Single pred

# Fundus_img = cv2.imread(Patient103)
# Fundus_img = cv2.resize(Fundus_img,(IMG_DIM,IMG_DIM))
# Fundus_img = np.expand_dims(Fundus_img, axis=0) 
# dummy_datagen=ImageDataGenerator(rescale=1./255, preprocessing_function=load_ben_color)
# dummy_generator = dummy_datagen.flow(Fundus_img, y=None, batch_size=1, seed=7)
