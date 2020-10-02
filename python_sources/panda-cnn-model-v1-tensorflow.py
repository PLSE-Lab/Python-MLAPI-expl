#!/usr/bin/env python
# coding: utf-8

# ## tensorflow CNN implimentation
# 
# Hi all,
# 
# this notebook is for anyone who like to start with tensorflow based model for this competition. this show how to use tensorflow hub for transfer learning and tf addons for metrics.
# 
# 
# utilizing tensorflow addons for CohenKappa
# 
# utilizing tensorflow hub for finetuning models
# 
# ### Note look at version 1 for output
# 
# Video
# https://www.youtube.com/playlist?list=PLz2_wbmGj3n6-fLQ9WmzQgKpXVWIG7ncX

# In[ ]:


import pandas as pd
import os
import matplotlib.pyplot as plt
import skimage.io
import random
import numpy as np
import cv2
import tensorflow_hub as hub
import tensorflow as tf


# In[ ]:


get_ipython().system('pip install tensorflow_addons')
import tensorflow_addons as tfa


# ## Data reading

# In[ ]:


train_df = pd.read_csv("/kaggle/input/prostate-cancer-grade-assessment/train.csv")
train_df.head()


# In[ ]:


train_img_list  = os.listdir("/kaggle/input/prostate-cancer-grade-assessment/train_images") 
train_mask_list = os.listdir("/kaggle/input/prostate-cancer-grade-assessment/train_label_masks")
print(train_mask_list[0])
# to get only the image id and not the mask extension
train_mask_list = [filename.split('_')[0] for filename in train_mask_list ]
print(train_mask_list[0])


# ## custom generator

# In[ ]:


def ImageGenerator(df,batch_size = 2,img_path = None,mask_path = None,n_classes=2,size=(256,256)):
    df_iterator = df.iterrows()
    while True:
        count = 0
        image_list = []
        mask_list = []
        label_list = []
        while count < batch_size:
            try:
                image_id, row = next(df_iterator)
                #print(image_id)
                image_path = img_path + image_id +'.tiff'
                mask_img_path = mask_path + image_id + "_mask.tiff"
                im = skimage.io.MultiImage(image_path)            
                label_list.append(tf.keras.utils.to_categorical(int(row.isup_grade), n_classes))
                image_list.append(cv2.resize(im[-1],size))
                count += 1
            except StopIteration:
                del df_iterator
                df_iterator = df.iterrows()
        yield np.array(image_list)/255.,np.array(label_list)
            
            


# ## Model creation

# In[ ]:


from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


# Get the model.
def model_dense_fn():
    
    n_classes = 6

    input_img = tf.keras.Input(shape=(512,512,3), name='color_sample')

    densenet121 = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_tensor=input_img, input_shape=None,
        pooling=None, classes=1000
    )
    glob = layers.GlobalAveragePooling2D()(densenet121.output)
    out = layers.Dense(n_classes,activation='softmax',  name='prediction')(glob)
    model = keras.Model(inputs=input_img, outputs=out)
    # Instantiate an optimizer.
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)   
    return model, optimizer, loss_fn


# In[ ]:


def model_hub_bit():
    
    n_classes = 6
    do_fine_tuning = True
    MODULE_HANDLE = "https://tfhub.dev/google/bit/s-r101x1/1"
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(512,512) + (3,)),
        hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(n_classes,activation='softmax',  name='prediction')
    ])
    model.build((None,)+(512,512)+(3,))
    model.summary()
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
    return model, optimizer, loss_fn


# In[ ]:


model, optimizer, loss_fn = model_hub_bit()


# In[ ]:


n_classes = 6
model.compile(
    optimizer=optimizer, loss=loss_fn, metrics=['accuracy',tfa.metrics.CohenKappa(num_classes=n_classes,weightage='quadratic')])


# In[ ]:


model.summary()


# ## Generator initialization

# In[ ]:


train_df.head()


# In[ ]:


df_train_corrected = train_df.sample(frac=1).set_index("image_id")
BATCH = 16

train_gen = ImageGenerator(df_train_corrected[:-1000],batch_size = BATCH,img_path = "/kaggle/input/prostate-cancer-grade-assessment/train_images/",
                   mask_path = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/",
                   n_classes=6,size=(512,512))
valid_gen = ImageGenerator(df_train_corrected[-1000:],batch_size = BATCH,img_path = "/kaggle/input/prostate-cancer-grade-assessment/train_images/",
                   mask_path = "/kaggle/input/prostate-cancer-grade-assessment/train_label_masks/",
                   n_classes=6,size=(512,512))


# In[ ]:


checkpoint_filepath = 'check/train'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_cohen_kappa",
    save_best_only=True)


# In[ ]:


EPOCH = 30
v = model.fit_generator(
    train_gen, steps_per_epoch=9000/BATCH, epochs=EPOCH,
    validation_data=valid_gen, validation_steps=1000/BATCH, callbacks = [model_checkpoint_callback]
)


# ## Testing

# In[ ]:


def ImageGenerator_test(df,batch_size = 2,img_path = None,mask_path = None,n_classes=2,size=(256,256)):
    df_iterator = df.iterrows()
    
    
    while True:
        count = 0
        image_list = []
        mask_list = []
        image_id_list = []
        while count < batch_size:
            try:
                image_id, row = next(df_iterator)

                image_path = img_path + image_id +'.tiff'

                image_id_list.append(image_id)
                im = skimage.io.MultiImage(image_path)
                
                image_list.append(cv2.resize(im[-1],size))

                count += 1
            except StopIteration:
                del df_iterator
                df_iterator = df.iterrows()

        yield np.array(image_list)/255.,image_id_list


# In[ ]:


folder_name = 'train'
if os.path.exists(f'../input/prostate-cancer-grade-assessment/{folder_name}_images'):
    sub = df_train_corrected[-1000:]#pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv') 
    #sub = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv') 
    test_img_list  = os.listdir(f"../input/prostate-cancer-grade-assessment/{folder_name}_images")
    
    df_test = df_train_corrected[-1000:]#pd.read_csv(f"../input/prostate-cancer-grade-assessment/{folder_name}.csv",index_col="image_id")
    BATCH = 16
    test_gen = ImageGenerator_test(df_test,batch_size = BATCH,img_path = f"../input/prostate-cancer-grade-assessment/{folder_name}_images/",
                   mask_path = "../input/prostate-cancer-grade-assessment/test_label_masks/",n_classes=6,size=(512,512))

    prediction_list = []
    image_id_list = []
    for i in range (int(1100/BATCH)):
        #print('testing')
        v = next(test_gen)
        image_id_list.extend(v[1])

        prediction = np.argmax(model.predict(v[0]),axis = 1)
        prediction_list.extend(prediction)


    d = {'image_id': image_id_list, 'isup_grade': prediction_list}
    df = pd.DataFrame(data=d)
    df.head()    
    #sub = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')   
    #sub.drop(labels= 'isup_grade',axis=1,inplace=True)
    dff = pd.merge(sub,df,on='image_id')
    dff.drop_duplicates(subset ="image_id", 
                         keep = "first",inplace = True) 
    
    sub = dff


# In[ ]:


original = sub['isup_grade_x'].values
prediction = sub['isup_grade_y'].values


# In[ ]:


from sklearn.metrics import cohen_kappa_score


# In[ ]:


qwk = cohen_kappa_score(original, prediction, labels=None, weights= 'quadratic', sample_weight=None)
print(qwk)


# In[ ]:





# In[ ]:


get_ipython().system('zip -r check.zip /kaggle/working/check/')

