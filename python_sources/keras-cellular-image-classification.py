#!/usr/bin/env python
# coding: utf-8

# A concatenated EffNet approach to the Recursion Cellular Image Classification competition
# ======
# In this approach to the Recursion Cellular Image Classification competition, there are two datasets used:
# 1. The original Recursion Cellular Image Classification data for this competition
# 2. The Recursion Cellular Image Classification: 128 data from the first version of this kernel
# 
# **Note that the options GPU and Internet have to be checked.**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, sys, random,copy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, concatenate
from keras.utils import np_utils, Sequence
from PIL import Image
from PIL import ImageFilter
from sklearn.model_selection import train_test_split
get_ipython().system('pip install efficientnet')
import efficientnet
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


# After importing the necessary modules one loads the test data...

# In[ ]:


test_data = pd.read_csv("../input/recursion-cellular-image-classification/test.csv")
print("Shape of test_data:", test_data.shape)
test_data.head()


# ... and the training data into pandas dataframes.

# In[ ]:


train_data = pd.read_csv("../input/recursion-cellular-image-classification/train.csv")
print("Shape of train_data:", train_data.shape)
train_data.head()


# To load the data in memory efficient batches later on an image data generator is necessary. Since I started out with the original dataset that needed a custom generator, I kept and modified my self-written generator instead of using the keras ImageDataGenerator class.
# 
# To load the data from an id_code the get_input function is necessary, which uses the PIL Image interface and has two options for either loading from the training or test directory.

# In[ ]:


def get_input(id_code,site,train=True):
    if train==True:
        base_path = '../input/recursion-cellular-image-classification-128/train/train'
    else:
        base_path = '../input/recursion-cellular-image-classification-128/test/test'
    img = Image.open(base_path+"/"+id_code+"_s"+str(site)+".jpeg")
    return(img)


# The ImgGen inherits the keras.utils.Sequence class. It takes a pandas dataframe as input with the id_codes as index and the sirna values as the first column. The generator supports shuffling at startup and using a custom preprocessing function for image transformation purposes like augmentation. Its default mode is using training data, else it loads the test data.

# In[ ]:


class ImgGen (Sequence):
    def __init__(self, label_file, batch_size = 32,preprocess=(lambda x: x),train=True,shuffle=False):
        if shuffle==True:
            self.label_file = label_file.sample(frac=1)
        else:
            self.label_file = label_file
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.train = train
        self.x = list(self.label_file.index)
        if self.train==True:
            self.y = list(self.label_file[self.label_file.columns[0]])

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.train==True:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            
            x1 = [np.array(self.preprocess(get_input(id_code,1))).reshape(128,128,3) / 255 for id_code in batch_x] 
            x2 = [np.array(self.preprocess(get_input(id_code,2))).reshape(128,128,3) / 255 for id_code in batch_x]
            y = [sirna for sirna in batch_y]
            return [np.array(x1),np.array(x2)], np.array(y)
        else:
            x1 = [np.array(self.preprocess(get_input(id_code,1,train=False))).reshape(128,128,3) / 255 for id_code in batch_x] 
            x2 = [np.array(self.preprocess(get_input(id_code,2,train=False))).reshape(128,128,3) / 255 for id_code in batch_x] 
            return [np.array(x1),np.array(x2)]


# To make the model more robust during training image augmentation is used. Since I don't know which transformations result in misleading images only small modifications like blur, rotation and rank filters are used.

# In[ ]:


def augment(image):
    random_transform = random.randint(-1,4)
    if random_transform==0:
        image = image.rotate(random.randint(-5,5))
    if random_transform==1:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    if random_transform==2:
        image = image.filter(ImageFilter.RankFilter(size=3, rank=1))
    if random_transform==3:
        image = image.filter(ImageFilter.MedianFilter(size=3))
    if random_transform==4:
        image = image.filter(ImageFilter.MaxFilter(size=3))
    return image


# Since all cell images are taken from two sites there are two input channels that are processed by an EfficientNetB1. To shorten the fitting process the pretrained model from https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b1_imagenet_1000_notop.h5 is used which is trained on the imagenet data.
# 
# After pooling they are concatenated and result in a dense layer to output the classification result for all 1108 sirna classes. 

# In[ ]:


def create_model():
    effnet = efficientnet.EfficientNetB1(weights='imagenet',include_top=False,input_shape=(128, 128, 3))
    site1 = Input(shape=(128,128,3))
    site2 = Input(shape=(128,128,3))
    x = effnet(site1)
    x = GlobalAveragePooling2D()(x)
    x = Model(inputs=site1, outputs=x)
    y = effnet(site2)
    y = GlobalAveragePooling2D()(y)
    y = Model(inputs=site2, outputs=y)
    combined = concatenate([x.output, y.output])
    z = Dropout(0.5)(combined)
    z = Dense(1108, activation='softmax')(z)
    model = Model(inputs=[x.input, y.input], outputs=z)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    model.summary()
    
    return model

model = create_model()


# At first the model is trained and validated via instances of ImgGen on the whole training data. Afterwards the model with the best weights is refined on each cell type and then saved in 4 different checkpoints.

# In[ ]:


phases = ['All','HEPG2','HUVEC','RPE','U2OS']
batch_size = 128
test_size = 0.025

for phase in phases:
    
    print('Start phase %s.' % (phase))
    
    filepath = 'ModelCheckpoint_'+phase+'.h5'
    print("Set filepath to %s." % (filepath))
    
    callback = [
        ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        ]
    
    if phase != 'All':
        model.load_weights('ModelCheckpoint_All.h5')
        print("Successfully loaded weights of ModelCheckpoint_All.h5")
    
    if phase != 'All':
        label_file_source = train_data[train_data['id_code'].str.contains(phase)]
        print("Successfully created label_file_source for phase %s." % (phase))
    else:
        label_file_source = train_data
        print("Successfully created label_file_source for phase All.")
    
    label_file = pd.DataFrame(index=label_file_source['id_code'],data=list(label_file_source['sirna']),columns=['sirna'])
    train, val = train_test_split(label_file, test_size=test_size)
    train_gen = ImgGen(train,batch_size=batch_size,shuffle=True,preprocess=augment)
    val_gen = ImgGen(val,batch_size=batch_size,shuffle=True,preprocess=augment)
    
    history = model.fit_generator(train_gen, 
                              steps_per_epoch=len(train)//batch_size, 
                              epochs=25, 
                              verbose=1, 
                              validation_data=val_gen,
                              validation_steps=len(val)//batch_size,
                              callbacks=callback
                             )


# Finally each cell type on the test data is predicted separately. This is done by loading the corresponding refined weigths and then predicting on a filtered list.
# The 4 prediction arrays are then combined into the submission file.

# In[ ]:


submission = pd.DataFrame()
id_codes = []

for phase in phases[1:]:
    test_label_file = pd.DataFrame(index=(test_data[test_data['id_code'].str.contains(phase)]['id_code']))
    
    model.load_weights('ModelCheckpoint_'+phase+'.h5')
    print("Successfully loaded weights of ModelCheckpoint_%s.h5" % (phase))
    test_generator = ImgGen(test_label_file,batch_size=batch_size,train=False)
    if phase == phases[1]:
        predictions = model.predict_generator(test_generator,verbose=1)
    else:
        predictions = np.append(predictions,model.predict_generator(test_generator,verbose=1), axis=0)
    id_codes += list(test_label_file.index)
    
submission['id_code'] = id_codes
submission['sirna'] = predictions.argmax(axis=-1)
submission.to_csv("submission.csv",index=False)

print(pd.read_csv("submission.csv"))

