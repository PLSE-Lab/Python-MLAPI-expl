#!/usr/bin/env python
# coding: utf-8

# ## Additional changes
# I have used CNN as classifier, but this kernel use CAM(Class Activation Map) as classifier and Activation feature map. 
# Most of them are same, Little different at Network. and than we can see how CAM activate.
# 
# 
# ## Introduction
# Hi, It's my first dataset kernel. <br>
# This kernel forked from WM-811k Wafermap[https://www.kaggle.com/ashishpatel26/wm-811k-wafermap].<br>
# This dataset has various wafer resolution with class imbalanced. so I just consider specific subset wafer that has 26x26 resolution.<br> 
# and solve imbalance problem using 2D convolutional autoencoder. then, classfy faulty case labels.

# In[1]:


import os
from os.path import join

import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
import keras
from keras import layers, Input, models
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

datapath = join('data', 'wafer')

print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")


# ### Read data

# In[2]:


df=pd.read_pickle("../input/LSWMD.pkl")
df.info()


# * The dataset comprises **811,457 wafer maps**, along with additional information such as **wafer die size**, **lot name** and **wafer index**. 
# 
# * The training / test set were already split by domain experts, but in this kernel we ignore this info and we re-divided the dataset into training set and test set by hold-out mehtod which will be introduced in later section.

# >Target distribution

# In[3]:


df.head()


# In[4]:


df.tail()


# * The dataset were collected from **47,543 lots** in real-world fab. However, **47,543 lots x 25 wafer/lot =1,157,325 wafer maps ** is larger than **811,457 wafer maps**. 
# 
# * Let's see what happened. 

# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


uni_Index=np.unique(df.waferIndex, return_counts=True)
plt.bar(uni_Index[0],uni_Index[1], color='gold', align='center', alpha=0.5)
plt.title(" wafer Index distribution")
plt.xlabel("index #")
plt.ylabel("frequency")
plt.xlim(0,26)
plt.ylim(30000,34000)
plt.show()


# * The figure shows that not all lots have perfect 25 wafer maps and it may caused by **sensor failure** or other unknown problems.
# 
# * Fortunately, we do not need wafer index feature in our classification so we can just drop the variable. 

# In[6]:


df = df.drop(['waferIndex'], axis = 1)


# * We can not get much information from the wafer map column but we can see the die size for each instance is different. 
# 
# * We create a new variable **'waferMapDim'** for wafer map dim checking.
# 

# In[7]:


def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)
df.sample(5)


# ## Get sub wafer with specific resolution.
# get wafers have (26, 26) resolution. rarrange wafer nd-array with fautly case label.<br>
# some wafer has null label, skip it.

# In[8]:


sub_df = df.loc[df['waferMapDim'] == (26, 26)]
sub_wafer = sub_df['waferMap'].values

sw = np.ones((1, 26, 26))
label = list()

for i in range(len(sub_df)):
    # skip null label
    if len(sub_df.iloc[i,:]['failureType']) == 0:
        continue
    sw = np.concatenate((sw, sub_df.iloc[i,:]['waferMap'].reshape(1, 26, 26)))
    label.append(sub_df.iloc[i,:]['failureType'][0][0])


# In[9]:


x = sw[1:]
y = np.array(label).reshape((-1,1))


# In[10]:


mask_x = np.zeros((24, 24))
dummy_x = cv2.resize(x[0], (24,24))
mask_x[dummy_x == 1] = 1 
mask_x[dummy_x == 2] = 1 
mask_x = mask_x.reshape((1, 24,24))


# In[11]:


# check dimension
print('x shape : {}, y shape : {}'.format(x.shape, y.shape))


# plot 1st data for check.

# In[12]:


# plot 1st data
plt.imshow(x[0])
plt.show()

# check faulty case
print('Faulty case : {} '.format(y[0]))


# We will use 2D Convolutional Autoencoder, extend dimension for channel.

# In[13]:


#add channel
x = x.reshape((-1, 26, 26, 1))


# Make faulty case list, and check how classes imbalanced.

# In[14]:


faulty_case = np.unique(y)
print('Faulty case list : {}'.format(faulty_case))


# In[15]:


faulty_case_dict =dict()


# In[16]:


for i, f in enumerate(faulty_case) :
    print('{} : {}'.format(f, len(y[y==f])))
    faulty_case_dict[i] = f


# Wafer data's each pixels have a categorical variable that express 0 : not wafer, 1 : normal, 2 : faulty. <br>
# Extend extra dimension with one-hot-encoded categorical data as channel. <br>
# **that idea from Data Science & Business Analytics Lab, School of Industrial Management Engineering, College of Engineering, Korea University**[http://dsba.korea.ac.kr/main]

# In[17]:


# One-hot-Encoding faulty categorical variable as channel
new_x = np.zeros((len(x), 26, 26, 3))

for w in range(len(x)):
    for i in range(26):
        for j in range(26):
            new_x[w, i, j, int(x[w, i, j])] = 1


# In[18]:


#check new x dimension
new_x.shape


# ## Convolutional Autoencoder for augmentation.
# As solving class imbalanced problem, we need for data augmentation. <br>
# The wafer data is image data. so we use convolutional autoencoder.

# In[19]:


# parameter
epoch=15
batch_size=1024


# In[20]:


# Encoder
input_shape = (26, 26, 3)
input_tensor = Input(input_shape)
encode = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_tensor)

latent_vector = layers.MaxPool2D()(encode)

# Decoder
decode_layer_1 = layers.Conv2DTranspose(64, (3,3), padding='same', activation='relu')
decode_layer_2 = layers.UpSampling2D()
output_tensor = layers.Conv2DTranspose(3, (3,3), padding='same', activation='sigmoid')

# connect decoder layers
decode = decode_layer_1(latent_vector)
decode = decode_layer_2(decode)

ae = models.Model(input_tensor, output_tensor(decode))
ae.compile(optimizer = 'Adam',
              loss = 'mse',
             )


# Check summary

# In[21]:


ae.summary()


# In[22]:


# start train
ae.fit(new_x, new_x,
       batch_size=batch_size,
       epochs=epoch,
       verbose=2)


# In[23]:


# Make encoder model with part of autoencoder model layers
encoder = models.Model(input_tensor, latent_vector)


# In[24]:


# Make decoder model with part of autoencoder model layers
decoder_input = Input((13, 13, 64))
decode = decode_layer_1(decoder_input)
decode = decode_layer_2(decode)

decoder = models.Model(decoder_input, output_tensor(decode))


# In[25]:


# Encode original faulty wafer
encoded_x = encoder.predict(new_x)


# In[26]:


# Add noise to encoded latent faulty wafers vector.
noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64))


# In[27]:


# check original faulty wafer data
plt.imshow(np.argmax(new_x[3], axis=2))


# In[28]:


# check new noised faulty wafer data
noised_gen_x = np.argmax(decoder.predict(noised_encoded_x), axis=3)
plt.imshow(noised_gen_x[3])


# In[29]:


# check reconstructed original faulty wafer data
gen_x = np.argmax(ae.predict(new_x), axis=3)
plt.imshow(gen_x[3])


# ## Data augmentation
# We made convolutional autoencoder model for data augmentation.<br>
# I just want data has 2000 samples for each case. Let's augment data for all faulty case.

# In[30]:


# augment function define
def gen_data(wafer, label):
    # Encode input wafer
    encoded_x = encoder.predict(wafer)
    
    # dummy array for collecting noised wafer
    gen_x = np.zeros((1, 26, 26, 3))
    
    # Make wafer until total # of wafer to 2000
    for i in range((2000//len(wafer)) + 1):
        noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64)) 
        noised_gen_x = decoder.predict(noised_encoded_x)
        gen_x = np.concatenate((gen_x, noised_gen_x), axis=0)
    # also make label vector with same length
    gen_y = np.full((len(gen_x), 1), label)
    
    # return date without 1st dummy data.
    return gen_x[1:], gen_y[1:]


# In[31]:


# Augmentation for all faulty case.
for f in faulty_case : 
    # skip none case
    if f == 'none' : 
        continue
    
    gen_x, gen_y = gen_data(new_x[np.where(y==f)[0]], f)
    new_x = np.concatenate((new_x, gen_x), axis=0)
    y = np.concatenate((y, gen_y))


# In[32]:


print('After Generate new_x shape : {}, new_y shape : {}'.format(new_x.shape, y.shape))


# In[33]:


for f in faulty_case :
    print('{} : {}'.format(f, len(y[y==f])))


# In[34]:


# choice index without replace.
none_idx = np.where(y=='none')[0][np.random.choice(len(np.where(y=='none')[0]), size=11000, replace=False)]


# In[35]:


# delete choiced index data.
new_x = np.delete(new_x, none_idx, axis=0)
new_y = np.delete(y, none_idx, axis=0)


# In[36]:


print('After Delete "none" class new_x shape : {}, new_y shape : {}'.format(new_x.shape, new_y.shape))


# In[37]:


for f in faulty_case :
    print('{} : {}'.format(f, len(new_y[new_y==f])))


# In[38]:


# make string label data to numerical data
for i, l in enumerate(faulty_case):
    new_y[new_y==l] = i
    
# one-hot-encoding
new_y = to_categorical(new_y)


# In[39]:


# split data train, test
x_train, x_test, y_train, y_test = train_test_split(new_x, new_y,
                                                    test_size=0.33,
                                                    random_state=2019)


# In[40]:


print('Train x : {}, y : {}'.format(x_train.shape, y_train.shape))
print('Test x: {}, y : {}'.format(x_test.shape, y_test.shape))


# ## Class Activation Map Model
# The data is ready. As wafer data is image. simply use cnn for classification.<br>
# 
# ### Make model
# define create model function, because we will validate model with sklearn kfold cross validation.

# In[41]:


input_shape = (26, 26, 3)
input_tensor = Input(input_shape)

def create_model():
    global input_tensor
    
    conv = layers.Conv2D(32, (3,3), activation='relu', name='conv1')(input_tensor)
    padding = layers.ZeroPadding2D(padding=(1, 1), name='padding1')(conv)
    conv = layers.Conv2D(64, (3,3), activation='relu', name='conv2')(padding)
    padding = layers.ZeroPadding2D(padding=(1, 1), name='padding2')(conv)
    conv = layers.Conv2D(128, (3,3), activation='relu', name='conv3')(padding)
    padding = layers.ZeroPadding2D(padding=(1, 1), name='padding3')(conv)
    conv_out = layers.Conv2D(256, (3,3), activation='relu', name='conv4')(padding)

    gap_layer = layers.GlobalAveragePooling2D(name='GAP')
    output_layer = layers.Dense(9, activation='softmax', name='output')

    aver_pool = gap_layer(conv_out)
    output_tensor = output_layer(aver_pool)

    model = models.Model(input_tensor, output_tensor)

    model.compile(optimizer='Adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

    return model


# ### Cross validate model
# Using sklearn KFold Cross validation, we validate our simple cnn.

# In[42]:


# Make keras model to sklearn classifier.
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1024, verbose=2) 
# 3-Fold Crossvalidation
kfold = KFold(n_splits=3, shuffle=True, random_state=2019) 
results = cross_val_score(model, x_train, y_train, cv=kfold)
# Check 3-fold model's mean accuracy
print('Class Activation Map Cross validation score : {:.4f}'.format(np.mean(results)))


# Our model seems quite a good model.

# In[43]:


history = model.fit(x_train, y_train,
         validation_data=[x_test, y_test],
         epochs=50,
         batch_size=batch_size,
         )


# In[44]:


# accuracy plot 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Visualize How CAM activates

# In[55]:


#set target wafer number
target_wafer_num = 11
# predict 
prob = model.model.predict(x_test[target_wafer_num].reshape(1,26,26,3))


# In[56]:


aver_output = model.model.layers[7]
aver_model = models.Model(input_tensor, aver_output.output)
cam_result = aver_model.predict(x_test[target_wafer_num].reshape(1, 26, 26, 3))


# In[57]:


weight_result = model.model.layers[-1].get_weights()[0]


# In[65]:


def make_cam(cam_result, weight_result): 
    cam_arr = np.zeros((1,24, 24))
    for row in range(0,9):
        cam = np.zeros((1, 24, 24))
        for i, w in enumerate(weight_result[:, row]):
            cam += (w*cam_result[0,:,:,i]).reshape(24,24)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam[mask_x == 0] = 0
        cam_arr = np.concatenate((cam_arr, cam))
    return cam_arr[1:]

def display_activation(cam_arr, prob, wafer): 
    fig, ax = plt.subplots(9, 1, figsize=(50, 50))
    count = 0
    cam_arr[np.percentile(cam_arr, 0.8) > cam_arr] = 0
    for row in range(0,9):
        ax[row].imshow(np.argmax(wafer, axis=2))
        ax[row].imshow(cam_arr[row],cmap='Reds', alpha=0.7)
        ax[row].set_title('class : ' + faulty_case_dict[count]+', prob : {:.4f}'.format(prob[:, count][0]*100) + '%')
        count += 1


# In[66]:


faulty_case_dict[np.argmax(y_test[11])]


# In[67]:


plt.imshow(np.argmax(x_test[target_wafer_num], axis=2))
print('faulty case : {}'.format(faulty_case_dict[np.argmax(y_test[target_wafer_num])]))


# In[68]:


cam_result.shape


# In[69]:


cam_arr = make_cam(cam_result, weight_result)


# In[70]:


prob.shape


# In[71]:


display_activation(cam_arr, prob, x_test[target_wafer_num])

