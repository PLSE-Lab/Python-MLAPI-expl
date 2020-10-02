#!/usr/bin/env python
# coding: utf-8

# ****This is the KERAS CNN implementation for the MALARIA CELL IMAGES DATASET with 95% accuracy********
# 
# ANY FEEDBACK IN THE COMMENTS WILL BE HIGHLY APPRECIATED.

# Breakdown of this notebook:
# 
# 1. Loading the dataset: Load the data and import the libraries.
# 1. Data Preprocessing: 
#      * Reading the images,labels stored in 2 folders(Parasitized,Uninfected).
#      * Plotting the Uninfected and Parasitized images with their respective labels.
#      * Normalizing the image data.
#      * Train,test split
# 1. Data Augmentation: Augment the train and validation data using ImageDataGenerator
# 1. Creating and Training the Model: Create a cnn model in KERAS.
# 1. Evaluation: Display the plots from the training history.
# 1. Submission: Run predictions with model.predict, and create confusion matrix.

# In[224]:


import warnings
warnings.filterwarnings('ignore')
from __future__ import absolute_import, division, print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image
import os
from keras.utils import to_categorical
from keras import backend as K
from keras import layers
from keras.preprocessing.image import save_img
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Flatten,Dropout,Concatenate,GlobalAveragePooling2D,Lambda,ZeroPadding2D
from keras.layers import SeparableConv2D,BatchNormalization,MaxPooling2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
print(os.listdir("../input/cell_images/cell_images"))


# In[ ]:


infected = os.listdir('../input/cell_images/cell_images/Parasitized/') 
uninfected = os.listdir('../input/cell_images/cell_images/Uninfected/')


# **DATA PREPROCESSING:-**

# 
# The images from infected folder are read one by one.
# 
# The images are resized to (64x64) and stored to data list.
# 
# The images in infected folder are assigned the label=1 and stored in labels list.
# 
# The process is similar for uninfected folder but here the label=0. 
# 
# 

# In[ ]:


data = []
labels = []

for i in infected:
    try:
    
        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((64 , 64))
        data.append(np.array(resize_img))
        label = to_categorical(1, num_classes=2)
        labels.append(label)
        
    except AttributeError:
        print('')
    
for u in uninfected:
    try:
        
        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+u)
        image_array = Image.fromarray(image , 'RGB')
        resize_img = image_array.resize((64 , 64))
        data.append(np.array(resize_img))
        label = to_categorical(0, num_classes=2)
        labels.append(label)
        
    except AttributeError:
        print('')


# In[ ]:


data = np.array(data)
labels = np.array(labels)

np.save('Data' , data)
np.save('Labels' , labels)


# In[ ]:


print('Cells : {} | labels : {}'.format(data.shape , labels.shape))


# **PLOTS FOR INFECTED AND UNINFECTED CELL:-**

# In[ ]:


plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(data[0])
plt.title('Infected Cell')
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(data[15000])
plt.title('Uninfected Cell')
plt.xticks([]) , plt.yticks([])

plt.show()


# In[225]:


n = np.arange(data.shape[0])
np.random.shuffle(n)
data = data[n]
labels = labels[n]


# In[ ]:


data = data.astype(np.float32)
labels = labels.astype(np.int32)
data = data/255


# **TRAIN,TEST,SPLIT**

# In[ ]:


from sklearn.model_selection import train_test_split

train_x , eval_x , train_y , eval_y = train_test_split(data , labels , 
                                            test_size = 0.2 ,
                                            random_state = 111)

# eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
#                                                     test_size = 0.2 , 
#                                                     random_state = 111)


# In[ ]:


print('train data shape {} ,eval data shape {} '.format(train_x.shape, eval_x.shape))


# As you can see the train and test are highly imbalanced after our data preprocessing.
# 
# So, data augmentation will help us in getting results for our test set.

# **DATA AUGMENTATION**

# Data augmentation is a powerful technique which helps in almost every case for improving the robustness of a model. But augmentation can be much more helpful where the dataset is imbalanced. You can generate different samples of undersampled class in order to try to balance the overall distribution.
# 
# Here we can implement both horizontal and vertical flips because that will not alter the result.A horizontally or vertically flipped infected cell still remains an infected cell.

# In[ ]:


train_aug = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,  
    zoom_range=0.2,        
    horizontal_flip=True,
    vertical_flip=True)  

val_aug= ImageDataGenerator(
    rescale=1./255)

train_gen = train_aug.flow(
    train_x,
    train_y,
    batch_size=16)

val_gen = val_aug.flow(
    eval_x,
    eval_y,
    batch_size=16)


# **Function for plotting loss,accuracy**

# In[ ]:


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()


# **MODEL**

# *LAYERS:-*
# 
# Depthwise SeparableConv is a good replacement for Conv layer. It introduces lesser number of parameters as compared to normal convolution and as different filters are applied to each channel, it captures more information.
# 
# BatchNormalization and Dropout are used to prevent overfitting.
# 
# Input shape is taken as 64x64 as also shown in the preprocessing.

# In[ ]:


def ConvBlock(model, layers, filters,name):
    for i in range(layers):
        model.add(SeparableConv2D(filters, (3, 3), activation='relu',name=name))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
def FCN():
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(64, 64, 3)))
    ConvBlock(model, 1, 64,'block_1')
    ConvBlock(model, 1, 128,'block_2')
    ConvBlock(model, 1, 256,'block_3')
    ConvBlock(model, 1, 512,'block_4')
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='sigmoid'))
    return model

model = FCN()
model.summary()

# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# **CALLBACKS**

# Note that the monitor for the callbacks is val_loss.

# In[ ]:


#-------Callbacks-------------#
best_model_weights = './base.model'
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_weights_only=False,
    period=1
)
earlystop = EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=7,
    verbose=2,
    mode='min'
)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=40,
    verbose=1, 
    mode='auto',
    cooldown=1 
)

callbacks = [checkpoint,earlystop,reduce]


# In[ ]:


opt = SGD(lr=1e-4,momentum=0.99)
opt1 = Adam(lr=2e-4)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
    
history = model.fit_generator(
    train_gen, 
    steps_per_epoch  = 5000, 
    validation_data  = val_gen,
    validation_steps = 2000,
    epochs = 10, 
    verbose = 1,
    callbacks=callbacks
)


# In[ ]:


show_final_history(history)
model.load_weights(best_model_weights)
model_score = model.evaluate_generator(val_gen,steps=50)
print("Model Test Loss:",model_score[0])
print("Model Test Accuracy:",model_score[1])
model.save('malaria.h5')


# **PREDICTIONS**

# In[222]:


preds = model.predict(eval_x, batch_size=16)
preds = np.argmax(preds, axis=-1)

# Original labels
orig_test_labels = np.argmax(eval_y, axis=-1)

print(orig_test_labels.shape)
print(preds.shape)

print(np.unique(orig_test_labels))
print(np.unique(preds))


# In[221]:



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(orig_test_labels , preds)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Infected'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Infected'], fontsize=16)
plt.show()

