#!/usr/bin/env python
# coding: utf-8

# ## This is the KERAS CNN implementation for the CHEST X RAY IMAGES with > 93% validation accuracy and >85% test set accuracy**
# 
# #### ANY FEEDBACK IN THE COMMENTS WILL BE HIGHLY APPRECIATED.

# 
# 
# ### Breakdown of this notebook:
# 
# 1. Loading the dataset: Load the data and import the libraries.
# 2. Data Preprocessing:
#      * Reading the images stored in 3 folders(Train,Val,Test).
#      * Plotting the NORMAL and PNEUMONIA images with their respective labels.
# 3. Data Augmentation: Augment the train,validation and test data using ImageDataGenerator
# 4. Creating and Training the Model: Create a CNN model in KERAS.
# 5. Evaluation: Display the plots from the training history.
# 6. Prediction: Run predictions with model.predict
# 7. Conclusion: Comparing original labels with predicted labels and calculating recall score

# In[35]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[33]:


import keras
import matplotlib.pyplot as plt
from glob import glob 
from keras.models import Sequential 
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D,Input,SeparableConv2D
from keras.preprocessing.image import ImageDataGenerator #Data augmentation and preprocessing
from keras.utils import to_categorical 
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
import cv2
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,classification_report


# ### Exploring the directories in our dataset

# In[14]:


print(os.listdir("../input/chest_xray/chest_xray"))


# In[15]:


path_train = "../input/chest_xray/chest_xray/train"
path_val = "../input/chest_xray/chest_xray/val"
path_test = "../input/chest_xray/chest_xray/test"


# ### Example plots of images in NORMAL and PNEOMONIA folder

# In[16]:


plt.figure(1, figsize = (15 , 7))
plt.subplot(1 , 2 , 1)
img = glob(path_train+"/PNEUMONIA/*.jpeg") #Getting an image in the PNEUMONIA folder
img = np.asarray(plt.imread(img[0]))
plt.title('PNEUMONIA X-RAY')
plt.imshow(img)

plt.subplot(1 , 2 , 2)
img = glob(path_train+"/NORMAL/*.jpeg") #Getting an image in the NORMAL folder
img = np.asarray(plt.imread(img[0]))
plt.title('NORMAL CHEST X-RAY')
plt.imshow(img)

plt.show()


# ### AUGMENTATION ON TRAINING, VALIDATION, TEST DATA

# Data augmentation is a powerful technique which helps in almost every case for improving the robustness of a model. But augmentation can be much more helpful where the dataset is imbalanced. You can generate different samples of undersampled class in order to try to balance the overall distribution.

# In[17]:



train_gen = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip=True)

val_gen = ImageDataGenerator(rescale=1./255)

train_batch = train_gen.flow_from_directory(path_train,
                                            target_size = (224, 224),
                                            classes = ["NORMAL", "PNEUMONIA"],
                                            class_mode = "categorical")
val_batch = val_gen.flow_from_directory(path_val,
                                        target_size = (224, 224),
                                        classes = ["NORMAL", "PNEUMONIA"],
                                        class_mode = "categorical")
test_batch = val_gen.flow_from_directory(path_test,
                                         target_size = (224, 224),
                                         classes = ["NORMAL", "PNEUMONIA"],
                                         class_mode = "categorical")

print(train_batch.image_shape)


# ### Creating the CNN model
# 
# * I have used Keras's Functional API to build the Sequential model.I find it to be a better and easier way to deine a Convolutional Neural Net Model.Below is the reference to Functional API documentation by KERAS:-
# https://keras.io/getting-started/functional-api-guide/
# 
# * In the model, use Depthwise "SeparableConv" layer,which is less computationally expensive than standard "CONV2D" layer.The convolution operation in "SeparableConv2D" layer is applied to a single channel at a time, unlike normal convolution where the operation is applied to all the channels at once. With that, the number of parameters and multiplications to be done are reduced, making it faster than normal convolution. In practice that's the advantage of using it, it's really helpful in large neural net structures, MobileNet and Xception for example are based on this type of convolution. I'll recommend you watch this video, helped me a lot when I was studying. https://www.youtube.com/watch?v=T7o3xvJLuHk
# 
# 

# In[18]:


def build_model():
    input_img = Input(shape=train_batch.image_shape, name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=input_img, outputs=x)
    
    return model


# ### Function for getting accuracy and loss plots 

# In[19]:


def create_plots(history):
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


# In[20]:


model= build_model()
model.summary()


# ### Here I have used 4 callbacks to get the best model
# You can experiment by changing the number of epochs and changing the monitoring parameters. However, increasing the number of epochs increases the computation time but gives better results in visualizing the plots and getting the best model.

# In[21]:


batch_size = 16
epochs = 50
early_stop = EarlyStopping(patience=25,
                           verbose = 2,
                           monitor='val_loss',
                           mode='auto')

checkpoint = ModelCheckpoint(
    filepath='best_model',
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    verbose = 1)

reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=5,
    verbose=1, 
    mode='auto',
    min_delta=0.0001, 
    cooldown=1, 
    min_lr=0.0001
)

model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=Adam(lr=0.0001))

history = model.fit_generator(epochs=epochs,
                              callbacks=[early_stop,checkpoint,reduce],
                              shuffle=True,
                              validation_data=val_batch,
                              generator=train_batch,
                              steps_per_epoch=500,
                              validation_steps=10,
                              verbose=2)


# ### Loss and accuracy plots

# In[22]:


create_plots(history)


# ### Getting the images and labels from test data

# In[51]:


original_test_label=[]
images=[]

test_normal=Path("../input/chest_xray/chest_xray/test/NORMAL") 
normal = test_normal.glob('*.jpeg')
for i in normal:
    img = cv2.imread(str(i))
#     print("normal",img)
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (224,224))
    except Exception as e:
        print(str(e))
    images.append(img)
    label = to_categorical(0, num_classes=2)
    original_test_label.append(label)

test_pneumonia = Path("../input/chest_xray/chest_xray/test/PNEUMONIA")
pneumonia = test_pneumonia.glob('*.jpeg')
for i in pneumonia:
    img = cv2.imread(str(i))
#     print("pneumonia",img)
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        img = cv2.resize(img, (224,224))
    except Exception as e:
        print(str(e))
    images.append(img)
    label = to_categorical(1, num_classes=2)
    original_test_label.append(label)    

    
images = np.array(images)
original_test_label = np.array(original_test_label)
print(original_test_label.shape)


orig_test_labels = np.argmax(original_test_label, axis=-1)
# print(orig_test_labels)
# print(p)



# ### Prediction on test set images

# In[54]:


p = model.predict(images, batch_size=16)
preds = np.argmax(p, axis=-1)
print(preds.shape)


# ### Evaluation of model on test set

# In[48]:


test_loss, test_score = model.evaluate_generator(test_batch,steps=100)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)


# ### Validation Accuracy and Recall score

# In[56]:


print("Accuracy: " + str(history.history['val_acc'][-1:]))


# In[57]:


recall_score(orig_test_labels,preds)

