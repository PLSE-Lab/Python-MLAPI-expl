#!/usr/bin/env python
# coding: utf-8

# # <a id="1">VGG-16/19</a>

# ## Loading images

# Packages:

# In[ ]:


import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, load_img

print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/"))


# Parameters:

# In[ ]:


PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'

num_classes  = 2
sample_size  = 25000
IMG_size     = 224
batch_size   = 50
epoch_num    = 50


# Extract pictures from zip:

# In[ ]:


train_img_path = os.path.join(PATH, "train.zip")
test_img_path  = os.path.join(PATH, "test.zip")

import zipfile
with zipfile.ZipFile(train_img_path, "r") as z:
   z.extractall(".")
with zipfile.ZipFile(test_img_path, "r") as z:
   z.extractall(".")


# Prepare data frame:

# In[ ]:


filenames  = os.listdir("./train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[ ]:


df.head()


# In[ ]:


df.tail()


# ## Prepare train and validation data generator

# In[ ]:


df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

train_df, val_df = train_test_split(df, test_size=0.4, random_state=2020)

train_df  = train_df.reset_index(drop=True)
val_df    = val_df.reset_index(drop=True)
train_num = train_df.shape[0]
val_num   = val_df.shape[0]


# In[ ]:


train_df['category'].value_counts().plot.bar()


# Training generator:

# In[ ]:


datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(
x_col = "filename",
y_col = "category",
dataframe = train_df,
directory = "./train/",
batch_size = batch_size,
shuffle    = True,
class_mode = "categorical",
target_size = (IMG_size, IMG_size))


# Validation generator:

# In[ ]:


val_generator = datagen.flow_from_dataframe(
x_col = "filename",
y_col = "category",
dataframe = val_df,
directory = "./train/",
batch_size = batch_size,
shuffle    = True,
class_mode = "categorical",
target_size = (IMG_size, IMG_size))


# ## <a id="13">VGG-16 model</a>

# Packages:

# In[ ]:


import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Build VGG-16 model:
# 
# * CNN model contains 16 layers in which weights and bias parameters are learned.
# * 13 convolutional layers are stacked one by one and 3 dense layers for classification.
# * the dense layers comprises of 4096, 4096, 2 nodes each.
# * dense layers action = ReLU + ReLU + Softmax.
# 
# ![](https://tech.showmax.com/2017/10/convnet-architectures/image_0-8fa3b810.png)

# In[ ]:


model_vgg16 = Sequential()

# CONV3-64 + POOL2
model_vgg16.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-128 + POOL2
model_vgg16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-256 + POOL2
model_vgg16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-512 + POOL2
model_vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-512 + POOL2
model_vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg16.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# DENSE
model_vgg16.add(Flatten())
model_vgg16.add(Dense(units=4096,activation="relu"))
model_vgg16.add(Dense(units=4096,activation="relu"))
model_vgg16.add(Dense(units=2, activation="softmax"))


# Compile VGG-16 model: 
# * using Adam optimiser to reach global minimum, with learning rate = 0.00001.
# * loss function set as categorical_crossentropy, metrics = accuracy.
# * total 134,268,738 parameters, with trainable parameters 134,268,738.

# In[ ]:


opt = Adam(lr = 0.00001)

model_vgg16.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model_vgg16.summary()


# Callbacks: ModelCheckpoint and EarlyStopping method.
# 
# * monitoring validation accuracy by passing **val_acc** to ModelCheckpoint.
# * The model will only be saved to disk if val_acc in current epoch is greater than the last epoch
# * passing val_acc to EarlyStopping, set patience = 10 means the model will stop if there is no rise in val_acc in 10 epochs.

# In[ ]:


checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Training VGG-16 model:

# In[ ]:


hist_vgg16 = model_vgg16.fit_generator(
    generator = train_generator, 
    epochs = epoch_num,
    validation_data  = val_generator,
    validation_steps = val_num//batch_size,
    steps_per_epoch  = train_num//batch_size,
    callbacks = [checkpoint,early])


# Visualize validation acc and loss:

# In[ ]:


def plot_acc_los(model_history):
    hist = model_history.history
    acc = hist['accuracy']
    los = hist['loss']
    val_acc = hist['val_accuracy']
    val_los = hist['val_loss']
    epochs = range(len(acc))
    f,  ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, label='Training accuracy')
    ax[0].plot(epochs, val_acc, label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, los, label='Training loss')
    ax[1].plot(epochs, val_los, label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()


# In[ ]:


plot_acc_los(hist_vgg16)


# ## Predictions

# Prepare test data:

# In[ ]:


test_filenames = os.listdir("./test/")

test_df  = pd.DataFrame({'filename': test_filenames})
test_num = test_df.shape[0]


# Test generator:

# In[ ]:


test_generator = datagen.flow_from_dataframe(
x_col = "filename",
y_col = None,
dataframe = test_df,
directory = "./test/",
batch_size = batch_size,
shuffle    = False,
class_mode = None,
target_size = (IMG_size, IMG_size))


# Get predictions:

# In[ ]:


predict = model_vgg16.predict_generator(test_generator, steps = np.ceil(test_num/batch_size))


# Creat labels:

# In[ ]:


test_df['category'] = np.argmax(predict, axis=-1)
test_df['category'] = test_df['category'].replace({ 1: 'dog', 0: 'cat' })

test_df['category'].value_counts().plot.bar()


# Samples with predicted labels:

# In[ ]:


sample_test = test_df.head(9)
sample_test.head()

plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./test/" + filename, target_size = (IMG_size,IMG_size))
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.title("Predicted:" + format(category))
    plt.axis('off')
plt.tight_layout()

plt.show()


# ## <a id="15">VGG-19 model</a>

# * Difference between VGG-16 and VGG-19: extra CONV3-512 layer.

# In[ ]:


model_vgg19 = Sequential()

# CONV3-64 + POOL2
model_vgg19.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model_vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-128 + POOL2
model_vgg19.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-256 + POOL2
model_vgg19.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-512 + POOL2
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-512 + POOL2
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# CONV3-512 + POOL2
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model_vgg19.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# DENSE
model_vgg19.add(Flatten())
model_vgg19.add(Dense(units=4096,activation="relu"))
model_vgg19.add(Dense(units=4096,activation="relu"))
model_vgg19.add(Dense(units=2, activation="softmax"))


# Compiling VGG-19 model:

# In[ ]:


model_vgg19.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
model_vgg19.summary()


# Using the same callbacks and train VGG-19 model:

# In[ ]:


checkpoint = ModelCheckpoint("vgg19_300.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early      = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

hist_vgg19 = model_vgg19.fit_generator(
    generator = train_generator, 
    epochs = epoch_num,
    validation_data  = val_generator,
    validation_steps = val_num//batch_size,
    steps_per_epoch  = train_num//batch_size,
    callbacks = [checkpoint,early])    


# Visualize validation acc and loss:

# In[ ]:


plot_acc_los(hist_vgg19)


# ## <a id="16">Pre-trained VGG-16 model</a>

# Pretrained model: 
# * Importing from Keras and weights from ImagNet.
# * Adding 2 dense layers with 4096 units each and 1 softmax layer with 2 units.

# In[ ]:


from keras import layers, models, optimizers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(IMG_size, IMG_size, 3))

model_pre_vgg16 = models.Sequential()
model_pre_vgg16.add(conv_base)

model_pre_vgg16.add(Flatten())
model_pre_vgg16.add(Dense(units=4096,activation="relu"))
model_pre_vgg16.add(Dense(units=4096,activation="relu"))
model_pre_vgg16.add(Dense(units=2, activation="softmax"))

model_pre_vgg16.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


# Training this model on our data set:

# In[ ]:


hist_pre_vgg16 = model_pre_vgg16.fit_generator(
    generator = train_generator, 
    epochs = 10,
    validation_data  = val_generator,
    validation_steps = val_num//batch_size,
    steps_per_epoch  = train_num//batch_size)


# Visualiz validation acc and loss:

# In[ ]:


plot_acc_los(hist_pre_vgg16)  


# # <a id="2">ResNet / InceptionNet</a>

# ## Loading and data preparing

# Packages and parameters:

# In[ ]:


import os, cv2, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from random import shuffle 

PATH = '/kaggle/input/dogs-vs-cats-redux-kernels-edition/'
FOLDER_TRAIN = './train/'
FOLDER_TEST  = './test/'
IMG_SIZE     = 224
NUM_CLASSES  = 2
SAMPLE_SIZE  = 25000


# Loading images:

# In[ ]:


train_img_path = os.path.join(PATH, "train.zip")
test_img_path  = os.path.join(PATH, "test.zip")

import zipfile
with zipfile.ZipFile(train_img_path, "r") as z:
   z.extractall(".")
with zipfile.ZipFile(test_img_path, "r") as z:
   z.extractall(".")

train_img_list = os.listdir("./train/")[0: SAMPLE_SIZE]
test_img_list  = os.listdir("./test/")


# Define functions for processing image data:

# In[ ]:


def label_pet(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return [1,0]
    elif pet == 'dog': return [0,1]
    
def process_data(data_img_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_img_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_pet(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df

def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split('.')[-3])
    sns.countplot(labels)
    plt.title('Cats vs Dogs')


# Data exploration:

# In[ ]:


plot_image_list_count(train_img_list)    

train = process_data(train_img_list, FOLDER_TRAIN)


# Preparing training and validation data:

# In[ ]:


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array([i[1] for i in train])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.4, random_state = 2020)


# ## <a id="22">ResNet50 model</a>

# Packages and parameters:

# In[ ]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D,Dropout

BATCH_SIZE = 50
EPOCH_NUM  = 10


# Build and complie ResNet50 model:
# 
# * Importing pre-trained ResNet50 model from Keras, weights from ImageNet.
# * Adding additional layer of Dense with softmax activation function.
# * The first layer is not trainable, used pre-trained model.
# * Compling this model with using sigmoid optimizer.
# * Total 23,591,810 parameters, 23,538,690 trainable.
# 
# ![](https://i.stack.imgur.com/gI4zT.png)

# In[ ]:


model_RN50 = Sequential()

model_RN50.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
model_RN50.add(Dense(NUM_CLASSES, activation='softmax'))
model_RN50.layers[0].trainable = True

model_RN50.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model_RN50.summary()


# Training ResNet50 model on our data set:

# In[ ]:


hist_RN50 = model_RN50.fit(X_train, y_train,
                  batch_size = BATCH_SIZE,
                  epochs  = EPOCH_NUM,
                  verbose = 1,
                  validation_data = (X_val, y_val))


# Visualize validation acc and loss:

# In[ ]:


plot_acc_loss(hist_RN50)    


# Numeric validation acc and loss:

# In[ ]:


score = model_RN50.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


# ## Predictions:

# Getting predictions and show sample images:

# In[ ]:


test = process_data(test_img_list, FOLDER_TEST, False)

f, ax = plt.subplots(5,5, figsize=(15,15))
for i,data in enumerate(test[:25]):
    img_data = data[0]
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    model_out = model_RN50.predict([data])[0]
    
    if np.argmax(model_out) == 1: 
        str_predicted='Dog'
    else: 
        str_predicted='Cat'
    ax[i//5, i%5].imshow(orig)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Predicted:{}".format(str_predicted))    
plt.show()


# ## <a id="24">InceptionNet model</a>

# Build and compile InceptionNet model:
# 
# ![](https://hackathonprojects.files.wordpress.com/2016/09/inception_implement.png?w=649&h=337)
# 
# * Inception modules performed as local network topology in InceptionNets.
# * Inception modules acts as the multi-level feature extractor in which convolutions of different sizes are obtained to create a diversified feature map.
# 
# ![](https://cdn-images-1.medium.com/max/2000/1*uXfC5fcbDsL0TJG4T8PsVw.png)
# 
# * Total 22 layers in pretrained model.
# * Importing pre-trained InceptionV3 model from Keras, weights from ImageNet.
# * Adding 1 global avg pooling layer, 1 ReLU dense layer with 1024 units and softmax dense layer with 2 units.
# * Using sigmoid optimizer to compile InceptionNet model.

# In[ ]:


from tensorflow.keras.applications import InceptionV3

Incep = InceptionV3(weights=INCEP_PATH, include_top=False)
x     = Incep.output
x_pool  = GlobalAveragePooling2D()(x)
x_dense = Dense(1024,activation='relu')(x_pool)
final_pred  = Dense(NUM_CLASSES,activation='softmax')(x_dense)

model_Incep = Model(inputs=Incep.input,outputs=final_pred)

model_Incep.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model_Incep.summary


# Training Inception model on our data set:

# In[ ]:


hist_Incep = model_Incep.fit(X_train, y_train,
                  batch_size = BATCH_SIZE,
                  epochs  = EPOCH_NUM,
                  verbose = 1,
                  validation_data = (X_val, y_val))


# Visualize validation acc and loss:

# In[ ]:


plot_acc_loss(hist_Incep)


# Numeric validation acc and loss:

# In[ ]:


score = model_Incep.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


# # <a id="3">References</a>
# 
# [1] Dogs vs. Cats Redux: Kernels Edition, https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition  
# [2] CNN Arthitectures: VGG, Resnet, InceptionNet, XceptionNet, https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl
# [3] Cats or Dogs - Using CNN with Transfer Learning, https://www.kaggle.com/gpreda/cats-or-dogs-using-cnn-with-transfer-learning
