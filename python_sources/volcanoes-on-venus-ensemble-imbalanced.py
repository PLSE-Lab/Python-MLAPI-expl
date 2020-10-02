#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense, Flatten, Average
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as prep_inputVgg16
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as prep_inputResNet50
from keras.optimizers import Adam, RMSprop

sns.set()


# In[ ]:


np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
np.set_printoptions(edgeitems=10)


# In[ ]:


TRAIN_PATH = '../input/volcanoesvenus/volcanoes_train/'
TEST_PATH = '../input/volcanoesvenus/volcanoes_test/'


# In[ ]:


IMAGE_HEIGHT_TARGET = 110
IMAGE_WIDTH_TARGET = 110


# In[ ]:


# Load train data
train_images = pd.read_csv(TRAIN_PATH + 'train_images.csv', header=None)
train_labels = pd.read_csv(TRAIN_PATH + 'train_labels.csv', header=None)


# In[ ]:


# Load test data
test_images = pd.read_csv(TEST_PATH + 'test_images.csv', header=None)
test_labels = pd.read_csv(TEST_PATH + 'test_labels.csv', header=None)


# #### Check data

# In[ ]:


train_images.head()


# In[ ]:


train_labels.head()


#  We can see that the first line is the header!
#  We must remove it

# In[ ]:


train_labels = train_labels.drop([0])


# We can see that when we have Volcanoe value 0 , we have all the rest columns as NaN (because there is no volcanoe)
# We want to check if we have any nan values when there is a volcanoe (so index is 1)

# In[ ]:


nulls = []
for i in range(len(train_labels.index)):
    if (train_labels.iloc[i][0] == 1):
        nulls.append(train_labels.isnull().iloc[i][0])


# In[ ]:


# count nan values (true) in list
count_nans = sum(nulls)
count_nans


# So, we don't have any nan values when we have a volcanoe

# In[ ]:


# Do the same for test labels
nulls = []
for i in range(len(test_labels.index)):
    if (test_labels.iloc[i][0] == 1):
        nulls.append(test_labels.isnull().iloc[i][0])


# In[ ]:


count_nans = sum(nulls)
count_nans


# We don't have an nan values in test set either.

# In[ ]:


# Do the same for train and test images.
train_images.isnull().values.any()


# In[ ]:


test_images.isnull().values.any()


# We are ok with our dataset , so we can continue our analysis

# How many Volcanoes

# In[ ]:


ax = sns.countplot(data = train_labels,x=train_labels[0][1:])
ax.set(xlabel='Volcanoes', ylabel='Count')


# We can see that we have an unbalanced set of data.Non volcanoes are 6000 and volcanoes are 1000.We must take that into acocunt when we are going to design our model.

# What kind of type

# In[ ]:


ax = sns.countplot(data = train_labels,x=train_labels[1][1:])
ax.set(xlabel='Type', ylabel='Count')


# Number of volcanoes 

# In[ ]:


ax = sns.countplot(data = train_labels,x=train_labels[3][1:])
ax.set(xlabel='Number of volcanoes', ylabel='Count')


# ## Multi label classification

# **We are going to perfrom multilabel categorical classification (each sample can have several classes).**
# 
# **In order to do so , we are going to take into account only the categories where a volcanoe exists because in that case we have
# type, radius and number of volcanoes.**
# 
# **In the other cases all these values are nan.**

# **Extract only the cases where volcanoe exists**

# In[ ]:


indices_train = np.where(train_labels.iloc[:, 0].astype(np.float) == 1)


# In[ ]:


train_labels.iloc[indices_train].shape


# In[ ]:


train_images.iloc[indices_train].shape


# So, we have 1000 volcanoes and the rest 6000 are no volcanoes

# In[ ]:


train_labels.head()


# We are going to replace the strings columns data with more convenient

# **Replace all float nans with string nans in order for not to have any problems with values nan in the classifier.**

# In[ ]:


train_labels = train_labels.fillna('nan')


# **Replace 0 or 1 with No or Yes**

# In[ ]:


train_labels.iloc[:, 0] = (train_labels.iloc[:, 0]).str.replace('0', 'No')
train_labels.iloc[:, 0] = (train_labels.iloc[:, 0]).str.replace('1', 'Yes')


# ** Replace 1, 2, 3, 4 with Type1, 2, 3, 4**

# In[ ]:


train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('1', 'Type 1')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('2', 'Type 2')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('3', 'Type 3')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('4', 'Type 4')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('nan', 'Type nan')


# **Replace nan value with Nb volcanoes nan**

# In[ ]:


train_labels.iloc[:, 3] = (train_labels.iloc[:, 3]).str.replace('nan', 'Nb volcanoes nan')


# In[ ]:


train_labels[:10]


# **Create a labels list which will hold all the available labels**

# In[ ]:


labels = []
for idx in range(len(train_labels)):
    # index 0: Volcanoe or not
    # index 1: Type
    # index 3: Nb of volcanoes
    labels.append([train_labels.iloc[:, 0].values.item(idx), train_labels.iloc[:, 1].values.item(idx), train_labels.iloc[:, 3].values.item(idx)]) 
    
labels = np.array(labels)


# In[ ]:


#Show a few labels
labels[:4]


# In[ ]:


# Binarize the labels
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)


# **We can see all the classes**
# 
# **We see the order by which the labels are given.Note,that in order to find out if we have a volcanoe we must check the last index**

# In[ ]:


mlb.classes_


# In[ ]:


# Check the binarized labels
labels[:4]


# **We can see for example the first line.
# The last index show us that there is a volcanoe , the 4th index from the end, that it is of type 3 and the first index , shows that we have only 1 volcanoe.**

# **Split data into train and validation sets**

# In[ ]:


def data():
    X_train, X_val, y_train, y_val  = train_test_split(train_images.values,
                                                       labels,
                                                       test_size=0.2,
                                                       stratify=labels,
                                                       random_state=1340)
        
    return X_train, X_val, y_train, y_val


# In[ ]:


X_train, X_val, y_train, y_val = data()


# In[ ]:


X_train_res = X_train.reshape((-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))
X_val_res = X_val.reshape((-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))


# In[ ]:


# Stack, in order to have 3 channels
X_train_vggnet = np.stack((np.squeeze(X_train_res),) * 3, -1)
X_val_vggnet = np.stack((np.squeeze(X_val_res),) * 3, -1)


# In[ ]:


# Preprocess input
X_train_vggnet = prep_inputVgg16(X_train_vggnet)
X_val_vggnet = prep_inputVgg16(X_val_vggnet)


# **Use data augmentation**

# In[ ]:


train_data_gen = ImageDataGenerator(horizontal_flip=True,
                                    rotation_range=40,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2)


# **We are going to use the vggnet16**
# 
# **Lets's create a class**

# In[ ]:


class VGGNet:
    @staticmethod
    def build(width, height, depth, classes, final_activ):
        # Initialize the model to use channels last
        input_shape = (height, width, depth)
        
        # In case where channels first is used
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            
        # Load pretrained weights
        imagenet_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_vgg16 = VGG16(include_top=False, weights=imagenet_weights, input_shape=input_shape)
        last_layer = base_vgg16.output
        
        x = Flatten()(last_layer)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        preds_base_vgg16 = Dense(classes, activation=final_activ)(x)
        
        # Before compiling and train the model it is very important to freeze the convolutional base (resnet base).That means, preventing the weights from being updated during training.
        # If you omit this step, then the representations that were learned previously by the convolutional base will be modified during training.
        base_vgg16.trainable = False
        
        model_vgg16 = Model(base_vgg16.input, preds_base_vgg16)
               
        return model_vgg16
    
    def train(model, X, y, batch_size, epochs, class_weights, k_fold, loss, optimizer, metrics, model_checkpoint, early_stopping):
            
        # use k-fold cross validation test
        histories = []
        nb_validation_samples = len(X) // k_fold
        for fold in range(k_fold):
            x_training_data = np.concatenate([X[:nb_validation_samples * fold], X[nb_validation_samples * (fold + 1):]])
            y_training_data = np.concatenate([y[:nb_validation_samples * fold], y[nb_validation_samples * (fold + 1):]])

            x_validation_data = X[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]
            y_validation_data = y[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]

            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            history = model.fit_generator(train_data_gen.flow(x_training_data, y_training_data, batch_size=batch_size),
                                                              validation_data=[x_validation_data, y_validation_data],
                                                              epochs = epochs,
                                                              shuffle=True,
                                                              verbose=2,
                                                              class_weight=class_weights,
                                                              steps_per_epoch = int(len(X_train) / batch_size),
                                                              validation_steps =int(len(X_val) / batch_size),
                                                              callbacks=[model_checkpoint, early_stopping])
            histories.append(history)
        
        return histories, model


# In[ ]:


model_vgg16 = VGGNet.build(IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 3, len(mlb.classes_), 'sigmoid')


# In[ ]:


final_activation = 'sigmoid'
batch_size = 32
epochs = 100
k_fold = 3
loss = 'binary_crossentropy'
adam = Adam(lr=0.0001)
optimizer = adam
metrics = ['accuracy']
early_stopping = EarlyStopping(patience=10, verbose=1)
model_vgg16_checkpoint = ModelCheckpoint('./b_32_relu_optim_adam.hdf5', verbose=1, save_best_only=True)


# **Define the class weights that we are going to use with our model because it has imbalanced set of data**

# In[ ]:


class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[:, -1]), y_train[:, -1]) #-1 is the lats index (volcanoe or not)


# In[ ]:


# Concatenate train and test data
X_data = np.concatenate((X_train_vggnet, X_val_vggnet))
y_data = np.concatenate((y_train, y_val))


# **Take only the last index (volcanoe or not)**

# In[ ]:


history_vgg16, model_vgg16 = VGGNet.train(model_vgg16,
                                          X_data,
                                          y_data,
                                          batch_size,
                                          epochs,
                                          class_weights,
                                          k_fold,
                                          loss,
                                          optimizer,
                                          metrics,
                                          model_vgg16_checkpoint,
                                          early_stopping)


# In[ ]:


fig, axes = plt.subplots(k_fold, 2, figsize=(20, 12))

for i in range(k_fold):
    
    axes[i, 0].plot(history_vgg16[i].epoch, history_vgg16[i].history['loss'], label='Train loss')
    axes[i, 0].plot(history_vgg16[i].epoch, history_vgg16[i].history['val_loss'], label='Val loss')
    axes[i, 0].legend()

    axes[i, 1].plot(history_vgg16[i].epoch, history_vgg16[i].history['acc'], label = 'Train acc')
    axes[i, 1].plot(history_vgg16[i].epoch, history_vgg16[i].history['val_acc'], label = 'Val acc')
    axes[i, 1].legend()

 
plt.tight_layout()


# In[ ]:


class ResNet:
    @staticmethod
    def build(width, height, depth, classes, final_activ):
        # Initialize the model to use channels last
        input_shape = (height, width, depth)
        
        # In case where channels first is used
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            
        # Load pretrained weights
        imagenet_weights = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_resnet = ResNet50(include_top=False, weights=imagenet_weights, input_shape=input_shape)
        last_layer = base_resnet.output
        
        x = Flatten()(last_layer)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        preds_base_resnet = Dense(classes, activation=final_activ)(x)
        
        # Before compiling and train the model it is very important to freeze the convolutional base (resnet base).That means, preventing the weights from being updated during training.
        # If you omit this step, then the representations that were learned previously by the convolutional base will be modified during training.
        base_resnet.trainable = False
        
        model_resnet = Model(base_resnet.input, preds_base_resnet)
               
        return model_resnet
    
    def train(model, X, y, batch_size, epochs, class_weights, k_fold, loss, optimizer, metrics, model_checkpoint, early_stopping):
        
        # use k-fold cross validation test
        histories = []
        nb_validation_samples = len(X) // k_fold
        for fold in range(k_fold):
            x_training_data = np.concatenate([X[:nb_validation_samples * fold], X[nb_validation_samples * (fold + 1):]])
            y_training_data = np.concatenate([y[:nb_validation_samples * fold], y[nb_validation_samples * (fold + 1):]])
            
            x_validation_data = X[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]
            y_validation_data = y[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]
            
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            
            history = model.fit_generator(train_data_gen.flow(x_training_data, y_training_data, batch_size=batch_size),
                                                              validation_data=[x_validation_data, y_validation_data],
                                                              epochs = epochs,
                                                              shuffle=True,
                                                              verbose=2,
                                                              class_weight=class_weights,
                                                              steps_per_epoch = int(len(X_train) / batch_size),
                                                              validation_steps =int(len(X_val) / batch_size),
                                                              callbacks=[model_checkpoint, early_stopping])
            histories.append(history)
            
        return histories, model


# In[ ]:


model_resnet = ResNet.build(IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 3, len(mlb.classes_), 'sigmoid')


# In[ ]:


X_train_resnet = np.stack((np.squeeze(X_train_res),) * 3, -1)
X_val_resnet = np.stack((np.squeeze(X_val_res),) * 3, -1)


# In[ ]:


X_train_resnet = prep_inputResNet50(X_train_resnet)
X_val_resnet = prep_inputResNet50(X_val_resnet)


# In[ ]:


# Concatenate train and test data
X_data_resnet = np.concatenate((X_train_resnet, X_val_resnet))
y_data_resnet = np.concatenate((y_train, y_val))


# In[ ]:


model_checkpoint_resnet = ModelCheckpoint('./b_32_relu_optim_adam_resnet.hdf5', verbose=1, save_best_only=True)


# In[ ]:


history_resnet, model_resnet = ResNet.train(model_resnet,
                                            X_data_resnet,
                                            y_data_resnet,
                                            batch_size,
                                            epochs,
                                            class_weights,
                                            k_fold,
                                            loss,
                                            optimizer,
                                            metrics,
                                            model_checkpoint_resnet,
                                            early_stopping)


# In[ ]:


fig, axes = plt.subplots(k_fold, 2, figsize=(20, 12))

for i in range(k_fold):
    
    axes[i, 0].plot(history_resnet[i].epoch, history_resnet[i].history['loss'], label='Train loss')
    axes[i, 0].plot(history_resnet[i].epoch, history_resnet[i].history['val_loss'], label='Val loss')
    axes[i, 0].legend()

    axes[i, 1].plot(history_resnet[i].epoch, history_resnet[i].history['acc'], label = 'Train acc')
    axes[i, 1].plot(history_resnet[i].epoch, history_resnet[i].history['val_acc'], label = 'Val acc')
    axes[i, 1].legend()

 
plt.tight_layout()


# **Uncomment the `load_weights` lines in order to load the best saved weights. (for some reason kernel couldn't load them even though I have them in my output)**

# In[ ]:


# Load the best saved model
#model_vgg16.load_weights(filepath='../input/volcanoes-on-venus-ensemble-imbalanced/b_32_relu_optim_adam.hdf5')
#model_resnet.load_weights(filepath='../input/volcanoes-on-venus-ensemble-imbalanced/b_32_relu_optim_adam_resnet.hdf5')


# ### Create an esemble model.
# 
# #### Define a function where we take the average of our two best models.

# In[ ]:


def ensemble(models):
    input_image = Input(shape=(IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 3))
    
    vgg16_out = models[0](input_image)
    resnet_out = models[1](input_image)

    output = Average()([vgg16_out, resnet_out])
    model = Model(input_image, output)
    
    return model


# In[ ]:


# Combine all models
models = [model_vgg16, model_resnet]
ensemble_model = ensemble(models)


# In[ ]:


test_images.head()


# In[ ]:


test_labels.head()


# We can see that the first line is the header!
# We must remove it

# In[ ]:


test_labels = test_labels.drop([0])


# Apply multilabelbinarizer in test data as we did in train data

# In[ ]:


# Replace all float nans with string nans.
test_labels = test_labels.fillna('nan')
# Replace 0 or 1 with No or Yes
test_labels.iloc[:, 0] = (test_labels.iloc[:, 0]).str.replace('0', 'No')
test_labels.iloc[:, 0] = (test_labels.iloc[:, 0]).str.replace('1', 'Yes')


# In[ ]:


# Replace 1,2,3,4 with Type1,2,3,4
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('1', 'Type 1')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('2', 'Type 2')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('3', 'Type 3')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('4', 'Type 4')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('nan', 'Type nan')


# In[ ]:


test_labels.iloc[:, 3] = (test_labels.iloc[:, 3]).str.replace('nan', 'Nb volcanoes nan')


# **Before proceeding with mlb I noticed that in the X_test data the number of volcanoes go up to 3 , not 5 as 
# in the training data.This results in giving 11 classes instead of 13 as in train mlb.
# So, we are going to copy 2 rows which contain 4 and 5 nb of volcanoes from train data to test data.
# The 3rd column which contains the number of volcanoes, it is of type string and contains 
# either 'Nb volcanoes nan' either the nb of volcanoes in string format.
#  So we must take into consideration only the nb of volcanoes**

# In[ ]:


tmp = train_labels.iloc[:, 3].values
idx, = np.where(tmp != 'Nb volcanoes nan')
idx_greater = idx[tmp[idx].astype(int) > 3]


# In[ ]:


idx_greater


# Update the test images and labels with 2 new rows which contain 4 and 5 nb of volcanoes

# In[ ]:


series_list_images = [pd.Series(train_images.iloc[425, :], index=test_images.columns ) ,
                      pd.Series(train_images.iloc[1513, :], index=test_images.columns )]

series_list_labels = [pd.Series(train_labels.iloc[425, :], index=test_labels.columns ) ,
                      pd.Series(train_labels.iloc[1513, :], index=test_labels.columns )]

test_images_full = test_images.append(series_list_images , ignore_index=True)
test_labels_full = test_labels.append(series_list_labels , ignore_index=True)


# Create a labels list which will hold all the available labels

# In[ ]:


labels_test = []
for idx in range(len(test_labels_full)):
    # index 0: Volcanoe or not
    # index 1: Type
    # index 3: Nb of volcanoes
    labels_test.append([test_labels_full.iloc[:, 0].values.item(idx), test_labels_full.iloc[:, 1].values.item(idx), test_labels_full.iloc[:, 3].values.item(idx)]) 
    
labels_test = np.array(labels_test)


# In[ ]:


# Binarize the labels
labels_test = mlb.fit_transform(labels_test)


# In[ ]:


# Check classes
mlb.classes_


# In[ ]:


X_test = test_images_full
y_test = labels_test


# In[ ]:


# Reshape test data and create 3 channels
X_test = X_test.values.reshape((-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))
X_test = np.stack((np.squeeze(X_test),) * 3, -1)


# In[ ]:


# Preprocess data
X_test = prep_inputVgg16(X_test)


# In[ ]:


# predict on validation and test data
pred_val = ensemble_model.predict(X_val_vggnet) 
pred_test = ensemble_model.predict(X_test, batch_size=batch_size)


# In[ ]:


# Squeeze one dimension to be able to plot
X_train_squeeze = X_train_vggnet.squeeze()
y_train_squeeze = y_train.squeeze()
pred_val_squeeze = pred_val.squeeze()
X_val_squeeze = X_val_vggnet.squeeze()
y_val_squeeze = y_val.squeeze()
X_test_squeeze = X_test.squeeze()


# #### Create a function in order to be able to scale images from original to target normalization.

# In[ ]:


def scale_image(input_data, min_orig, max_orig, min_target, max_target):
    orig_range = max_orig - min_orig
    target_range = max_target - min_target
    scaled_data = np.array((input_data - min_orig) / float(orig_range))
    return min_target + (scaled_data * target_range)


# Denormalize our image in order to properly show it in plot

# In[ ]:


X_test_denorm = scale_image(X_test_squeeze, X_test_squeeze.min(), X_test_squeeze.max(), 0, 1)


# In[ ]:


plt.rc('text', usetex=False)
max_images = 6

fig, axes = plt.subplots(max_images//2, 2, figsize=(22, 18))
axes = axes.ravel()

idxlist = [0, 1, 2, 3, 4, 5]
for i in  range(max_images):   

    #idx = np.random.randint(0, len(X_test)-1)
    idx = idxlist[i]
    
    axes[i].grid(False)
    axes[i].imshow(X_test_denorm[idx], cmap='Greens')
    axes[i].set_title(mlb.inverse_transform(y_test[idx:idx+1, :]))
    
    label_img = []
    for (label, p) in zip(mlb.classes_, pred_test[idx]):
        #label_img.append("{0}: {1}%".format(label.astype(np.str), (p * 100).astype(np.float32) ))
        label_img.append((p * 100).astype(np.float32))
        
    text = 'Volcanoe\nYes: {0:.2f}% No: {1:.2f}%\nType\n1: {2:.2f}% 2: {3:.2f}% 3: {4:.2f}% 4: {5:.2f}% nan: {6:.2f}%\n            Nb.Volcanoes\n1: {7:.2f}% 2: {8:.2f}% 3: {9:.2f}% 4: {10:.2f}% 5: {11:.2f}% nan: {12:.2f}%'            .format(label_img[12], label_img[6], label_img[7], label_img[8], label_img[9], label_img[10], label_img[11], label_img[0], label_img[1],                   label_img[2], label_img[3], label_img[4], label_img[5])
    
    axes[i].text(55, 95, text , size=12, ha="center", va="center",
            bbox=dict( fc=(1., 1., 0.8), alpha=0.7))
    
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()


# If you want to play with the plot you must change the `axes[i].text(55, 95,..`  the positions 55, 95 in order for the results to show up. Also, uncomment the idx and use `idx = np.random.randint(0, len(X_test)-1)` in order to get random samples.

# ### Now let's deal with regression in order to find out the radius of the volcanoe.

# In[ ]:


class Regression:
    @staticmethod
    def build(X):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=(X.shape[1],)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))
       
        return model
    
    def train(X, y, k, optimizer, batch_size, epochs):
                    
        # use k-fold cross validation test
        kfold = KFold(n_splits=k, shuffle=True, random_state=1340)
        histories = []
        for train, test in kfold.split(X):
            model = Regression.build(X)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
            
            history = model.fit(X[train],
                                y[train],
                                validation_data=[X[test], y[test]],
                                batch_size=batch_size,
                                epochs = epochs,
                                verbose=0)
            
            # evaluate the model
            #val_mse, val_mae = model.evaluate(X[test], y[test], verbose=0)
            mae_history = history.history['val_mean_absolute_error']
            histories.append(mae_history)
                    
        return histories, model


# **Volcanoes images
# Find the indices where we have a volcanoe,since only volcanoes have radius.**

# In[ ]:


indices_radius, = np.where(train_labels.iloc[:, 0] == 'Yes')
X_volcanoes_radius = train_images.iloc[indices_radius]


# In[ ]:


# Volcanoes labels
y_volcanoes_radius = train_labels.iloc[indices_radius]
# take only the radius
y_volcanoes_radius = y_volcanoes_radius.iloc[:, 2]


# **Split data into train and validation sets**

# In[ ]:


def data_radius():
    X_train, X_val, y_train, y_val  = train_test_split(X_volcanoes_radius,
                                                       y_volcanoes_radius.values.astype(np.float32).reshape(-1, 1),
                                                       test_size=0.2,
                                                       random_state=1340)
        
    return X_train, X_val, y_train, y_val


# In[ ]:


X_train_radius, X_val_radius, y_train_radius, y_val_radius = data_radius()


# In[ ]:


# Concatenate train and test data
X_data_radius = np.concatenate((X_train_radius, X_val_radius))
y_data_radius = np.concatenate((y_train_radius, y_val_radius))


# In[ ]:


# initialize scaler
scaler = StandardScaler()
# scale data
scaler.fit(X_data_radius)
X_std_radius = scaler.transform(X_data_radius)


# In[ ]:


batch_size_reg = 16
epochs = 200
k = 3
rmsprop = RMSprop(lr=0.001)

history_mae, model_reg = Regression.train(X_std_radius, y_data_radius, k, rmsprop, batch_size_reg, epochs)


# Lets's check the radius values range in order to have a grasp for comparing to mean absolute error

# In[ ]:


y_data_check = y_data_radius.astype(np.float32)


# In[ ]:


# Find out max radius value
np.amax(y_data_check)


# In[ ]:


# Find out min radius value
np.amin(y_data_check)


# In[ ]:


# Let's plot the mae
avg_mae = [np.mean([x[i] for x in history_mae]) for i in range(epochs)]

plt.plot(range(1, len(avg_mae) + 1), avg_mae)
plt.xlabel('Epochs')
plt.ylabel('Val mae')
plt.show()


# ### Process test data

# **Let's work with the indices where we predicted that we have a volcanoe with possibilty >= 50%**

# In[ ]:


indices_test,  = np.where(pred_test[:, 12] >= 0.5)
# Volcanoes test images
X_volcanoes_test = test_images_full.iloc[indices_test]
# scale test data
X_volcanoes_test_std = scaler.transform(X_volcanoes_test.values.astype(np.float32))


# In[ ]:


# Volcanoes test labels
y_volcanoes_test = test_labels_full.iloc[indices_test]
# take only the radius
y_volcanoes_test = y_volcanoes_test.iloc[:, 2]


# **We can see that in test data we have nan values in cases where we don't have a volcanoe.**
# **We can replace nan values with zero value for radius when we don't have a volcanoe**

# In[ ]:


y_volcanoes_test = pd.to_numeric(y_volcanoes_test, errors='coerce')
y_volcanoes_test = y_volcanoes_test.fillna(0)


# ### Evaluate regression model on test data

# In[ ]:


test_mse, test_mae = model_reg.evaluate(X_volcanoes_test_std, y_volcanoes_test.values.astype(np.float32))


# In[ ]:


test_mse


# In[ ]:


test_mae


# Mean absolute error returns a value between 8.5-9 which isn't so good. I would expect a value of 1-2 to be good.

# In[ ]:


pred_regression = model_reg.predict(X_volcanoes_test_std)


# In[ ]:


# Check a few values
for i in range(100):
    idx = np.random.randint(0, len(y_volcanoes_test.values)-1)
    print('Real: {0}\t Pred: {1:.2f}'.format(y_volcanoes_test.values[idx], pred_regression.squeeze()[idx]))


# ### Conclusion

# We saw that the vggnet and resnet did a pretty good job to classify if an image is volcanoe or not. We had an accuracy of around ! Predicting the type of volcanoe was not that accurate but in general I think it was good. The regression model didn't succeed to predict accurate the radius of the volcanoe. I have tried various combinations and various models but I couldn't make a good model for that. Maybe the information included in the images is not enough to predict the radius?Or maybe due to my inexperience I can't find a good solution.
