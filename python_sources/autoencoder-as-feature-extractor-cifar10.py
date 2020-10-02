#!/usr/bin/env python
# coding: utf-8

# > ## Introduction:
# 
# The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class (5000 for training and 1000 for testing purpose). In this notebook, I try to implement different autoencoders as feature extractor and using those features as inputs to a classifier model, I try to predict the different classes of images in the CIFAR-10 dataset. There is also one more handicap. I used only 50% of the images (3000 images per class) for training and the rest 50% for testing for the following 3 classes: bird, deer and truck. So, the training data is imbalanced.
# 
# I used both general convolutional autoencoder and [U-net](https://arxiv.org/pdf/1505.04597.pdf) model as autoencoder for feature extraction purpose. For the classifier model, I used both simple stacked dense layers and also convolution layers and then dense layers. I used hyperopt library to find out the optimized hyperparameters for the classifier model. I also used the class-weight function from the sklearn.utils library for using different class weights to tackle the imbalanced training data.
# 
# I found that, U-net architecture (with connector layers) is vastly superior as autoencoders compared to general convolutional autoencoders. But using the features generated from the U-net architecture resulted in low accuracy for the classifier models.

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

import keras
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPool1D, CuDNNLSTM, Reshape
from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate
from keras.datasets import cifar10
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
import keras.backend as K
from keras.objectives import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, RobustScaler, StandardScaler

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# ## Load Data:
# 
# I used the keras datasets library to load the training and testing data.

# In[ ]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# I created the following dictionary for using it later in visualizations.

# In[ ]:


dict = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer', 5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}


# ## Creating imbalanced data:
# 
# I just used the first 2000 images of the bird, deer and truck classes from the training data as test data. In this process, I has 3000 images in both training and test dataset for these 3 classes.

# In[ ]:


x_test_extra = []
y_test_extra = []
x_train_final = []
y_train_final = []
count = [0, 0, 0]
for i, j in zip(x_train, y_train):
    if (j==2):
        if(count[0]<2000):
            x_test_extra.append(i)
            y_test_extra.append(j)
            count[0]+=1
        else:
            x_train_final.append(i)
            y_train_final.append(j)
    elif (j==4):
        if(count[1]<2000):
            x_test_extra.append(i)
            y_test_extra.append(j)
            count[1]+=1
        else:
            x_train_final.append(i)
            y_train_final.append(j)
    elif (j==9):
        if(count[2]<2000):
            x_test_extra.append(i)
            y_test_extra.append(j)
            count[2]+=1
        else:
            x_train_final.append(i)
            y_train_final.append(j)
    else:
        x_train_final.append(i)
        y_train_final.append(j)
        
x_test_extra = np.array(x_test_extra)
y_test_extra = np.array(y_test_extra)
x_train_final = np.array(x_train_final)
y_train_final = np.array(y_train_final)


# In[ ]:


x_test_final = np.append(x_test_extra, x_test, axis=0)
y_test_final = np.append(y_test_extra, y_test, axis=0)


# ## Data Normalization:
# 
# Data was normalized because neural networks work better with normalized data.

# In[ ]:


#x_train_final = x_train    ## These code were used to check model performances with balanced dataset.
#x_test_final = x_test
#y_train_final = y_train
#y_test_final = y_test
x_train_final = x_train_final.astype('float32')
x_test_final = x_test_final.astype('float32')
x_train_final = x_train_final / 255
x_test_final = x_test_final / 255


# ## Validation split:
# 
# I used 20% of the training data as validation set. Validation data was chosen randomly.

# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data
x_train, x_valid, y_trainf, y_validf = train_test_split(x_train_final, y_train_final, test_size=0.2, random_state=42, shuffle= True)


# ## Target conversion to categorical:
# 
# The target variable was converted to one-hot encoded data using the utils.to_categorical function of the keras library.

# In[ ]:


y_train = keras.utils.to_categorical(y_trainf, 10)
y_valid = keras.utils.to_categorical(y_validf, 10)
y_test_one_hot = keras.utils.to_categorical(y_test_final, 10)


# ## Necessary functions:

# In[ ]:


def create_block(input, chs): ## Convolution block of 2 layers
    x = input
    for i in range(2):
        x = Conv2D(chs, 3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
    return x

##############################

## Here, I compute the class weights for using in different models. 
## This is to order our model to emphasize more on classes with less training data.
class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(y_trainf), 
                y_trainf.reshape(y_trainf.shape[0]))

class_weights

##############################

def showOrigDec(orig, dec, num=10):  ## function used for visualizing original and reconstructed images of the autoencoder model
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(orig[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(dec[300*i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
        
def show_test(m, d):  ## function used for visualizing the predicted and true labels of test data
    plt.figure(figsize =(40,8))
    for i in range(5):
        ax = plt.subplot(1, 5, i+1)
        test_image = np.expand_dims(d[1810*i+5], axis=0)
        test_result = m.predict(test_image)
        plt.imshow(x_test_final[1810*i+5])
        index = np.argsort(test_result[0,:])
        plt.title("Pred:{}, True:{}".format(dict[index[9]], dict[y_test_final[1810*i+5][0]]))
    plt.show()
    
def show_test2(m, d):  ## function used for visualizing the predicted and true labels of test data
    plt.figure(figsize =(40,8))
    for i in range(5):
        ax = plt.subplot(1, 5, i+1)
        test_image = np.expand_dims(d[1810*i+5], axis=0)
        test_result = m.predict(test_image)[1]
        plt.imshow(x_test_final[1810*i+5])
        index = np.argsort(test_result[0,:])
        plt.title("Pred:{}, True:{}".format(dict[index[9]], dict[y_test_final[1810*i+5][0]]))
    plt.show()
    
def report(predictions): ## function used for creating a classification report and confusion matrix
    cm=confusion_matrix(y_test_one_hot.argmax(axis=1), predictions.argmax(axis=1))
    print("Classification Report:\n")
    cr=classification_report(y_test_one_hot.argmax(axis=1),
                                predictions.argmax(axis=1), 
                                target_names=list(dict.values()))
    print(cr)
    plt.figure(figsize=(12,12))
    sns.heatmap(cm, annot=True, xticklabels = list(dict.values()), yticklabels = list(dict.values()), fmt="d")
    
def loss_function(y_true, y_pred):  ## loss function for using in autoencoder models
    mses = mean_squared_error(y_true, y_pred)
    return K.sum(mses, axis=(1,2))


# ## Simple Convolution Model as Classifier:
# 
# Here, I have a model that uses several convolution layers stacked and followed by a dense layer with 10 output nodes and softmax activation. It is used as a single classifier model that I can use as a benchmark model. Dropout layers are used for reducing overfitting.

# In[ ]:


def full_conv():
    input = Input((32,32,3))
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    #x = Dropout(0.2)(x)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    #x = Dropout(0.3)(x)
    block3 = create_block(x, 128)
    #x = MaxPool2D(2)(block3)
    x = Dropout(0.4)(block3)
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)
    return Model(input, output)

conv_model = full_conv()
conv_model.summary()


# In[ ]:


#training
batch_size = 512
epochs=50
opt_rms = Adadelta()
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=opt_rms,
                   metrics=['accuracy'])


# In[ ]:


def run_conv_model(data_aug):
    er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
    lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
    callbacks = [er, lr]
    
    if not data_aug:
        history = conv_model.fit(x_train, y_train, batch_size=512,
                                 epochs=epochs,
                                 verbose=1, callbacks=callbacks,
                                 validation_data=(x_valid,y_valid),
                                 class_weight=class_weights)
    else:
        train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_set_ae = train_datagen.flow(x_train, y_train, batch_size=512)

        validation_datagen = ImageDataGenerator()
        validation_set_ae = validation_datagen.flow(x_valid, y_valid, batch_size=512)
        
        history = conv_model.fit_generator(train_set_ae,
                                           epochs=epochs,
                                           steps_per_epoch=np.ceil(x_train.shape[0]/512),
                                           verbose=1, callbacks=callbacks,
                                           validation_data=(validation_set_ae),
                                           validation_steps=np.ceil(x_valid.shape[0]/512),
                                           class_weight=class_weights)
        
        return history


# In[ ]:


run_conv_model(1)


# In[ ]:


print('Test accuracy for benchmark model= {}'.format(conv_model.evaluate(x_test_final, y_test_one_hot)[1]))


# In[ ]:


show_test(conv_model, x_test_final)


# In[ ]:


predictions = conv_model.predict(x_test_final)
report(predictions)


# From the confusion matrix, we can see that, the model works quite good with only 50 epochs for all the classes inspite of the imbalanced data. It only had problem identifying birds correctly. I found that, the result was worse when I did not use class weights. The model accuracy is also improved by augmenting data.

# ## Autoencoder Model:

# In[ ]:


def unet():  ## I commented several layers of the model for descreasing model complexity as the results were almost same
    input = Input((32,32,3))
    
    # Encoder
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    #block3 = create_block(x, 64)
    #x = MaxPool2D(2)(block3)
    #block4 = create_block(x, 128)
    
    # Middle
    #x = MaxPool2D(2)(block2)
    middle = create_block(x, 128)
    
    # Decoder
    #x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)
    #x = Concatenate()([block4, x])
    #x = create_block(x, 128)
    #x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
    #x = Concatenate()([block3, x])
    #x = create_block(x, 64)
    x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)
    x = Concatenate()([block2, x])
    x = create_block(x, 64)
    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)
    x = Concatenate()([block1, x])
    x = create_block(x, 32)
    
    # output
    x = Conv2D(3, 1)(x)
    output = Activation("sigmoid")(x)
    
    return Model(input, middle), Model(input, output)

def general_ae():
    input = Input((32,32,3))
    
    # Encoder
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    
    #Middle
    middle = create_block(x, 128)
    
    # Decoder
    up1 = UpSampling2D((2,2))(middle)
    block3 = create_block(up1, 64)
    #up1 = UpSampling2D((2,2))(block3)
    up2 = UpSampling2D((2,2))(block3)
    block4 = create_block(up2, 32)
    #up2 = UpSampling2D((2,2))(block4)
    
    # output
    x = Conv2D(3, 1)(up2)
    output = Activation("sigmoid")(x)
    return Model(input, middle), Model(input, output)


# In[ ]:


def run_ae(m):  ## function for choosing unet/general autoencoder
    if m=='unet':
        encoder, model = unet()
    elif m=='ae':
        encoder, model = general_ae()
        
    return encoder, model


# ## Implementing U-Net:

# In[ ]:


encoder_unet, model_unet = run_ae('unet')
model_unet.compile(SGD(1e-3, 0.9), loss=loss_function)
model_unet.summary()


# In[ ]:


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
history = model_unet.fit(x_train, x_train, 
                         batch_size=512,
                         epochs=100,
                         verbose=1,
                         validation_data=(x_valid, x_valid),
                         shuffle=True, callbacks=callbacks,
                         class_weight=class_weights)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


recon_test_unet = model_unet.predict(x_test_final)
recon_valid_unet = model_unet.predict(x_valid)


# In[ ]:


showOrigDec(x_valid, recon_valid_unet)


# In[ ]:


showOrigDec(x_test_final, recon_test_unet)


# ## Implement Convolutional AE:

# In[ ]:


encoder_ae, model_ae = run_ae('ae')
model_ae.compile(SGD(1e-3, 0.9), loss=loss_function)
model_ae.summary()


# In[ ]:


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
history = model_ae.fit(x_train, x_train, 
                       batch_size=512,
                       epochs=100,
                       verbose=1,
                       validation_data=(x_valid, x_valid),
                       shuffle=True, callbacks=callbacks,
                       class_weight=class_weights)


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


recon_test_ae = model_ae.predict(x_test_final)
recon_valid_ae = model_ae.predict(x_valid)


# In[ ]:


showOrigDec(x_valid, recon_valid_ae)


# In[ ]:


showOrigDec(x_test_final, recon_test_ae)


# As we can see, unet architecture is far better in terms of reconstructing the data.

# ## Extracting bottleneck features to use as inputs in the classifier model:

# In[ ]:


gist_train_unet = encoder_unet.predict(x_train)
gist_valid_unet = encoder_unet.predict(x_valid)
gist_test_unet = encoder_unet.predict(x_test_final)

gist_train_ae = encoder_ae.predict(x_train)
gist_valid_ae = encoder_ae.predict(x_valid)
gist_test_ae = encoder_ae.predict(x_test_final)


# ## Classifier Models:

# In[ ]:


def classifier_dense(inp):
    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))
    #x = MaxPool2D()(input)
    x = Flatten()(input)
    #x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.64)(x)
    x = Dense(50, activation='relu')(x)
    #x = Reshape((-1, 1))(x)
    #x = Conv1D(128, (3,), activation='relu', padding='same')(x)
    #x = MaxPool1D()(x)
    #x = CuDNNLSTM(64)(x)
    #x = Flatten()(x)
    x = Dropout(0.4)(x)
    output = Dense(10, activation='softmax')(x)
    return Model(input, output)

def classifier_conv(inp):
    input = Input((inp.shape[1], inp.shape[2], inp.shape[3]))
    x = Conv2D(1024, 3, padding="same")(input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, 3, padding="same")(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.35)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.69)(x)
    output = Dense(10, activation='softmax')(x)
    return Model(input, output)


# In[ ]:


def run_cls(m, inp):  ## function for choosing dense/convolutional classifier model
    if m=='dense':
        classifier = classifier_dense(inp)
    elif m=='conv':
        classifier = classifier_conv(inp)
        
    return classifier


# ## Convolutional AE with convolutional NN as classifier:

# In[ ]:


decoder_ae_conv = run_cls('conv', gist_train_ae)
decoder_ae_conv.compile(loss='categorical_crossentropy',
                        optimizer=Adadelta(),
                        metrics=['accuracy'])
decoder_ae_conv.summary()


# In[ ]:


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
hist1 = decoder_ae_conv.fit(gist_train_ae, y_train, batch_size=512, epochs=100, 
                            validation_data = (gist_valid_ae, y_valid),
                            shuffle=True, callbacks=callbacks,
                            class_weight=class_weights)


# In[ ]:


plt.plot(hist1.history['acc'])
plt.plot(hist1.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


print('Test accuracy for AE_conv model= {}'.format(decoder_ae_conv.evaluate(gist_test_ae, y_test_one_hot)[1]))


# In[ ]:


show_test(decoder_ae_conv, gist_test_ae)


# In[ ]:


predictions = decoder_ae_conv.predict(gist_test_ae)
report(predictions)


# ## Convolutional AE with simple NN as classifier:

# In[ ]:


decoder_ae_dense = run_cls('dense', gist_train_ae)
decoder_ae_dense.compile(loss='categorical_crossentropy',
                         optimizer=Adadelta(),
                         metrics=['accuracy'])
decoder_ae_dense.summary()


# In[ ]:


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
hist1 = decoder_ae_dense.fit(gist_train_ae, y_train, batch_size=512, epochs=100, 
                             validation_data = (gist_valid_ae, y_valid),
                             shuffle=True, callbacks=callbacks,
                             class_weight=class_weights)


# In[ ]:


plt.plot(hist1.history['acc'])
plt.plot(hist1.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


print('Test accuracy for AE_dense model= {}'.format(decoder_ae_dense.evaluate(gist_test_ae, y_test_one_hot)[1]))


# In[ ]:


show_test(decoder_ae_dense, gist_test_ae)


# In[ ]:


predictions = decoder_ae_dense.predict(gist_test_ae)
report(predictions)


# ## Unet with convolutional NN as classifier:

# In[ ]:


decoder_un_conv = run_cls('conv', gist_train_unet)
decoder_un_conv.compile(loss='categorical_crossentropy',
                         optimizer=Adadelta(),
                         metrics=['accuracy'])
decoder_un_conv.summary()


# In[ ]:


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
hist1 = decoder_un_conv.fit(gist_train_unet, y_train, batch_size=512, epochs=100, 
                            validation_data = (gist_valid_unet, y_valid),
                            shuffle=True, callbacks=callbacks,
                            class_weight=class_weights)


# In[ ]:


plt.plot(hist1.history['acc'])
plt.plot(hist1.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


print('Test accuracy for Unet_conv model= {}'.format(decoder_un_conv.evaluate(gist_test_unet, y_test_one_hot)[1]))


# In[ ]:


show_test(decoder_un_conv, gist_test_unet)


# In[ ]:


predictions = decoder_un_conv.predict(gist_test_unet)
report(predictions)


# ## Unet with simple NN as classifier:

# In[ ]:


decoder_un_dense = run_cls('dense', gist_train_unet)
decoder_un_dense.compile(loss='categorical_crossentropy',
                         optimizer=Adadelta(),
                         metrics=['accuracy'])
decoder_un_dense.summary()


# In[ ]:


er = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
hist1 = decoder_un_dense.fit(gist_train_unet, y_train, batch_size=512, epochs=100, 
                             validation_data = (gist_valid_unet, y_valid),
                             shuffle=True, callbacks=callbacks,
                             class_weight=class_weights)


# In[ ]:


plt.plot(hist1.history['acc'])
plt.plot(hist1.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[ ]:


print('Test accuracy for Unet_dense model= {}'.format(decoder_un_dense.evaluate(gist_test_unet, y_test_one_hot)[1]))


# In[ ]:


show_test(decoder_un_dense, gist_test_unet)


# In[ ]:


predictions = decoder_un_dense.predict(gist_test_unet)
report(predictions)


# ## Multi-output Model:

# In[ ]:


def end_to_end():  ## I commented several layers of the model for descreasing model complexity as the results were almost same
    input = Input((32,32,3))
    
    # Encoder
    block1 = create_block(input, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    #block3 = create_block(x, 64)
    #x = MaxPool2D(2)(block3)
    #block4 = create_block(x, 128)
    
    # Middle
    #x = MaxPool2D(2)(block2)
    middle = create_block(x, 128)
    
    # Decoder
    #x = Conv2DTranspose(128, kernel_size=2, strides=2)(middle)
    #x = Concatenate()([block4, x])
    #x = create_block(x, 128)
    #x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
    #x = Concatenate()([block3, x])
    #x = create_block(x, 64)
    x = Conv2DTranspose(64, kernel_size=2, strides=2)(middle)
    x = Concatenate()([block2, x])
    x = create_block(x, 64)
    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)
    x = Concatenate()([block1, x])
    x = create_block(x, 32)
    
    # reconstruction
    x = Conv2D(3, 1)(x)
    recon = Activation("sigmoid", name='autoencoder')(x)
    
    #classification 
    c = Conv2D(1024, 3, padding="same")(middle)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(128, 3, padding="same")(c)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.4)(c)
    c = Flatten()(c)
    c = Dense(512, activation='relu')(c)
    c = Dropout(0.35)(c)
    c = Dense(100, activation='relu')(c)
    c = Dropout(0.69)(c)
    classify = Dense(10, activation='softmax', name='classification')(c)
    
    outputs = [recon, classify]
    
    return Model(input, outputs)


# In[ ]:


multimodel = end_to_end()
multimodel.compile(loss = {'classification': 'categorical_crossentropy', 'autoencoder': loss_function}, 
                  loss_weights = {'classification': 0.9, 'autoencoder': 0.1}, 
                  optimizer = SGD(lr= 0.01, momentum= 0.9),
                  metrics = {'classification': ['accuracy'], 'autoencoder': []})


# In[ ]:


er = EarlyStopping(monitor='val_classification_acc', patience=10, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_classification_acc', factor=0.2, patience=5, min_delta=0.0001)
callbacks = [er, lr]
hist_mul = multimodel.fit(x_train, [x_train,y_train], batch_size=512, epochs=100, 
                          validation_data = (x_valid, [x_valid,y_valid]),
                          shuffle=True, callbacks=callbacks)
#                           class_weight=class_weights


# In[ ]:


recon_test_e2e = multimodel.predict(x_test_final)[0]
recon_valid_e2e = multimodel.predict(x_valid)[0]


# In[ ]:


showOrigDec(x_valid, recon_valid_e2e)


# In[ ]:


showOrigDec(x_test_final, recon_test_e2e)


# In[ ]:


predictions = multimodel.predict(x_test_final)[1]
report(predictions)


# In[ ]:


show_test2(multimodel, x_test_final)


# Here, we trained a model that learns to generate the images and also classify them at the same time. We used a multi-output model for this task. The encoder part of the autoencoder model is shared for both the task to create similarity with how we did the classification in previous models where we used autoencoders as feature extractor. Also, we used loss_weights to give emphasis on the model to learn classification better. The autoencoder part of the model uses U-net architecture.
# 
# This multi-output model performs better than models where autoencoders were used as feature extractors, but not better than simple cnn models.

# ## Final Verdict:
# 
# 1. Although, U-net is vastly superior as autoencoder compared to the convolutional autoencoder; the bottleneck features extracted from this model performs badly while classifying. The bottleneck features extracted from the simple convolutional AE model performs better in terms of classifications.
# 
# 2. Convolution model with dense layer works better than stacked dense layers as classifier model.
# 
# 3. Multioutput model, that share the encoder part of the autoencoder, works better than learning the autoencoder first and then learning to classify.
# 
# 4. The baseline model without any autoencoder outperforms all the model.
# 
# 5. The model can not classify the classes properly that had less training data.

# ## Future Works:
# 
# I have to experiment more with different classifier models and different hyperparameters. The extracted features are expected to have the most important gist of data of the images. So, I expected models with AE to outperform the baseline model. Also, although U-net model can almost perfectly reconstruct even images in the test dataset inspite of data imbalance, the bottleneck features extracted from it as input to different classifier models performed worst, which was a shock to me.

# # Extra:

# ## Using sklearn models instead of neural networks:
# 
# The following code snippets were used for checking with different sklearn models as classifiers instead of neural networks. I used these to see if tree based models or svm performed better than neural networks to classify test images. But I found that neural networks performed better. Also, svm took a long time to run.

# In[ ]:


# def solvers(func):
#     scaler_classifier = MinMaxScaler(feature_range=(0.0, 1.0))
#     pipe = Pipeline(steps=[("scaler_classifier", scaler_classifier),
#                            ("classifier", func)])

#     pipe = pipe.fit(gist_train.reshape(gist_train.shape[0], -1), y_trainf)
#     acc = pipe.score(gist_test.reshape(gist_test.shape[0], -1), y_test_final)
#     predict = pipe.predict(gist_test.reshape(gist_test.shape[0], -1))
    
#     return acc, predict


# In[ ]:


# lr = LogisticRegression(C=5e-1, random_state=666, solver='lbfgs', multi_class='multinomial')
# rf = RandomForestClassifier(random_state=666)
# knn = KNeighborsClassifier()
# svc = svm.SVC()


# In[ ]:


# acc_lr, pred_lr = solvers(lr)
# acc_lr


# In[ ]:


# acc_rf, pred_rf = solvers(rf)
# acc_rf


# In[ ]:


# acc_knn, pred_knn = solvers(knn)
# acc_knn


# In[ ]:


# acc_svc, pred_svc = solvers(svc)
# acc_svc


# ## Hyperparameter Optimization:
# 
# The following code was used for hyperparameter optimization of the classifier model. The code was updated during various iterations to suit for different types of models used.

# In[ ]:


# space = {
#             'units1': hp.choice('units1', [256,512,1024]),
#             'units2': hp.choice('units2', [128,256,512]),
#             'units4': hp.choice('units4', [256,512,1024]),
#             'units5': hp.choice('units5', [50,64,100,128]),
#             'dropout1': hp.uniform('dropout1', .25,.75),
#             'dropout2': hp.uniform('dropout2', .25,.75),
#             'batch_size' : hp.choice('batch_size', [64,128,256,512]),
         
#             'nb_epochs' :  200,
#             'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
#             'activation': 'relu'
#         }


# In[ ]:


# def f_nn(params):   
#     from keras.models import Sequential
#     from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization
#     from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate
#     from keras.optimizers import Adadelta, Adam, rmsprop
#     import sys

#     print ('Params testing: ', params)
#     model = Sequential()
#     model.add(Conv2D(params['units1'], 3, padding="same", activation="relu"))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D())
#     model.add(Conv2D(params['units2'], 3, padding="same", activation="relu"))
#     model.add(BatchNormalization())
#     model.add(MaxPool2D())   

#     model.add(Flatten())
#     model.add(Dense(output_dim=params['units4'], activation="relu"))
#     model.add(Dropout(params['dropout1']))
#     model.add(Dense(output_dim=params['units5'], activation="relu"))
#     model.add(Dropout(params['dropout2']))
#     model.add(Dense(10))
#     model.add(Activation('softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

#     model.fit(gist_train, y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

#     acc = model.evaluate(gist_valid, y_valid)[1]
#     print('Accuracy:', acc)
#     sys.stdout.flush() 
#     return {'loss': -acc, 'status': STATUS_OK}


# trials = Trials()
# best = fmin(f_nn, space, algo=tpe.suggest, max_evals=5, trials=trials)
# print('best: ')
# print(best)


# In[ ]:




