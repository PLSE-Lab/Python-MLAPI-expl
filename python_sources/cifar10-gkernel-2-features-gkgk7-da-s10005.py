#!/usr/bin/env python
# coding: utf-8

# corresponding to keras 2.2

# In[ ]:


#!pip install keras==2.1.6


# In[ ]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


rs = 10005


# In[ ]:


import datetime
now = datetime.datetime.now()
print(now)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import os.path
import itertools
from itertools import chain

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import cluster, datasets, mixture
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

import tensorflow as tf

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D, GlobalAveragePooling2D,     GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
from keras.models import Model, Sequential
from keras import losses
from keras.callbacks import BaseLogger, ProgbarLogger, Callback, History
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras import initializers
from keras.metrics import categorical_accuracy
from keras.constraints import maxnorm, non_neg
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import keras


# In[ ]:


from PIL import Image
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm


# In[ ]:


from keras_ex.gkernel import GaussianKernel, GaussianKernel2, GaussianKernel3


# ## Load data

# In[ ]:


src_dir = '../input/cifar10-object-recognition-in-images-zip-file'
train_zip = os.path.join(src_dir, 'train_test/train.zip')
test_zip = os.path.join(src_dir, 'train_test/test.zip')


# In[ ]:


train_labels = pd.read_csv(os.path.join(src_dir, "trainLabels.csv"))
print(train_labels.shape)
train_labels.head(10)


# In[ ]:


id_key = dict([ee for ee in enumerate(np.unique(train_labels.label.values))])
id_key


# In[ ]:


key_id = dict([(ee[1], ee[0]) for ee in enumerate(np.unique(train_labels.label.values))])
key_id


# In[ ]:


y_train0 = np.array([key_id[ee] for ee in train_labels.label.values])
y_train0


# In[ ]:


test_labels = pd.read_csv(os.path.join(src_dir, "sampleSubmission.csv"))
print(test_labels.shape)
test_labels.head()


# In[ ]:


from zipfile import ZipFile


# In[ ]:


trainImg_list = []
with ZipFile(train_zip, 'r') as myzip:
    for ii in train_labels.id.values:
        with myzip.open('train/'+str(ii)+'.png') as tgt:
            img = Image.open(tgt)
            img_array = np.asarray(img)
            trainImg_list.append(img_array)


# In[ ]:


x_train0 = np.stack(trainImg_list).astype('float32') / 255.0
x_train0.shape


# In[ ]:


testImg_list = []
with ZipFile(test_zip, 'r') as myzip:
    for ii in test_labels.id.values:
        with myzip.open('test/'+str(ii)+'.png') as tgt:
            img = Image.open(tgt)
            img_array = np.asarray(img)
            testImg_list.append(img_array)


# In[ ]:


x_test = np.stack(testImg_list).astype('float32') / 255.0
x_test.shape


# In[ ]:


plt.imshow(x_train0[0])


# In[ ]:


plt.imshow(x_test[0])


# In[ ]:


y_cat_train0 = to_categorical(y_train0)
print(y_cat_train0.shape)


# In[ ]:


nrows=10
ncols=10
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = x_train0[ii*ncols + jj]
        iplt.imshow(img_array)


# In[ ]:


nrows=10
ncols=12
fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

for ii in range(nrows):
    idx = (y_train0 == ii)
    target_img = x_train0[idx][:ncols]
    for jj in range(ncols):
        iplt = subs[ii, jj]
        img_array = target_img[jj]
        iplt.imshow(img_array)


# In[ ]:


x_train = x_train0
y_train = y_train0
y_cat_train = y_cat_train0


# ## Create model

# In[ ]:


def make_trainable_false(model, trainable=False):
    layers = model.layers
    for ilayer in layers:
        ilayer.trainable = trainable
    return

class TrainableCtrl(object):
    
    def __init__(self, model_dic):
        self.model_dic = model_dic
        self.trainable_dic = {}
        self.get_trainable()
        
    def get_trainable(self):
        for k in self.model_dic:
            model = self.model_dic[k]
            res = []
            for ilayer in model.layers:
                res.append(ilayer.trainable)
            self.trainable_dic[k] = res
    
    def set_trainable_false(self, model_key):
        model = self.model_dic[model_key]
        make_trainable_false(model)
    
    def set_trainable_true(self, model_key):
        model = self.model_dic[model_key]
        for ii, ilayer in enumerate(model.layers):
            ilayer.trainable = self.trainable_dic[model_key][ii]


# In[ ]:


img_shape = x_train.shape[1:]
img_dim = np.array(img_shape).prod()
print(img_dim)

nn = 256*2 # output dim of img_cnvt

num_cnvt_lm = 2
num_cls = 10


# In[ ]:


n = 3
depth = n * 9 + 2


# In[ ]:


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


# In[ ]:


def _resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    x1 = GlobalMaxPooling2D()(x)
    x2 = GlobalAveragePooling2D()(x)
    x = concatenate([x1, x2])
    # v2 has BN-ReLU before Pooling
#     x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
#     x = AveragePooling2D(pool_size=8)(x)
    
    # Instantiate model.
    model = Model(inputs=inputs, outputs=x, name='model_img_converter')
    return model


# ### image_converter

# In[ ]:


model_img_cnvt = _resnet_v2(input_shape=img_shape, depth=depth)
model_img_cnvt.summary()


# In[ ]:


# SVG(model_to_dot(model_img_cnvt).create(prog='dot', format='svg'))


# ### gkernel2

# In[ ]:


def make_model_gkernel(nn=nn, num_lm=num_cnvt_lm, random_state=0, scale=1):
    inp = Input(shape=(nn,), name='inp')
    oup = inp
    
    np.random.seed(random_state)
    #init_wgt = (np.random.random_sample((num_lm, nn))-0.5) * scale
    init_wgt = np.random.random_sample((num_lm, nn))
    
    weights2 = [init_wgt, np.log(np.array([1/(2*nn*0.1*scale)]))]
    oup = GaussianKernel3(num_landmark=num_lm, num_feature=nn, weights=weights2, name='gkernel')(oup)
#     weights2 = [np.log(np.array([1/(2*nn*0.1*scale)]))]
#     oup = GaussianKernel2(init_wgt, weights=weights2, name='gkernel')(oup)
    model = Model(inp, oup, name='model_gkernel')
    return init_wgt, model

lm_gkernel, model_gkernel = make_model_gkernel(random_state=rs)
model_gkernel.summary()


# In[ ]:


print(lm_gkernel.shape)

df = pd.DataFrame(lm_gkernel[:,:5])
df.head()
fig = sns.pairplot(df, markers=['o'], height=2.2, diag_kind='hist')


# ### output layer

# In[ ]:


# def get_circle(nn=10, rs=None):
#     np.random.seed(rs)
#     idx = np.pi*2*np.arange(nn-1)/(nn-1)
#     idx += 2*np.pi*np.random.random(1)
#     init_wgt = np.c_[np.cos(idx), np.sin(idx)]
#     init_wgt = np.vstack([init_wgt, np.array([0,0])])
#     return np.random.permutation(init_wgt)
#     return init_wgt
def get_circle(nn=10, rs=None):
    np.random.seed(rs)
    idx = np.pi*2*np.arange(nn)/nn
    idx += 2*np.pi*np.random.random(1)
    idx = np.random.permutation(idx)
    #return idx
    init_wgt = np.c_[np.cos(idx), np.sin(idx)]
    return init_wgt


# In[ ]:


init_circle = get_circle(nn=num_cls, rs=rs)
init_circle = init_circle*0.8/2 + 0.5
print(init_circle)

df = pd.DataFrame(init_circle)
df['cls'] = ['c'+str(ee) for ee in range(num_cls)]
df.head()
fig = sns.pairplot(df, markers='o', size=2.2, diag_kind='hist', hue='cls')
axes = fig.axes
axes[0,0].set_xlim(0, 1)
axes[0,0].set_ylim(0, 1)
axes[1,1].set_xlim(0, 1)
axes[1,1].set_ylim(0, 1)


# In[ ]:


def make_models_out(init_heart, nn=num_cnvt_lm, num_cls=num_cls):
    inp = Input(shape=(nn,), name='inp')
    # oup = Dense(num_cls, activation='sigmoid')(inp)
#     init_wgt = np.random.random_sample((num_cls, nn))
#     weights = [init_wgt, np.log(np.array([1/(2*nn*0.1)]))]
#     oup = GaussianKernel3(num_landmark=num_cls, num_feature=nn, weights=weights, name='gkernel3')(inp)
    weights = [np.log(np.array([1/(2*nn*0.1)]))]
    oup = GaussianKernel2(init_heart, weights=weights, name='gkernel_out')(inp)
    model = Model(inp, oup, name='model_out')
    return model

model_out = make_models_out(init_circle)
model_out.summary()


# In[ ]:


# model_dic = {
#     'model_img_cnvt': model_img_cnvt,
#     'model_gkernel2': model_gkernel,
#     'model_out': model_out
# }
# train_ctrl = TrainableCtrl(model_dic)


# In[ ]:


def make_modelz(img_shape, model_img_cnvt, model_gkernel2, model_out):
    inp = Input(shape=img_shape, name='inp')
    oup = model_img_cnvt(inp)
    oup = model_gkernel2(oup)
    oup1 = model_out(oup)
    pre_model = Model(inp, oup1)
    pre_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return {
        'pre_model': pre_model,
        'model_img_cnvt': model_img_cnvt,
        'model_gkernel2': model_gkernel2,
        'model_out': model_out,
    }

models = make_modelz(img_shape, model_img_cnvt, model_gkernel, model_out)
models['pre_model'].summary()


# In[ ]:





# In[ ]:


THRESHOLD = 0.5

# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras

K_epsilon = K.epsilon()
def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K_epsilon)
    r = tp / (tp + fn + K_epsilon)

    f1 = 2*p*r / (p+r+K_epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    #y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K_epsilon)
    r = tp / (tp + fn + K_epsilon)

    f1 = 2*p*r / (p+r+K_epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)


# In[ ]:


'''
Thanks Iafoss.
pretrained ResNet34 with RGBY
https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
'''
gamma = 2.0
epsilon = K.epsilon()
def focal_loss(y_true, y_pred):
    pt = y_pred * y_true + (1-y_pred) * (1-y_true)
    pt = K.clip(pt, epsilon, 1-epsilon)
    CE = -K.log(pt)
    FL = K.pow(1-pt, gamma) * CE
    loss = K.sum(FL, axis=1)
    return loss


# ## Train

# In[ ]:


# train_ctrl.set_trainable_true('model_img_cnvt')
# train_ctrl.set_trainable_false('model_gkernel2')
# train_ctrl.set_trainable_false('model_out')

models['pre_model'].compile(loss=focal_loss,
                            optimizer='adam',
                            metrics=['categorical_accuracy', 'binary_accuracy', f1])
models['pre_model'].summary()


# In[ ]:


def lr_schedule(epoch):
    lr0 = 0.001
    epoch1 = 64
    epoch2 = 64
    epoch3 = 64
    epoch4 = 64
    
    if epoch<epoch1:
        lr = lr0
    elif epoch<epoch1+epoch2:
        lr = lr0/2
    elif epoch<epoch1+epoch2+epoch3:
        lr = lr0/4
    elif epoch<epoch1+epoch2+epoch3+epoch4:
        lr = lr0/8
    else:
        lr = lr0/16
    
    if divmod(epoch,4)[1] == 3:
        lr *= (1/8)
    elif divmod(epoch,4)[1] == 2:
        lr *= (1/4)
    elif divmod(epoch,4)[1] == 1:
        lr *= (1/2)
    elif divmod(epoch,4)[1] == 0:
        pass
    print('Learning rate: ', lr)
    return lr

# filepath = 'cifar10_model.{epoch:03d}.h5'
# filepath2 = 'cifar10_model.best.h5'

# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_categorical_accuracy',
#                              verbose=1,
#                              save_best_only=True)
# checkpoint2 = ModelCheckpoint(filepath=filepath2,
#                              monitor='val_categorical_accuracy',
#                              verbose=1,
#                              save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

#callbacks = [checkpoint, lr_scheduler, checkpoint2]
callbacks = [lr_scheduler]


# In[ ]:


datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=10,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)


# In[ ]:


# hst = models['pre_model'].fit(x_train, y_cat_train,
#                               epochs=10, batch_size=32, verbose=2)
it = datagen.flow(x_train, y_cat_train, batch_size=128)
hst = models['pre_model'].fit_generator(it, steps_per_epoch=len(it),
                                        #validation_data=(x_val, y_cat_val),
                                        epochs=128, verbose=2,
                                        #epochs=4, verbose=2,
                                        callbacks=callbacks)


# In[ ]:


hst_history = hst.history


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].set_title('loss')
ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["binary_accuracy"], label="binary_accuracy")
ax[2].set_title('f1_score')
ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["f1"], label="f1 score")
ax[0].legend()
ax[1].legend()
ax[2].legend()


# In[ ]:


# fig, ax = plt.subplots(1, 2, figsize=(20,5))
# ax[0].set_title('acc')
# ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["categorical_accuracy"], label="categorical_accuracy")
# ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["val_categorical_accuracy"], label="val_categorical_accuracy")
# ax[1].set_title('f1')
# ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["f1"], label="f1 score")
# ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["val_f1"], label="val_f1 score")
# ax[0].legend()
# ax[1].legend()


# In[ ]:


models['pre_model'].save_weights('cifar10_model.h5')


# In[ ]:





# In[ ]:


init_circle2 = get_circle(nn=num_cls, rs=rs)
init_circle2 = init_circle2*0.6/2 + 0.5
print(init_circle2)

df = pd.DataFrame(init_circle2)
df['cls'] = ['c'+str(ee) for ee in range(num_cls)]
df.head()
fig = sns.pairplot(df, markers='o', size=2.2, diag_kind='hist', hue='cls')
axes = fig.axes
axes[0,0].set_xlim(0, 1)
axes[0,0].set_ylim(0, 1)
axes[1,1].set_xlim(0, 1)
axes[1,1].set_ylim(0, 1)


# In[ ]:


model_out = make_models_out(init_circle2)
model_out.summary()


# In[ ]:


models = make_modelz(img_shape, model_img_cnvt, model_gkernel, model_out)
models['pre_model'].summary()


# In[ ]:


'''load saved weights'''
models['pre_model'].load_weights('cifar10_model.h5', by_name=False)


# In[ ]:


models['pre_model'].compile(loss=focal_loss,
                            optimizer='adam',
                            metrics=['categorical_accuracy', 'binary_accuracy', f1])
models['pre_model'].summary()


# In[ ]:


def lr_schedule(epoch):
    epoch +=128
    lr0 = 0.001
    epoch1 = 64
    epoch2 = 64
    epoch3 = 64
    epoch4 = 64
    
    if epoch<epoch1:
        lr = lr0
    elif epoch<epoch1+epoch2:
        lr = lr0/2
    elif epoch<epoch1+epoch2+epoch3:
        lr = lr0/4
    elif epoch<epoch1+epoch2+epoch3+epoch4:
        lr = lr0/8
    else:
        lr = lr0/16
    
    if divmod(epoch,4)[1] == 3:
        lr *= (1/8)
    elif divmod(epoch,4)[1] == 2:
        lr *= (1/4)
    elif divmod(epoch,4)[1] == 1:
        lr *= (1/2)
    elif divmod(epoch,4)[1] == 0:
        pass
    print('Learning rate: ', lr)
    return lr

# filepath = 'cifar10_model.{epoch:03d}.h5'
# filepath2 = 'cifar10_model.best.h5'

# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_categorical_accuracy',
#                              verbose=1,
#                              save_best_only=True)
# checkpoint2 = ModelCheckpoint(filepath=filepath2,
#                              monitor='val_categorical_accuracy',
#                              verbose=1,
#                              save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# callbacks = [checkpoint, lr_scheduler, checkpoint2]
callbacks = [lr_scheduler]


# In[ ]:


# hst = models['pre_model'].fit(x_train, y_cat_train,
#                               epochs=10, batch_size=32, verbose=2)
it = datagen.flow(x_train, y_cat_train, batch_size=128)
hst = models['pre_model'].fit_generator(it, steps_per_epoch=len(it),
                                        #validation_data=(x_val, y_cat_val),
                                        epochs=64, verbose=2,
                                        #epochs=4, verbose=2,
                                        callbacks=callbacks)


# In[ ]:


'''merge hst_history'''
for k in hst.history:
    hst_history[k].extend(hst.history[k])


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].set_title('loss')
ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["binary_accuracy"], label="binary_accuracy")
ax[2].set_title('f1_score')
ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["f1"], label="f1 score")
ax[0].legend()
ax[1].legend()
ax[2].legend()


# In[ ]:


models['pre_model'].save_weights('cifar10_model.h5')


# In[ ]:





# In[ ]:


init_circle3 = get_circle(nn=num_cls, rs=rs)
init_circle3 = init_circle3*0.4/2 + 0.5
print(init_circle3)

df = pd.DataFrame(init_circle3)
df['cls'] = ['c'+str(ee) for ee in range(num_cls)]
df.head()
fig = sns.pairplot(df, markers='o', size=2.2, diag_kind='hist', hue='cls')
axes = fig.axes
axes[0,0].set_xlim(0, 1)
axes[0,0].set_ylim(0, 1)
axes[1,1].set_xlim(0, 1)
axes[1,1].set_ylim(0, 1)


# In[ ]:


model_out = make_models_out(init_circle3)
model_out.summary()


# In[ ]:


models = make_modelz(img_shape, model_img_cnvt, model_gkernel, model_out)
models['pre_model'].summary()


# In[ ]:


'''load saved weights'''
models['pre_model'].load_weights('cifar10_model.h5', by_name=False)


# In[ ]:


models['pre_model'].compile(loss=focal_loss,
                            optimizer='adam',
                            metrics=['categorical_accuracy', 'binary_accuracy', f1])
models['pre_model'].summary()


# In[ ]:


def lr_schedule(epoch):
    epoch += 128+64
    lr0 = 0.001
    epoch1 = 64
    epoch2 = 64
    epoch3 = 64
    epoch4 = 64
    
    if epoch<epoch1:
        lr = lr0
    elif epoch<epoch1+epoch2:
        lr = lr0/2
    elif epoch<epoch1+epoch2+epoch3:
        lr = lr0/4
    elif epoch<epoch1+epoch2+epoch3+epoch4:
        lr = lr0/8
    else:
        lr = lr0/16
    
    if divmod(epoch,4)[1] == 3:
        lr *= (1/8)
    elif divmod(epoch,4)[1] == 2:
        lr *= (1/4)
    elif divmod(epoch,4)[1] == 1:
        lr *= (1/2)
    elif divmod(epoch,4)[1] == 0:
        pass
    print('Learning rate: ', lr)
    return lr

# filepath = 'cifar10_model.{epoch:03d}.h5'
# filepath2 = 'cifar10_model.best.h5'

# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_categorical_accuracy',
#                              verbose=1,
#                              save_best_only=True)
# checkpoint2 = ModelCheckpoint(filepath=filepath2,
#                              monitor='val_categorical_accuracy',
#                              verbose=1,
#                              save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# callbacks = [checkpoint, lr_scheduler, checkpoint2]
callbacks = [lr_scheduler]


# In[ ]:


# hst = models['pre_model'].fit(x_train, y_cat_train,
#                               epochs=10, batch_size=32, verbose=2)
it = datagen.flow(x_train, y_cat_train, batch_size=128)
hst = models['pre_model'].fit_generator(it, steps_per_epoch=len(it),
                                        #validation_data=(x_val, y_cat_val),
                                        epochs=128, verbose=2,
                                        #epochs=4, verbose=2,
                                        callbacks=callbacks)


# In[ ]:


'''merge hst_history'''
for k in hst.history:
    hst_history[k].extend(hst.history[k])


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20,5))
ax[0].set_title('loss')
ax[0].plot(list(range(len(hst_history["loss"]))), hst_history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(list(range(len(hst_history["loss"]))), hst_history["binary_accuracy"], label="binary_accuracy")
ax[2].set_title('f1_score')
ax[2].plot(list(range(len(hst_history["loss"]))), hst_history["f1"], label="f1 score")
ax[0].legend()
ax[1].legend()
ax[2].legend()


# In[ ]:


models['pre_model'].save_weights('cifar10_model.h5')


# In[ ]:


# '''load saved weights'''
# models['pre_model'].load_weights(filepath2, by_name=False)


# In[ ]:


pred_img_cnvt = model_img_cnvt.predict(x_train0, batch_size=128, verbose=1)
print(pred_img_cnvt.shape)
pred_img_cnvt


# In[ ]:


df = pd.DataFrame(pred_img_cnvt[:,:5])
df['cls'] = ['c'+str(ee) for ee in y_train0]
df.head()
fig = sns.pairplot(df, markers='o', hue='cls', height=2.2, diag_kind='hist')


# In[ ]:


df = pd.DataFrame(np.vstack([pred_img_cnvt, lm_gkernel])[:,:5])
df['cls'] = ['c'+str(ee) for ee in y_train0] + ['LM']*lm_gkernel.shape[0]
df.head()
fig = sns.pairplot(df, markers=['.']*num_cls + ['s'], hue='cls', height=2.2, diag_kind='hist')


# In[ ]:


y_pred0 = models['pre_model'].predict(x_train0, batch_size=128, verbose=1)
y_pred0.shape


# In[ ]:


print(f1_score(y_train0, np.argmax(y_pred0, axis=1), average='macro'))
print(classification_report(y_train0, np.argmax(y_pred0, axis=1)))
confusion_matrix(y_train0, np.argmax(y_pred0, axis=1))


# In[ ]:


y_pred_test = models['pre_model'].predict(x_test, batch_size=128, verbose=0)
y_pred_test.shape


# In[ ]:


# print(f1_score(y_test, np.argmax(y_pred_test, axis=1), average='macro'))
# print(classification_report(y_test, np.argmax(y_pred_test, axis=1)))
# confusion_matrix(y_test, np.argmax(y_pred_test, axis=1))


# In[ ]:


pred_gkernel = model_gkernel.predict(pred_img_cnvt, batch_size=128, verbose=1)
print(pred_gkernel.shape)


# In[ ]:


df = pd.DataFrame(pred_gkernel)
df.columns = ["comp_1", "comp_2"]
df['cls'] = ['c'+str(ee) for ee in y_train0]
print(df.head())
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
sns.lmplot("comp_1", "comp_2", hue="cls", data=df, fit_reg=False, markers='.')
# fig = sns.pairplot(df, markers='.', hue='cls', height=2.2, diag_kind='hist')


# In[ ]:


df = pd.DataFrame(np.vstack([pred_gkernel, init_circle3]))
df.columns = ["comp_1", "comp_2"]
df['cls'] = ['c'+str(int(ee)) for ee in y_train0] + ['LM']*(init_circle3.shape[0])
print(df.head())

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
sns.lmplot("comp_1", "comp_2", hue="cls", data=df, fit_reg=False, markers=['.']*10 + ['s'])
# fig = sns.pairplot(df, markers=['.']*num_cls+['s'], hue='cls', height=2.2, diag_kind='hist')


# In[ ]:





# In[ ]:


df_pred1 = pd.DataFrame({'label': y_train0})
df_pred1 = pd.concat([df_pred1, pd.DataFrame(y_pred0)], axis=1)
print(df_pred1.shape)
df_pred1.to_csv('proba.csv', index=False)
df_pred1.head()


# In[ ]:


#df_pred1_test = pd.DataFrame({'label': y_test})
df_pred1_test = pd.DataFrame(y_pred_test)
print(df_pred1_test.shape)
df_pred1_test.to_csv('proba_test.csv', index=False)
df_pred1_test.head()


# In[ ]:


submit_csv = test_labels.copy()
submit_csv.label = [id_key[ee] for ee in np.argmax(y_pred_test, axis=1)]

submit_csv.to_csv('submit.csv', index=False)
submit_csv.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




