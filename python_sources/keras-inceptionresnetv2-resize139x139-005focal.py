#!/usr/bin/env python
# coding: utf-8

# * no data expand, but augment  
# * try batch_size=128  
# * pre-train
# * tanh
# * focul loss

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import os
import shutil
print(os.listdir("../input"))


# In[ ]:


ls -la ../input/keras-pretrained-models


# In[ ]:


try:
    os.makedirs('/tmp/.keras/datasets')
except FileExistsError:
    pass


# In[ ]:


try:
    shutil.copytree("../input/keras-pretrained-models", "/tmp/.keras/models")
except FileExistsError:
    pass


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

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Lambda,     Conv1D, Conv2D, Conv3D,     Conv2DTranspose,     AveragePooling1D, AveragePooling2D,     MaxPooling1D, MaxPooling2D, MaxPooling3D,     GlobalAveragePooling1D,     GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D,     LocallyConnected1D, LocallyConnected2D,     concatenate, Flatten, Average, Activation,     RepeatVector, Permute, Reshape, Dot,     multiply, dot, add,     PReLU,     Bidirectional, TimeDistributed,     SpatialDropout1D,     BatchNormalization
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


# In[ ]:


from PIL import Image
from zipfile import ZipFile
import h5py
import cv2
from tqdm import tqdm


# In[ ]:


src_dir = '../input/human-protein-atlas-image-classification'


# In[ ]:


train_labels = pd.read_csv(os.path.join(src_dir, "train.csv"))
print(train_labels.shape)
train_labels.head(10)


# In[ ]:


test_labels = pd.read_csv(os.path.join(src_dir, "sample_submission.csv"))
print(test_labels.shape)
test_labels.head()


# In[ ]:


def show_arr(arr, nrows = 1, ncols = 4, figsize=(15, 5)):
    fig, subs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ii in range(ncols):
        iplt = subs[ii]
        try:
            img_array = arr[:,:,ii]
            if ii == 0:
                cp = 'Greens'
            elif ii == 1:
                cp = 'Blues'
            elif ii == 2:
                cp = 'Reds'
            else:
                cp = 'Oranges'
            iplt.imshow(img_array, cmap=cp)
        except:
            pass


# In[ ]:


def get_arr0(Id, test=False):
    def fn(Id, color, test=False):
        if test:
            tgt = 'test'
        else:
            tgt = 'train'
        with open(os.path.join(src_dir, tgt, Id+'_{}.png'.format(color)), 'rb') as fp:
            img = Image.open(fp)
            arr = (np.asarray(img) / 255.)
        return arr
    res = []
    for icolor in ['green', 'blue', 'red', 'yellow']:
        arr0 = fn(Id, icolor, test)
        res.append(arr0)
    arr = np.stack(res, axis=-1)
    return arr


# In[ ]:


arr = get_arr0('00008af0-bad0-11e8-b2b8-ac1f6b6435d0', test=True)
print(arr.shape)
show_arr(arr)


# In[ ]:


arr = get_arr0('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0')
print(arr.shape)
show_arr(arr)


# In[ ]:


SH = (139, 139)
ID_LIST_TRAIN = train_labels.Id.tolist()


# ### load cache

# In[ ]:


img_cache_train = np.load('../input/139x139-resized-numpy-array/img_cache_train_resize_139x139.npz')['x']


# In[ ]:


img_cache_train.shape


# In[ ]:


def get_arr(Id, test=False):
    if test:
        arr = get_arr0(Id, test=True)
        arr = cv2.resize(arr, SH).astype('float32')
    else:
        ii = ID_LIST_TRAIN.index(Id)
        arr = img_cache_train[ii]
        arr = arr.astype('float32')
    return arr


# In[ ]:


arr = get_arr('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0')
arr.shape


# In[ ]:


show_arr(arr)


# In[ ]:


arr = get_arr('00008af0-bad0-11e8-b2b8-ac1f6b6435d0', test=True)
print(arr.shape)
show_arr(arr)


# In[ ]:


y_cat_train_dic = {}
for icat in range(28):
    target = str(icat)
    y_cat_train_5 = np.array([int(target in ee.split()) for ee in train_labels.Target.tolist()])
    y_cat_train_dic[icat] = y_cat_train_5


# In[ ]:


up_sample = {}
for k in y_cat_train_dic:
    v = y_cat_train_dic[k].sum()
    up_sample[k] = np.ceil((train_labels.shape[0]/28) / v)

up_sample


# In[ ]:


up_sample2 = list(zip(*sorted(list(up_sample.items()), key=lambda x: x[0])))[1]
up_sample2 = np.array(up_sample2)
up_sample2


# In[ ]:


import random

class Seq(object):
    sections = None
    index = None
    
    def __init__(self, df, extend=False, aug=False, test=False, batch_size=32):
        self.shaffle = None
        self.extend = extend
        self.aug = aug
        self.test = test
        self.batch_size = batch_size
        self.df = df
        
        # proccess
        self.ids = self.df.Id.tolist()
        self.reversed = sorted(range(SH[0]), reverse=True)
        
        # estimate self length
        self.initialize_it()
        self.len = 1
        for _ in self.it:
            self.len += 1
        
        self.initialize_it()
    
    def initialize_it(self):
        if self.shaffle:
            '''not implemented yet'''
            raise NotImplementedError
            #random.seed(self.state)
            #random.shuffle(self.ids)
        
        self.it = iter(range(0, len(self.ids), self.batch_size))
        self.idx_next = self.it.__next__()
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = self.idx_next
        self.ids_part = self.ids[idx:((idx+self.batch_size) if idx+self.batch_size<len(self.ids) else len(self.ids))]
        res = self.getpart(self.ids_part)
        try:
            self.idx_next = self.it.__next__()
        except StopIteration:
            self.initialize_it()
        return res
    
    def __getitem__(self, id0):
        arr, tgts = self.get_data(id0)
        cat = self.convert_tgts(tgts)
        return arr, cat
    
    k_list = list(range(4))
    def random_transform(self, arr):
        k = random.choice(self.k_list)
        arr0 = np.rot90(arr, k=k)
        if random.randint(0,1):
            arr0 = arr0[self.reversed,:,:]
        if random.randint(0,1):
            arr0 = arr0[:,self.reversed,:]
        return arr0
    
    def convert_tgts(self, tgts):
        try:
            cats = to_categorical(tgts, num_classes=28)
            cat = cats.sum(axis=0)
        except TypeError:
            cat = np.zeros((28,))
        return cat
    
    def get_data(self, id0):
        arr = get_arr(id0, test=self.test)
        
        try:
            y0 = (self.df.Target[self.df.Id == id0]).tolist()[0]
            y1 = y0.split()
            y = [int(ee) for ee in y1]
        except AttributeError:
            y = None
        return arr, y
    
    def getpart(self, ids):
        xs = []
        ys = []
        for id0 in ids:
            self.extend_data(id0, xs, ys)
        
        x = np.stack(xs)
        y = np.stack(ys)
        x_dummy = np.zeros((len(x), 1))
        x_ret = {
            'input': x,
            'input_cls': y,
        }
        y_ret = {
            'path_fit_cls_img': x_dummy,
            'path_cls_img_cls': y,
            'path_fit_imgA': x_dummy,
            'path_fit_img_cls_img': x_dummy,
            'path_cls_cls': y,
            'path_img_cls': y,
            'path_img_img_cls': y,
            'path_fit_cls_img_imgE': x_dummy,
        }
        return (x_ret, y_ret)
    
    def split(self, arr, sections=sections):
        res0 = np.vsplit(arr, sections)
        res = [np.hsplit(ee, sections) for ee in res0]
        res = list(chain.from_iterable(res))
        return res
    
    def extend_data(self, id0, xs, ys):
        arr0, cat = self[id0]
        
        # data augmentation
        if self.extend:
            mm = up_sample2[cat==1].max()
            mm = int(mm)
            #print(mm)
            for ii in range(mm):
                if self.aug:
                    img = self.random_transform(arr0)
                else:
                    img = arr0
                xs.append(img.flatten())
                ys.append(cat)
        else:
            if self.aug:
                img = self.random_transform(arr0)
            else:
                img = arr0
            xs.append(img.flatten())
            ys.append(cat)


# ### make model

# In[ ]:


from keras import applications


# In[ ]:


model_resnet = applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(139,139,3),
    pooling='avg',
    classes=None)


# In[ ]:


model_resnet.summary()


# In[ ]:


def make_trainable_false(model_resnet, trainable=False):
    layers = model_resnet.layers
    for ilayer in layers:
        ilayer.trainable = trainable
    return


# In[ ]:


model_resnet.layers[1].get_weights()[0][:,:,:,0]


# In[ ]:


img_shape = (139, 139, 4)
img_dim = np.array(img_shape).prod()
print(img_dim)


# In[ ]:


def make_model_cnvt(img_dim, img_shape):
    '''==============================
    inputs
    =============================='''
    inp = Input(shape=(img_dim,))
    oup = Reshape(img_shape)(inp)
    #oup = Conv2D(3, kernel_size=1, strides=1, padding='same')(oup)
    #oup = Conv2D(3, kernel_size=1, strides=1, padding='same', activation='sigmoid')(oup)
    oup = Conv2D(3,
                 kernel_size=1,
                 strides=1,
                 padding='same',
                 activation='tanh',
                 kernel_regularizer=regularizers.l2(1e-4))(oup)
    #kernel_regularizer=regularizers.l2(1e-4)
    model_cnvt = Model(inp, oup, name='model_cnvt')
    return model_cnvt


# In[ ]:


model_cnvt = make_model_cnvt(img_dim, img_shape)
model_cnvt.summary()


# In[ ]:


def make_model_classifier(input_dim=1536):
    inp_cls = Input((input_dim,))
    oup_cls = Dense(28)(inp_cls)
    oup_cls = Activation('sigmoid')(oup_cls)
    model_classifier = Model(inp_cls, oup_cls, name='classifier')
    return model_classifier


# In[ ]:


model_classifier = make_model_classifier()
model_classifier.summary()


# In[ ]:


def make_model(img_dim, model_cnvt, model_resnet, model_classifier):
    '''==============================
    inputs
    =============================='''
    inp = Input(shape=(img_dim,), name='input')
    oup = model_cnvt(inp)
    oup = model_resnet(oup)
    oup = model_classifier(oup)
    oup = Activation('linear', name='path_cls_cls')(oup)
    
    model = Model(inp, oup, name='model')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy', 'binary_accuracy'])
    
    return {
        'model_classifier': model_classifier,
        'model_resnet': model_resnet,
        'model_cnvt': model_cnvt,
        'model': model
    }

models = make_model(img_dim, model_cnvt, model_resnet, model_classifier)
models['model'].summary()


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


# ## #1
#  * frozen
#  * model_resnet.trainable=False

# In[ ]:


model_resnet.layers[1].get_weights()[0][:,:,:,0]


# In[ ]:


make_trainable_false(model_resnet, trainable=False)
models['model'].compile(loss=focal_loss,
                        optimizer='adam',
                        metrics=['categorical_accuracy', 'binary_accuracy'])


# In[ ]:


model_resnet.layers[1].get_weights()[0][:,:,:,0]


# In[ ]:


def lr_schedule(epoch):
    lr = 1e-3
    if epoch == 0:
        pass
    elif epoch == 1:
        pass
    elif epoch == 2:
        pass
    else:
        lr *= 0.1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]


# In[ ]:


seq = Seq(train_labels, extend=False, aug=True, batch_size=128)
print(len(seq))

hst = models['model'].fit_generator(seq, epochs=5,
                              steps_per_epoch=len(seq),
                              callbacks=callbacks)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hst.epoch, hst.history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(hst.epoch, hst.history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(hst.epoch, hst.history["binary_accuracy"], label="binary_accuracy")
ax[0].legend()
ax[1].legend()


# In[ ]:


model_resnet.layers[1].get_weights()[0][:,:,:,0]


# In[ ]:


seq = Seq(train_labels, extend=False, aug=False, batch_size=32)
print(len(seq))
xs, ys = next(seq)
print(xs['input'].shape)
y_pred = models['model_cnvt'].predict(xs['input'])
print(y_pred.shape)


# In[ ]:


y_pred[0]


# In[ ]:


show_arr(y_pred[0])


# In[ ]:


Image.fromarray(np.uint8((y_pred[0]+1)/2*255))


# ### predict and submit

# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
len(seq_pred)


# In[ ]:


pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)


# In[ ]:


def calc_threshold(pred):
    ### calc threshold
    threshold_dic = {}
    for idx in tqdm(range(28)):
        m = 0
        for ii in range(100):
            threshold0 = ii*0.01
            f1_val = f1_score(y_cat_train_dic[idx], threshold0<(pred[:,idx]))
            if m < f1_val:
                threshold_dic[idx] = threshold0+0.005
                m = f1_val
    return threshold_dic


# In[ ]:


threshold_dic = calc_threshold(pred)


# In[ ]:


threshold_dic


# In[ ]:


seq_test = Seq(test_labels, test=True, aug=False, batch_size=128)
seq_test


# In[ ]:


pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)


# In[ ]:


def make_test(pred):
    test_labels1 = test_labels.copy()
    test_labels1['Predicted'] = [str(ee) for ee in np.argmax(pred, axis=1)]
    print(test_labels1.head())
    #test_labels1.to_csv(fn0, index=False)
    
    test_labels2 = test_labels1.copy()
    for ii in range(test_labels2.shape[0]):
        threshold = list(zip(*sorted(list(threshold_dic.items()), key=lambda x:x[0], reverse=False)))[1]
        idx = threshold < pred[ii,:]
        tgt = test_labels2['Predicted'][ii]
        tgt = [tgt] + [str(ee) for ee in np.arange(28)[idx]]
        tgt = set(tgt)
        tgt = ' '.join(tgt)
        test_labels2['Predicted'][ii] = tgt
    print(test_labels2.head())
    #test_labels2.to_csv(fn, index=False)
    return test_labels1, test_labels2


# In[ ]:


test_labels1_1, test_labels1_2 = make_test(pred_test)


# In[ ]:


test_labels1_2.head()


# In[ ]:


test_labels1_2.to_csv('InceptionResNetV1_2.csv', index=False)


# ## save weights for later loading

# In[ ]:


'''save weights for later loading'''
No = 1
models['model_cnvt'].save_weights('model_{}_cnvt.h5'.format(No))
models['model_resnet'].save_weights('model_{}_resnet.h5'.format(No))
models['model_classifier'].save_weights('model_{}_classifier.h5'.format(No))


# In[ ]:


ls -la


# load test

# In[ ]:


# model_cnvt = make_model_cnvt(512*512*4, (512,512,4))
# model_cnvt.summary()


# In[ ]:


# model_cnvt.layers[2].get_weights()


# In[ ]:


# model_cnvt.load_weights('model_1_cnvt.h5')
# model_cnvt.layers[2].get_weights()


# In[ ]:


# model_classifier = make_model_classifier()
# model_classifier.summary()


# In[ ]:


# model_classifier.layers[1].get_weights()[1]


# In[ ]:


# model_classifier.load_weights('model_1_classifier.h5')
# model_classifier.layers[1].get_weights()[1]


# In[ ]:


# model_resnet = applications.inception_resnet_v2.InceptionResNetV2(
#     include_top=False,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=(512,512,3),
#     pooling='avg',
#     classes=None)
# model_resnet.summary()


# In[ ]:


# models2 = make_model(512*512*4, model_cnvt, model_resnet, model_classifier)
# models2['model'].summary()


# In[ ]:





# ## #2
# * batch_size = 64
# * model_resnet.trainable=True
# 
# make model_resnet    
# make model with model_cnvt, model_resnet and model_classifier

# In[ ]:


model_resnet = applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(139,139,3),
    pooling='avg',
    classes=None)
# model_resnet.summary()


# In[ ]:


models = make_model(img_dim, model_cnvt, model_resnet, model_classifier)
models['model'].summary()


# In[ ]:


models['model'].compile(loss=focal_loss,
                        optimizer='adam',
                        metrics=['categorical_accuracy', 'binary_accuracy'])


# In[ ]:


seq = Seq(train_labels, aug=True, batch_size=64)
print(len(seq))

hst = models['model'].fit_generator(seq, epochs=5,
                              steps_per_epoch=len(seq),
                              callbacks=[])


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hst.epoch, hst.history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(hst.epoch, hst.history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(hst.epoch, hst.history["binary_accuracy"], label="binary_accuracy")
ax[0].legend()
ax[1].legend()


# In[ ]:


model_resnet.layers[1].get_weights()[0][:,:,:,0]


# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
len(seq_pred)
pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)
threshold_dic = calc_threshold(pred)
threshold_dic


# In[ ]:


seq_test = Seq(test_labels, test=True, aug=False, batch_size=128)
seq_test
pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)
test_labels2_1, test_labels2_2 = make_test(pred_test)
test_labels2_2.head()


# In[ ]:


test_labels2_2.to_csv('InceptionResNetV2_2.csv', index=False)


# In[ ]:


'''save weights for later loading'''
No = 2
models['model_cnvt'].save_weights('model_{}_cnvt.h5'.format(No))
models['model_resnet'].save_weights('model_{}_resnet.h5'.format(No))
models['model_classifier'].save_weights('model_{}_classifier.h5'.format(No))


# ## #3
# batch_size = 128

# In[ ]:


seq = Seq(train_labels, aug=True, batch_size=128)
print(len(seq))

hst = models['model'].fit_generator(seq, epochs=5,
                              steps_per_epoch=len(seq),
                              callbacks=[])


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hst.epoch, hst.history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(hst.epoch, hst.history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(hst.epoch, hst.history["binary_accuracy"], label="binary_accuracy")
ax[0].legend()
ax[1].legend()


# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
len(seq_pred)
pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)
threshold_dic = calc_threshold(pred)
threshold_dic


# In[ ]:


seq_test = Seq(test_labels, test=True, aug=False, batch_size=128)
seq_test
pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)
test_labels3_1, test_labels3_2 = make_test(pred_test)
test_labels3_2.head()


# In[ ]:


test_labels3_2.to_csv('InceptionResNetV3_2.csv', index=False)


# In[ ]:


'''save weights for later loading'''
No = 3
models['model_cnvt'].save_weights('model_{}_cnvt.h5'.format(No))
models['model_resnet'].save_weights('model_{}_resnet.h5'.format(No))
models['model_classifier'].save_weights('model_{}_classifier.h5'.format(No))


# ## #4
# batch_size = 128

# In[ ]:


def lr_schedule(epoch):
    lr = 1e-4
    print('Learning rate: ', lr)
    return lr


# In[ ]:


lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]

seq = Seq(train_labels, aug=True, batch_size=128)
print(len(seq))

hst = models['model'].fit_generator(seq, epochs=5,
                              steps_per_epoch=len(seq),
                              callbacks=callbacks)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hst.epoch, hst.history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(hst.epoch, hst.history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(hst.epoch, hst.history["binary_accuracy"], label="binary_accuracy")
ax[0].legend()
ax[1].legend()


# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
len(seq_pred)
pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)
threshold_dic = calc_threshold(pred)
threshold_dic


# In[ ]:


seq_test = Seq(test_labels, test=True, aug=False, batch_size=128)
seq_test
pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)
test_labels4_1, test_labels4_2 = make_test(pred_test)
test_labels4_2.head()


# In[ ]:


test_labels4_2.to_csv('InceptionResNetV4_2.csv', index=False)


# In[ ]:


'''save weights for later loading'''
No = 4
models['model_cnvt'].save_weights('model_{}_cnvt.h5'.format(No))
models['model_resnet'].save_weights('model_{}_resnet.h5'.format(No))
models['model_classifier'].save_weights('model_{}_classifier.h5'.format(No))


# ## #5

# In[ ]:


def lr_schedule(epoch):
    lr = 1e-4
    print('Learning rate: ', lr)
    return lr


# In[ ]:


lr_scheduler = LearningRateScheduler(lr_schedule)
callbacks = [lr_scheduler]

seq = Seq(train_labels, aug=True, batch_size=128)
print(len(seq))

hst = models['model'].fit_generator(seq, epochs=5,
                              steps_per_epoch=len(seq),
                              callbacks=callbacks)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hst.epoch, hst.history["loss"], label="Train loss")
ax[1].set_title('acc')
ax[1].plot(hst.epoch, hst.history["categorical_accuracy"], label="categorical_accuracy")
ax[1].plot(hst.epoch, hst.history["binary_accuracy"], label="binary_accuracy")
ax[0].legend()
ax[1].legend()


# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
len(seq_pred)
pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)
threshold_dic = calc_threshold(pred)
threshold_dic


# In[ ]:


seq_test = Seq(test_labels, test=True, aug=False, batch_size=128)
seq_test
pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)
test_labels5_1, test_labels5_2 = make_test(pred_test)
test_labels5_2.head()


# In[ ]:


test_labels5_2.to_csv('InceptionResNetV5_2.csv', index=False)


# In[ ]:


'''save weights for later loading'''
No = 5
models['model_cnvt'].save_weights('model_{}_cnvt.h5'.format(No))
models['model_resnet'].save_weights('model_{}_resnet.h5'.format(No))
models['model_classifier'].save_weights('model_{}_classifier.h5'.format(No))


# In[ ]:


ls -la


# In[ ]:





# In[ ]:


seq = Seq(train_labels, extend=False, aug=False, batch_size=32)
print(len(seq))
xs, ys = next(seq)
print(xs['input'].shape)
y_pred = models['model_cnvt'].predict(xs['input'])
print(y_pred.shape)


# In[ ]:


y_pred[0]


# In[ ]:


show_arr(y_pred[0])


# In[ ]:


Image.fromarray(np.uint8((y_pred[0]+1)/2*255))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




