#!/usr/bin/env python
# coding: utf-8

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


os.makedirs('/tmp/.keras/datasets')


# In[ ]:


shutil.copytree("../input/keras-pretrained-models", "/tmp/.keras/models")


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


# ### cache train data

# In[ ]:


# ### CACHE
# img_cache_train = np.zeros((train_labels.shape[0], 139, 139, 4), dtype=np.float16)
# ID_LIST_TRAIN = train_labels.Id.tolist()


# In[ ]:


# for ii, id0 in enumerate(tqdm(ID_LIST_TRAIN)):
#     arr = get_arr0(id0)
#     img_cache_train[ii] = cv2.resize(arr[:], SH)


# In[ ]:


# img_cache_train.shape


# In[ ]:


# np.savez_compressed('img_cache_train_resize_139x139', x=img_cache_train)


# ### load cache

# In[ ]:


img_cache_train = np.load('../input/139x139-resized-numpy-array/img_cache_train_resize_139x139.npz')['x']


# In[ ]:


img_cache_train.shape


# In[ ]:





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
    
    def __init__(self, df, aug=True, test=False, batch_size=32):
        self.shaffle = None
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
            if self.aug:
                self.extend_data(id0, xs, ys)
            else:
                img, cat = self[id0]
                xs.append(img.flatten())
                ys.append(cat)
        
        x = np.stack(xs)
        y = np.stack(ys)
        x_ret = x
        y_ret = y
        return (x_ret, y_ret)
    
    def split(self, arr, sections=sections):
        res0 = np.vsplit(arr, sections)
        res = [np.hsplit(ee, sections) for ee in res0]
        res = list(chain.from_iterable(res))
        return res
    
    def extend_data(self, id0, xs, ys):
        arr0, cat = self[id0]
        
        # data augmentation
        mm = up_sample2[cat==1].max()
        mm = int(mm)
        #print(mm)
        for ii in range(mm):
            img = self.random_transform(arr0)
            xs.append(img.flatten())
            ys.append(cat)


# In[ ]:


seq = Seq(train_labels, batch_size=32)
seq


# In[ ]:


print(len(seq.ids))
len(seq)


# In[ ]:


arr, y = seq.get_data('ad5a4858-bb9d-11e8-b2b9-ac1f6b6435d0')
print(arr.shape)
show_arr(arr)


# In[ ]:


x, y = seq['ad5a4858-bb9d-11e8-b2b9-ac1f6b6435d0']
print(x.shape)
show_arr(x)


# In[ ]:


xs, ys = next(seq)


# In[ ]:


xs.shape


# In[ ]:


show_arr(xs[0].reshape((139,139,4)))


# In[ ]:


show_arr(xs[1].reshape((139,139,4)))


# In[ ]:


show_arr(xs[2].reshape((139,139,4)))


# In[ ]:


show_arr(xs[3].reshape((139,139,4)))


# In[ ]:


show_arr(xs[6].reshape((139,139,4)).astype('float32'))


# In[ ]:


show_arr(xs[7].reshape((139,139,4)).astype('float32'))


# ### test_labels, test=True, aug=False

# In[ ]:


seq = Seq(test_labels, test=True, aug=False, batch_size=32)
seq


# In[ ]:


print(len(seq.ids))
len(seq)


# In[ ]:


xs, ys = next(seq)


# In[ ]:


xs.shape


# In[ ]:


show_arr(xs[0].reshape((139,139,4)).astype('float32'))


# In[ ]:


show_arr(xs[1].reshape((139,139,4)).astype('float32'))


# In[ ]:


show_arr(xs[2].reshape((139,139,4)).astype('float32'))


# ### make model

# In[ ]:


from keras import applications
# model_resnet = applications.resnet50.ResNet50(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000)


# In[ ]:


# from keras import applications
# model_resnet = applications.inception_resnet_v2.InceptionResNetV2(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000)


# In[ ]:


# model_resnet.summary()


# In[ ]:


model_resnet = applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(139,139,3),
    pooling='avg',
    classes=None)


# In[ ]:


# model_resnet.summary()


# In[ ]:


img_shape = (139, 139, 4)
img_dim = np.array(img_shape).prod()
print(img_dim)


# In[ ]:


def make_model(model_resnet):
    '''==============================
    inputs
    =============================='''
    inp = Input(shape=(img_dim,), name='input')
    oup = Reshape(img_shape)(inp)
    oup = Conv2D(3, kernel_size=2, strides=1, padding='same')(oup)
    model_cnvt = Model(inp, oup)
    
    oup = model_resnet(oup)
    oup = Dense(28)(oup)
    oup = Activation('sigmoid', name='cls')(oup)
    
    model = Model(inp, oup, name='model')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy', 'binary_accuracy'])
    
    return {
        'model_resnet': model_resnet,
        'model_cnvt': model_cnvt,
        'model': model
    }

models = make_model(model_resnet)
models['model'].summary()


# In[ ]:


models['model_cnvt'].summary()


# In[ ]:


'''aug=False'''
seq = Seq(train_labels, aug=False, batch_size=32)
print(len(seq))

models['model'].fit_generator(seq, epochs=1,
                              steps_per_epoch=len(seq),
                              callbacks=[])


# ### focal loss

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


# In[ ]:


models['model'].compile(loss=focal_loss,
                        optimizer='adam',
                        metrics=['categorical_accuracy', 'binary_accuracy'])


# In[ ]:


'''aug=False'''
seq = Seq(train_labels, aug=False, batch_size=32)
print(len(seq))

models['model'].fit_generator(seq, epochs=1,
                              steps_per_epoch=len(seq),
                              callbacks=[])


# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
seq_pred
xs, ys = next(seq_pred)
xs.shape


# In[ ]:


y_pred = models['model'].predict(xs)
y_pred.shape


# In[ ]:


y_pred[:3]


# In[ ]:





# In[ ]:


# seq = Seq(train_labels, aug=True, batch_size=32)
# print(len(seq))

# models['model'].fit_generator(seq, epochs=10,
#                               steps_per_epoch=len(seq),
#                               callbacks=[])


# ### predict and submit

# In[ ]:


# seq_pred = Seq(train_labels, test=False, aug=False, batch_size=128)
# seq_pred


# In[ ]:


# pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)


# In[ ]:


# seq_test = Seq(test_labels, test=True, aug=False, batch_size=128)
# seq_test


# In[ ]:


# pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)


# In[ ]:




