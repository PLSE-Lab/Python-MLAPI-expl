#!/usr/bin/env python
# coding: utf-8

# * Please refer to the comments for the overview.

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
import shutil
print(os.listdir("../input"))


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

import tensorflow as tf

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


SH_ALL = (512, 512)
SH = (256, 256)
ID_LIST_TRAIN = train_labels.Id.tolist()


# In[ ]:


# def get_arr(Id, test=False, spl=2):
#     if test:
#         arr = get_arr0(Id, test=True)
#     else:
#         arr = get_arr0(Id)
#     arr = arr[:].astype('float32')
#     res = []
#     arr2 = [res.extend(np.hsplit(ee, spl)) for ee in np.vsplit(arr, spl)]
#     return np.stack(res)


# In[ ]:


# get_arr('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0').shape


# In[ ]:


# arr = get_arr('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0')
# print(arr.shape)
# show_arr(arr[0])
# show_arr(arr[1])
# show_arr(arr[2])
# show_arr(arr[3])


# In[ ]:


# arr = get_arr('00008af0-bad0-11e8-b2b8-ac1f6b6435d0', test=True)
# print(arr.shape)
# show_arr(arr[0])
# show_arr(arr[1])
# show_arr(arr[2])
# show_arr(arr[3])


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
    SPLIT_LIST = [
        np.array(range(256)),
        np.array(range(256)) + 256
    ]
    
    def __init__(self, df, extend=False, aug=False, test=False, batch_size=32):
        self.shaffle = None
        self.extend = extend
        self.aug = aug
        self.test = test
        self.batch_size = batch_size
        self.df = df
        
        # proccess
        self.ids = self.df.Id.tolist()
        self.reversed = sorted(range(SH_ALL[0]), reverse=True)
        
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
        arr = get_arr0(id0, test=self.test)
        
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
    
    def split(self, arr):
        res = []
        for row_idx in self.SPLIT_LIST:
            for col_idx in self.SPLIT_LIST:
                #print(arr.shape, row_idx, col_idx)
                res.append(arr[:, col_idx][row_idx].flatten())
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
                img0 = self.random_transform(arr0)
            else:
                img0 = arr0
            res = self.split(img0)
            res = np.concatenate(res)
            xs.append(res)
            ys.append(cat)


# In[ ]:


seq = Seq(train_labels, extend=False, aug=True, batch_size=8)
print(len(seq))


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
print(xs['input'].shape)
print(xs['input_cls'].shape)


# In[ ]:


show_arr(xs['input'][0].reshape((4,256,256,4))[0])


# In[ ]:


show_arr(xs['input'][0].reshape((4,256,256,4))[1])


# ### make model

# In[ ]:


from keras import applications


# In[ ]:


def make_trainable_false(model_resnet, trainable=False):
    layers = model_resnet.layers
    for ilayer in layers:
        ilayer.trainable = trainable
    return


# In[ ]:


img_shape = tuple(list(SH) + [4])
print(img_shape)
img_dim = 4 * np.array(img_shape).prod()
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
                 kernel_regularizer=regularizers.l2(0))(oup)
    #kernel_regularizer=regularizers.l2(1e-4)
    model_cnvt = Model(inp, oup, name='model_cnvt')
    return model_cnvt


# In[ ]:


def make_model_classifier(input_dim=1536):
    inp_cls = Input((input_dim,))
    oup_cls = Dense(28)(inp_cls)
    oup_cls = Activation('sigmoid')(oup_cls)
    model_classifier = Model(inp_cls, oup_cls, name='classifier')
    return model_classifier


# In[ ]:


def make_model(img_dim, model_cnvt, model_resnet, model_classifier):
    inp0 = Input(shape=(int(img_dim/4),), name='input0')
    oup0 = model_cnvt(inp0)
    oup0 = model_resnet(oup0)
    oup0 = model_classifier(oup0)
    oup0 = Activation('linear', name='path_cls_cls')(oup0)
    model0 = Model(inp0, oup0, name='model0')
    
    '''==============================
    inputs
    =============================='''
    def fn(x, idx):
        x1 = K.permute_dimensions(x, (1,0,2,))
        x2 = K.gather(x1, idx)
        return x2
    inp = Input(shape=(img_dim,), name='input')
    oup = Reshape((4,int(img_dim/4)))(inp)
    img0 = Lambda(lambda x: fn(x, 0))(oup)
    img1 = Lambda(lambda x: fn(x, 1))(oup)
    img2 = Lambda(lambda x: fn(x, 2))(oup)
    img3 = Lambda(lambda x: fn(x, 3))(oup)
    oup = Lambda(lambda x: K.stack(x, 1))([model0(img0), model0(img1), model0(img2), model0(img3)])
    oup = GlobalMaxPooling1D()(oup)
    oup = Activation('linear', name='path_cls_cls')(oup)
    model = Model(inp, oup, name='model')
    
    return {
        'model_classifier': model_classifier,
        'model_resnet': model_resnet,
        'model_cnvt': model_cnvt,
        'model': model,
        'model0': model0
    }


# In[ ]:


# models = make_model(img_dim, model_cnvt, model_resnet, model_classifier)
# models['model0'].summary()


# In[ ]:


# models['model'].summary()


# In[ ]:


model_cnvt = make_model_cnvt(int(img_dim/4), img_shape)
model_cnvt.summary()


# In[ ]:


model_cnvt.layers[2].get_weights()


# In[ ]:


model_cnvt.load_weights('../input/resnet50-4x256-max-1/model_1_cnvt.h5')
model_cnvt.layers[2].get_weights()


# In[ ]:


model_resnet = applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=list(SH)+[3],
    pooling='avg',
    classes=None)


# In[ ]:


# model_resnet.summary()


# In[ ]:


model_resnet.layers[2].get_weights()[0][0,0,0]


# In[ ]:


model_resnet.load_weights('../input/resnet50-4x256-max-1/model_1_resnet.h5')
model_resnet.layers[2].get_weights()[0][0,0,0]


# In[ ]:


model_classifier = make_model_classifier(1024*2)
model_classifier.summary()


# In[ ]:


model_classifier.layers[1].get_weights()[0][0]


# In[ ]:


model_classifier.load_weights('../input/resnet50-4x256-max-1/model_1_classifier.h5')
model_classifier.layers[1].get_weights()[0][0]


# In[ ]:


models = make_model(img_dim, model_cnvt, model_resnet, model_classifier)
models['model0'].summary()


# In[ ]:


models['model'].summary()


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


# In[ ]:


class LearningRateReducer(Callback):

    def __init__(self, schedule,
                 cycle_epoch=2, steps_per_epoch=100,
                 start_lr=0.001, end_lr=0.0001, verbose=0):
        super(LearningRateReducer, self).__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch
        self.cycle_epoch = cycle_epoch
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.verbose = verbose

    def on_train_begin(self, epoch, logs=None):
        self.lrf = {'lr': [], 'loss': []}
        self.epoch = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    
    def on_batch_begin(self, batch, logs=None):
        #print(batch)
        lr = self.schedule(self.epoch, batch,
                           cycle_epoch=self.cycle_epoch,
                           steps_per_epoch=self.steps_per_epoch,
                           start_lr=self.start_lr, end_lr=self.end_lr)
        K.set_value(self.model.optimizer.lr, lr)
        self.lrf['lr'].append(lr)
        return
    
    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.lrf['loss'].append(loss)


# In[ ]:


def lr_schedule(epoch, batch,
                cycle_epoch=2, steps_per_epoch=100,
                start_lr=0.002, end_lr=0.0002):
    by = (np.log(start_lr) - np.log(end_lr)) / (steps_per_epoch*cycle_epoch-1)
    ii = divmod(epoch, cycle_epoch)[1]*steps_per_epoch + batch # use amari
    lr = np.exp(np.log(start_lr) - ii * by)
    #print('Learning rate: ', lr)
    return lr


# In[ ]:


LEARNING_RATE = 0.001
print(datetime.datetime.now() - now)


# In[ ]:


models['model'].compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['categorical_accuracy', 'binary_accuracy', f1])


# ## #1

# In[ ]:


seq = Seq(train_labels, extend=False, aug=True, batch_size=8)
print(len(seq))

lr_scheduler = LearningRateReducer(lr_schedule,
                                   cycle_epoch=1, steps_per_epoch=len(seq),
                                   start_lr=LEARNING_RATE, end_lr=LEARNING_RATE/10)
callbacks = [lr_scheduler]

hst = models['model'].fit_generator(seq, epochs=1,
                              steps_per_epoch=len(seq),
                              callbacks=callbacks)


# In[ ]:


plt.subplots(1, 1, figsize=(10,10))
plt.plot(lr_scheduler.lrf['lr'], lr_scheduler.lrf["loss"])
plt.xscale('log')
plt.title('learning rate')


# In[ ]:


seq = Seq(train_labels, extend=False, aug=False, batch_size=32)
print(len(seq))
xs, ys = next(seq)
print(xs['input'].shape)
xs['input'].reshape((32,4,-1)).shape
y_pred = models['model_cnvt'].predict(xs['input'].reshape((32,4,-1))[0])
print(y_pred.shape)


# In[ ]:


y_pred[0]


# In[ ]:


tmp = np.vstack(
    [
        np.hstack([y_pred[0], y_pred[1]]),
        np.hstack([y_pred[2], y_pred[3]])
    ])
tmp.shape


# In[ ]:


show_arr(tmp)


# In[ ]:


Image.fromarray(np.uint8((tmp+1)/2*255))


# ### predict and submit

# In[ ]:


seq_pred = Seq(train_labels, test=False, aug=False, batch_size=32)
len(seq_pred)


# In[ ]:


pred = models['model'].predict_generator(seq_pred, steps=len(seq_pred), verbose=1)
pred.shape


# In[ ]:


def calc_threshold(pred):
    ### calc threshold
    threshold_dic = {}
    for idx in tqdm(range(28)):
        threshold_dic[idx] = 0.5
        m = 0
        for ii in range(100):
            threshold0 = ii*0.01
            f1_val = f1_score(y_cat_train_dic[idx], threshold0<(pred[:,idx]))
            if m < f1_val:
                threshold_dic[idx] = threshold0+0.005
                m = f1_val
    return threshold_dic


# In[ ]:


import warnings
warnings.filterwarnings('ignore')

threshold_dic = calc_threshold(pred)
threshold_dic


# In[ ]:


'''use threshold_dic'''
for ii in range(28):
    print(ii, f1_score(y_cat_train_dic[ii], threshold_dic[ii]<pred[:,ii]))


# In[ ]:


'''threshold = 0.5'''
for ii in range(28):
    print(ii, f1_score(y_cat_train_dic[ii], 0.5<pred[:,ii]))


# In[ ]:


seq_test = Seq(test_labels, test=True, aug=False, batch_size=32)
seq_test


# In[ ]:


pred_test = models['model'].predict_generator(seq_test, steps=len(seq_test), verbose=1)
pred_test.shape


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


df_pred1 = pd.DataFrame(pred)
df_pred1.columns = ['cls'+ str(ii) for ii in range(28)]
df_pred1 = pd.concat([train_labels.copy(), df_pred1], axis=1)

df_pred1.head()
df_pred1.to_csv('proba_{}.csv'.format(No), index=False)


# In[ ]:


df_pred1_test = pd.DataFrame(pred_test)
df_pred1_test.columns = ['cls'+ str(ii) for ii in range(28)]
df_pred1_test = pd.concat([test_labels.copy(), df_pred1_test], axis=1)

df_pred1_test.head()
df_pred1_test.to_csv('proba_test_{}.csv'.format(No), index=False)


# In[ ]:


print(datetime.datetime.now() - now)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




