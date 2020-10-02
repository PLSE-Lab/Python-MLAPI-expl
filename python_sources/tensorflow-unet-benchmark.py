#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is an attempt to create a benchmark using Tensorflow and the Unet. **The 0.38 score is because the kernel is not training at all (actually, it predicts all 0 at the output).** Any help to solve this problem would be really appreciated.

# ## Imports

# In[ ]:


import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import gc
import time
from sklearn.model_selection import train_test_split
from skimage.feature import canny
from skimage import exposure
import matplotlib.pyplot as plt


# ## Usefull functions

# In[ ]:


def print_progress(it, mIoU,loss):
    # Calculate the accuracy on the training-set.
    now = time.strftime("%c")
    print("Iteration " + str(it) + " --- mIoU: " + str(mIoU) + " --- Loss: " + str(loss) + " --- " + now);


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def create_submission(pred,test_fns,score, threshold):
    pred_dict = {fn[:-4]: RLenc(np.round(pred[i, :, :, 0] > threshold)) for i, fn in enumerate(test_fns)};
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_'+str(score)+'.csv')


def transform_images(x_train,y_train,depths):
    x = [i for i in x_train];
    y = [i for i in y_train];
    d = [i for i in depths];

    for i in range(len(x_train)):
        (h, w) = x_train[i].shape[:2];
        center = (w / 2, h / 2);
        # flip h
        x.append(np.array(cv2.flip(x_train[i],0)))
        y.append(np.array(cv2.flip(y_train[i],0)))
        d.append(depths[i]);
        # flip v
        x.append(np.array(cv2.flip(x_train[i], 1)))
        y.append(np.array(cv2.flip(y_train[i], 1)))
        d.append(depths[i]);
        # flip h & v
        x.append(np.array(cv2.flip(x_train[i], -1)))
        y.append(np.array(cv2.flip(y_train[i], -1)))
        d.append(depths[i]);
        '''
        # rotate 90
        M = cv2.getRotationMatrix2D(center,90,1.0)
        x.append(np.array(cv2.warpAffine(x_train[i], M,(h,w))))
        y.append(np.array(cv2.warpAffine(y_train[i], M,(h,w))))
        d.append(depths[i]);
        # rotate 180
        M = cv2.getRotationMatrix2D(center, 180, 1.0)
        x.append(np.array(cv2.warpAffine(x_train[i], M, (h, w))))
        y.append(np.array(cv2.warpAffine(y_train[i], M, (h, w))))
        d.append(depths[i]);
        # rotate 270
        M = cv2.getRotationMatrix2D(center, 270, 1.0)
        x.append(np.array(cv2.warpAffine(x_train[i], M, (h, w))))
        y.append(np.array(cv2.warpAffine(y_train[i], M, (h, w))))
        d.append(depths[i]);
        '''
    return x, y, d;


def conv_layer(input, filters, kernel_size, strides, k_init, k_reg, activation=tf.nn.relu, dropout=0.,
               p_type=None,p_size=(2,2),p_stride=(2,2)):

    l = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides, activation=activation,
                         kernel_initializer=k_init, kernel_regularizer=k_reg,padding='same');

    if p_type == None:
        d = tf.layers.dropout(inputs=l, rate=dropout);
    elif p_type == 'avg':
        a = tf.layers.average_pooling2d(inputs=l, pool_size=p_size, strides=p_stride);
        d = tf.layers.dropout(inputs=a, rate=dropout);
    elif p_type == 'max':
        m = tf.layers.max_pooling2d(inputs=l, pool_size=p_size, strides=p_stride);
        d = tf.layers.dropout(inputs=m, rate=dropout);

    print("Layer created with shape: " + str(d.shape))

    return d;


def convt_layer(input, filters, kernel_size, strides, k_init, k_reg, activation=tf.nn.relu, dropout=0.,
               p_type=None,p_size=(2,2),p_stride=(2,2)):

    l = tf.layers.conv2d_transpose(inputs=input, filters=filters, kernel_size=kernel_size, strides=strides,
                                   activation=activation,kernel_initializer=k_init, kernel_regularizer=k_reg, padding='same');

    if p_type == None:
        d = tf.layers.dropout(inputs=l, rate=dropout);
    elif p_type == 'avg':
        a = tf.layers.average_pooling2d(inputs=l, pool_size=p_size, strides=p_stride);
        d = tf.layers.dropout(inputs=a, rate=dropout);
    elif p_type == 'max':
        m = tf.layers.max_pooling2d(inputs=l, pool_size=p_size, strides=p_stride);
        d = tf.layers.dropout(inputs=m, rate=dropout);

    print("Layer created with shape: " + str(d.shape))

    return d;


# ## Data class

# In[ ]:


class Data():
    def __init__(self):
        self.width = 128;
        self.channels = 1;
        self.x_train = None;
        self.y_train = None;
        self.depths = None;
        self.x_val = None;
        self.y_val = None;
        self.x_test = None;
        self.next_batch = 0;
        self.end_epoch = False;
        self.test_fns = None;
        self.feats = None;

    def load_train(self):
        print("Loading train data...");
        TRAIN_IMAGE_DIR = '../input/train/images/'
        TRAIN_MASK_DIR = '../input/train/masks/'

        train_fns = os.listdir(TRAIN_IMAGE_DIR)

        depths = pd.read_csv('../input/depths.csv');
        max_depth = np.max(depths['z'].values);
        print('Max depth: ' + str(max_depth));
        depths['z'] = np.array(depths['z'].values)/max_depth;

        x_train = [np.array(cv2.resize(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE),(128,128)),dtype=np.uint8) for p in train_fns]
        x_train = [exposure.equalize_adapthist(x) for x in x_train];
        x_train = np.array(x_train / 255
        #x_train = np.expand_dims(x_train, axis=3)

        y_train = [np.array(cv2.resize(cv2.imread(TRAIN_MASK_DIR + p, cv2.IMREAD_GRAYSCALE),(128,128)),dtype=np.uint8) for p in train_fns]
        y_train = np.array(y_train) / 255
        #y_train = np.expand_dims(y_train, axis=3)

        self.width = x_train.shape[1];

        depths = [np.full((self.width,self.width),depths[depths['id'] == p[:-4]]['z'].values) for p in train_fns];
        #depths = np.expand_dims(depths, axis=3)

        #self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
        #                                                                      random_state=23, test_size=0.2)

        x_train, y_train, depths = transform_images(x_train,y_train,depths);

        feats = [canny(x) for x in x_train];

        self.x_train = np.array(x_train);
        del x_train
        gc.collect()
        self.y_train = np.array(y_train);
        del y_train
        gc.collect()
        self.depths = np.array(depths);
        del depths
        gc.collect()
        self.feats = np.array(feats);
        del feats
        gc.collect()

        self.x_train = np.expand_dims(self.x_train, axis=3)
        self.y_train = np.expand_dims(self.y_train, axis=3)
        self.depths = np.expand_dims(self.depths, axis=3)
        self.feats = np.expand_dims(self.feats, axis=3)
        self.channels = self.x_train.shape[3];

        self.shuffleTrainData();

        print("Finish loading!");
        print("x_train shape:");
        print(self.x_train.shape);
        print("y_train shape:");
        print(self.y_train.shape);
        print("depths shape:");
        print(self.depths.shape);
        print("feats shape:");
        print(self.feats.shape);

    def load_test(self):
        del self.x_train, self.y_train
        gc.collect()
        self.x_train = None;
        self.y_train = None;
        gc.collect()

        print("Loading test data...");
        TEST_IMAGE_DIR = '../input/test/images/'

        self.test_fns = os.listdir(TEST_IMAGE_DIR)

        depths = pd.read_csv('../input/depths.csv');
        max_depth = np.max(depths['z'].values);
        print('Max depth: ' + str(max_depth));
        depths['z'] = np.array(depths['z'].values) / max_depth;

        self.depths = depths;

        self.next_batch = 0;
        self.end_epoch = False;

        print("Finish loading!");

    def getNextTrainBatch(self,batch_size):
        init = int(self.next_batch * batch_size);
        end = int(init + batch_size);

        self.next_batch += 1;

        if end > len(self.x_train):
            end = int(len(self.x_train));
            init = int(end - batch_size);
            self.end_epoch = True;
            self.next_batch = 0;

        x = self.x_train[init:end];
        y = self.y_train[init:end];
        z = self.depths[init:end];
        f = self.feats[init:end];

        return x, y, z, f;

    def getNextTestBatch(self,batch_size):
        TEST_IMAGE_DIR = '../input/test/images/';
        init = int(self.next_batch * batch_size);
        end = int(init + batch_size);

        self.next_batch += 1;

        if end >= len(self.test_fns):
            end = int(len(self.test_fns));
            self.end_epoch = True;
            self.next_batch = 0;

        test_fns = self.test_fns[init:end];
        #print("Length test_fns: " + str(len(test_fns)));
        x_test = [
            np.array(cv2.resize(cv2.imread(TEST_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), (128, 128)), dtype=np.uint8) for p
            in test_fns]

        x_test = [exposure.equalize_adapthist(x) for x in x_test];

        x_test = np.array(x_test) / 255

        feats = [canny(x) for x in x_test];

        x_test = np.expand_dims(x_test, axis=3)
        #print("Length x_test: " + str(len(x_test)));
        width = x_test.shape[1];

        depths = [np.full((width, width), self.depths[self.depths['id'] == p[:-4]]['z'].values) for p in test_fns];
        depths = np.expand_dims(depths, axis=3)

        feats = np.expand_dims(feats, axis=3)

        return x_test, depths, feats, np.arange(init,end,1);

    def shuffleTrainData(self):
        rng_state = np.random.get_state();
        np.random.shuffle(self.x_train);
        np.random.set_state(rng_state);
        np.random.shuffle(self.y_train);
        np.random.set_state(rng_state);
        np.random.shuffle(self.depths);
        np.random.set_state(rng_state);
        np.random.shuffle(self.feats);
        self.next_batch = 0;
        return;


# ## Parameters and data loading

# In[ ]:


batch_size = 100;
dropout_rate = .0;
learning_rate = .01;
lr_decay = .9;
reg = .0;
max_epochs = 1;
salt_threshold = .5;

data = Data();
data.load_train();


# ## Building the computational graph

# In[ ]:


tf.reset_default_graph();
tf.logging.set_verbosity(tf.logging.ERROR)
# inputs
with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, data.width, data.width, data.channels], name='x');
    y_true = tf.placeholder(tf.float32, shape=[None, data.width, data.width, data.channels], name='y_true');
    z = tf.placeholder(tf.float32, shape=[None, data.width, data.width, data.channels], name='z');
    feats = tf.placeholder(tf.float32, shape=[None, data.width, data.width, 1], name='feats');
    lr = tf.placeholder(tf.float32, name='learning_rate');
    dropout = tf.placeholder(tf.float32, name='dropout');

x_ = tf.concat([x,z,feats],axis=3);

print("Total input shape: " + str(x_.shape));

tf_batchsize = tf.shape(x)[0];
xavier = tf.contrib.layers.xavier_initializer();
reg_regr = tf.contrib.layers.l2_regularizer(scale=reg)

# conv layers
with tf.variable_scope('encoder'):
    e1 = conv_layer(input=x_, filters=8, kernel_size=(3,3), strides=(1,1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    e2 = conv_layer(input=e1, filters=8, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout,p_type='avg');

    e3 = conv_layer(input=e2, filters=16, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    e4 = conv_layer(input=e3, filters=16, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout, p_type='avg');

    e5 = conv_layer(input=e4, filters=32, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    e6 = conv_layer(input=e5, filters=32, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout, p_type='avg');

    e7 = conv_layer(input=e6, filters=64, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    e8 = conv_layer(input=e7, filters=64, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout, p_type='avg');

    e9 = conv_layer(input=e8, filters=128, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    e10 = conv_layer(input=e9, filters=128, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

with tf.variable_scope('decoder'):
    d1 = convt_layer(input=e10, filters=64, kernel_size=(2,2), strides=(2,2), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

    c1 = tf.concat([d1,e7],axis=3);
    d2 = conv_layer(input=c1, filters=64, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    d3 = conv_layer(input=d2, filters=64, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                         dropout=dropout);

    d4 = convt_layer(input=d3, filters=32, kernel_size=(2,2), strides=(2,2), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

    c2 = tf.concat([d4,e5],axis=3);
    d5 = conv_layer(input=c2, filters=32, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    d6 = conv_layer(input=d5, filters=32, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

    d7 = convt_layer(input=d6, filters=16, kernel_size=(2,2), strides=(2,2), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

    c3 = tf.concat([d7, e3], axis=3);
    d8 = conv_layer(input=c3, filters=16, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    d9 = conv_layer(input=d8, filters=16, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

    d10 = convt_layer(input=d9, filters=8, kernel_size=(2, 2), strides=(2, 2), k_init=xavier, k_reg=reg_regr,
                         dropout=dropout);

    c4 = tf.concat([d10, e1], axis=3);
    d11 = conv_layer(input=c4, filters=8, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);
    d12 = conv_layer(input=d11, filters=8, kernel_size=(3, 3), strides=(1, 1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout);

with tf.variable_scope('output'):
    y_pred = convt_layer(input=d12, filters=1, kernel_size=(1,1), strides=(1,1), k_init=xavier, k_reg=reg_regr,
                        dropout=dropout,activation=tf.nn.sigmoid);

with tf.variable_scope('loss'):
    y_true_flat = tf.reshape(y_true,[tf_batchsize,-1]);
    y_pred_flat = tf.reshape(y_pred, [tf_batchsize, -1]);
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true_flat,
                                               logits=y_pred_flat);
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred);

with tf.variable_scope('metrics'):
    IoU, IoU_op = tf.metrics.mean_iou(labels=tf.cast(y_true,tf.int32),
                                          predictions=tf.to_int32(y_pred > salt_threshold),num_classes=2,name='mIoU');

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='op').minimize(loss);


# ## Initializing variables

# In[ ]:


session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())


# ## Starting training

# In[ ]:


print("Starting training...");

epoch = 1;
lr = learning_rate;
step = 0;
mIoU = [];
mIoU_over_steps = [];
xent = [];
loss_over_steps = [];

feed_dict_val = {'input/x:0': data.x_val, 'input/y_true:0': data.y_val, 'input/dropout:0': 0};

while epoch <= max_epochs:

    x_batch, y_batch, z_batch, f_batch = data.getNextTrainBatch(batch_size);

    feed_dict_train = {'input/x:0': x_batch, 'input/y_true:0': y_batch, 'input/z:0': z_batch,
                           'input/feats:0': f_batch,'input/learning_rate:0': lr,
                           'input/dropout:0': dropout_rate};

    _ = session.run(['op'], feed_dict=feed_dict_train);
    _ = session.run([IoU_op], feed_dict=feed_dict_train);
    mIoU,xent = session.run([IoU,loss], feed_dict=feed_dict_train);
    mIoU_over_steps.append(mIoU);
    loss_over_steps.append(xent);

    if step % 10 == 0:
            #_ = session.run([IoU_op], feed_dict=feed_dict_val);
            #mIoU, xent = session.run([IoU, loss], feed_dict=feed_dict_val);
        print_progress(step, np.mean(mIoU_over_steps), np.mean(loss_over_steps));
        mIoU_over_steps = [];
        loss_over_steps = [];
        plt.imshow(y_batch[0]);
    if data.end_epoch:
        print('End of epoch ' + str(epoch));
        epoch += 1;
        lr *= lr_decay;
        data.end_epoch = False;
        data.shuffleTrainData();

    step += 1;

del data
gc.collect()


# ## Predicting

# In[ ]:


data = Data();
data.load_test();
sub = np.zeros((len(data.test_fns), data.width, data.width, data.channels));

while not data.end_epoch:
    x, z, f, idx = data.getNextTestBatch(batch_size);
    feed_dict_test = {'input/x:0': x, 'input/z:0': z, 'input/feats:0': f, 'input/dropout:0': 0};
    sub[idx] = session.run([y_pred],feed_dict_test);

session.close();

subs_resized = [];

for i in range(len(sub)):
    subs_resized.append(np.array(cv2.resize(sub[i],dsize=(101,101)),dtype=np.int8));

subs_resized = np.array(subs_resized);
subs_resized = np.expand_dims(subs_resized, axis=3)
print(subs_resized.shape)

create_submission(subs_resized,data.test_fns,mIoU,salt_threshold)

