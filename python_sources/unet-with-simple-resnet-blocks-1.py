#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########################
# Setup & Configuration  
#######################
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style('white')

plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
get_ipython().run_line_magic('matplotlib', 'inline')


import warnings 
warnings.filterwarnings("ignore")
#######################

import os, sys, cv2, random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

import imgaug as ia
from imgaug import augmenters as iaa

from keras.layers.core import Lambda
from keras.models import Model, load_model, save_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import (Input, Dense, Dropout, Conv2D, Conv2DTranspose, 
                          BatchNormalization, Activation, GlobalAveragePooling2D,
                          MaxPooling2D, concatenate, Reshape, Add, multiply)


from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img

import time
t_start = time.time()


# In[ ]:


#####################
# Global Constants
##################
basic_name = f'Unet_resnet'
save_model_name = basic_name + '.model'
submission_file = basic_name + '.csv'

TRAIN_IMAGE_DIR = '../input/train/images/'
TRAIN_MASK_DIR = '../input/train/masks/'
TEST_IMAGE_DIR = '../input/test/images/'

img_size = 101
seed=1994

batch_size  = 128
epochs = 120


# In[ ]:


###########
# Metrics
#########
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
        
    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    intersection = temp1[0]
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
        
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

##################################

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

###############
# BINARY LOSSES
###############
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.elu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss, strict=True, name="loss")
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    logits = y_pred
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

def symmetric_lovasz(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    loss = ((lovasz_hinge(y_pred, y_true, per_image = True, ignore = None)) +             (lovasz_hinge(-y_pred, 1 - y_true, per_image = True, ignore = None))) /2
    return loss


# In[ ]:


# Loading of training/testing ids
train_df = pd.read_csv("../input/train.csv", index_col="id", usecols=[0])
train_df["images"] = [np.array(load_img(TRAIN_IMAGE_DIR + "{}.png".format(idx), color_mode = "grayscale")) / 255 
                      for idx in tqdm_notebook(train_df.index)]

train_df["masks"] = [np.array(load_img(TRAIN_MASK_DIR + "{}.png".format(idx), color_mode = "grayscale")) / 255 
                     for idx in tqdm_notebook(train_df.index)]

train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size, 2)

def cov_to_class(val):    
    for i in range(0, 11): 
        if val * 10 <= i : return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

train_df.head()


# In[ ]:


fig, axs = plt.subplots(1, 2, figsize=(15,5))
sns.distplot(train_df.coverage, kde=False, ax=axs[0], color= '#123456')
sns.distplot(train_df.coverage_class, bins=10, kde=False, ax=axs[1], color= '#123456')
plt.suptitle("Salt coverage")
axs[0].set_xlabel("Coverage")
axs[1].set_xlabel("Coverage class");


# ## Create train/validation split stratified by salt coverage

# In[ ]:


# Create train/validation split stratified by salt coverage
X = np.array(train_df.images.tolist()).reshape(-1, img_size, img_size, 1)
y = np.array(train_df.masks.tolist()).reshape(-1, img_size, img_size, 1)
x_train, x_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2, 
    stratify=train_df.coverage_class,
    random_state=seed
)


# In[ ]:


def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation==True: x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3))
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate: x = BatchActivate(x)
    return x

def squeeze_excite_block_cSE(input_, ratio=2):
    init = input_

    filters = K.int_shape(init)[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(se)

    x = multiply([init, se])
    return x

def squeeze_excite_block_sSE(input_):
    sSE_scale = Conv2D(1, (1, 1), activation='sigmoid', padding="same", use_bias = True)(input_)
    return multiply([input_, sSE_scale])

def unet_layer(blockInput, num_filters, use_csSE_ratio = 2):
    x = Conv2D(num_filters, (3, 3), activation=None, padding="same")(blockInput)
    x = residual_block(x, num_filters )
    x = residual_block(x, num_filters , batch_activate = True)

    if use_csSE_ratio > 0:
        sSEx = squeeze_excite_block_sSE(x)
        cSEx = squeeze_excite_block_cSE(x,ratio = use_csSE_ratio ) #modified 10/10/2018
        x = Add()([sSEx, cSEx])

    return x


# In[ ]:


# Build Model
def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    
    # 101 -> 50
    conv1 = unet_layer(input_layer,start_neurons * 1,use_csSE_ratio=2)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(DropoutRatio/3)(pool1)
    
    # 50 -> 25
    conv2 = unet_layer(pool1, start_neurons * 2,use_csSE_ratio=2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(DropoutRatio/2)(pool2)
    
    # 25 -> 12
    conv3 = unet_layer(pool2, start_neurons * 4,use_csSE_ratio=2)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)
    
    # 12 -> 6
    conv4 = unet_layer(pool3, start_neurons * 8,use_csSE_ratio=2)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)
    ##############
    
    # Middle
    convm = Conv2D(start_neurons*16, (3,3), activation=None, padding='same')(pool4)
    convm = residual_block(convm, start_neurons*16)
    convm = residual_block(convm, start_neurons*16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons*8, (3,3), strides=(2,2), padding='same')(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons*8, (3,3), activation=None, padding='same')(uconv4)
    uconv4 = residual_block(uconv4, start_neurons*8)
    uconv4 = residual_block(uconv4, start_neurons*8, True)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons*4, (3,3), strides=(2,2), padding='valid')(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons*4, (3,3), activation=None, padding='same')(uconv3)
    uconv3 = residual_block(uconv3, start_neurons*4)
    uconv3 = residual_block(uconv3, start_neurons*4, True)
    
    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons*2, (3,3), strides=(2,2), padding='same')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio)(uconv2)
    
    uconv2 = Conv2D(start_neurons*2, (3,3), activation=None, padding='same')(uconv2)
    uconv2 = residual_block(uconv2, start_neurons*2)
    uconv2 = residual_block(uconv2, start_neurons*2, True)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons*1, (3,3), strides=(2,2), padding='valid')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio)(uconv1)
    
    uconv1 = Conv2D(start_neurons*1, (3,3), activation=None, padding='same')(uconv1)
    uconv1 = residual_block(uconv1, start_neurons*1)
    uconv1 = residual_block(uconv1, start_neurons*1, True)
    
    output_layer_noActi = Conv2D(1, (1,1), padding='same', activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)
    
    return output_layer


# ## Data augmentation

# In[ ]:


def do_augmentation(seqs, seq2_train, X_train, y_train):

    seq_det = seqs.to_deterministic()
    X_train_aug = seq_det.augment_image(X_train)
    X_train_aug = seq2_train.augment_image(X_train_aug)
    
    y_train_aug = seq_det.augment_image(y_train)

    if y_train_aug.shape != (101, 101):
        X_train_aug = ia.imresize_single_image(X_train_aug, (101, 101), interpolation="linear")
        y_train_aug = ia.imresize_single_image(y_train_aug, (101, 101), interpolation="nearest")

    return np.array(X_train_aug), np.array(y_train_aug)

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
    iaa.Fliplr(0.5),

    iaa.OneOf([
        iaa.Noop(),
        iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0), backend="cv2"),
        iaa.Noop(),
        iaa.CropAndPad(
            percent=(-0.2, 0.2),
            pad_mode="reflect",
            pad_cval=0,
            keep_size=False
        ),
    ])
])
seq_train = iaa.Sequential(
    sometimes(iaa.Multiply((0.8, 1.2))),
    sometimes(iaa.Add((-0.2, 0.2))),
    sometimes(iaa.OneOf([
        iaa.AdditiveGaussianNoise(scale=(0, 0.05)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
    ]))
)


def make_image_gen(features, labels, batch_size=32):
    all_batches_index = np.arange(0, features.shape[0])
    out_images = []
    out_masks = []
    
    while True:
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            c_img, c_mask = do_augmentation(seq, seq_train, features[index], labels[index])

            out_images += [c_img]
            out_masks += [c_mask]
            if len(out_images) >= batch_size:
                yield np.stack(out_images, 0), np.stack(out_masks, 0)
                out_images, out_masks = [], []


# In[ ]:


# model
input_layer = Input((img_size, img_size, 1))
output_layer = build_model(input_layer, 16,0.5)

model1 = Model(input_layer, output_layer)
model1.compile(loss="binary_crossentropy", optimizer=Adam(lr = 0.005), metrics=[my_iou_metric])

#model1.summary()


# In[ ]:


early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',patience=20, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_my_iou_metric', mode='max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode='max', factor=0.5, patience=10, min_lr=0.0001, verbose=1)

train_generator = make_image_gen(x_train, y_train,batch_size)

t_model1_start = time.time()
history = model1.fit_generator(
    train_generator,
    steps_per_epoch = x_train.shape[0]*2//batch_size,
    validation_data=[x_valid, y_valid], 
    epochs=epochs,
    callbacks = [early_stopping, model_checkpoint, reduce_lr],
    verbose = 1
)
t_model1_end = time.time()
print(f"Run time = {(t_model1_end-t_model1_start)/3600} hours")


# In[ ]:


model1 = load_model(save_model_name, custom_objects={'my_iou_metric':my_iou_metric})

# remove activation layer and use lovasz loss
input_x = model1.layers[0].input
output_layer = model1.layers[-1].input

model2 = Model(input_x, output_layer)
model2.compile(loss=symmetric_lovasz, optimizer=Adam(lr=0.01), metrics=[my_iou_metric_2])


# In[ ]:


early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=30, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=10, min_lr=0.00005, verbose=1)

train_generator = make_image_gen(x_train, y_train,batch_size//2)

t_model2_start = time.time()
history = model2.fit_generator(
             train_generator,
             steps_per_epoch = x_train.shape[0]*2//batch_size,
             validation_data=[x_valid, y_valid], 
             epochs=epochs + int(epochs*(1/5)),
             callbacks = [early_stopping, model_checkpoint, reduce_lr],
             verbose = 1
)
t_model2_end = time.time()
print(f"Run time = {(t_model2_end - t_model2_start)/3600} hours")


# In[ ]:


fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
ax_score.legend();


# In[ ]:


model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2, 'symmetric_lovasz': symmetric_lovasz})


# In[ ]:


def predict_result(model,x_test,img_size): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size, img_size)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size, img_size)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2

preds_valid = predict_result(model,x_valid,img_size)


# In[ ]:


## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious);


# In[ ]:


threshold_best_index = np.argmax(ious)
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, 'xr', label='Best threshold')
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend();


# In[ ]:


def rle_encode(im):
    '''
    im: numpy array, 1-mask, 0-background
    Returns run length as string
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


test_images = os.listdir(TEST_IMAGE_DIR)
x_test = np.array([(np.array(load_img(TEST_IMAGE_DIR + "{}".format(idx), color_mode = "grayscale"))) / 255 
                   for idx in tqdm_notebook(test_images)]).reshape(-1, img_size, img_size, 1)

preds_test = predict_result(model,x_test,img_size)
pred_dict = {idx[:10]: rle_encode(np.round(preds_test[i] > threshold_best)) 
             for i, idx in enumerate(tqdm_notebook(test_images))};


# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)


# In[ ]:


t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")


# In[ ]:




