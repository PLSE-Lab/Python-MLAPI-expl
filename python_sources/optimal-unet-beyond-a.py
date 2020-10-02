#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# --- Necessary Libraries --- #
import os
import sys
import random
import time
import _pickle
import pandas as pd
import numpy as np
from itertools import chain
from datetime import datetime
from tqdm import tqdm_notebook, tnrange
# -- Graphing -- #
import seaborn as sns
import matplotlib.pyplot as plt
# -- Sci Kit Learn and Sci Kit Image -- # 
from sklearn.model_selection import train_test_split
from skimage.io import (
    imread, imshow, concatenate_images
)
from skimage.transform import resize
from skimage.morphology import label

# --- TensorFlow ---
import tensorflow as tf

# --- Keras Imports --- # 
from keras.models import Model, load_model
from keras.layers import (
    Input,Dropout,BatchNormalization,
    Activation,Add, LeakyReLU,
    UpSampling2D, Reshape, GaussianNoise,
    Wrapper, Dense
)
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from keras import backend as K
from keras.regularizers import l2
from keras.preprocessing.image import (
    ImageDataGenerator, array_to_img, img_to_array,
    load_img
)
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model
from keras import optimizers
from keras.optimizers import TFOptimizer


# In[ ]:


# DropConnect Wrapper
class DropConnect(Wrapper):
    def __init__(self, layer, prob=1., **kwargs):
        self.prob = prob
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if 0. < self.prob < 1.:
            self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if 0. < self.prob < 1.:
            self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
            self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
        return self.layer.call(x)


# In[ ]:


# d = datetime.fromtimestamp(1537315160) Our source Time Stamp
# Function to Ensure Reproducible Results
def set_random(number):
    os.environ['PYTHONHASHSEED'] = '6072019'
    np.random.seed(number)
    random.seed(number)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    tf.set_random_seed(number)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


# In[ ]:


# Functions to facilitate image
# upsizing and downsizing
def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)


# In[ ]:


# Introduce swish function supposedly better performance
# than ReLU
# swish: sigmoid x * x .804 lb
def swish(x):
    return (K.sigmoid(x) * x )


# In[ ]:


# dropconnect(X,P) -- 
# DropConnect implementation using tensorFlow
def dropconnect(input_value, prob):
    return tf.nn.dropout(input_value, keep_prob=prob) * prob


# In[ ]:


# Metric Functions
#Score the model and do a threshold optimization by the best IoU.
# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels, y_pred = y_true_in, y_pred_in
    true_objects, pred_objects = 2, 2
    # Jiaxin fin that if all zeros, then, the background is treated as object
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
    intersection = temp1[0]
    
    # Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
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
    y_pred_in = y_pred_in > 0.5 # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float64)
    return metric_value


# In[ ]:


"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# In[ ]:


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test1 = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test2 = np.array([ np.fliplr(x) for x in preds_test2_refect] )
    preds_avg = (preds_test1 +preds_test2)/2
    return preds_avg


# # Model Architecture

# In[ ]:


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('swish')(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = Activation('swish')(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


# In[ ]:


# Build model
# for use model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # For as much reproducibility as possible
    # Applied to dropout layers
    # 101 -> 50
    prob_dc = 1 - (DropoutRatio/4)
    #conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = DropConnect(Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same"), prob =prob_dc)(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = Activation('swish')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #pool1 = Dropout(DropoutRatio/4)(pool1)
    #x = DropConnect(Dense(64, activation='relu'), prob=0.5)(x)
    #prob = 1 - (DropoutRatio/4)
    #pool1 = Lambda(dropconnect,arguments={"prob":prob})(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = Activation('swish')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio/2)(pool2)
    #prob = 1 - (DropoutRatio/2)
    #pool2 = Lambda(dropconnect,arguments={"prob":prob})(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = Activation('swish')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio/2)(pool3)
    #prob = 1 - (DropoutRatio/2)
    #pool3 = Lambda(dropconnect,arguments={"prob":prob})(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = Activation('swish')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio/2)(pool4)
    #prob = 1 - (DropoutRatio/2)
    #pool4 = Lambda(dropconnect,arguments={"prob":prob})(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16)
    convm = Activation('swish')(convm)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio/2)(uconv4)
    #prob = 1 - (DropoutRatio/2)
    #uconv4 = Lambda(dropconnect,arguments={"prob":prob})(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = Activation('swish')(uconv4)
    
    # 12 -> 25
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio/2)(uconv3)
    #prob = 1 - (DropoutRatio/2)
    #uconv3 = Lambda(dropconnect,arguments={"prob":prob})(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = Activation('swish')(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(DropoutRatio/2)(uconv2)
    #prob = 1 - (DropoutRatio/2)
    #uconv2 = Lambda(dropconnect,arguments={"prob":prob})(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = Activation('swish')(uconv2)
    
    # 50 -> 101
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #prob = 1 - (DropoutRatio/2)
    #uconv1 = Lambda(dropconnect,arguments={"prob":prob})(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = Activation('swish')(uconv1)
    
    #uconv1 = Dropout(DropoutRatio/4)(uconv1)
    prob = 1 - (DropoutRatio/4)
    uconv1 = Lambda(dropconnect,arguments={"prob":prob})(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer


# In[ ]:


plt.style.use('seaborn-white')
sns.set_style("white")
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))


# In[ ]:


# Set all seeds and states for reproducibel
# results can be further optimized if
# a state object is saved and used
set_random(1537315160)
#Initialize Image dimension Variables
img_size_ori = 101
img_size_target = 101
# Set Directory to grab from (needed when making predictions)
DATA_DIR = "../input/tgsdata/"
# Load in the train images and their masks for the training of models
x_train2 = np.load("../input/intermediatetgs/x_train2.npy")
x_valid = np.load("../input/intermediatetgs/x_valid.npy")
y_train2 = np.load("../input/intermediatetgs/y_train2.npy")
y_valid = np.load("../input/intermediatetgs/y_valid.npy")
print("Train Data:")
print("X: ",x_train2.shape,"Y: ",y_train2.shape)
print("Validation Data:")
print("X: ",x_valid.shape,"Y: ",y_valid.shape)
# add swish to list of string options in activation
# Needed for Keras
get_custom_objects().update({"swish": swish})


# In[ ]:


#fig, axs = plt.subplots(1, 2, figsize=(15,5))
#sns.distplot(train.coverage, kde=False, ax=axs[0])
#sns.distplot(train.coverage_class, bins=10, kde=False, ax=axs[1])
#plt.suptitle("Salt coverage")
#axs[0].set_xlabel("Coverage")
#axs[1].set_xlabel("Coverage class")


# In[ ]:


#Plotting the depth distributions
#sns.distplot(train.z, label="Train")
#sns.distplot(test.z, label="Test")
#plt.legend()
#plt.title("Depth distribution")


# In[ ]:


# Build the model then compile it using our custom metric
input_layer = Input((img_size_target, img_size_target, 2))
output_layer = build_model(input_layer, 16,0.6)
model = Model(input_layer, output_layer)
# Optimize optimizer
adam_opt = optimizers.Adam(lr=0.001,
                           beta_1=0.9,beta_2=0.999,
                           epsilon=None,
                           amsgrad=False,clipnorm=1.0
                          )
#sgd_opt = optimizers.SGD(lr=0.005)
model.compile(loss="binary_crossentropy", optimizer=adam_opt, metrics=[my_iou_metric])
#model.summary()


# In[ ]:


# Call backs 
# EarlyStopping: will stop after 20 epochs of no change in iou metric
# ReduceLROnPlaeau: will reduce learning rate by 0.2 after 5 epochs of no change in iou metric
early_stopping = EarlyStopping(monitor='val_my_iou_metric', mode = 'max',
                               patience=35, verbose=1)

reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric', mode = 'max',factor=0.97,
                              patience=4, min_lr=0, verbose=1)

# Save best model according to iou metric
model_checkpoint = ModelCheckpoint("./unet_best1.model",monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)

# Declare number of epochs and batch size
epochs = 200
batch_size = 32
# Train Model and set to history variable for graphing purposes
# Setting Shuffle should continur to ensure reproducibility in
# later trails
history = model.fit(x_train2, 
                    y_train2,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    shuffle=True,
                    batch_size=batch_size,
                    callbacks=[early_stopping, model_checkpoint, reduce_lr], 
                    verbose=1
                   )


# In[ ]:


# summarize history for loss
plt.plot(history.history['my_iou_metric'][1:])
plt.plot(history.history['val_my_iou_metric'][1:])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','Validation'], loc='upper left')
plt.show()


# In[ ]:


fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")


# In[ ]:


#model = load_model("./unet_best1.model",custom_objects={'my_iou_metric': my_iou_metric})


# In[ ]:


preds_valid = predict_result(model,x_valid,img_size_target)
preds_valid2 = np.array([downsample(x) for x in preds_valid])

y_valid2 = np.array([downsample(x) for x in y_valid])


# In[ ]:


## Scoring for last model
thresholds = np.linspace(0.3, 0.7, 31)
ious = np.array([iou_metric_batch(y_valid2, np.int32(preds_valid2 > threshold)) for threshold in tqdm_notebook(thresholds)])
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]


# In[ ]:


# Plot threshold and iou
plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()


# In[ ]:


#del(x_train2)
#del(x_valid)
#del(y_train2)
#del(y_valid)


# In[ ]:


x_test = np.load("../input/intermediatetgs/x_test.npy")


# In[ ]:


preds_test = predict_result(model,x_test,img_size_target)


# In[ ]:


#test = pd.read_hdf(DATA_DIR + "tgs_salt.h5", key="test")
# load test indexes
import _pickle
with open("../input/intermediatetgs/test_index.obj", "rb") as f:
    indexes = _pickle.load(f)


# In[ ]:


t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(indexes))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")


# In[ ]:


sub = pd.DataFrame.from_dict(pred_dict,orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub.to_csv("submission4.csv")


# In[ ]:


# Return png representation
#plot_model(model, to_file='model_version_2.png')

