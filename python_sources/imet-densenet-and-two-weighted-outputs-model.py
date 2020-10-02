#!/usr/bin/env python
# coding: utf-8

# ## imports

# In[ ]:


import os
import re

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

import numpy as np
import pandas as pd

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *

from keras import backend as K
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from tqdm import tqdm


# ## load data

# In[ ]:


train_images = os.listdir("../input/imet-2019-fgvc6/train/")
test_images = os.listdir("../input/imet-2019-fgvc6/test/")

print("number of train images: ", len(train_images))
print("number of test  images: ", len(test_images))


# In[ ]:


train = pd.read_csv("../input/imet-2019-fgvc6/train.csv")
train.head()


# In[ ]:


labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")
labels.head()


# In[ ]:


labels.tail()


# In[ ]:


cultures = [x for x in labels.attribute_name.values if x.startswith("culture")]
tags = [x for x in labels.attribute_name.values if x.startswith("tag")]


# In[ ]:


len(cultures), len(tags)


# In[ ]:


def split_culture_tag(x):
    cultures_ = list()
    tags_ = list()
    for i in x.split(" "):
        if int(i) <= len(cultures):
            cultures_.append(i)
        else:
            tags_.append(str(int(i) - len(cultures)))
    if not cultures_:
        cultures_.append(str(len(cultures)))
    if not tags_:
        tags_.append(str(len(tags)))
    return " ".join(cultures_), " ".join(tags_)


# In[ ]:


culture_ids = list()
tag_ids = list()

for v in tqdm(train.attribute_ids.values):
    c, t = split_culture_tag(v)
    culture_ids.append(c)
    tag_ids.append(t)


# In[ ]:


train["culture_ids"] = culture_ids
train["tag_ids"] = tag_ids

train.head()


# In[ ]:


num_classes_c = len(cultures) + 1
num_classes_t = len(tags) + 1

print(num_classes_c, num_classes_t)


# In[ ]:


labels_map = {v:i for i, v in zip(labels.attribute_id.values, labels.attribute_name.values)}
labels_map_rev = {i:v for i, v in zip(labels.attribute_id.values, labels.attribute_name.values)}

num_classes = len(labels_map)
print("{} categories".format(num_classes))


# In[ ]:


submission = pd.read_csv("../input/imet-2019-fgvc6/sample_submission.csv")
submission.head()


# ## prepare X and y

# In[ ]:


def obtain_y_c(ids):
    y = np.zeros(num_classes_c)
    for idx in ids.split(" "):
        y[int(idx)] = 1
    return y

def obtain_y_t(ids):
    y = np.zeros(num_classes_t)
    for idx in ids.split(" "):
        y[int(idx)] = 1
    return y


# In[ ]:


paths = ["../input/imet-2019-fgvc6/train/{}.png".format(x) for x in train.id.values]

targets_c = np.array([obtain_y_c(y) for y in train.culture_ids.values])
targets_t = np.array([obtain_y_t(y) for y in train.tag_ids.values])


# ### image generator

# In[ ]:


class ImageGenerator(Sequence):
    
    def __init__(self, paths, targets_c, targets_t, batch_size, shape, augment=False):
        self.paths = paths
        self.targets_c = targets_c
        self.targets_t = targets_t
        self.batch_size = batch_size
        self.shape = shape
        self.augment = augment
        
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        y = np.zeros((self.batch_size, num_classes, 1))
        for i, path in enumerate(batch_paths):
            x[i] = self.__load_image(path)
        y_c = self.targets_c[idx * self.batch_size : (idx + 1) * self.batch_size]
        y_t = self.targets_t[idx * self.batch_size : (idx + 1) * self.batch_size]
        return x, [y_c, y_t]
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, (self.shape[0], self.shape[1]))
        image = preprocess_input(image)
        if self.augment:
            seq = iaa.Sequential([
                iaa.OneOf([
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.5),
                    iaa.CropAndPad(percent=(-0.25, 0.25)),
                    iaa.Crop(percent=(0, 0.1)),
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])
            ], random_order=True)
            image = seq.augment_image(image)
        return image


# ### train test split

# In[ ]:


batch_size = 64

train_paths, val_paths, train_targets_c, val_targets_c, train_targets_t, val_targets_t = train_test_split(paths, 
                                                                      targets_c,
                                                                      targets_t,
                                                                      test_size=0.1, 
                                                                      random_state=1029)

train_gen = ImageGenerator(train_paths, train_targets_c, train_targets_t, batch_size=batch_size, shape=(224,224,3), augment=False)
val_gen = ImageGenerator(val_paths, val_targets_c, val_targets_t, batch_size=batch_size, shape=(224,224,3), augment=False)


# ## build model

# In[ ]:


inp = Input((224, 224, 3))
backbone = DenseNet121(input_tensor=inp,
                       weights="../input/densenet-keras/DenseNet-BC-121-32-no-top.h5",
                       include_top=False)
x = backbone.output
x = GlobalAveragePooling2D()(x)

y_c = Dense(1024, activation="relu")(x)
y_c = Dropout(0.5)(y_c)
y_c = Dense(num_classes_c, activation="sigmoid", name="cultures_out")(y_c)

y_t = Dense(2048, activation="relu")(x)
y_t = Dropout(0.5)(y_t)
y_t = Dense(num_classes_t, activation="sigmoid", name="tags_out")(y_t)


model = Model(inp, [y_c, y_t])


# In[ ]:


losses = {
    "cultures_out": 'binary_crossentropy',
    "tags_out": 'binary_crossentropy'
}
   
loss_weights = {
   "cultures_out": 1.0,
   "tags_out": 4.0
}


# ### f_score for Keras

# In[ ]:


def f_score(y_true, y_pred, threshold=0.1, beta=2):
    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1+beta**2) * ((precision * recall) / ((beta**2)*precision + recall))


def tp_score(y_true, y_pred, threshold=0.1):
    tp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )
    tp = K.sum(K.cast(K.all(tp_3d, axis=1), 'int32'))
    return tp


def fp_score(y_true, y_pred, threshold=0.1):
    fp_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(K.abs(y_true - K.ones_like(y_true)))), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.greater(y_pred, K.constant(threshold)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=-1
    )
    fp = K.sum(K.cast(K.all(fp_3d, axis=1), 'int32'))
    return fp


def fn_score(y_true, y_pred, threshold=0.1):
    fn_3d = K.concatenate(
        [
            K.cast(K.expand_dims(K.flatten(y_true)), 'bool'),
            K.cast(K.expand_dims(K.flatten(K.abs(K.cast(K.greater(y_pred, K.constant(threshold)), 'float') - K.ones_like(y_pred)))), 'bool'),
            K.cast(K.ones_like(K.expand_dims(K.flatten(y_pred))), 'bool')
        ], axis=1
    )
    fn = K.sum(K.cast(K.all(fn_3d, axis=1), 'int32'))
    return fn


def precision_score(y_true, y_pred, threshold=0.1):
    tp = tp_score(y_true, y_pred, threshold)
    fp = fp_score(y_true, y_pred, threshold)
    return tp / (tp + fp)


def recall_score(y_true, y_pred, threshold=0.1):
    tp = tp_score(y_true, y_pred, threshold)
    fn = fn_score(y_true, y_pred, threshold)
    return tp / (tp + fn)


# In[ ]:


checkpoint = ModelCheckpoint('model.h5', 
                             monitor='val_tags_out_f_score', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max', 
                             save_weights_only=True)

reduce_lr = ReduceLROnPlateau(monitor='val_tags_out_f_score', factor=0.2,
                              patience=1, verbose=1, mode='max',
                              min_delta=0.0001, cooldown=2, min_lr=1e-7)

early_stop = EarlyStopping(monitor="val_tags_out_f_score", mode="max", patience=5)


# In[ ]:


model.compile(
    loss=losses,
    loss_weights=loss_weights,
    optimizer=Adam(1e-03),
    metrics=['acc', f_score])


# In[ ]:


history = model.fit_generator(generator=train_gen, 
                              steps_per_epoch=len(train_gen), 
                              validation_data=val_gen, 
                              validation_steps=len(val_gen),
                              epochs=20,
                              callbacks=[checkpoint, reduce_lr, early_stop])


# In[ ]:


plt.rcParams['figure.figsize'] = (6,6)

c_fscore = history.history['cultures_out_f_score']
val_c_fscore = history.history['val_cultures_out_f_score']
t_fscore = history.history['tags_out_f_score']
val_t_fscore = history.history['val_tags_out_f_score']

epochs = range(1, len(c_fscore) + 1)

plt.title('Training and validation culture f2 score')
plt.plot(epochs, c_fscore, 'red', label='Training f_score')
plt.plot(epochs, val_c_fscore, 'blue', label='Validation f_score')
plt.legend()

plt.title('Training and validation tag f2 score')
plt.plot(epochs, t_fscore, 'red', label='Training f_score')
plt.plot(epochs, val_t_fscore, 'blue', label='Validation f_score')
plt.legend()

plt.show()


# In[ ]:


model.load_weights("./model.h5")


# ## prediction

# ### test image generator

# In[ ]:


class TestImageGenerator(Sequence):
    
    def __init__(self, paths, batch_size, shape):
        self.paths = paths
        self.batch_size = batch_size
        self.shape = shape
        
    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        x = np.zeros((len(batch_paths), self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            x[i] = self.__load_image(path)
        return x
    
    def __iter__(self):
        for item in (self[i] for i in range(len(self))):
            yield item
            
    def __load_image(self, path):
        image = cv2.imread(path)
        image = cv2.resize(image, (self.shape[0], self.shape[1]))
        image = preprocess_input(image)
        return image


# ### do prediction

# In[ ]:


test_paths = ["../input/imet-2019-fgvc6/test/{}.png".format(x) for x in submission.id.values]
test_gen = TestImageGenerator(test_paths, batch_size=batch_size, shape=(224,224,3))

predicts = model.predict_generator(test_gen, verbose=1)


# In[ ]:


predicts[0].shape, predicts[1].shape


# In[ ]:


val_predicts = model.predict_generator(val_gen, verbose=1)


# In[ ]:


best_threshold_c = 0.
best_score_c = 0.

for threshold in tqdm(np.arange(0, 0.5, 0.01)):
    f2_score = fbeta_score(val_targets_c, np.array(val_predicts[0]) > threshold, beta=2, average='samples')
    if f2_score > best_score_c:
        best_score_c = f2_score
        best_threshold_c = threshold


# In[ ]:


best_threshold_t = 0.
best_score_t = 0.

for threshold in tqdm(np.arange(0, 0.5, 0.01)):
    f2_score = fbeta_score(val_targets_t, np.array(val_predicts[1]) > threshold, beta=2, average='samples')
    if f2_score > best_score_t:
        best_score_t = f2_score
        best_threshold_t = threshold


# In[ ]:


print("culture classifier: best threshold: {} best score: {}".format(best_threshold_c, best_score_c))
print("tag     classifier: best threshold: {} best score: {}".format(best_threshold_t, best_score_t))


# In[ ]:


def classifier(probs, th_c, th_t):
    c = list()
    
    # culture classifier
    a = np.array(probs[0] > th_c, dtype=np.int8)
    b = np.where(a == 1)[0]
    for idx in b.tolist():
        if idx != len(cultures):
            c.append(str(idx))
            
    # tag classifier
    a = np.array(probs[1] > th_t, dtype=np.int8)
    b = np.where(a == 1)[0]
    for idx in b.tolist():
        if idx != len(cultures) + len(tags):
            c.append(str(idx + len(cultures)))

    return " ".join(c)


# In[ ]:


predictions = list()

for probs in tqdm(zip(predicts[0], predicts[1])):
    predictions.append(classifier(probs, best_threshold_c, best_threshold_t))


# In[ ]:


len(predictions)


# In[ ]:


n = 6

img = cv2.imread(test_paths[n])
plt.imshow(img)

a = np.array(predicts[0][n]>best_score_c, dtype=np.int8)
b = np.where(a==1)[0]
for idx in b.tolist():
    if idx != len(cultures):
        print(labels_map_rev[idx])
    
a = np.array(predicts[1][n]>best_score_t, dtype=np.int8)
b = np.where(a==1)[0]
for idx in b.tolist():
    if idx != len(cultures) + len(tags):
        print(labels_map_rev[idx + len(cultures)])


# ### submission

# In[ ]:


submission["attribute_ids"] = np.array(predictions)
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.shape


# In[ ]:


get_ipython().system('head submission.csv')


# In[ ]:


submission_df = submission.copy()
submission_df.n_cate = submission.attribute_ids.apply(lambda x: len(x.split(" ")))
_ = submission_df.n_cate.value_counts().sort_index().plot.bar()

