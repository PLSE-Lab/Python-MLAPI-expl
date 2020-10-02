#!/usr/bin/env python
# coding: utf-8

# tips:
# - https://github.com/rasbt/python-machine-learning-book/blob/master/code/optional-py-scripts/ch04.py
# - https://stackoverflow.com/questions/45932773/why-there-are-strips-in-the-predicted-image-in-keras
# - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# - https://medium.com/@fromtheast/implement-fit-generator-in-keras-61aa2786ce98
# - https://www.kaggle.com/theblackcat/loading-bson-data-for-keras-fit-generator
# - https://github.com/abnera/image-classifier/blob/master/code/fine_tune.py#L124
# - https://www.kaggle.com/vfdev5/data-visualization-and-analysis/notebook
# - https://www.kaggle.com/saptak7/2-layer-cnn-adam-optimizer-and-5-epochs
# - https://www.kaggle.com/bguberfain/naive-keras-cdiscount/notebook
# 
# iterator
# - https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
# - https://spark-in.me/post/bird-voice-recognition-eight
# - https://www.kaggle.com/vfdev5/random-item-access
# - https://www.kaggle.com/ezietsman/keras-convnet-with-fit-generator
# - https://gist.github.com/faroit/92ba12373440d092e1096967b530a5b8
# - https://github.com/fchollet/keras/blob/master/tests/keras/utils/data_utils_test.py
# - https://stanford.edu/~shervine/blog/keras-generator-multiprocessing.html
# - https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
# - http://anandology.com/blog/using-iterators-and-generators/
# 
# models
# - https://gogul09.github.io/software/flower-recognition-deep-learning
# - https://shuaiw.github.io/2017/03/09/smaller-faster-deep-learning-models.html
# - https://www.kaggle.com/drn01z3/mxnet-xgboost-baseline-lb-0-57
# - https://www.kaggle.com/drn01z3/resnet50-features-xgboost
# - http://blog.kaggle.com/2016/04/28/yelp-restaurant-photo-classification-winners-interview-1st-place-dmitrii-tsybulevskii/
# - https://github.com/u1234x1234/kaggle-yelp-restaurant-photo-classification
# - http://blog.kaggle.com/2016/08/24/avito-duplicate-ads-detection-winners-interview-1st-place-team-devil-team-stanislav-dmitrii/
# - https://www.kaggle.com/zfturbo/python-xgboost-starter/code
# - https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge
# - https://www.kaggle.com/algila/inception-v3-and-k-fold-in-python-0-98996
# - https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn
# 
# 
# ensembles
# - https://mlwave.com/kaggle-ensembling-guide/
# 
# label
# - http://sloth.readthedocs.io/en/latest/
# 
# to order
# - https://github.com/LowikC/kaggle_cdiscount
# - https://github.com/rdoume/kaggle_cdiscount
# - https://github.com/fgmehlin/kaggle_cdiscount
# - https://github.com/Cuongvn08/kaggle_cdiscount_image_classify/blob/master/train.py
# - https://github.com/petrosgk/Kaggle-Cdiscount-Image-Classification-Challenge
# - https://github.com/lidalei/cdiscount
# - https://github.com/shawnxiaow1118/Kaggle_Cdiscounts_Image_Classification
# - https://github.com/xkumiyu/chainer-cdiscount-kernel
# - https://github.com/ilcauchy/cdiscount
# - https://github.com/DeepLearningSandbox/DeepLearningSandbox/blob/master/transfer_learning/fine-tune.py
# - https://github.com/abnera/image-classifier/blob/master/code/fine_tune.py
# 
# - https://github.com/knjcode/mxnet-finetuner

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.data import imread 
import json
import bson
import io


# In[ ]:


import os
INPUT_PATH = os.path.join('..', 'input', 'cdiscount-image-classification-challenge')

train_example_path = os.path.join(INPUT_PATH, 'train_example.bson')
train_path = os.path.join(INPUT_PATH, 'train.bson')

#train_example_path = "../input/cdiscount-image-classification-challenge/train_example.bson"
#train_path = "../input/cdiscount-image-classification-challenge/train.bson"


# In[ ]:


import os.path as path

def split():
    ids = []
    categories = []
    for example in bson.decode_file_iter(open(train_path, 'rb')):
        ids.append(example['_id'])
        categories.append(example['category_id'])
    return ids, categories
        
ids, categories = split()


# In[ ]:


import random

_valid_percent=0.1
size = len(ids)
train_size = int(size*(1-_valid_percent))
valid_size = (size-train_size)
split = ['train']*train_size + ['valid']*valid_size
random.shuffle(split)
split_df = pd.DataFrame({'id': ids, 'split': split})
split_df = split_df.set_index(['id'])
split_df.head(11)


# In[ ]:


#if self._split.loc[example['_id']]['split'] == name:
#for row in 
#split_df.loc[11]['split'] == 'valid'


# In[ ]:


from sklearn.cross_validation import train_test_split
ids_train,ids_test = train_test_split(ids,test_size=0.1)
print(len(ids_train),len(ids_test))
print(ids_train[0], ids_test[0])

#X_train,X_test,Y_train,Y_test = train_test_split(X,dummy_y,test_size=0.3)
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[ ]:


#10 in ids, 10 in ids_train, 10 in ids_test


# In[ ]:


#data.category_id.values
#path = "../input/cdiscount-image-classification-challenge/train_example.bson"
category_names = pd.read_csv(os.path.join(INPUT_PATH, 'category_names.csv'))
print(len(category_names['category_id'].unique()))
category_names.head()


# In[ ]:


num_classes = 5270


# In[ ]:


from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(category_names['category_id'])
#encoded_y = encoder.transform(y)
#dummy_y = np_utils.to_categorical(encoded_y, num_classes=num_classes)
#dummy_y.shape


# In[ ]:


#encoder.fit([1000010653])
np_utils.to_categorical(encoder.transform([1000010653]), num_classes=num_classes).shape


# In[ ]:


im_size = 180

batch_size = 32
batch_size_validation = 64

epochs_top = 10
epochs = 20

steps_per_epoch_top = 5
steps_per_epoch = 10

validation_steps_top=2
validation_steps=2

#minibatch_size = batch_size
#words_per_epoch = 6362906+706990
#6362906 

#val_words = 706990

#steps = 7069896/32
#steps = 220,934.25

#validation_steps = val_words // minibatch_size
#validation_steps = 706990//32
#validation_steps = 22,093.4375

#steps_per_epoch = (words_per_epoch - val_words) // minibatch_size
#steps_per_epoch = (7069896 - 706990) // 32
#steps_per_epoch = 198,840.8125

# https://github.com/fchollet/keras/blob/master/examples/image_ocr.py
#val_words = int(words_per_epoch * (val_split))
#val_split = 0.2
#val_split=words_per_epoch - val_words


# In[ ]:


import cv2

def _imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_AREA)
    return np.float32(x) / 255.

#X = np.empty((num_images, im_size, im_size, 3), dtype=np.float32)
#y = []

def load_image(pic, target):
    x = _imread(pic['picture'])
    x = img2feat(x)
#    bar.update()
    
    return x, target

def generate_arrays_from_file(path, batch_size):
    batch_features = np.zeros((batch_size, im_size, im_size, 3))
    batch_labels = np.zeros((batch_size,1))
    while 1:
        f = open(path,'rb')
        data_iter = bson.decode_file_iter(f)
        for i, d in enumerate(data_iter):
            target = d['category_id']
            for e, pic in enumerate(d['imgs']):
                x, target = load_image(pic, target)
#                yield x, np_utils.to_categorical(encoder.transform([target]), num_classes=num_classes)[0]


                encoded_y = encoder.transform([target])
                dummy_y = np_utils.to_categorical(encoded_y, num_classes=num_classes)
                yield np.array([x]), dummy_y
#                yield ({'input_2': np.array([x])}, {'dense_2': dummy_y})
#                yield batch_features, batch_labels
        f.close()


# In[ ]:


def convert2onehot(category_id):
    encoded_y = encoder.transform([category_id])
    dummy_y = np_utils.to_categorical(encoded_y, num_classes=num_classes)
    return True, dummy_y[0]
#convert2onehot(1000010653)


# In[ ]:


# https://www.kaggle.com/theblackcat/loading-bson-data-for-keras-fit-generator

from random import randint
    
def data_generator(path, ids, batch_size=128, start_image=0, name=''):
    count_product = 0
    images = []
    y_label = []
    while True:
        f = open(path,'rb')
        data_iter = bson.decode_file_iter(f)
        count = 0
        for c, d in enumerate(data_iter):
#            if d['_id'] not in ids:
            if split_df.loc[d['_id']]['split'] == name:
#                print(name, 'not in', d['_id'])
                continue
#            print(name, 'in', d['_id'])
            category_id = d['category_id']
            if count_product < start_image:
                count_product += 1
                continue
            success, one_hot = convert2onehot(category_id)
            if not success:
                print("id conversion failed")
                continue
            for e, pic in enumerate(d['imgs']):
#                picture = imread(io.BytesIO(pic['picture']))
                picture, _ = load_image(pic, category_id)
                images.append(picture)
                y_label.append(one_hot)
                count += 1
            if count >= batch_size:
                count = 0
                y_label = np.asarray(y_label)
                images = np.asarray(images)
                '''
                    since shuffle in fit function will not work here, 
                    a batch shuffle mechnism is added 
                '''
                for i,image in enumerate(images[:int(batch_size/2)]):
                    j = randint(0,batch_size-1)
                    y_temp = y_label[i]
                    img_temp = image
                    images[i] = images[j]
                    y_label[i] = y_label[j]
                    images[j] = img_temp
                    y_label[j] = y_temp
                yield images, y_label
                # just to be sure past batch are removed from the memory
                del images
                del y_label
                images = []
                y_label = []
#                return
#list(data_generator(batch_size=2))


# In[ ]:


#from sklearn.cross_validation import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,dummy_y,test_size=0.3)
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


# In[ ]:


import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

K.set_image_dim_ordering('tf')


# In[ ]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras import backend as K

input_tensor = Input(shape=(im_size, im_size, 3))

# create the base pre-trained model
#base_model = InceptionV3(weights=None, include_top=False, input_tensor=input_tensor)
base_model = InceptionV3(weights=None, include_top=False, input_shape=(im_size, im_size, 3))

keras_models_dir = "../input/keras inception v3 notop v0.5"
base_model.load_weights('%s/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' % keras_models_dir)


# In[ ]:


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# add a new top layer
#x = base_model.output
#x = Flatten()(x)
#predictions = Dense(17, activation='sigmoid')(x)

#x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
#x = Flatten(name='flatten')(x)
#predictions = Dense(36, activation='softmax', name='predictions')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# train the model on the new data for a few epochs
if True:
    #model.fit_generator(...)
#    batch_size = 32
#    epochs = 10
#    generator=generate_arrays_from_file(path)
    generator = data_generator(train_path, ids_train, batch_size=batch_size, name='train')
    generator_validation = data_generator(train_path, ids_test, batch_size=batch_size_validation, name='valid')
    hist = model.fit_generator(
                  generator=generator,
                  steps_per_epoch=steps_per_epoch_top,
                  epochs=epochs_top,
                  verbose=1,
                  callbacks = None,
                  validation_data=generator_validation,
                  validation_steps=validation_steps_top,
#                  validation_data=(X_test, Y_test)
                  )
else:
#    batch_size = 32
#    epochs = 10
    hist = model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=epochs_top,
                  verbose=1,
                  callbacks = None,
                  validation_data=(X_test, Y_test))


# In[ ]:


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# In[ ]:


# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#model.fit_generator(...)
if True:
    #model.fit_generator(...)
#    batch_size = 32
#    epochs = 10
#    generator=generate_arrays_from_file(path)
    generator = data_generator(train_path, ids_train, batch_size=batch_size, name='train')
    generator_validation = data_generator(train_path, ids_test, batch_size=batch_size_validation, name='valid')
    hist = model.fit_generator(
                  generator=generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=epochs,
                  verbose=1,
                  callbacks = None,
                  validation_data=generator_validation,
                  validation_steps=validation_steps,
#                  validation_data=(X_test, Y_test)
                  )
else:
#    batch_size = 32
#    epochs = 10
    hist = model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  callbacks = None,
                  validation_data=(X_test, Y_test))


# In[ ]:


def plot_train(hist):
    h = hist.history
    if 'acc' in h:
        meas='acc'
        loc='lower right'
    else:
        meas='loss'
        loc='upper right'
    plt.plot(hist.history[meas])
    plt.plot(hist.history['val_'+meas])
    plt.title('model '+meas)
    plt.ylabel(meas)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc=loc)
plot_train(hist)


# In[ ]:


#hist.history


# In[ ]:


# https://www.kaggle.com/saptak7/2-layer-cnn-adam-optimizer-and-5-epochs
# https://www.kaggle.com/bguberfain/naive-keras-cdiscount/notebook

from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count
num_cpus = cpu_count()

submission = pd.read_csv(
    '../input/cdiscount-image-classification-challenge/sample_submission.csv', 
    index_col='_id')

#most_frequent_guess = 1000018296
#submission['category_id'] = most_frequent_guess 

num_images_test = 100
bar = tqdm_notebook(total=num_images_test * 1)

with open('../input/cdiscount-image-classification-challenge/test.bson', 'rb') as f,          concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)

    future_load = []

    for i,d in enumerate(data):
        if i >= num_images_test:
              break
#        future_load.append(executor.submit(load_image, d['imgs'][0]['picture'], d['_id'], bar))
        future_load.append(executor.submit(load_image, d['imgs'][0], d['_id']))
        
#        print("Starting future processing")
    for future in concurrent.futures.as_completed(future_load):
        x, _id = future.result()
        y_cat = encoder.inverse_transform(np.argmax(model.predict(x[None])[0]))
#        print(y_cat)
#        y_cat = rev_labels[np.argmax(model.predict(x[None])[0])]
#        if y_cat == -1:
#            y_cat = most_frequent_guess

        bar.update()
        submission.loc[_id, 'category_id'] = y_cat
print('Finished')


# In[ ]:


submission.to_csv('new_submission.csv.gz', compression='gzip')


# In[ ]:


#encoder.inverse_transform(model.predict(x[None]))
#encoder.inverse_transform(np.argmax(model.predict(x[None])[0]))


# First kernel try in keras 
# Lot more to come!!
# Stay Advance :)
