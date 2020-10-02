#!/usr/bin/env python
# coding: utf-8

# # Overview
# The goal is to make a nice retinopathy model by using a pretrained inception v3 as a base and retraining some modified final layers with attention
# 
# This can be massively improved with 
# * high-resolution images
# * better data sampling
# * ensuring there is no leaking between training and validation sets, ```sample(replace = True)``` is real dangerous
# * better target variable (age) normalization
# * pretrained models
# * attention/related techniques to focus on areas

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # showing and rendering figures
# io related
from skimage.io import imread
import os
from glob import glob
# not needed in Kaggle, but required in Jupyter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


batch_size = 64
LEARNING_RATE = 3e-4
IMG_SIZE = (224, 224) # slightly smaller than vgg16 normally expects
EPOCHS = 15


# In[ ]:


base_image_dir = os.path.join('..', 'input')
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
all_images = glob(os.path.join(base_image_dir, 'images', '*', '*'))
print(len(all_images), 'images found')
full_food_df = pd.DataFrame(dict(path = all_images))
food_cat = LabelEncoder()
full_food_df['category'] = full_food_df['path'].map(lambda x: x.split('/')[-2].replace('_', ' '))
food_cat.fit(full_food_df['category'].values)
full_food_df['cat_vec'] = full_food_df['category'].map(lambda x: to_categorical(food_cat.transform([x]), num_classes=len(food_cat.classes_))[0])
full_food_df.sample(3)


# # Examine the distribution of classes

# In[ ]:


full_food_df.groupby(['category']).size().plot.bar(figsize = (10, 5))


# # Split Data into Training and Validation

# In[ ]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(full_food_df, 
                                   test_size = 0.25, 
                                   random_state = 2018,
                                   stratify = full_food_df['category'])
print('train', train_df.shape[0], 'validation', valid_df.shape[0])


# In[ ]:


import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import preprocess_input
import numpy as np
def tf_image_loader(out_size, 
                      horizontal_flip = True, 
                      vertical_flip = False, 
                     random_brightness = True,
                     random_contrast = True,
                    random_saturation = True,
                    random_hue = True,
                      color_mode = 'rgb',
                       preproc_func = preprocess_input,
                       on_batch = False):
    def _func(X):
        with tf.name_scope('image_augmentation'):
            with tf.name_scope('input'):
                X = tf.image.decode_png(tf.read_file(X), channels = 3 if color_mode == 'rgb' else 0)
                X = tf.image.resize_images(X[:,:,::-1], out_size)
            with tf.name_scope('augmentation'):
                if horizontal_flip:
                    X = tf.image.random_flip_left_right(X)
                if vertical_flip:
                    X = tf.image.random_flip_up_down(X)
                if random_brightness:
                    X = tf.image.random_brightness(X, max_delta = 0.15)
                if random_saturation:
                    X = tf.image.random_saturation(X, lower = 0.5, upper = 2)
                if random_hue:
                    X = tf.image.random_hue(X, max_delta = 0.15)
                if random_contrast:
                    X = tf.image.random_contrast(X, lower = 0.75, upper = 1.5)
                return preproc_func(X)
    if on_batch: 
        # we are meant to use it on a batch
        def _batch_func(X, y):
            return tf.map_fn(_func, X), y
        return _batch_func
    else:
        # we apply it to everything
        def _all_func(X, y):
            return _func(X), y         
        return _all_func
    
def tf_augmentor(out_size,
                intermediate_size = (480, 480),
                 intermediate_trans = 'crop',
                 batch_size = 16,
                   horizontal_flip = True, 
                  vertical_flip = False, 
                 random_brightness = True,
                 random_contrast = True,
                 random_saturation = True,
                    random_hue = True,
                  color_mode = 'rgb',
                   preproc_func = preprocess_input,
                   min_crop_percent = 0.001,
                   max_crop_percent = 0.005,
                   crop_probability = 0.5,
                   rotation_range = 10):
    
    load_ops = tf_image_loader(out_size = intermediate_size, 
                               horizontal_flip=horizontal_flip, 
                               vertical_flip=vertical_flip, 
                               random_brightness = random_brightness,
                               random_contrast = random_contrast,
                               random_saturation = random_saturation,
                               random_hue = random_hue,
                               color_mode = color_mode,
                               preproc_func = preproc_func,
                               on_batch=False)
    def batch_ops(X, y):
        batch_size = tf.shape(X)[0]
        with tf.name_scope('transformation'):
            # code borrowed from https://becominghuman.ai/data-augmentation-on-gpu-in-tensorflow-13d14ecf2b19
            # The list of affine transformations that our image will go under.
            # Every element is Nx8 tensor, where N is a batch size.
            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if rotation_range > 0:
                angle_rad = rotation_range / 180 * np.pi
                angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
                transforms += [tf.contrib.image.angles_to_projective_transforms(angles, intermediate_size[0], intermediate_size[1])]

            if crop_probability > 0:
                crop_pct = tf.random_uniform([batch_size], min_crop_percent, max_crop_percent)
                left = tf.random_uniform([batch_size], 0, intermediate_size[0] * (1.0 - crop_pct))
                top = tf.random_uniform([batch_size], 0, intermediate_size[1] * (1.0 - crop_pct))
                crop_transform = tf.stack([
                      crop_pct,
                      tf.zeros([batch_size]), top,
                      tf.zeros([batch_size]), crop_pct, left,
                      tf.zeros([batch_size]),
                      tf.zeros([batch_size])
                  ], 1)
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), crop_probability)
                transforms += [tf.where(coin, crop_transform, tf.tile(tf.expand_dims(identity, 0), [batch_size, 1]))]
            if len(transforms)>0:
                X = tf.contrib.image.transform(X,
                      tf.contrib.image.compose_transforms(*transforms),
                      interpolation='BILINEAR') # or 'NEAREST'
            if intermediate_trans=='scale':
                X = tf.image.resize_images(X, out_size)
            elif intermediate_trans=='crop':
                X = tf.image.resize_image_with_crop_or_pad(X, out_size[0], out_size[1])
            else:
                raise ValueError('Invalid Operation {}'.format(intermediate_trans))
            return X, y
    def _create_pipeline(in_ds):
        batch_ds = in_ds.map(load_ops, num_parallel_calls=4).batch(batch_size)
        return batch_ds.map(batch_ops)
    return _create_pipeline


# In[ ]:


def flow_from_dataframe(idg, 
                        in_df, 
                        path_col,
                        y_col, 
                        shuffle = True, 
                        color_mode = 'rgb'):
    files_ds = tf.data.Dataset.from_tensor_slices((in_df[path_col].values, 
                                                   np.stack(in_df[y_col].values,0)))
    in_len = in_df[path_col].values.shape[0]
    while True:
        if shuffle:
            files_ds = files_ds.shuffle(in_len) # shuffle the whole dataset
        
        next_batch = idg(files_ds).repeat().make_one_shot_iterator().get_next()
        for i in range(max(in_len//32,1)):
            # NOTE: if we loop here it is 'thread-safe-ish' if we loop on the outside it is completely unsafe
            yield K.get_session().run(next_batch)


# In[ ]:


core_idg = tf_augmentor(out_size = IMG_SIZE, 
                        color_mode = 'rgb', 
                        vertical_flip = True,
                        crop_probability=0.0, # crop doesn't work yet
                        rotation_range = 0,
                        batch_size = batch_size) 
valid_idg = tf_augmentor(out_size = IMG_SIZE, color_mode = 'rgb', 
                         crop_probability=0.0, 
                         horizontal_flip = False, 
                         vertical_flip = False, 
                         random_brightness = False,
                         random_contrast = False,
                         random_saturation = False,
                         random_hue = False,
                         rotation_range = 0,
                        batch_size = batch_size)

train_gen = flow_from_dataframe(core_idg, train_df, 
                             path_col = 'path',
                            y_col = 'cat_vec')

valid_gen = flow_from_dataframe(valid_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'cat_vec') # we can use much larger batches for evaluation


# # Validation Set
# We do not perform augmentation at all on these images

# In[ ]:


t_x, t_y = next(valid_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(np.clip(c_x+127, 0, 255).astype(np.uint8))
    c_ax.set_title('{}'.format(food_cat.classes_[np.argmax(c_y, -1)]))
    c_ax.axis('off')


# # Training Set
# These are augmented and a real mess

# In[ ]:


t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(np.clip(c_x+127, 0, 255).astype(np.uint8))
    c_ax.set_title('{}'.format(food_cat.classes_[np.argmax(c_y, -1)]))
    c_ax.axis('off')


# # Basic VGG Model

# In[ ]:


from keras.applications.vgg16 import VGG16 as PTModel
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
in_lay = Input(t_x.shape[1:])
base_pretrained_model = PTModel(input_shape = t_x.shape[1:], 
                                include_top = True, 
                                weights = None, # 'imagenet')
                                classes=t_y.shape[-1]
                               )

food_model = base_pretrained_model


# In[ ]:


from keras.metrics import top_k_categorical_accuracy
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import Adam
def top_5_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=5)

food_model.compile(optimizer = Adam(lr=LEARNING_RATE), 
                   loss = 'categorical_crossentropy',
                   metrics = ['categorical_accuracy', top_5_accuracy])
food_model.summary()


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('food')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=6) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


food_model.fit_generator(train_gen, 
                           steps_per_epoch = 100, #train_df.shape[0]//batch_size,
                           validation_data = valid_gen, 
                           validation_steps = 10, #valid_df.shape[0]//batch_size,
                              epochs = EPOCHS, 
                              callbacks = callbacks_list,
                             workers = 0, # tf-generators are not thread-safe
                             use_multiprocessing=False, 
                             max_queue_size = 0
                            )


# In[ ]:


# load the best version of the model
food_model.load_weights(weight_path)
food_model.save('full_food_model.h5')


# In[ ]:


import gc
gc.enable()
gc.collect()


# In[ ]:


##### create one fixed dataset for evaluating
from tqdm import tqdm_notebook
# fresh valid gen
valid_gen = flow_from_dataframe(valid_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'cat_vec') 
vbatch_count = min(5, (valid_df.shape[0]//batch_size-1))
out_size = vbatch_count*batch_size
test_X = np.zeros((out_size,)+t_x.shape[1:], dtype = np.float32)
test_Y = np.zeros((out_size,)+t_y.shape[1:], dtype = np.float32)
for i, (c_x, c_y) in zip(tqdm_notebook(range(vbatch_count)), 
                         valid_gen):
    j = i*batch_size
    test_X[j:(j+c_x.shape[0])] = c_x
    test_Y[j:(j+c_x.shape[0])] = c_y


# # Evaluate the results
# Here we evaluate the results by loading the best version of the model and seeing how the predictions look on the results. We then visualize spec

# In[ ]:


from sklearn.metrics import accuracy_score, classification_report
pred_Y = food_model.predict(test_X, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = np.argmax(test_Y, -1)
print('Accuracy on Test Data: %2.2f%%' % (100*accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat, target_names = food_cat.classes_))


# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
fig, ax1 = plt.subplots(1,1, figsize = (20, 20))
sns.heatmap(confusion_matrix(test_Y_cat, pred_Y_cat), 
            annot=False, fmt="d", cbar = False, cmap = plt.cm.Blues, vmax = test_X.shape[0]//16, ax = ax1)


# In[ ]:




