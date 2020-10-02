#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook aims to organize the data and hack Keras so that we can train a model in a fairly simple way. The aim here is to get a model working that can reliably segment the images into objects and then we can make a model that handles grouping the objects into categories based on the labels. As you will see the Keras requires a fair bit of hackery to get it to load images from a dataframe and then get it to read the label images correctly (uint16 isn't supported well). Once that is done, training a U-Net model is really easy.
# 
# ## Focus
# The focus here is to get the vehicles as accurately as possible without looking at the other classes. We can also try to differentiate between the various class

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.segmentation import mark_boundaries
DATA_DIR = os.path.join('..', 'input')


# In[2]:


class_str = """car, 33
motorbicycle, 34
bicycle, 35
person, 36
rider, 37
truck, 38
bus, 39
tricycle, 40
others, 0
rover, 1
sky, 17
car_groups, 161
motorbicycle_group, 162
bicycle_group, 163
person_group, 164
rider_group, 165
truck_group, 166
bus_group, 167
tricycle_group, 168
road, 49
siderwalk, 50
traffic_cone, 65
road_pile, 66
fence, 67
traffic_light, 81
pole, 82
traffic_sign, 83
wall, 84
dustbin, 85
billboard, 86
building, 97
bridge, 98
tunnel, 99
overpass, 100
vegatation, 113
unlabeled, 255"""
class_dict = {v.split(', ')[0]: int(v.split(', ')[-1]) for v in class_str.split('\n')}
# we will just try to find moving things
car_classes = [ 'bus',  'car', 'bus_group', 'car_groups', 'truck', 'truck_group']
car_idx = [v for k,v in class_dict.items() if k in car_classes]
def read_label_image(in_path):
    idx_image = imread(in_path)//1000
    return np.isin(idx_image.ravel(), car_idx).reshape(idx_image.shape).astype(np.float32)


# In[3]:


group_df = pd.read_csv('../input/label-analysis/label_breakdown.csv', index_col = 0)
# fix the paths
group_df['color'] = group_df['color'].map(lambda x: x.replace('/input/', '/input/cvpr-2018-autonomous-driving/'))
group_df['label'] = group_df['label'].map(lambda x: x.replace('/input/', '/input/cvpr-2018-autonomous-driving/'))
group_df.sample(3)


# 1. # Let's train with very vehicle images
# Here we select a group of images with lots of vehicle in them to make the dataset less imbalanced.

# In[4]:


def total_car_vol(in_row):
    out_val = 0.0
    for k in car_classes:
        out_val += in_row[k]
    return out_val
group_df['total_vehicle'] = group_df.apply(total_car_vol,1)
group_df['total_vehicle'].plot.hist(bins = 50, normed = True)
train_df = group_df.sort_values('total_vehicle', ascending = False).head(1000)
train_df['total_vehicle'].plot.hist(bins = 50, normed = True)
print(train_df.shape[0], 'rows')


# # Explore the training set
# Here we can show the training data image by image to see what exactly we are supposed to detect with the model

# In[5]:


sample_rows = 6
fig, m_axs = plt.subplots(sample_rows, 3, figsize = (20, 6*sample_rows))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2, ax3), (_, c_row) in zip(m_axs, train_df.sample(sample_rows).iterrows()):
    c_img = imread(c_row['color'])
    l_img = read_label_image(c_row['label'])
    ax1.imshow(c_img)
    ax1.set_title('Color')
    
    ax2.imshow(l_img, cmap = 'nipy_spectral')
    ax2.set_title('Labels')
    xd, yd = np.where(l_img)
    bound_img = mark_boundaries(image = c_img, label_img = l_img, color = (1,0,0), background_label = 255, mode = 'thick')
    ax3.imshow(bound_img[xd.min():xd.max(), yd.min():yd.max(),:])
    ax3.set_title('Cropped Overlay')


# In[6]:


from sklearn.model_selection import train_test_split
train_split_df, valid_split_df = train_test_split(train_df, random_state = 2018, test_size = 0.25)
print('Training Images', train_split_df.shape[0])
print('Holdout Images', valid_split_df.shape[0])


# In[7]:


from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
IMG_SIZE = (512, 512) # many of the ojbects are small so 512x512 lets us see them
img_gen_args = dict(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range = 0.05, 
                              width_shift_range = 0.02, 
                              rotation_range = 3, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range = 0.05)
rgb_gen = ImageDataGenerator(preprocessing_function = preprocess_input, **img_gen_args)
lab_gen = ImageDataGenerator(**img_gen_args)


# In[8]:


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, seed = None, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways: seed: {}'.format(seed))
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                              seed = seed,
                                    **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# ## Replace PIL with scikit-image 
# This lets us handle the 16bit numbers well in the instanceIds image. This is incredibly, incredibly hacky, please do not use this code outside of this kernel.

# In[9]:


import keras.preprocessing.image as KPImage
from PIL import Image
class pil_image_awesome():
    @staticmethod
    def open(in_path):
        if 'instanceIds' in in_path:
            # we only want to keep the positive labels not the background
            return Image.fromarray(read_label_image(in_path))
        else:
            return Image.open(in_path)
    fromarray = Image.fromarray
KPImage.pil_image = pil_image_awesome


# # Create the generators
# We want to generate parallel streams of images and labels

# In[10]:


from skimage.filters.rank import maximum
from scipy.ndimage import zoom
def lab_read_func(in_path):
    bin_img = (imread(in_path)>1000).astype(np.uint8)
    x_dim, y_dim = bin_img.shape
    max_label_img = maximum(bin_img, np.ones((x_dim//IMG_SIZE[0], y_dim//IMG_SIZE[1])))
    return np.expand_dims(zoom(max_label_img, (IMG_SIZE[0]/x_dim, IMG_SIZE[1]/y_dim), order = 3), -1)


def train_and_lab_gen_func(in_df, batch_size = 8, seed = None):
    if seed is None:
        seed = np.random.choice(range(1000))
    train_rgb_gen = flow_from_dataframe(rgb_gen, in_df, 
                             path_col = 'color',
                            y_col = 'id', 
                            target_size = IMG_SIZE,
                             color_mode = 'rgb',
                            batch_size = batch_size,
                                   seed = seed)
    train_lab_gen = flow_from_dataframe(lab_gen, in_df, 
                             path_col = 'label',
                            y_col = 'id', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = batch_size,
                                   seed = seed)
    for (x, _), (y, _) in zip(train_rgb_gen, train_lab_gen):
        yield x, y
    
train_and_lab_gen = train_and_lab_gen_func(train_split_df, batch_size = 8)
valid_and_lab_gen = train_and_lab_gen_func(valid_split_df, batch_size = 8)


# In[11]:


(rgb_batch, lab_batch) = next(valid_and_lab_gen)

sample_rows = 4
fig, m_axs = plt.subplots(sample_rows, 3, figsize = (20, 6*sample_rows))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2, ax3), rgb_img, lab_img in zip(m_axs, rgb_batch, lab_batch):
    # undoing the vgg correction is tedious
    r_rgb_img = np.clip(rgb_img+110, 0, 255).astype(np.uint8)
    ax1.imshow(r_rgb_img)
    ax1.set_title('Color')
    ax2.imshow(lab_img[:,:,0], cmap = 'nipy_spectral')
    ax2.set_title('Labels')
    if lab_img.max()>0.1:
        xd, yd = np.where(lab_img[:,:,0]>0)
        bound_img = mark_boundaries(image = r_rgb_img, label_img = lab_img[:,:,0], 
                                    color = (1,0,0), background_label = 255, mode = 'thick')
        ax3.imshow(bound_img[xd.min():xd.max(), yd.min():yd.max(),:])
        ax3.set_title('Cropped Overlay')


# In[12]:


fcl_size = 256
out_depth = 2


# In[13]:


from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Dropout, Flatten, Reshape, Dense
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

# Build U-Net model
inputs = Input(IMG_SIZE+(3,))
s = BatchNormalization()(inputs) # we can learn the normalization step
s = Dropout(0.5)(s)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

c5_min = Conv2D(out_depth, (3, 3), activation='relu', padding='same') (c5)

# fully connected component for spatial sensitivity
flat_c5 = Dropout(0.5)(Flatten()(c5_min))
fcl_c5 = Dropout(0.5)(Dense(fcl_size)(flat_c5))
out_shape = c5._keras_shape[1:3]+(out_depth,)
fcl_c5_imgflat = Dense(np.prod(out_shape))(fcl_c5)
fcl_img = Reshape(out_shape)(fcl_c5_imgflat)
new_c5 = concatenate([c5, fcl_img])

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (new_c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.summary()


# In[14]:


import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec))

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_bce_loss(y_true, y_pred):
    return 0.5*binary_crossentropy(y_true, y_pred)-dice_coef(y_true, y_pred)

model.compile(optimizer = 'adam', 
                   loss = dice_bce_loss, 
                   metrics = [dice_coef, 'binary_accuracy', 'mse'])


# In[15]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('unet')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=5) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[16]:


# reset the generators so they all have different seeds when multiprocessing lets loose
from IPython.display import clear_output
batch_size = 8
train_and_lab_gen = train_and_lab_gen_func(train_split_df, batch_size = batch_size)
valid_and_lab_gen = train_and_lab_gen_func(valid_split_df, batch_size = batch_size)
model.fit_generator(train_and_lab_gen, 
                    steps_per_epoch = 2048//batch_size,
                    validation_data = valid_and_lab_gen,
                    validation_steps = 256//batch_size,
                    epochs = 4, 
                    workers = 2,
                    use_multiprocessing = True,
                    callbacks = callbacks_list)
clear_output()


# In[ ]:


model.load_weights(weight_path)
model.save('vehicle_unet.h5')


# In[19]:


# Show the performance on a small batch since we delete the other messages
eval_out =  model.evaluate_generator(valid_and_lab_gen, steps=8)
clear_output()


# In[20]:


print('Loss: %2.2f, DICE: %2.2f, Accuracy %2.2f%%, Mean Squared Error: %2.2f' % (eval_out[0], eval_out[1], eval_out[2]*100, eval_out[3]))


# # Showing the results
# Here we can preview the output of the model on a few examples

# In[24]:


(rgb_batch, lab_batch) = next(valid_and_lab_gen)
sample_rows = 8
fig, m_axs = plt.subplots(sample_rows, 5, figsize = (20, 6*sample_rows))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2, ax2_pred, ax3, ax3_pred), rgb_img, lab_img in zip(m_axs, rgb_batch, lab_batch):
    # undoing the vgg correction is tedious
    r_rgb_img = np.clip(rgb_img+110, 0, 255).astype(np.uint8)
    lab_pred = model.predict(np.expand_dims(rgb_img, 0))[0]
    
    ax1.imshow(r_rgb_img)
    ax1.set_title('Color')
    ax2.imshow(lab_img[:,:,0], cmap = 'bone_r')
    ax2.set_title('Labels')
    ax2_pred.imshow(lab_pred[:,:,0], cmap = 'bone_r')
    ax2_pred.set_title('Pred Labels')
    if lab_img.max()>0.1:
        xd, yd = np.where(lab_img[:,:,0]>0)
        bound_img = mark_boundaries(image = r_rgb_img, label_img = lab_img[:,:,0], 
                                    color = (1,0,0), background_label = 255, mode = 'thick')
        ax3.imshow(bound_img[xd.min():xd.max(), yd.min():yd.max(),:])
        ax3.set_title('Cropped Overlay')
        bound_pred = mark_boundaries(image = r_rgb_img, label_img = (lab_pred[:,:,0]>0.5).astype(int), 
                                    color = (1,0,0), background_label = 0, mode = 'thick')
        ax3_pred.imshow(bound_pred[xd.min():xd.max(), yd.min():yd.max(),:])
        ax3_pred.set_title('Cropped Prediction')
fig.savefig('trained_model.png')


# In[ ]:




