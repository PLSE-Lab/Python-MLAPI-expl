#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import pandas as pd

import os

print(os.listdir('/kaggle/input/3d-object-detection-for-autonomous-vehicles'))


# In[ ]:


get_ipython().system('pip install pyquaternion')


# In[ ]:


import json
import os.path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import itertools
from skimage.morphology import convex_hull_image


# In[ ]:


class Table:
    def __init__(self, data):
        self.data = data
        self.index = {x['token']: x for x in data}


DATA_ROOT = '/kaggle/input/3d-object-detection-for-autonomous-vehicles/'


def load_table(name, root=os.path.join(DATA_ROOT, 'train_data')):
    with open(os.path.join(root, name), 'rb') as f:
        return Table(json.load(f))

    
scene = load_table('scene.json')
sample = load_table('sample.json')
sample_data = load_table('sample_data.json')
ego_pose = load_table('ego_pose.json')
calibrated_sensor = load_table('calibrated_sensor.json')


# In[ ]:


train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv')).set_index('Id')


# Translations of coordinates from https://www.kaggle.com/lopuhin/lyft-3d-join-all-lidars-annotations-from-scratch

# In[ ]:


def rotate_points(points, rotation, inverse=False):
    assert points.shape[1] == 3
    q = Quaternion(rotation)
    if inverse:
        q = q.inverse
    return np.dot(q.rotation_matrix, points.T).T
    
def apply_pose(points, cs, inverse=False):
    """ Translate (lidar) points to vehicle coordinates, given a calibrated sensor.
    """
    points = rotate_points(points, cs['rotation'])
    points = points + np.array(cs['translation'])
    return points

def inverse_apply_pose(points, cs):
    """ Reverse of apply_pose (we'll need it later).
    """
    points = points - np.array(cs['translation']) 
    points = rotate_points(points, np.array(cs['rotation']), inverse=True)
    return points

def get_annotations(token):
    annotations = np.array(train_df.loc[token].PredictionString.split()).reshape(-1, 8)
    return {
        'point': annotations[:, :3].astype(np.float32),
        'wlh': annotations[:, 3:6].astype(np.float32),
        'rotation': annotations[:, 6].astype(np.float32),
        'cls': np.array(annotations[:, 7]),
    }


# Helpers to rotate bounding box points

# In[ ]:


import copy

import math

def rotate(origin, point, angle):
    ox, oy, _ = origin
    px, py, pz = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy, pz]


def make_box_coords(center, wlh, rotation, ep):

    planar_wlh = copy.deepcopy(wlh)
    planar_wlh = planar_wlh[[1,0,2]]

    bottom_center = copy.deepcopy(center)
    bottom_center[-1] = bottom_center[-1] - planar_wlh[-1] / 2

    bottom_points = []
    bottom_points.append(bottom_center + planar_wlh * [1, 1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [-1, -1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [1, -1, 0] / 2)
    bottom_points.append(bottom_center + planar_wlh * [-1, 1, 0] / 2)
    bottom_points = np.array(bottom_points)

    rotated_bottom_points = []
    for point in bottom_points:
        rotated_bottom_points.append(rotate(bottom_center, point, rotation))

    rotated_bottom_points = np.array(rotated_bottom_points)
    rotated_top_points = rotated_bottom_points + planar_wlh * [0,0,1]

    box_points = np.concatenate([rotated_bottom_points, rotated_top_points], axis=0)

    box_points = inverse_apply_pose(box_points, ep)
    
    return box_points


# Helpers to get training data in raster format. To each sample we can create raster 1000x1000x100 image as 3-dimensional histogram of points. Mask is bounding boxes of cars. There is get_crop_positive function to create small crops from sample image to put it in neural network 

# In[ ]:


def get_sample_data(sample_token):
    lidars = []
    for x in sample_data.data:
        if x['sample_token'] == sample_token and 'lidar' in x['filename']:
            lidars.append(x)

    lidars_data = [
        # here, sorry
        np.fromfile(os.path.join(DATA_ROOT, x['filename'].replace('lidar/', 'train_lidar/')), dtype=np.float32)
        .reshape(-1, 5)[:, :3] for x in lidars]


    all_points = []
    all_colors = []
    for points, lidar in zip(lidars_data, lidars):
        cs = calibrated_sensor.index[lidar['calibrated_sensor_token']]
        points = apply_pose(points, cs)
        all_points.append(points)
    all_points = np.concatenate(all_points)


    ego_pose_token, = {x['ego_pose_token'] for x in lidars}
    ep = ego_pose.index[ego_pose_token]
    annotations = get_annotations(sample_token)
    car_centers = annotations['point'][annotations['cls'] == 'car']
    car_wlhs = annotations['wlh'][annotations['cls'] == 'car']
    car_rotations = annotations['rotation'][annotations['cls'] == 'car']


    all_boxes = []
    for k in range(len(car_centers)):
        center = car_centers[k]
        wlh = car_wlhs[k]
        rotation = car_rotations[k]

        box_coords = make_box_coords(center, wlh, rotation, ep)
        all_boxes.append(box_coords)

    all_boxes = np.array(all_boxes)    

    car_centers = inverse_apply_pose(car_centers, ep)
    
    return all_points, all_boxes, car_centers


def get_sample_raster(all_points, all_boxes): 
    x_bounds = np.linspace(-100, 100, 1001)
    y_bounds = np.linspace(-100, 100, 1001)
    z_bounds = np.linspace(-10, 10, 101)

    sample_hist = np.histogramdd(all_points[:], [x_bounds, y_bounds, z_bounds])[0]
    sample_mask = np.zeros((len(x_bounds)-1, len(y_bounds)-1, len(z_bounds)-1))



    for box in all_boxes:
        x_min, y_min, z_min = box.min(axis=0)
        x_max, y_max, z_max = box.max(axis=0)

        x_box_bound_cnt = int(1001 / 200 * (x_max - x_min))
        y_box_bound_cnt = int(1001 / 200 * (y_max - y_min))
        z_box_bound_cnt = int(101 / 20 * (z_max - z_min))

        box_hist = np.histogramdd(box, [np.linspace(x_min, x_max, x_box_bound_cnt),
                                        np.linspace(y_min, y_max, y_box_bound_cnt),
                                        np.linspace(z_min, z_max, z_box_bound_cnt)])[0]

        box_mask = convex_hull_image(box_hist)


        x_start_idx = np.where(x_bounds > x_min)[0][0]
        y_start_idx = np.where(y_bounds > y_min)[0][0]
        z_start_idx = np.where(z_bounds > z_min)[0][0]


        x_cnt = min(sample_mask.shape[0] - x_start_idx - 1, x_box_bound_cnt - 1)
        y_cnt = min(sample_mask.shape[1] - y_start_idx - 1, y_box_bound_cnt - 1)
        z_cnt = min(sample_mask.shape[2] - z_start_idx - 1, z_box_bound_cnt - 1)

        sample_mask[x_start_idx:x_start_idx+x_cnt,
                   y_start_idx:y_start_idx+y_cnt,
                   z_start_idx:z_start_idx+z_cnt] = sample_mask[x_start_idx:x_start_idx+x_cnt,
                                                                           y_start_idx:y_start_idx+y_cnt,
                                                                           z_start_idx:z_start_idx+z_cnt] + box_mask[:x_cnt, :y_cnt, :z_cnt]

    return sample_hist, sample_mask, (x_bounds, y_bounds, z_bounds)


def get_crop_positive(sample_hist, sample_mask, bounds, car_centers, crop_size=(64, 64, 32)):
    
    half_x_size = crop_size[0] // 2
    half_y_size = crop_size[1] // 2
    half_z_size = crop_size[2] // 2
    
    (x_bounds, y_bounds, z_bounds) = bounds
    if len(car_centers) > 0:
        idx = np.random.choice(range(len(car_centers)))
        x_center, y_center, z_center = car_centers[idx]
    else:
        x_center, y_center = np.random.randint(-30, 30, 2)
        z_center = np.random.randint(-10, 10)

    x_center, y_center, z_center = [x_center, y_center, z_center] + np.random.randint(-3, 3, 3)

    x_center = min(x_center, 100 - np.abs(x_bounds[-1] - x_bounds[-2]) * half_x_size - 1)
    x_center = max(x_center, -100 + np.abs(x_bounds[-1] - x_bounds[-2]) * half_x_size + 1)

    y_center = min(y_center, 100 - np.abs(y_bounds[-1] - y_bounds[-2]) * half_y_size - 1)
    y_center = max(y_center, -100 + np.abs(y_bounds[-1] - y_bounds[-2]) * half_y_size + 1)

    z_center = min(z_center, 10 - np.abs(z_bounds[-1] - z_bounds[-2]) * half_z_size - 1)
    z_center = max(z_center, -10 + np.abs(z_bounds[-1] - z_bounds[-2]) * half_z_size + 1)




    x_center_idx = np.where(x_bounds > x_center)[0][0]
    y_center_idx = np.where(y_bounds > y_center)[0][0]
    z_center_idx = np.where(z_bounds > z_center)[0][0]

        
    crop_hist = sample_hist[x_center_idx-half_x_size:x_center_idx+half_x_size,
                            y_center_idx-half_y_size:y_center_idx+half_y_size,
                            z_center_idx-half_z_size:z_center_idx+half_z_size]
    
    crop_mask = sample_mask[x_center_idx-half_x_size:x_center_idx+half_x_size,
                            y_center_idx-half_y_size:y_center_idx+half_y_size,
                            z_center_idx-half_z_size:z_center_idx+half_z_size]

    return crop_hist, crop_mask > 0


# # Data example

# In[ ]:


sample_token = train_df.reset_index()['Id'].values[35]

all_points, all_boxes, car_centers = get_sample_data(sample_token)


# Full raster image for one sample

# In[ ]:


sample_hist, sample_mask, bounds = get_sample_raster(all_points, all_boxes)


# Crop from this image

# In[ ]:


crop_hist, crop_mask = get_crop_positive(sample_hist, sample_mask, bounds, car_centers, crop_size=(128,128,64))


# ## Source point cloud for full sample

# In[ ]:


boxes_coords = np.concatenate(all_boxes, axis=0)


plt.figure(figsize=(25,15))
plt.scatter(all_points[:, 0], all_points[:, 1],s=[0.1]*len(all_points))
#plt.scatter(car_centers[nearest_idxs, 0], car_centers[nearest_idxs, 1],s=[15]*len(nearest_idxs),color='r')
plt.scatter(boxes_coords[:, 0], boxes_coords[:, 1],s=[15]*len(boxes_coords),color='r')


# ## Source point cloud for small region with car

# In[ ]:


ann_idx = 1

center_point = car_centers[ann_idx]
x_min = center_point[0] - 5
x_max = center_point[0] + 5
y_min = center_point[1] - 5
y_max = center_point[1] + 5
z_min= center_point[2] - 5
z_max = center_point[2] + 5


area_mask = (all_points[:, 0] > x_min) * (all_points[:, 0] < x_max) * (all_points[:, 1] > y_min) * (all_points[:, 1] < y_max) * (all_points[:, 2] > z_min) * (all_points[:, 2] < z_max)
area_mask = np.where(area_mask)[0]


fig = pyplot.figure(figsize=(25,15))
ax = Axes3D(fig)
ax.scatter(all_points[area_mask, 0], all_points[area_mask, 1], all_points[area_mask, 2])

#ax.scatter([center_point[0]], [center_point[1]], [center_point[2]], color='k', s=[100])
ax.scatter(all_boxes[ann_idx][:, 0], all_boxes[ann_idx][:, 1], all_boxes[ann_idx][:, 2], color='r', s=[100])



pyplot.show()


# ## Rastered crop of a small region

# Show only top projection of 3d image

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(20,20))
axes[0].imshow(crop_hist.sum(axis=-1))
axes[1].imshow(crop_mask.sum(axis=-1))
    
plt.show()


# # Neural net part

# ### Create generator for nn

# In[ ]:


tokens = train_df.reset_index()['Id'].values

def generator(tokens, crop_size, batch_size):
    while True:
        sample_token = np.random.choice(tokens)
        all_points, all_boxes, car_centers = get_sample_data(sample_token)
        sample_hist, sample_mask, bounds = get_sample_raster(all_points, all_boxes)
        
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            crop_hist, crop_mask = get_crop_positive(sample_hist, sample_mask, bounds, car_centers, crop_size=(128, 128, 64))
            crop_hist, crop_mask = get_crop_positive(sample_hist, sample_mask, bounds, car_centers, crop_size=crop_size)

            x_batch.append(crop_hist)
            y_batch.append(crop_mask)
        
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        
        x_batch = np.expand_dims(x_batch, axis=1)
        y_batch = np.expand_dims(y_batch, axis=1)
        
        yield x_batch , y_batch


# Train and validation generators

# In[ ]:


train_loader = generator(tokens[:15000], (64,64,32), 16)
val_loader = generator(tokens[15000:], (64,64,32), 16)


# In[ ]:


for x_batch, y_batch in train_loader:
    break


# In[ ]:


i = 0
fig, axes = plt.subplots(1, 2, figsize=(20,20))
axes[0].imshow(x_batch[i].sum(axis=(0, -1)))
axes[1].imshow(y_batch[i].sum(axis=(0, -1)))
    
plt.show()


# ## Simple 3d Unet

# In[ ]:


from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam
import keras
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint

K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32,
                  batch_normalization=False, activation_name="sigmoid"):
    
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):

    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        from keras_contrib.layers.normalization import InstanceNormalization

        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)


# In[ ]:


model = unet_model_3d((1, 64,64,32))
adam = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy',
        optimizer=adam,
        metrics=['acc'])


# ### Training takes ~1 hour, so load ready weights

# In[ ]:



# model.fit_generator(generator=train_loader,
#                     steps_per_epoch=100,
#                     epochs=20,
#                     verbose=1,
#                     callbacks=[keras.callbacks.ModelCheckpoint('%d.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)],
#                     validation_data=val_loader,
#                     validation_steps=50,
#                     class_weight=None,
#                     max_queue_size=10,
#                     use_multiprocessing=False,
#                     shuffle=True,
#                     initial_epoch=0)


# In[ ]:


get_ipython().system("wget 'https://vk.com/doc77582890_516136970?hash=83735ce0a1fe5fe100&dl=078b1ac75312cff898' -O  weights.h5")


# In[ ]:


model.load_weights('weights.h5')


# In[ ]:


for x_batch, y_batch in val_loader:
    break 
    
pred = model.predict(x_batch)


# ## Vizualize predictions in all 3 projections

# In[ ]:


i = 3
fig, axes = plt.subplots(1, 4, figsize=(35,20))
axes[0].imshow(x_batch[i].sum(axis=(0, -1)))
axes[1].imshow(y_batch[i].sum(axis=(0, -1)))
axes[2].imshow((pred[i]  ).sum(axis=(0, -1)))
axes[3].imshow((pred[i] > 0.5 ).sum(axis=(0, -1)))
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(35,20))
axes[0].imshow(x_batch[i].sum(axis=(0, 1)).T)
axes[1].imshow(y_batch[i].sum(axis=(0, 1)).T)
axes[2].imshow((pred[i]  ).sum(axis=(0, 1)).T)
axes[3].imshow((pred[i] > 0.5 ).sum(axis=(0, 1)).T)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(35,20))
axes[0].imshow(x_batch[i].sum(axis=(0, 2)).T)
axes[1].imshow(y_batch[i].sum(axis=(0, 2)).T)
axes[2].imshow((pred[i]  ).sum(axis=(0, 2)).T)
axes[3].imshow((pred[i] > 0.5 ).sum(axis=(0, 2)).T)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




