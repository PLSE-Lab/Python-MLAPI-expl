#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# 

# ## Setup

# In[ ]:


import os
from IPython.display import FileLink


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = Path('../input')
SAMPLE_RATE = 60

import doctest
import copy
import functools
# tests help notebooks stay managable
def autotest(func):
    globs = copy.copy(globals())
    globs.update({func.__name__: func})
    doctest.run_docstring_examples(
        func, globs, verbose=True, name=func.__name__)
    return func


# In[ ]:


import h5py
from tqdm import tqdm
hdf5_path = BASE_DIR / 'tc_processed.h5'
print('%2.2f MB' % (os.stat(hdf5_path).st_size//1024/1024.0))
all_data = {}
with h5py.File(hdf5_path, 'r') as hf:
  for k1 in hf.keys():
    
    if True:
      all_data[k1] = {}
      for k2 in hf[k1].keys():
        all_data[k1][k2] = hf[k1][k2].value
print(list(all_data.keys()))
walking_keys = [k1 for k1 in all_data.keys() if 'walking' in k1]
hp_feet_keys = ['Head', 'Hips', 'LeftFoot', 'RightFoot']


# In[ ]:


sample_name = '04_10-walking2'
test_data = all_data[sample_name]


# In[ ]:


fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(111, projection='3d')
for c_col in test_data.keys():
  if ('_' not in c_col) and (c_col != 'frame'):
    xyz_vec = test_data[c_col]
    ax1.plot(xyz_vec[:, 0], 
            xyz_vec[:, 2],
            xyz_vec[:, 1], '-', label=c_col)
ax1.legend()
ax1.set_title(sample_name)


# ## Preview Plots
# Make sure the data is loaded correctly and that we can visualize well

# In[ ]:


x_grid, y_grid = 3, 3
fig = plt.figure(figsize=(30, 30))
for i, sample_name in enumerate(all_data.keys(), 1):
  ax1 = fig.add_subplot(100*x_grid+10*y_grid+i, projection='3d')
  test_data = all_data[sample_name]
  for c_col in test_data.keys():
    if c_col in hp_feet_keys:
      xyz_vec = test_data[c_col]
      ax1.plot(xyz_vec[:, 0], 
              xyz_vec[:, 2],
              xyz_vec[:, 1], '-', label=c_col)
  ax1.legend()
  ax1.set_title(sample_name)
  if i==x_grid*y_grid:
    break


# ### Head Centric Plots

# In[ ]:


x_grid, y_grid = 3, 3
fig = plt.figure(figsize=(30, 30))
for i, sample_name in enumerate(all_data.keys(), 1):
  ax1 = fig.add_subplot(100*x_grid+10*y_grid+i, projection='3d')
  test_data = all_data[sample_name]
  ax1.plot([0], [0], [0], 'ks', ms=10, label='Head')
  for c_col in hp_feet_keys[1:]:
    xyz_vec = test_data[c_col].copy()
    head_xyz = test_data['Head']
    ax1.plot(xyz_vec[:, 0]-head_xyz[:, 0], 
            xyz_vec[:, 2]-head_xyz[:, 2],
            xyz_vec[:, 1]-head_xyz[:, 1], 
             '-', 
             label=c_col)
  ax1.legend()
  ax1.set_title(sample_name)
  if i==x_grid*y_grid:
    break


# In[ ]:


fig, m_axs = plt.subplots(3, 12, figsize=(30, 10))
for c_axs, sample_name in zip(m_axs.T, all_data.keys()):
  test_data = all_data[sample_name]
  
  for c_ax, (i, c_x) in zip(c_axs, enumerate('xyz')):
    c_ax.axhline([0], c='k', label='Head')
    for c_col in hp_feet_keys[1:]:
      xyz_vec = test_data[c_col]
      head_xyz = test_data['Head']
      c_ax.plot(xyz_vec[:, i]-head_xyz[:, i], '-', 
                 label='{}'.format(c_col, c_x))
    c_ax.legend()
    c_ax.set_title(f'{sample_name}: {c_x}')


# # Training Model
# 

# In[ ]:


FEET_RELATIVE_TO_HEAD = True
TRAIN_SEQ_LEN = 180
TRAIN_STEPS = 64000
VALID_STEPS = 2048
RANDOM_SEED = 2019
EPOCHS = 150
BATCH_SIZE = 256
GAUSS_NOISE = 0.75
DROPOUT_RATE = 0.5
STACK_UNITS = 4
BASE_DEPTH = 64
DILATION_COUNT = 6
USE_NON_WALKING = True
AUGMENT_WALKS = False
MISSING_HEADPOSE = False


# ## Training Idea
# - Given head ($\vec{H}(t_0)$) and feet ($\vec{F}_{\textrm{left}}(t_0)$ and $\vec{F}_{\textrm{right}}(t_0)$) positions at $t_0$
# - We redefine
# - We define $\vec{H}_{\Delta}(s) = \vec{H}(t_0+s)-\vec{H}(t_0)$
# - We try and predict $\vec{F}_{(L,R)\Delta}(s) = \vec{F}_{(L, R)}(t_0+s)-\vec{F}_{(L, R)}(t_0)$
# - Take 2 seconds of head data ($\vec{H}-\vec{H}_0$) and try 

# In[ ]:


x_grid, y_grid = 2, 2
fig = plt.figure(figsize=(10, 10))
for i, sample_name in enumerate(reversed(sorted(all_data.keys())), 1):
  ax1 = fig.add_subplot(100*x_grid+10*y_grid+i, projection='3d')
  test_data = {k: v[:240] for k,v in all_data[sample_name].items()} # 2 seconds
  #ax1.plot([0], [0], [0], 'ks', ms=10, label='Head')
  head_xyz = test_data['Head']
  fl_xyz = test_data['LeftFoot']
  fr_xyz = test_data['RightFoot']
  #ax1.plot([0], [0], [0], 'ks', ms=10, label='Head')
  
  
  mid_foot_xyz = (fl_xyz-fr_xyz)/2+fr_xyz
  
  init_body_vec = mid_foot_xyz[0, :]-head_xyz[0, :]
  
  body_vec = np.tile(np.expand_dims(init_body_vec, 0), 
                     (head_xyz.shape[0], 1))+head_xyz
  
  fl_xyz -= body_vec
  ax1.plot(fl_xyz[:, 0], 
          fl_xyz[:, 2],
          fl_xyz[:, 1], 
           '-', 
           label='Left foot')
  
  fr_xyz -= body_vec
  ax1.plot(fr_xyz[:, 0], 
          fr_xyz[:, 2],
          fr_xyz[:, 1], 
           '-', 
           label='Right foot')
  ax1.legend()
  ax1.set_title(sample_name)
  ax1.view_init(0, 0)
  if i==x_grid*y_grid:
    break


# In[ ]:


plt.plot(fl_xyz[:, 2], fl_xyz[:, 1])


# In[ ]:


@autotest
def diffpad(in_x, 
            n=1,
            axis=0,
            starting_value=0):
  """
  Run diff and pad the results to keep the same 
  If the starting_value is the same then np.cumsum should exactly undo diffpad
  >>> diffpad([1, 2, 3], axis=0)
  array([0, 1, 1])
  >>> np.cumsum(diffpad([1, 2, 3], axis=0, starting_value=1))
  array([1, 2, 3])
  >>> diffpad(np.cumsum([0, 1, 2]), axis=0)
  array([0, 1, 2])
  >>> diffpad(np.eye(3), axis=0)
  array([[ 0.,  0.,  0.],
         [-1.,  1.,  0.],
         [ 0., -1.,  1.]])
  >>> diffpad(np.eye(3), axis=1)
  array([[ 0., -1.,  0.],
         [ 0.,  1., -1.],
         [ 0.,  0.,  1.]])
  """
  assert axis>=0, "Axis must be nonneggative"
  d_x = np.diff(in_x, n=n, axis=axis)
  return np.pad(d_x, [(n, 0) if i==axis else (0,0) 
                      for i, _ in enumerate(np.shape(in_x))], 
                mode='constant', constant_values=starting_value)


# In[ ]:


def generate_batch(data_dict, # type: Dict[str, Dict[str, np.array]]
                   seq_len=TRAIN_SEQ_LEN, # type: Optional[int]
                   debug=True,
                   include_offset=False
                  ):
  """
  The basic generator for taking random small chunks from all the data
  It produces the X (head pose) and (y1, y2) foot pose (left and right)
  
  arguments:
    data_dict: a dictionary with all of the data in it organized by experiment, sensor, array
    seq_len: an optional number of how many samples (60hz) to cut the data (None means use the whole data)
    debug: shows a plot
    include_offset: return initial foot positions
  
  """
  data_keys = list(data_dict.keys())
  while True:
    sample_name = np.random.choice(data_keys)
    pt_count = data_dict[sample_name]['Head'].shape[0]
    if seq_len is None:
      pt_start = 0
      pt_end = pt_count
    else:
      pt_start = np.random.choice(range(pt_count-seq_len))
      pt_end = pt_start+seq_len
    test_data = {k: v[pt_start:pt_end] for k,v in data_dict[sample_name].items()} 
    head_xyz = test_data['Head']
    fl_xyz = test_data['LeftFoot']
    fr_xyz = test_data['RightFoot']
    
    mid_foot_xyz = (fl_xyz-fr_xyz)/2+fr_xyz
    init_body_vec = mid_foot_xyz[0:1, :]-head_xyz[0:1, :]
    projected_body_vec = np.tile(init_body_vec, (head_xyz.shape[0], 1))+head_xyz
    if FEET_RELATIVE_TO_HEAD:
      fl_xyz_body = fl_xyz-projected_body_vec
      fr_xyz_body = fr_xyz-projected_body_vec
    else:
      fl_xyz_body = fl_xyz
      fr_xyz_body = fr_xyz
    
    # keep just the differential steps
    fl_xyz = diffpad(fl_xyz_body, axis=0)
    fr_xyz = diffpad(fr_xyz_body, axis=0)
    
    head_xyz_zero = head_xyz-np.tile(head_xyz[:1, :], (head_xyz.shape[0], 1))
    ft_xyz = np.concatenate([fl_xyz, fr_xyz], 1)
    if debug:
      fig = plt.figure(figsize=(20, 10))
      ax0 = fig.add_subplot(121, projection='3d')
      ax0.plot(head_xyz_zero[:, 0], 
          head_xyz_zero[:, 2],
          head_xyz_zero[:, 1], 
           '-', 
           label='Left foot')
      ax1 = fig.add_subplot(122, projection='3d')
      ax1.plot(fl_xyz[:, 0], 
          fl_xyz[:, 2],
          fl_xyz[:, 1], 
           '-', 
           label='Left foot')
      ax1.plot(fr_xyz[:, 0], 
          fr_xyz[:, 2],
          fr_xyz[:, 1], 
           '-', 
           label='Right foot')
      ax1.legend()
      ax1.set_title(sample_name)
      ax1.view_init(0, 0)
    if include_offset:
      #TODO: returning different things with different args is always a BAD IDEA
      yield head_xyz_zero, ft_xyz, (fl_xyz_body[0], fl_xyz_body[0])
    else:
      yield head_xyz_zero, ft_xyz


# In[ ]:


walk_gen = generate_batch({k: v for k,v in 
                           all_data.items() if 'walking' in k},
                         seq_len=TRAIN_SEQ_LEN, debug=False)


# In[ ]:


X, y = next(walk_gen)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
ax1.plot(X)
ax2.plot(y[:, :3], label='Left')
ax2.plot(y[:, 3:], label='Right')
ax2.legend()

ax3.plot(np.cumsum(y[:, :3], axis=0), label='Left')
ax3.plot(np.cumsum(y[:, 3:], axis=0), label='Right')
ax3.legend()


# In[ ]:


def batch_it(in_gen, batch_size=64):
  out_vals = []
  for c_vals in in_gen:
    out_vals += [c_vals]
    if len(out_vals)==batch_size:
      yield tuple([np.stack([c_row[i] for c_row in out_vals], 0) 
                   for i in range(len(c_vals))])
      out_vals = []
bat_watch_gen = batch_it(walk_gen)
tX, ty = next(bat_watch_gen)
print(tX.shape, ty.shape)


# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.hist(tX.ravel())
foot_bins = np.linspace(-2, 2, 20)
ax2.hist(ty[:, :, :3].ravel().clip(foot_bins.min(), foot_bins.max()), 
         foot_bins, label='Left', alpha=0.75, stacked=True)

ax2.hist(ty[:, :, 3:].ravel().clip(foot_bins.min(), foot_bins.max()), 
         foot_bins, label='Right', alpha=0.5, stacked=True)
ax2.legend()


# In[ ]:


from keras import models, layers
from keras.layers import CuDNNLSTM as LSTM
from keras.layers import LSTM
def simple_gait_mod(dil_rates, 
                    base_depth=16, 
                    stack_count=2,
                    crop_dist=10, 
                    seq_len=tX.shape[1],
                    use_gauss_noise=False
                   ):
  in_mod = layers.Input((seq_len, 3), name='HeadPose') # just headpose
  bn_in = layers.BatchNormalization()(in_mod)
  if use_gauss_noise:
    bn_in = layers.GaussianNoise(GAUSS_NOISE)(bn_in)
  c1 = layers.Conv1D(base_depth, (3,), 
                     padding='same', 
                     activation='relu')(bn_in)
  c2 = layers.Conv1D(base_depth, (3,), 
                     padding='same', 
                     activation='relu')(c1)
  
  stack_in = c2
  for i in range(stack_count):
    if i>0:
      if stack_in._keras_shape[-1]!=base_depth*2:
        c_out = [layers.Conv1D(base_depth*2, (1,), 
                       padding='same', 
                       activation='relu')(stack_in)]
      else:
        c_out = [stack_in]
    else:
      c_out = [stack_in]
    
    for c_d in range(0, dil_rates+1):
      c_out += [layers.Conv1D(base_depth*2, (3,), 
                              padding='same', 
                              activation='relu',
                              dilation_rate=2**c_d,
                              name='C1D_L{}_D{}'.format(i, 2**c_d))(stack_in)]
    if i>0:
      c_cat = layers.add(c_out)
    else:
      c_cat = layers.concatenate(c_out)
    stack_in = c_cat
  
  def _make_foot(in_feat, name):
    c_feat = layers.Conv1D(base_depth+4, (1,), 
                  padding='same', 
                  activation='relu', 
                  name='{}_features'.format(name))(in_feat)
    foot_out = layers.Conv1D(3, (1,), 
                  padding='same', 
                  activation='tanh', 
                  name='{}_output'.format(name))(c_feat)
    return foot_out
  c_left = _make_foot(c_cat, 'left')
  c_right = _make_foot(c_cat, 'right')
  c_feet = layers.concatenate([c_left, c_right], axis=-1)
  
  c_crop = layers.Cropping1D((crop_dist))(c_feet)
  c_pad = layers.ZeroPadding1D((crop_dist))(c_crop)
  return models.Model(inputs=[in_mod], 
                      outputs=[c_pad])

def lstm_gait_mod(n_lstms, base_depth=16, crop_dist=0, seq_len=tX.shape[1]):
  in_mod = layers.Input((seq_len, 3), name='HeadPose') # just headpose
  bn_in = layers.BatchNormalization()(in_mod)
  bn_in = layers.GaussianNoise(GAUSS_NOISE)(bn_in)
  c1 = layers.Conv1D(base_depth, (3,), padding='same', activation='relu')(bn_in)
  c2 = layers.Conv1D(base_depth, (3,), padding='same', activation='relu')(c1)
  last_layer = c2
  for k in range(n_lstms):
    c_lstm = LSTM(base_depth*2, return_sequences=True)
    last_layer = layers.Bidirectional(c_lstm)(last_layer)
    last_layer = layers.Dropout(DROPOUT_RATE)(last_layer)
  c_cat = last_layer
  
  def _make_foot(in_feat, name):
    c_feat = layers.Conv1D(base_depth+4, (1,), 
                  padding='same', 
                  activation='relu', 
                  name='{}_features'.format(name))(in_feat)
    foot_out = layers.Conv1D(3, (1,), 
                  padding='same', 
                  activation='tanh', 
                  name='{}_output'.format(name))(c_feat)
    
    foot_out = layers.Lambda(lambda x: x*2, name='{}_scaling'.format(name))(foot_out)
    
    return layers.Reshape((1, seq_len, 3))(foot_out)
  c_left = _make_foot(c_cat, 'left')
  c_right = _make_foot(c_cat, 'right')
  c_feet = layers.concatenate([c_left, c_right], axis=-1)
  if crop_dist>0:
    c_crop = layers.Cropping2D((0, crop_dist))(c_feet)
    c_pad = layers.ZeroPadding2D((0, crop_dist))(c_crop)
  else:
    c_pad = c_feet
  return models.Model(inputs=[in_mod], 
                      outputs=[c_pad],
                     name='LSTM_Stacked')


# In[ ]:


basic_gait = simple_gait_mod(stack_count=STACK_UNITS, 
                             dil_rates=DILATION_COUNT, 
                             base_depth=BASE_DEPTH)
#basic_gait = lstm_gait_mod(STACK_UNITS, base_depth=BASE_DEPTH)
basic_gait.summary()


# In[ ]:


from keras.utils import vis_utils
from IPython.display import SVG
SVG(vis_utils.model_to_dot(basic_gait, show_shapes=True).create_svg())


# In[ ]:


vis_utils.model_to_dot(basic_gait, show_shapes=True).write_png('model.png')
FileLink('model.png')


# In[ ]:


from keras import optimizers, losses
basic_gait.compile(optimizer=optimizers.Adam(lr=1e-3), 
                  loss=losses.mean_squared_error,
                  metrics=[losses.mean_absolute_error])


# ## Setup Training and Validation
# Use different people for training and validation to make the results more meaningful

# In[ ]:


all_data.keys()


# In[ ]:


all_train_keys = [k for k in all_data.keys() if k.split('_')[0] in ['01', '02', '03']]
all_valid_keys = [k for k in all_data.keys() if k.split('_')[0] in ['04', '05']]

if USE_NON_WALKING:
  train_keys = [k for k in all_train_keys]
  valid_keys = [k for k in all_valid_keys if 'walking' in k]
  
else:
  train_keys = [k for k in all_train_keys if 'walking' in k]
  valid_keys = [k for k in all_valid_keys if 'walking' in k]
print('Train:', train_keys)
print('Valid:', valid_keys)             


# In[ ]:


if AUGMENT_WALKS:
  raise NotImplementedError('Augmentation has not been implemented yet')
def _get_and_batch(in_keys, batch_size):
  return next(
      batch_it(
          generate_batch(
              {k: all_data[k] for k in in_keys},
              seq_len=TRAIN_SEQ_LEN, debug=False), 
          batch_size=batch_size)
  )
np.random.seed(RANDOM_SEED)
train_X, train_y = _get_and_batch(train_keys, TRAIN_STEPS)
valid_X, valid_y = _get_and_batch(valid_keys, VALID_STEPS)
print(train_X.shape, valid_X.shape)


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('footpose_lstm')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=15) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# ### Training Loop

# In[ ]:


from keras.utils import Sequence
class keras_walk_gen(Sequence):
  def __init__(self, x_vals, y_vals, batch_size):
    self._x = x_vals
    self._y = y_vals
    self._batch_size = batch_size
    
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(self._x.shape[0] / self._batch_size))
  
  def __getitem__(self, index):
    'Generate one batch of data'
    start_idx = index*self._batch_size
    x_batch = self._x[start_idx:(start_idx+self._batch_size)]
    y_batch = self._y[start_idx:(start_idx+self._batch_size)]
    x_batch += np.random.normal(0, GAUSS_NOISE, size=x_batch.shape)
    return x_batch, y_batch
        
  
  def on_epoch_end(self):
    idx = np.random.permutation(range(self._x.shape[0]))
    self._x = self._x[idx]
    self._y = self._y[idx]
    
noisy_gen = keras_walk_gen(train_X, train_y, BATCH_SIZE)


# In[ ]:


from IPython.display import clear_output
model_results = basic_gait.fit_generator(noisy_gen,
                                         validation_data=(valid_X, valid_y), 
                                         callbacks=callbacks_list,
                                         epochs=EPOCHS)
clear_output()
def show_results(model_hist):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
  ax1.plot(model_hist.history['loss'], label='Training')
  ax1.plot(model_hist.history['val_loss'], label='Validation')
  ax1.set_title('Loss')
  ax1.legend()
  ax2.semilogy(model_hist.history['mean_absolute_error'], label='Training')
  ax2.semilogy(model_hist.history['val_mean_absolute_error'], label='Validation')
  ax2.set_title('MAE')
  ax2.legend()
show_results(model_results)


# ### Showing Outputs

# In[ ]:


basic_gait.load_weights(weight_path)


# In[ ]:


get_ipython().run_cell_magic('time', '', '_ = basic_gait.predict(train_X[:8192], batch_size=512, verbose=True);')


# In[ ]:


v_idx = np.random.choice(range(valid_X.shape[0]))
X, y = valid_X[v_idx], valid_y[v_idx]
y_pred = basic_gait.predict(np.expand_dims(X, 0))[0]


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize=(15, 15))
for i, (ax1, ax2, ax3) in enumerate(m_axs.T):
  ax1.plot(X[:, i])
  ax1.set_title(f'Head Pose {"xyz"[i]}')
  ax2.plot(y[:,i], label='Left')
  ax2.plot(y_pred[:,i], '.-', label='Left Predicted')
  ax2.legend()
  ax2.set_title(f'Left: $\Delta {"xyz"[i]}$')

  ax3.plot(y[:,3+i], label='Right')
  ax3.plot(y_pred[:,3+i], label='Right Predicted')
  ax3.legend()
  ax3.set_title(f'Right: $\Delta {"xyz"[i]}$')


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize=(15, 15))
y_pred = basic_gait.predict(np.expand_dims(X, 0))[0]
for i, (ax1, ax2, ax3) in enumerate(m_axs.T):
  ax1.plot(X[:, i])
  ax1.set_title(f'Head Pose {"xyz"[i]}')
  ax2.plot(np.cumsum(y[:,i], 0), label='Left')
  ax2.plot(np.cumsum(y_pred[:,i], 0), '.-', label='Left Predicted')
  ax2.legend()
  ax2.set_title(f'Left: ${"xyz"[i]}$')

  ax3.plot(np.cumsum(y[:,3+i], 0), label='Right')
  ax3.plot(np.cumsum(y_pred[:,3+i]), label='Right Predicted')
  ax3.legend()
  ax3.set_title(f'Right: ${"xyz"[i]}$')


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))

ax1.plot(X[:, 0], X[:, 1], label='Head Pose')
ax1.set_title('XY Plot')

ax2.plot(X[:, 2], X[:, 1], label='Head Pose')
ax2.set_title('ZY Plot')

ax3.plot(np.cumsum(y[:,0], 0), 
         np.cumsum(y[:,1], 0), 'r-', label='Left')
ax3.plot(np.cumsum(y_pred[:,0], 0), 
         np.cumsum(y_pred[:,1], 0), 'r-.', label='Left Predicted')

ax3.plot(np.cumsum(y[:,3+0], 0), 
         np.cumsum(y[:,3+1], 0), 'g-', label='Right')
ax3.plot(np.cumsum(y_pred[:,3+0], 0), 
         np.cumsum(y_pred[:,3+1], 0), 'g-.', label='Right Predicted')

ax3.legend()
ax3.set_title('XY Feet')


ax4.plot(np.cumsum(y[:,2], 0), 
         np.cumsum(y[:,1], 0), 'r-', label='Left')
ax4.plot(np.cumsum(y_pred[:,2], 0), 
         np.cumsum(y_pred[:,1], 0), 'r-.', label='Left Predicted')

ax4.plot(np.cumsum(y[:,3+2], 0), 
         np.cumsum(y[:,3+1], 0), 'g-', label='Right')
ax4.plot(np.cumsum(y_pred[:,3+2], 0), 
         np.cumsum(y_pred[:,3+1], 0), 'g-.', label='Right Predicted')


ax3.legend()
ax3.set_title('ZY Feet')


# In[ ]:


np.random.seed(RANDOM_SEED-2)
v_idx = np.random.choice(range(valid_X.shape[0]))
X, y = valid_X[v_idx], valid_y[v_idx]
y_pred = basic_gait.predict(np.expand_dims(X, 0))[0]


# In[ ]:


fig = plt.figure(figsize=(20, 8))
body_height_estimate = 50
body_stance_estimate = 20
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
show_head_vec = X.copy()
show_head_vec[:, 1]+=body_height_estimate # above the fray

fl_pos = np.cumsum(y[:, :3], 0)
fr_pos = np.cumsum(y[:, 3:], 0)

if FEET_RELATIVE_TO_HEAD:
  fl_pos+=X
  fl_pos+=X

fr_pos[:,2]+=body_stance_estimate

ax1.plot(show_head_vec[:, 0], 
         show_head_vec[:, 2], 
         show_head_vec[:, 1], 
         'k-', ms=10, label='Head')

ax1.plot(fl_pos[:, 0], 
         fl_pos[:, 2], 
         fl_pos[:, 1], 
         'g-', ms=5, label='Left Foot')


ax1.plot(fr_pos[:, 0], 
         fr_pos[:, 2], 
         fr_pos[:, 1], 
         'r-', ms=5, label='Right Foot')

ax1.set_title('Recorded')
ax1.legend()


y_pred = basic_gait.predict(np.expand_dims(X, 0))[0]

fl_pred = np.cumsum(y_pred[:, :3], 0)
fr_pred = np.cumsum(y_pred[:, 3:], 0)

if FEET_RELATIVE_TO_HEAD:
  fl_pred+=X
  fr_pred+=X
  
fr_pred[:,2]+=body_stance_estimate


ax2.plot(show_head_vec[:, 0], 
         show_head_vec[:, 2], 
         show_head_vec[:, 1], 
         'k-', ms=5, label='Head')


ax2.plot(fl_pred[:, 0], 
         fl_pred[:, 2], 
         fl_pred[:, 1], 
         'g-', ms=5, label='Left Foot')


ax2.plot(fr_pred[:, 0], 
         fr_pred[:, 2], 
         fr_pred[:, 1], 
         'r-', ms=5, label='Right Foot')

ax2.set_title('Predicted')
ax2.legend()

ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())
ax1.view_init(30, 15)
ax2.view_init(30, 15)


# In[ ]:


vec_names = ['head', 'leftfoot_pos', 'rightfoot_pos', 'leftfoot_pred', 'rightfoot_pred']
pd.DataFrame([{'{}_{}'.format(v_name, c_x): c_v 
  for v_name, v_pos in zip(vec_names, vec_pos) 
  for c_x, c_v in zip('xyz', v_pos)} 
 for vec_pos in zip(show_head_vec, fl_pos, fr_pos, fl_pred, fr_pred)]).to_csv('demo.csv', index=False)
FileLink('demo.csv')


# In[ ]:


from matplotlib.animation import FuncAnimation
def _show_frame(i, ms=10, add_body=True):
  c_head = show_head_vec[i:i+1]
  for c_ax, [c_fl, c_fr] in zip([ax1, ax2], 
                  [(fl_pos[i:i+1], fr_pos[i:i+1]), 
                   (fl_pred[i:i+1], fr_pred[i:i+1])]):
    
    [c_line.remove() for c_line in c_ax.get_lines() if c_line.get_label().startswith('_')];
    if add_body:
      c_feet = (c_fr+c_fl)/2
      c_hips = c_feet + (c_head-c_feet)/2
      c_ax.plot(c_hips[:, 0], 
             c_hips[:, 2], 
             c_hips[:, 1], 'ks')
      
      c_ax.plot(
          [c_head[0, 0], c_hips[0, 0]],
          [c_head[0, 2], c_hips[0, 2]],
          [c_head[0, 1], c_hips[0, 1]],
          'm-', lw=10, alpha=0.75 
      )
      
      c_ax.plot(
          [c_hips[0, 0], c_fr[0, 0]],
          [c_hips[0, 2], c_fr[0, 2]],
          [c_hips[0, 1], c_fr[0, 1]],
          'm-', lw=3, alpha=0.75 
      )
      
      c_ax.plot(
          [c_hips[0, 0], c_fl[0, 0]],
          [c_hips[0, 2], c_fl[0, 2]],
          [c_hips[0, 1], c_fl[0, 1]],
          'm-', lw=3, alpha=0.75 
      )
    
    c_ax.plot(c_head[:, 0], 
             c_head[:, 2], 
             c_head[:, 1], 'ko', ms=2*ms)
  
    c_ax.plot(c_fl[:, 0], 
           c_fl[:, 2], 
           c_fl[:, 1], 
           'gs', ms=ms)
    c_ax.plot(c_fr[:, 0], 
           c_fr[:, 2], 
           c_fr[:, 1], 
           'rs', ms=ms)
    
      

out_anim = FuncAnimation(fig, _show_frame, range(0, show_head_vec.shape[0], 1))
out_anim.save('walk_recording.mp4', bitrate=8000, fps=8)
FileLink('walk_recording.mp4')


# In[ ]:


trained_model_path = 'model_weights.h5'
basic_gait.save_weights(trained_model_path)


# # Saving Experiments Infrastructure
# Save all the results in a json file

# ### Google Forms Code

# In[ ]:


from six.moves.urllib.parse import urlparse, parse_qs
from six.moves.urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
import json
import logging
from itertools import chain
import warnings
import pandas as pd
import json
import time, datetime
import string


def try_json(x):
  try:
    json.dumps(x)
    return True
  except Exception:
    return False
fetch_capital_globals = lambda : {k: globals()[k] for k in globals().keys() 
                                  if (k==k.upper()) & # all uppercase
                                  (len([clet for clet in k if clet in 
                                        string.ascii_uppercase])>3) # at least 3 letters
                                 }
fetch_json_globals = lambda : {k: v for k,v in fetch_capital_globals().items()
                               if try_json(v)}

def keyed_format(in_str, **kwargs):
  """simple format function that only replaces specific keys and works better with code"""
  new_str = in_str.replace('{', '{{').replace('}', '}}')
  for key in kwargs.keys():
        new_str = new_str.replace('{{%s}}' % key, '{%s}' % key)
  return new_str.format(**kwargs)

def get_questions(in_url):
    res = urlopen(in_url)
    soup = BeautifulSoup(res.read(), 'html.parser')

    def get_names(f):
        return [v for k, v in f.attrs.items() if 'label' in k]

    def get_name(f):
        return get_names(f)[0] if len(
            get_names(f)) > 0 else 'unknown'

    all_questions = soup.form.findChildren(
        attrs={'name': lambda x: x and x.startswith('entry.')})
    return {get_name(q): q['name'] for q in all_questions}
  
def submit_response(form_url, 
                    cur_questions, 
                    verbose=False, 
                    **answers):
    submit_url = form_url.replace('/viewform', '/formResponse')
    form_data = {'draftResponse': [],
                 'pageHistory': 0}
    for v in cur_questions.values():
        form_data[v] = ''
    for k, v in answers.items():
        if k in cur_questions:
            form_data[cur_questions[k]] = v
        else:
            warnings.warn('Unknown Question: {}'.format(k), RuntimeWarning)
    if verbose:
        print(form_data)
    user_agent = {'Referer': form_url,
                  'User-Agent': "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537\
                  .36 (KHTML, like Gecko) Chrome/28.0.1500.52 Safari/537.36"}
    return requests.post(submit_url, data=form_data, headers=user_agent)
  
def save_experiment_json(save_root, # type: Path
                    experiment_name, # type: str
                    metrics, # type: Dict[str, float]
                    model_json, # type: str
                    **kw_args
                   ):
  """
  Save the data to an existing json file in google drive
  """
  ts = time.gmtime()[:6]
  exp_dir = save_root / 'DeepLearning' / 'ExperimentResults'
  os.makedirs(exp_dir, exist_ok=True)
  exp_file = exp_dir / 'exp_log.json'
  if not exp_file.exists():
    with exp_file.open('w') as f:
      json.dump([], f)
  with exp_file.open('r') as f:
    try:
      old_experiments = json.load(f)
    except Exception as e:
      old_experiments = []
      print(e, 'cannot open file')
    
  with exp_file.open('w') as f:
    old_experiments.append({'name': experiment_name, 
                            'time': str(datetime.datetime(*ts)),
                            'timestamp': ts,
                            'model_json': model_json,
                            'extra_args': kw_args,
                            'metrics': metrics,
                            'parameters': fetch_json_globals()
                           })
    json.dump(old_experiments, f, indent='\t')
    

def save_experiment_gforms(gform_url, # type: str
                    experiment_name, # type: str
                           user_name, # type: str
                    metrics, # type: Dict[str, float]
                    model_dict, # type: str
                    model_name, # type: str       
                    **kw_args
                   ):
  q_dict = get_questions(gform_url)
  answers = {}
  for k, v in metrics.items():
     answers[k] = v
  bonus_args = fetch_json_globals()
  answers['experiment_name'] = experiment_name
  answers['parameters_json'] = json.dumps(bonus_args)
  answers['metrics_json'] = json.dumps(metrics)
  answers['extra_args_json'] = json.dumps(kw_args)
  answers['user_name'] = user_name
  answers['model_name'] = model_name
  answers['model_json'] = json.dumps(model_dict)
  submit_response(gform_url, q_dict, **answers)                


# In[ ]:


form_url = "https://docs.google.com/forms/d/e/1FAIpQLSf-9BoBCrCEX-aQ7k2p9VJsN8vOSWkGFsKWE7F3D3Mhe79Pmw/viewform?usp=sf_link"
q_dict = get_questions(form_url)
q_dict


# In[ ]:


metric_dict = dict(zip(basic_gait.metrics_names, 
                       basic_gait.evaluate(valid_X, valid_y, verbose=False)))
save_experiment_gforms(
    gform_url=form_url,
    experiment_name='Foot from Head',
    model_name=basic_gait.name,
    user_name='Kevin',
    metrics=metric_dict,
    model_dict=json.loads(basic_gait.to_json())
)


# In[ ]:


time_suffix = ''.join(['{:02d}'.format(x) for x in time.gmtime()[:6]])
basic_gait.save(str(BASE_DIR / '{}_{}.h5'.format(basic_gait.name, time_suffix)))


# # Apply for a full sequence
# We can take the model and apply it to a full sequence

# In[ ]:


big_X, big_y, feet_init = next(generate_batch({'just_one': all_data[valid_keys[0]]}, 
                                         seq_len=None, 
                                         debug=True, 
                                         include_offset=True
                                        ))


# In[ ]:


fancy_gait = lstm_gait_mod(3, base_depth=32, seq_len=big_X.shape[0])
fancy_gait.load_weights(trained_model_path)


# In[ ]:


X, y = big_X, np.stack(big_y, 0)
y_pred = fancy_gait.predict(np.expand_dims(big_X, 0))[0]


# In[ ]:


fig = plt.figure(figsize=(20, 8))
body_height_estimate = 50
body_stance_estimate = 20
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
show_head_vec = X.copy()
show_head_vec[:, 1]+=body_height_estimate # above the fray

fl_pos = np.cumsum(y[0], 0)
fr_pos = np.cumsum(y[1], 0)

if FEET_RELATIVE_TO_HEAD:
  fl_pos+=X
  fl_pos+=X

fr_pos[:,2]+=body_stance_estimate

ax1.plot(show_head_vec[:, 0], 
         show_head_vec[:, 2], 
         show_head_vec[:, 1], 
         'k-', ms=10, label='Head')

ax1.plot(fl_pos[:, 0], 
         fl_pos[:, 2], 
         fl_pos[:, 1], 
         'g-', ms=5, label='Left Foot')


ax1.plot(fr_pos[:, 0], 
         fr_pos[:, 2], 
         fr_pos[:, 1], 
         'r-', ms=5, label='Right Foot')

ax1.set_title('Recorded')
ax1.legend()


fl_pred = X+np.cumsum(y_pred[0], 0)
fr_pred = X+np.cumsum(y_pred[1], 0)

if FEET_RELATIVE_TO_HEAD:
  fl_pred+=X
  fr_pred+=X
  
fr_pred[:,2]+=body_stance_estimate


ax2.plot(show_head_vec[:, 0], 
         show_head_vec[:, 2], 
         show_head_vec[:, 1], 
         'k-', ms=5, label='Head')


ax2.plot(fl_pred[:, 0], 
         fl_pred[:, 2], 
         fl_pred[:, 1], 
         'g-', ms=5, label='Left Foot')


ax2.plot(fr_pred[:, 0], 
         fr_pred[:, 2], 
         fr_pred[:, 1], 
         'r-', ms=5, label='Right Foot')

ax2.set_title('Predicted')
ax2.legend()

ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())
ax1.view_init(30, 15)
ax2.view_init(30, 15)


# In[ ]:


def _show_frame(i, ms=10, add_body=True):
  c_head = show_head_vec[i:i+1]
  for c_ax, [c_fl, c_fr] in zip([ax1, ax2], 
                  [(fl_pos[i:i+1], fr_pos[i:i+1]), 
                   (fl_pred[i:i+1], fr_pred[i:i+1])]):
    
    [c_line.remove() for c_line in c_ax.get_lines() if c_line.get_label().startswith('_')];
    if add_body:
      c_feet = (c_fr+c_fl)/2
      c_hips = c_feet + (c_head-c_feet)/2
      c_ax.plot(c_hips[:, 0], 
             c_hips[:, 2], 
             c_hips[:, 1], 'ks')
      
      c_ax.plot(
          [c_head[0, 0], c_hips[0, 0]],
          [c_head[0, 2], c_hips[0, 2]],
          [c_head[0, 1], c_hips[0, 1]],
          'm-', lw=10, alpha=0.75 
      )
      
      c_ax.plot(
          [c_hips[0, 0], c_fr[0, 0]],
          [c_hips[0, 2], c_fr[0, 2]],
          [c_hips[0, 1], c_fr[0, 1]],
          'm-', lw=3, alpha=0.75 
      )
      
      c_ax.plot(
          [c_hips[0, 0], c_fl[0, 0]],
          [c_hips[0, 2], c_fl[0, 2]],
          [c_hips[0, 1], c_fl[0, 1]],
          'm-', lw=3, alpha=0.75 
      )
    
    c_ax.plot(c_head[:, 0], 
             c_head[:, 2], 
             c_head[:, 1], 'ko', ms=2*ms)
  
    c_ax.plot(c_fl[:, 0], 
           c_fl[:, 2], 
           c_fl[:, 1], 
           'gs', ms=ms)
    c_ax.plot(c_fr[:, 0], 
           c_fr[:, 2], 
           c_fr[:, 1], 
           'rs', ms=ms)
    
      

out_anim = FuncAnimation(fig, _show_frame, range(0, show_head_vec.shape[0], 3))
out_anim.save('big_walk.mp4', bitrate=8000, fps=8)
FileLink('big_walk.mp4')


# In[ ]:




