#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# The notebook below shows the workflow for training the first Gait model. The goal of the model is to predict steps (when at which foot) as well as foot positions vs time (as these are more fun to visualize). The input to the model is principally a time-series of headpose data (just translational) and depending on the parameters specified in the **Training Model** section various IMU signals. 

# ## Setup

# In[ ]:


import os
from IPython.display import FileLink
gd_path = '../input/'


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR =  Path(gd_path)
RESULTS_DIR = Path('.')

SAMPLE_RATE = 60

from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

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


DATA_DIRS = [BASE_DIR ]
hp_feet_keys = ['Head', 'Hips', 'LeftFoot', 'RightFoot']
all_meta_df = pd.concat([
    pd.read_csv(c_csv).assign(data_dir=c_dir.stem, 
                             hdf5_path=c_csv.with_name('{}.h5'.format(
                                 c_csv.stem.replace('_meta', '_processed')))
                             ) 
    for c_dir in DATA_DIRS for c_csv in c_dir.glob('*_meta.csv')
]).reset_index(drop=True)
print(all_meta_df['hdf5_path'].value_counts())
print(all_meta_df['walking_like'].map(lambda x: 'walking' if x else 'other activity').value_counts())
all_meta_df.sample(3)


# In[ ]:


import h5py
from functools import lru_cache
@lru_cache()
def get_hdf5_keys(path):
  with h5py.File(path, 'r') as hf:
    return list(hf.keys())
  
@lru_cache()
def get_key_length(
    data_path,
    key
):
  """Gets/Caches the length of the sequence so we can make running faster"""
  data_dict = read_key(data_path, key)
  return data_dict['Head'].shape[0]
  
def read_key(data_path, 
             key, 
             shuffle=True,
             verbose=False
            ):
  out_dict = {}
  key_list = get_hdf5_keys(data_path)
  with h5py.File(data_path, 'r') as hf:
    if shuffle:
      np.random.shuffle(key_list)
    for k1 in key_list:
      if key == k1.split('-')[0]:
        if verbose: print('Found Match', k1)
        for k2 in hf[k1].keys():
          out_dict[k2] = hf[k1][k2].value
        if 'frame' not in out_dict:
          out_dict['frame'] = np.arange(out_dict['Head'].shape[0])
        if verbose:
          print('Keys', [(k, v.shape) for k,v in out_dict.items()])
        
        return out_dict
  if verbose:
    print('No Match for {} in {}'.format(key, data_path))
  return out_dict
all_meta_df['is_available'] = all_meta_df.apply(lambda c_row: 
                                                c_row['full_id'] in {x.split('-')[0] 
                                                                     for x in 
                                                                     get_hdf5_keys(c_row['hdf5_path'])
                                                                    }, 1)
print(all_meta_df['is_available'].value_counts())
valid_meta_df = all_meta_df[all_meta_df['is_available']] # keep only valid


# In[ ]:


# make sure the reading function works
sample_df = valid_meta_df.  groupby(['walking_like', 'hdf5_path']).  apply(lambda x: x.sample(2)).  reset_index(drop=True).  sort_values('walking_like', ascending=False)
sample_df['vec_data'] = sample_df.  apply(lambda c_row: read_key(c_row['hdf5_path'], 
                               c_row['full_id'], 
                               verbose=True), 1);


# ### Scale
# Try and determine the scale factor for the VICON markers by looking at the range for all of the relevant walks
# 
# ` a volume measuring roughly 4x6m`

# In[ ]:


def center_vec(x):
  m_vec = np.mean(x, 
                  axis=(0,), 
                  keepdims=True)
  return x-np.tile(m_vec, (x.shape[0], 1))

all_pts_df = pd.DataFrame(np.concatenate([
    center_vec(c_dict[k])
                for c_dict in sample_df['vec_data'].values 
                for k in hp_feet_keys], 0),
             columns = ['x', 'y', 'z'])
all_pts_df.describe()


# In[ ]:


fig, ax1 = plt.subplots(1, 1)
all_pts_df.plot.scatter(x='x', y='z', ax=ax1)
#ax1.axis('equal')
#ax1.set_xlim(-200, 200)
#ax1.set_ylim(-300, 300)


# ## Preview Plots
# Make sure the data is loaded correctly and that we can visualize well

# In[ ]:


x_grid, y_grid = 2, 4
fig = plt.figure(figsize=(30, 30))
for i, (_, test_row) in enumerate(sample_df.iterrows(), 1):
  test_data = test_row['vec_data']
  ax1 = fig.add_subplot(100*x_grid+10*y_grid+i, projection='3d')
  for c_col in test_data.keys():
    if c_col in hp_feet_keys:
      xyz_vec = test_data[c_col]
      ax1.plot(xyz_vec[:, 0], 
              xyz_vec[:, 2],
              xyz_vec[:, 1], '-', label=c_col)
  ax1.legend()
  ax1.set_title('{activity_description}\n{walking_like}'.format(**test_row))
  if i==x_grid*y_grid:
    break


# # Add Step Track
# Here append a step track to the dataset (precomputing means we have all of the context)
# - `right_foot_down` 1 or 0 indicating if the right foot is on the ground
# - `step_event` 0 to 1 indicating the probability of a step

# In[ ]:


from scipy.signal import gaussian as gkern
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

@autotest
def process_foot_pos(left_foot, 
                     right_foot, 
                     sample_rate=SAMPLE_RATE,
                     kern_width=0.05
                    ):
  """Find steps using left and right foot vicon data. 
  
  :param left_foot: y position of left foot
  :param right_foot: y position of right foot
  >>> right_foot = [0, 0, 1, 1, 0, 0]
  >>> left_foot = [1, 1, 0, 0, 1, 1]
  >>> rfd, se = process_foot_pos(left_foot, right_foot, sample_rate=5)
  >>> rfd.astype(int)
  array([1, 1, 0, 0, 1, 1])
  >>> (se>0.5).astype(int)
  array([0, 0, 1, 0, 1, 0])
  """
  left_foot = np.array(left_foot)
  right_foot = np.array(right_foot)
  
  foot_right_down = ((left_foot-right_foot)>0).astype('float32')
  step_event = np.abs(diffpad(foot_right_down))
  conv_kern = gkern(np.clip(int(sample_rate*kern_width*4), 1, len(left_foot)), 
                            std=sample_rate*kern_width)
  step_event = np.convolve(step_event, conv_kern, mode='same').clip(0, 1)
  if len(step_event)>len(left_foot):
    x_d = len(step_event)-len(left_foot)
    l_pad = x_d//2
    r_pad = x_d-l_pad
    step_event = step_event[l_pad:-r_pad]
  return foot_right_down, step_event


# In[ ]:


@autotest
def create_foot_coord(left_foot, 
                     right_foot,
                      foot_var = None
                    ):
  """Create the foot coordinate system.
  
  :param left_foot: y position of left foot
  :param right_foot: y position of right foot
  :param foot_var: normalized variance for the foot
  >>> right_foot = np.array([0, 0, 1, 1, 0, 0]).reshape((-1, 1))
  >>> left_foot = np.array([1, 1, 0, 0, 1, 1]).reshape((-1, 1))
  >>> rfd = create_foot_coord(left_foot, right_foot)
  >>> rfd # doctest: +NORMALIZE_WHITESPACE
  array([[ 0.5, -0.5],
           [ 0.5, -0.5],
           [-0.5,  0.5],
           [-0.5,  0.5],
           [ 0.5, -0.5],
           [ 0.5, -0.5]])
  """
  mid_foot = left_foot*0.5+right_foot*0.5
  nfeet_pos = np.concatenate([left_foot-mid_foot, right_foot-mid_foot], 1)
  if foot_var is None:
    foot_var = np.std(nfeet_pos)
  nfeet_pos /= 2*foot_var
  
  return nfeet_pos


# In[ ]:


from scipy.signal import argrelextrema
@autotest
def get_local_maxi(s_vec, 
                   jitter_amount=1e-5, 
                   min_width=5, 
                   cutoff=None # type: Optional[float]
                  ): 
  # type: (...) -> List
  """Get the local maximums.
  
  The standard functions struggle with flat peaks
  
  >>> np.random.seed(2019)
  >>> get_local_maxi([0, 1, 1, 0])
  array([2])
  >>> get_local_maxi([1, 1, 1, 0])
  array([0])
  >>> get_local_maxi([1, 1, 1, 0, 1])
  array([0])
  >>> get_local_maxi([1, 0, 0, 0, 0, 1], min_width=1)
  array([0, 2, 5])
  >>> get_local_maxi([1, 0, 0, 0, 0, 1], min_width=1, cutoff=0.5)
  array([0, 5])
  """
  s_range = np.max(s_vec)-np.min(s_vec)
  if s_range==0:
    return []
  
  j_vec = np.array(s_vec)+    jitter_amount*s_range*np.random.uniform(-1, 1, size=np.shape(s_vec))
  max_idx = argrelextrema(j_vec, np.greater_equal, order=min_width)[0]
  if cutoff is not None:
    max_idx = np.array([k for k in max_idx if s_vec[k]>cutoff])
  return max_idx

@autotest
def calc_diff_step_vec(
    foot_pos_vec, # type: np.ndarray
    in_steps # type: List[int]
):
  # type: (...) -> np.ndarray
  """Calculates the step direction vector for each timepoint.
  
  >>> step_vec = np.array([[0,0], [3, 4], [0, 0], [4, 3]])
  >>> calc_diff_step_vec(step_vec, [0, 1])
  array([[0.6, 0.8],
         [0.6, 0.8],
         [0.6, 0.8],
         [0.6, 0.8]], dtype=float32)
  >>> calc_diff_step_vec(step_vec, [0, 1, 2])
  array([[ 0.6,  0.8],
         [-0.6, -0.8],
         [-0.6, -0.8],
         [-0.6, -0.8]], dtype=float32)
  >>> calc_diff_step_vec(step_vec, [1])
  array([[0.6, 0.8],
         [0.6, 0.8],
         [0.6, 0.8],
         [0.6, 0.8]], dtype=float32)
  >>> calc_diff_step_vec(step_vec, [])
  array([[0.8, 0.6],
         [0.8, 0.6],
         [0.8, 0.6],
         [0.8, 0.6]], dtype=float32)
  """
  step_vec = np.zeros_like(foot_pos_vec).astype('float32')
  if len(in_steps)<1:
    in_steps = [0, -1]
  if len(in_steps)<2:
    in_steps = [0]+in_steps
  d_step_pos = np.diff(foot_pos_vec[in_steps, :], n=1, axis=0)
  for i, (k, dvec) in enumerate(zip(in_steps, d_step_pos)):
    start_idx = 0 if i==0 else k
    # overwrite the values at an above k
    c_norm = np.linalg.norm(dvec)
    if c_norm>0:
      step_vec[start_idx:, :] = dvec/c_norm
  return step_vec


# In[ ]:


@autotest
def foot_movement_vector(
    left_foot_vec, 
    right_foot_vec,
    filter_width=5,
    verbose=False
):
  # type: (...) -> Dict[str, np.ndarray]
  """Calculate distance between feet at every timepoint.
  
  Computes the distance only along the direction of motion
  
  :param left_foot: y position of left foot
  :param right_foot: y position of right foot
  :param foot_var: normalized variance for the foot
  >>> np.random.seed(2019)
  >>> t_v = np.linspace(0, 2, 120)
  >>> foot_f = lambda offset=0, k=2: np.sin(k*2*np.pi*(t_v+offset))
  >>> right_foot = np.zeros((t_v.shape[0], 3), dtype='float32')
  >>> right_foot[:, 0] = t_v
  >>> left_foot = right_foot.copy()
  >>> right_foot[:, 1] = foot_f()
  >>> left_foot[:, 1] = foot_f(-0.25)
  >>> fv_vec = foot_movement_vector(left_foot, right_foot, filter_width=1, verbose=True) 
  lr_steps [31, 61, 91] [15, 46, 76, 106]
  >>> fv_vec['left_foot_proj'][0] # doctest: +NORMALIZE_WHITESPACE
  array([ 0.99500525, -0.09982233,  0.        ], dtype=float32)
  >>> fv_vec['right_foot_proj'][0] 
  array([ 0.8948159 , -0.44643548,  0.        ], dtype=float32)
  """
  right_foot_down, step_event = process_foot_pos(
      left_foot=left_foot_vec[:, 1],
      right_foot=right_foot_vec[:, 1]
  )
  step_list = get_local_maxi(step_event, cutoff=0.5)
  right_steps = []
  left_steps = []
  for k in step_list:
    lf_height = np.mean(left_foot_vec[k:k+filter_width, 1])
    rf_height = np.mean(right_foot_vec[k:k+filter_width, 1])
    if lf_height<=rf_height:
      # left step
      left_steps.append(k)
    else:
      # right step
      right_steps.append(k)
  
  dlf_vec = calc_diff_step_vec(left_foot_vec, left_steps)
  drf_vec = calc_diff_step_vec(right_foot_vec, right_steps)
  if verbose:
    print('lr_steps', left_steps, right_steps)
  
  return {'left_foot_proj': dlf_vec,
          'right_foot_proj': drf_vec
         }


# In[ ]:


@autotest
def foot_to_foot_distance(
    left_foot, 
    right_foot
):
  # type: (...) -> np.ndarray
  """Calculate distance between feet at every timepoint.
  
  :param left_foot: y position of left foot
  :param right_foot: y position of right foot
  :param foot_var: normalized variance for the foot
  >>> right_foot = np.array([0, 0, 1, 1, 2, 2]).reshape((-1, 1))
  >>> left_foot = np.array( [0, 1, 1, 2, 2, 3]).reshape((-1, 1))
  >>> foot_to_foot_distance(left_foot, right_foot) # doctest: +NORMALIZE_WHITESPACE
  array([0., 1., 0., 1., 0., 1.])
  >>> foot_to_foot_distance(left_foot, left_foot) # doctest: +NORMALIZE_WHITESPACE
  array([0., 0., 0., 0., 0., 0.])
  """
  return np.sqrt(np.sum(np.square(left_foot-right_foot), 1))


@autotest
def foot_to_foot_seperation(
    left_foot, 
    right_foot
):
  # type: (...) -> np.ndarray
  """Calculate distance between feet at every timepoint.
  
  Computes the distance only along the direction of motion
  
  :param left_foot: y position of left foot
  :param right_foot: y position of right foot
  :param foot_var: normalized variance for the foot
  >>> np.random.seed(2019)
  >>> base_foot = 100*np.tile(np.arange(10).reshape((-1, 1)),(1, 3))
  >>> right_foot = base_foot+np.random.uniform(-10, 10, size=base_foot.shape)
  >>> left_foot = base_foot+np.random.uniform(-10, 10, size=base_foot.shape)
  >>> raw_dist = foot_to_foot_distance(left_foot, right_foot) 
  >>> raw_dist.astype(int) # doctest: +NORMALIZE_WHITESPACE
  array([11, 11, 14,  4,  8, 20,  9, 11, 17, 15])
  >>> sep_dist = foot_to_foot_seperation(left_foot, right_foot)
  >>> sep_dist.astype(int) # doctest: +NORMALIZE_WHITESPACE
  array([ 7,  8, -7, -4,  6, -4,  0, -8,  4,  9])
  """
  mid_foot = left_foot*0.5+right_foot*0.5
  diff_vec = mid_foot[-1]-mid_foot[0]
  diff_vec /= np.linalg.norm(diff_vec)
  foot_diff = left_foot-right_foot  
  proj_len = np.dot(foot_diff, diff_vec)
  return (proj_len)

@autotest
def foot_pos_axis(left_foot, right_foot):
  """Return the foot position along the axis"""
  fv_vec = foot_movement_vector(left_foot, right_foot) 
  left_proj = (left_foot-right_foot)*fv_vec['left_foot_proj']
  left_xz_dot = left_proj[:, 0]+left_proj[:, 2]
  
  right_proj = (right_foot-left_foot)*fv_vec['right_foot_proj']
  right_xz_dot = right_proj[:, 0]+right_proj[:, 2]
  return left_xz_dot, right_xz_dot


# In[ ]:


fig, m_axs = plt.subplots(3, 4, figsize=(25, 10))
for (ax1, ax2, ax3), (_, test_row) in zip(m_axs.T, 
                            sample_df.iterrows()):
  sample_name = test_row['activity_description']
  max_len = test_row['vec_data']['LeftFoot'].shape[0]
  test_window = slice(0, 500 if max_len>500 else max_len)
  test_data = {k: v[test_window] for k,v in test_row['vec_data'].items()}
  
  ax1.plot(test_data['LeftFoot'][:, 1], label='Left')
  ax1.plot(test_data['RightFoot'][:, 1], label='Right')
  ax1.set_ylabel('Foot Position')
  ax1.set_title(sample_name)
  ax1.legend()
  
  right_foot_down, step_event = process_foot_pos(test_data['LeftFoot'][:, 1], 
                                                test_data['RightFoot'][:, 1])
  
  
  ax2.plot(right_foot_down, 'r-',
           label='Right Foot Down')
  
  ax2.plot(step_event, 'k-',
           label='Step Event')
  ax2.legend()
  lf_vec = test_data['LeftFoot']
  rf_vec = test_data['RightFoot']
  ax3.plot(lf_vec[:, 0], lf_vec[:, 2], 'g-', label='Left')
  ax3.plot(rf_vec[:, 0], rf_vec[:, 2], 'b-', label='Right')
  step_list = get_local_maxi(step_event, cutoff=0.5)
  right_steps = []
  left_steps = []
  for k in step_list:
    if np.mean(lf_vec[k:k+5, 1])<=np.mean(rf_vec[k:k+5, 1]):
      # left step
      ax1.axvline(k, c='r')
      ax3.plot(lf_vec[k, 0], lf_vec[k, 2], 'rs')
      left_steps.append(k)
    else:
      # right step
      ax1.axvline(k, c='k')
      ax3.plot(rf_vec[k, 0], rf_vec[k, 2], 'ks')
      right_steps.append(k)
  dlf_vec = calc_diff_step_vec(lf_vec, left_steps)
  drf_vec = calc_diff_step_vec(rf_vec, right_steps)
  qv_args = dict(angles='xy', color='m', alpha=0.5)
  ax3.quiver(lf_vec[::3, 0], lf_vec[::3, 2], 
             dlf_vec[::3, 0], dlf_vec[::3, 2], **qv_args)
  ax3.quiver(rf_vec[::3, 0], rf_vec[::3, 2], 
             drf_vec[::3, 0], drf_vec[::3, 2], **qv_args)
  ax3.legend()
  


# In[ ]:


fig, m_axs = plt.subplots(3, 4, figsize=(15, 5))
for (ax1, ax2, ax3), (_, test_row) in zip(m_axs.T, 
                            sample_df.iterrows()):
  sample_name = test_row['activity_description']
  test_data = test_row['vec_data']
  max_len = test_row['vec_data']['LeftFoot'].shape[0]
  test_window = slice(0, 500 if max_len>500 else max_len)
  right_foot_down, step_event = process_foot_pos(test_data['LeftFoot'][test_window, 1], 
                                                test_data['RightFoot'][test_window, 1])
  
  foot_coord = create_foot_coord(test_data['LeftFoot'][test_window, :], 
                                                test_data['RightFoot'][test_window, :])
  
  ax1.plot(test_data['LeftFoot'][test_window, 1], label='Left')
  ax1.plot(test_data['RightFoot'][test_window, 1], label='Right')
  ax1.set_ylabel('Foot Position')
  ax1.set_title(sample_name)
  ax1.legend()
  ax2.plot(right_foot_down, 'r-',
           label='Right Foot Down')
  
  ax2.plot(step_event, 'k-',
           label='Step Event')
  ax2.legend()
  
  lf_vec = test_data['LeftFoot'][test_window, :]
  rf_vec = test_data['RightFoot'][test_window, :]
  ax3.plot(foot_to_foot_distance(lf_vec, rf_vec), 'r-', label='Raw Distance')
  ax3.plot(foot_to_foot_seperation(lf_vec, rf_vec), 'k-', 
           label='Distance in DOM', alpha=0.5)
  ax3.plot(foot_pos_axis(lf_vec, rf_vec)[0], '-', label='Left Pos', alpha=0.5)
  ax3.plot(foot_pos_axis(lf_vec, rf_vec)[1], '-', label='Right Pos', alpha=0.5)
  ax3.legend()


# ### Modulate Head Bobbing
# Here we want to change the intensity of the head bobbing

# In[ ]:


from scipy.signal import detrend
@autotest
def scale_bobble(
    in_vec, # type: np.ndarray
    bobble_scale, # type: float
    axis=0
):
  # type: (...) -> np.ndarray
  """Scale the amount of bobble in headpose.
  
  :param bobble_scale: scalar for how much to multiply the amplitude
  >>> scale_bobble([0, 1, 0, 1, 0], 2)
  array([-0.4,  1.6, -0.4,  1.6, -0.4])
  """
  amp_vec = detrend(in_vec, axis=axis)
  trend_vec = in_vec-amp_vec
  return amp_vec*bobble_scale+trend_vec


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize=(15, 5))
for n_axs, (_, test_row) in zip(m_axs.T, 
                            sample_df.iterrows()):
  sample_name = test_row['activity_description']
  test_data = test_row['vec_data']
  test_window = slice(0, 100)
  in_vec = test_data['Head'][test_window, :]
  
  for i, (c_x, c_ax) in enumerate(zip('xyz', n_axs)):
    c_ax.plot(in_vec[:, i], label='Input Signal')
    new_vec = scale_bobble(in_vec, 2.5)
    c_ax.plot(new_vec[:, i], label='Add Bobble')
    c_ax.legend()


# In[ ]:


fig, m_axs = plt.subplots(3, 3, figsize=(30, 10))
for ax1, (_, test_row) in zip(m_axs.flatten(), 
                            sample_df.iterrows()):
  test_window = slice(0, 200)
  sample_name = test_row['activity_description']
  test_data = test_row['vec_data']
  right_foot_down, step_event = process_foot_pos(test_data['LeftFoot'][test_window, 1], 
                                                test_data['RightFoot'][test_window, 1])
  
  foot_coord = create_foot_coord(test_data['LeftFoot'][test_window, :], 
                                 test_data['RightFoot'][test_window, :])
  ax1.plot(foot_coord[:, :3], label='Left')
  ax1.plot(foot_coord[:, 3:], label='Right')
  ax1.plot(step_event, 'k-', label='Step')
  ax1.set_title(sample_name)
  ax1.legend()


# In[ ]:


# add to all tracks
def add_step_data(in_vec_dict):
  # type: (Dict[str, np.ndarray]) -> Dict[str, np.ndarray]
  """Add the step channels to the data."""
  
  right_foot_down, step_event = process_foot_pos(
      left_foot=in_vec_dict['LeftFoot'][:, 1],
      right_foot=in_vec_dict['RightFoot'][:, 1])
  in_vec_dict['foot_right_down'] = right_foot_down
  in_vec_dict['step_event'] = step_event
  
  foot_coord = create_foot_coord(left_foot=in_vec_dict['LeftFoot'], 
                                 right_foot=in_vec_dict['RightFoot'])
  left_coord, right_coord = foot_pos_axis(left_foot=in_vec_dict['LeftFoot'], 
                right_foot=in_vec_dict['RightFoot'])
  in_vec_dict['NormFeet'] = left_coord
  return in_vec_dict


# # Training Model

# ## Training Idea
# - Given head ($\vec{H}(t_0)$) and feet ($\vec{F}_{\textrm{left}}(t_0)$ and $\vec{F}_{\textrm{right}}(t_0)$) positions at $t_0$
# - We redefine
# - We define $\vec{H}_{\Delta}(s) = \vec{H}(t_0+s)-\vec{H}(t_0)$
# - We try and predict $\vec{F}_{(L,R)\Delta}(s) = \vec{F}_{(L, R)}(t_0+s)-\vec{F}_{(L, R)}(t_0)$
# - Take 2 seconds of head data ($\vec{H}-\vec{H}_0$) and try 

# ## Model Parameters
# 
# All of the global all capital variable names are wrapped up and included in the results dashboard (as long as they are easily json-serialiazable, no numpy arrays!)

# In[ ]:


# preprocessing
FEET_RELATIVE_TO_HEAD = True
FEET_RELATIVE_COORDS = False
USE_NON_WALKING = False
USE_MOCAP = False
USE_TOTALCAP = True
SMOOTH_FEET_TRACKS = False

# training
TRAIN_STEPS = 5000
VALID_STEPS = 4096
TRAIN_SEQ_LEN = 256

EPOCHS = 20
BATCH_SIZE = 128
STEP_LOSS_WEIGHT = 2.0
FEET_SEP_WEIGHT = 0.01
LEARNING_RATE = 1e-3

# augmentation
RANDOM_SEED = 2019
GAUSS_NOISE = 5e-3
DRIFT_NOISE = 2e-3
HEADPOSE_BOBBING = 2
AUGMENT_WALKS = True
AUGMENT_FREQUENCY = 2 # between 1/2 and 2x the normal speed
LOOP_CLOSURE = 0 # rate of loop closer (poisson probability per frame)
MISSING_HEADPOSE = False

# model settings
DROPOUT_SPATIAL = False
DROPOUT_EMPTY = True # no dropout at all
DIFF_HEADPOSE = True
DROPOUT_RATE = 0.25
STACK_UNITS = 3
BASE_DEPTH = 32 #32
DILATION_COUNT = 6 #6
AVG_POOL_SIZE = 3

USE_LSTM = False
USE_HOURGLASS = False
MERGE_FEET_FEATURES = False
CROP_DIST = 32
IMU_VARS = [] # ['Head_imu_acc', 'Head_imu_gyr'] # 'Hips_imu_acc', 'Hips_imu_gyr', 
SEPERATE_MODELS = False
PREDICT_FEET_SEP = True
USE_GPU = True

# results
EXPERIMENT_NAME = 'CombinedDataFeet'
results_form_url = "https://docs.google.com/forms/d/e/1FAIpQLSeVaGDwWuXAcSNLgKZhvzHPiRWIQkI71TNVszYk9n653uHUOg/viewform?usp=sf_link"
SAVE_VIDEO = False
SAVE_OTHER = False


# In[ ]:


if not SAVE_OTHER:
  FileLink = lambda *args, **kwargs: None
else:
  from google.colab.files import download as FileLink


# ## Batch Generators

# In[ ]:


def generate_batch(data_df, # type: pd.DataFrame
                   seq_len=TRAIN_SEQ_LEN, # type: Optional[int]
                   debug=True,
                   include_offset=False
                  ):
  """Creates batches by loading and preprocessing data.
  
  The basic generator for taking random small chunks from all the data
  It produces the X (head pose) and (y1, y2) foot pose (left and right)
  
  arguments:
    data_dict: a dictionary with all of the data in it organized by experiment, sensor, array
    seq_len: an optional number of how many samples (60hz) to cut the data (None means use the whole data)
    debug: shows a plot
    include_offset: return initial foot positions
  
  """
  data_rows = [c_row for _, c_row in data_df.iterrows()]
  while True:
    pt_count = -1
    cmp_seq_len = 0 if seq_len is None else seq_len
    while pt_count<=cmp_seq_len:
      # some datasets are too short
      c_idx = np.random.choice(range(len(data_rows)))
      c_row = data_rows[c_idx]
      # we can cache get_key_length better
      pt_count = get_key_length(c_row['hdf5_path'], c_row['full_id'])
      
    data_dict = read_key(c_row['hdf5_path'], c_row['full_id'], verbose=False)
    data_dict = add_step_data(data_dict)
    
    if seq_len is None:
      pt_start = 0
      pt_end = pt_count
    else:
      pt_start = np.random.choice(range(pt_count-seq_len))
      pt_end = pt_start+seq_len
    test_data = {k: v[pt_start:pt_end] for k,v in data_dict.items()} 
    head_xyz = test_data['Head']
    fl_xyz = test_data['LeftFoot']
    fr_xyz = test_data['RightFoot']
    
    f2f_sep = foot_to_foot_seperation(fl_xyz, fr_xyz)
    
    foot_right_down = test_data['foot_right_down']
    step_event = test_data['step_event']

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
    
    fl_s = np.sqrt(np.sum(np.power(diffpad(test_data['LeftFoot']), 2), 1))
    fr_s = np.sqrt(np.sum(np.power(diffpad(test_data['RightFoot']), 2), 1))
    
    head_xyz_zero = head_xyz-np.tile(head_xyz[:1, :], (head_xyz.shape[0], 1))
    
    
    if FEET_RELATIVE_COORDS:
      feet_norm = test_data['NormFeet']
      fl_xyz = feet_norm[:, :3]
      fr_xyz = feet_norm[:, 3:]
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
    x_vars = {'headpose': head_xyz_zero,
              'frame': test_data['frame']}
    for c_imu_var in IMU_VARS:
      x_vars[c_imu_var] = test_data[c_imu_var][:, :3]
    yield x_vars, {'foot_left': fl_xyz, 
                   'feet_dist': np.expand_dims(f2f_sep, -1),
                  'foot_right': fr_xyz,
                  'foot_left_vel': np.expand_dims(fl_s, -1),
                  'foot_right_vel': np.expand_dims(fr_s, -1),
                  'foot_right_down': np.expand_dims(foot_right_down, -1),
                  'step_event': np.expand_dims(step_event, -1),
                  'foot_left_offset': fl_xyz_body[0],
                  'foot_right_offset': fr_xyz_body[0]
             }


# In[ ]:


walk_gen = generate_batch(valid_meta_df[valid_meta_df['walking_like']],
                         seq_len=TRAIN_SEQ_LEN, debug=False)


# In[ ]:


X, y = next(walk_gen)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
for c_alpha, v_pts in zip([1.0, 0.25], [y['foot_right_down'][:, 0]>0, 
                                        y['foot_right_down'][:, 0]<1]):
  for c_color, (i, c_x) in zip(cycle(colors), enumerate('xyz')):
    ax1.plot(X['frame'][v_pts], X['headpose'][v_pts, i], '.', 
             
             color=c_color, alpha=c_alpha, label='{}'.format(c_x))
    ax2.plot(X['frame'][v_pts], y['foot_left'][v_pts, i], '.', 
             color=c_color, alpha=c_alpha, label='Left_{}'.format(c_x))
    ax2.plot(X['frame'][v_pts], y['foot_right'][v_pts, i], '+', 
             color=c_color, alpha=c_alpha, label='Right_{}'.format(c_x))
    ax2.set_title('Raw Foot')
    ax3.plot(X['frame'][v_pts], np.cumsum(y['foot_left'], axis=0)[v_pts, i], '.', 
             color=c_color, alpha=c_alpha, label='Left_{}'.format(c_x))
    ax3.plot(X['frame'][v_pts], np.cumsum(y['foot_right'], axis=0)[v_pts, i], '+', 
             color=c_color, alpha=c_alpha, label='Right_{}'.format(c_x))
    ax3.set_title('Integrated Foot')
  if c_alpha==0.25:
    ax1.legend()
    ax2.legend()
    ax3.legend()
for k in np.where(y['step_event']>0.95)[0]:
    ax1.axvline(X['frame'][k], color='k')


# In[ ]:


fig, (axm1) = plt.subplots(1, 1, figsize=(10, 5))
if FEET_RELATIVE_COORDS:
  axm1.plot(y['foot_right'][:,1])
  axm1.plot(y['foot_left'][:,1])
else:
  axm1.plot(np.cumsum(y['foot_right'][:, 1])+X['headpose'][:, 1])
  axm1.plot(np.cumsum(y['foot_left'][:, 1])+X['headpose'][:, 1])
axm1.plot(y['foot_right_down'])
axm1.plot(y['step_event'])


# In[ ]:


def batch_it(in_gen, batch_size=64):
  out_vals = []
  for c_vals in in_gen:
    out_vals += [c_vals]
    if len(out_vals)==batch_size:
      yield tuple([{k: np.stack([c_row[i][k] for c_row in out_vals], 0) 
                   for k in c_vals[i].keys()}
                   for i in range(len(c_vals))])
      out_vals = []
bat_watch_gen = batch_it(walk_gen)
tX, ty = next(bat_watch_gen)
print('x')
for c_key in tX.keys():
  print(c_key, tX[c_key].shape)
print('y')
for c_key in ty.keys():
  print(c_key, ty[c_key].shape)


# In[ ]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
ax1.hist(tX['headpose'][:, 1].ravel())
ax1.set_title('Head Pose Height')
foot_bins = np.linspace(-2, 2, 20)
ax2.hist(ty['foot_left'].ravel().clip(foot_bins.min(), foot_bins.max()), 
         foot_bins, label='Left', alpha=0.75, stacked=True)

ax2.hist(ty['foot_right'].ravel().clip(foot_bins.min(), foot_bins.max()), 
         foot_bins, label='Right', alpha=0.5, stacked=True)
ax2.legend()

ax3.hist(ty['foot_right_down'].ravel());
ax3.set_title('Right Down')

ax4.hist(ty['feet_dist'].ravel())
ax4.set_title('Foot Seperation')


# ## Build Model

# In[ ]:


from keras import models, layers
from keras import backend as K
if USE_GPU:
  from keras.layers import CuDNNLSTM as LSTM
else:
  from keras.layers import LSTM
if DROPOUT_SPATIAL:
  dropout_layer = layers.SpatialDropout1D
else:
  dropout_layer = layers.Dropout
if DROPOUT_EMPTY:
  # dropout is nothing
  dropout_layer = lambda *args, **kwargs: lambda x: x
@autotest
def diff_filter_layer(length=2, depth=1):
  """Calculates difference of input in 1D.
  
  >>> c_layer = diff_filter_layer()
  >>> t_in = layers.Input((5, 1))
  >>> t_out = c_layer(t_in)
  >>> t_model = models.Model(inputs=[t_in], outputs=[t_out])
  >>> t_model.predict(np.ones((1, 5, 1)))[0, :, 0]
  array([0., 0., 0., 0., 0.], dtype=float32)
  >>> t_model.predict(np.arange(5).reshape((1, 5, 1)))[0, :, 0]
  array([0., 1., 1., 1., 1.], dtype=float32)
  >>> c_layer_3d = diff_filter_layer(depth=3)
  >>> t_in_3d = layers.Input((4, 3))
  >>> t_out_3d = c_layer_3d(t_in_3d)
  >>> t_model_3d = models.Model(inputs=[t_in_3d], outputs=[t_out_3d])
  >>> t_model_3d.predict(np.ones((1, 4, 3)))[0, :, 0]
  array([0., 0., 0., 0.], dtype=float32)
  >>> fake_in = np.arange(12).reshape((1, 4, 3))
  >>> fake_in[0]
  array([[ 0,  1,  2],
         [ 3,  4,  5],
         [ 6,  7,  8],
         [ 9, 10, 11]])
  >>> t_model_3d.predict(fake_in)[0]
  array([[0., 0., 0.],
         [3., 3., 3.],
         [3., 3., 3.],
         [3., 3., 3.]], dtype=float32)
  """
  coef = np.zeros((length, depth, depth))
  i=length//2-1 # offset in middle
  for j in range(depth):
    coef[i,j,j] = -1
    coef[i+1,j,j] = 1
  c_layer = layers.Conv1D(depth, 
                          (coef.shape[0],), 
                          weights=[coef],
                          use_bias=False,
                          activation='linear',
                          padding='valid',
                          name='diff'
               )
  c_layer.trainable = False
  def _diff_module(x):
    diff_x = c_layer(x)
    needed_padding = length-1
    right_pad = np.clip(needed_padding//2-1, 0, 1e99).astype(int)
    left_pad = needed_padding-right_pad
    return layers.ZeroPadding1D((left_pad, right_pad), name='PaddingDiffEdges')(diff_x)
  return _diff_module
  
def _feet_sep_module(in_feat, crop_dist):
  c_feat = layers.Conv1D(128, (1,), 
                  padding='same', 
                  activation='relu', 
                  name='feet_sep_features')(in_feat)
  feet_dist_out = layers.Conv1D(1, (1,), 
                padding='same', 
                activation='tanh', 
                name='feet_sep_output')(c_feat)
  
  if crop_dist>0:
    c_crop = layers.Cropping1D((0, crop_dist))(feet_dist_out)
    c_pad = layers.ZeroPadding1D((0, crop_dist))(c_crop)
  else:
    c_pad = feet_dist_out
  # a fixed scaling of the output (x100, since maximum range is -1.0 to 1.0)
  scale_layer = layers.Conv1D(1, (1,), weights=[100.0*np.ones((1, 1, 1))],
                              use_bias=False, 
                              activation='linear',
                           name='feet_dist'
                          )

  scale_layer.trainable = False
  return scale_layer(c_pad)



@autotest
def stride_length_summarizer(in_stride_vec, smooth_window=60, crop_dist=90):
  """Calculates mean stride lengths from input.
  
  >>> sin_sig = 10*np.sin(np.linspace(0, 2*np.pi, 200)).reshape((1, 200, 1))
  >>> fd_i = layers.Input((None, 1), name='foot_dist_in')
  >>> out_len = stride_length_summarizer(fd_i)
  >>> d_mod = models.Model(inputs=[fd_i], outputs=[out_len])
  >>> d_mod.predict(sin_sig)[0, :]
  array([0.7834222, 9.47776  ], dtype=float32)
  >>> cos_sig = 10*np.cos(np.linspace(0, 2*np.pi, 200)).reshape((1, 200, 1))
  >>> d_mod.predict(cos_sig)[0, :]
  array([4.678287, 5.800243], dtype=float32)
  >>> long_sig = 10*np.cos(np.linspace(0, 16*np.pi, 1000)).reshape((1, 1000, 1))
  >>> d_mod.predict(long_sig)[0, :]
  array([7.977667, 7.821877], dtype=float32)
  """
  flip_layer = layers.Conv1D(1, 
                             kernel_size=1, 
                             use_bias=False, 
                             activation='linear',
                             weights=[-1*np.ones((1, 1, 1))]
                            )

  flip_layer.trainable=False
  stride_flip = flip_layer(in_stride_vec)
  comb = layers.concatenate([in_stride_vec, stride_flip], name='foot_and_inverse')
  max_stride = layers.MaxPool1D(smooth_window, strides=1, padding='same')(comb)
  stride_crop = layers.Cropping1D((crop_dist, 0))(max_stride)
  return layers.GlobalAveragePooling1D(name='Mean_Stride_Length_Estimates')(stride_crop)


# In[ ]:


import tensorflow as tf
from keras import backend as K
@autotest
def step_count_summarizer(step_event_vec, smooth_window=20, threshold=0.5):
  """Calculates step count from step events.
  
  >>> sin_sig = 10*np.sin(np.linspace(0, 2*np.pi, 100)).reshape((1, 100, 1))
  >>> fd_i = layers.Input((None, 1), name='step_event_in')
  >>> out_len = step_count_summarizer(fd_i)
  >>> d_mod = models.Model(inputs=[fd_i], outputs=[out_len])
  >>> d_mod.predict(sin_sig)[0, :]
  array([3], dtype=int32)
  >>> cos_sig = np.cos(np.linspace(0, 8*np.pi, 800)).reshape((1, 800, 1))
  >>> d_mod.predict(cos_sig)[0, :]
  array([16], dtype=int32)
  """
  
  is_step = layers.MaxPool1D(smooth_window, 
                             strides=(smooth_window,), 
                             padding='valid')(step_event_vec)
  
  return layers.Lambda(lambda x: 
                       K.sum(K.cast(x>threshold, 'int32'), 1), 
                       name='Step_Count')(is_step)


# ### Dilated Convolutions

# In[ ]:


def simple_gait_mod(
    in_node_names,
    dil_rates,
    base_depth=16,
    stack_count=2,
    crop_dist=CROP_DIST,
    seq_len=tX['headpose'].shape[1],
    suffix=''
):
  in_nodes = []
  in_prep_nodes = []
  
  for c_node in in_node_names:
    in_mod = layers.Input((seq_len, 3), name=c_node) 
    in_nodes += [in_mod]
    if DIFF_HEADPOSE and (c_node=='headpose'):
      in_mod = diff_filter_layer(depth=3)(in_mod)
    sd_in = dropout_layer(DROPOUT_RATE)(in_mod)
    bn_in = layers.BatchNormalization()(sd_in)
    c1 = layers.Conv1D(base_depth, (3,), 
                       padding='same', 
                       activation='relu')(bn_in)
    c2 = layers.Conv1D(base_depth, (3,), 
                       padding='same', 
                       activation='relu')(c1)
    in_prep_nodes += [c2]
  if len(in_prep_nodes)>1:
    c2 = layers.concatenate(in_prep_nodes, name='MergeSources')
  else:
    c2 = in_prep_nodes[0]
    
  stack_in = c2
  for i in range(stack_count):
    stack_in = dropout_layer(DROPOUT_RATE)(stack_in)
    if i>0:
      if stack_in._keras_shape[-1]!=base_depth*2:
        c_out = [layers.Conv1D(base_depth*2, (1,), 
                       padding='same', 
                       activation='linear')(stack_in)]
      else:
        c_out = [stack_in]
    else:
      c_out = [stack_in]
    
    for c_d in range(0, dil_rates+1):
      c_out += [layers.Conv1D(base_depth*2, (3,), 
                              padding='same', 
                              activation='linear',
                              dilation_rate=2**c_d,
                              name='C1D_L{}_D{}'.format(i, 2**c_d))(stack_in)]
    if (i==0) or (i==stack_count-1):
      c_cat = layers.concatenate(c_out)
    else:
      c_cat = layers.add(c_out)
    c_cat = layers.BatchNormalization(name='BN_L{}'.format(i))(c_cat)
    c_cat = layers.Activation('relu', name='Relu_L{}'.format(i))(c_cat)
    stack_in = c_cat
  
  def _make_foot(in_feat, name):
    c_feat = layers.Conv1D(base_depth+4, (1,), 
                  padding='same', 
                  activation='relu', 
                  name='{}_features'.format(name))(in_feat)
    last_conv_name = '{}_output'.format(name) if crop_dist>0 else 'foot_{}'.format(name)
    foot_out = layers.Conv1D(3, (1,), 
                  padding='same', 
                  activation='tanh', 
                  name=last_conv_name)(c_feat)
    if crop_dist>0:
      c_crop = layers.Cropping1D((crop_dist))(foot_out)
      c_pad = layers.ZeroPadding1D((crop_dist), name='foot_{}'.format(name))(c_crop)
    else:
      c_pad = foot_out
    return c_pad, c_feat
  
  c_left, c_left_feat = _make_foot(c_cat, 'left')
  c_right, c_right_feat = _make_foot(c_cat, 'right')
  
  
  if MERGE_FEET_FEATURES:
    # use the step output
    step_cat = layers.concatenate([c_cat, c_left_feat, c_right_feat])
  else:
    step_cat = c_cat
    
  step_cat = layers.AvgPool1D(pool_size=AVG_POOL_SIZE,
                              strides=1,
                              padding='same',
                              name='SmoothingStep')(step_cat)
  
  if crop_dist>0:
    c_crop = layers.Cropping1D((0, crop_dist))(step_cat)
    c_pad = layers.ZeroPadding1D((0, crop_dist))(c_crop)
  else:
    c_pad = step_cat
      
  
  dense_step_feat_a = layers.Conv1D(2*base_depth, (1,), 
                             padding='same', 
                             activation='relu')(c_pad)
  
  which_foot = layers.Conv1D(1, (1,), 
                  padding='same', 
                  activation='sigmoid', 
                  name='foot_right_down')(dense_step_feat_a)
  
  dense_step_feat_b = layers.Conv1D(2*base_depth, (1,), 
                             padding='same', 
                             activation='relu')(c_pad)
  
  step_event = layers.Conv1D(1, (1,), 
                  padding='same', 
                  activation='sigmoid', 
                  name='step_event')(dense_step_feat_b)
  
  out_nodes = [which_foot, step_event]
  
  if PREDICT_FEET_SEP:
    feet_sep = _feet_sep_module(step_cat, crop_dist=crop_dist)
    stride_len = stride_length_summarizer(feet_sep, crop_dist=crop_dist)
    out_nodes += [feet_sep, stride_len]
    
  out_nodes += [step_count_summarizer(step_event)]
    
  return models.Model(inputs=in_nodes, 
                      outputs=out_nodes,
                     name='C1D_Stacked{}'.format(suffix))


# ### Hourglass Sequence

# In[ ]:


def hourglass_gait_mod(
    in_node_names,
    dil_rates,
    base_depth=16, 
    stack_count=1,
    crop_dist=CROP_DIST,
    seq_len=tX['headpose'].shape[1],
    suffix=''
                   ):
  
  in_nodes = []
  in_prep_nodes = []
  
  for c_node in in_node_names:
    in_mod = layers.Input((seq_len, 3), name=c_node) 
    in_nodes += [in_mod]
    if DIFF_HEADPOSE and (c_node=='headpose'):
      in_mod = diff_filter_layer(depth=3)(in_mod)
    sd_in = dropout_layer(DROPOUT_RATE)(in_mod)
    bn_in = layers.BatchNormalization()(sd_in)
    c1 = layers.Conv1D(base_depth, (3,), 
                       padding='same', 
                       activation='relu')(bn_in)
    c2 = layers.Conv1D(base_depth, (3,), 
                       padding='same', 
                       activation='relu')(c1)
    in_prep_nodes += [c2]
  
  if len(in_prep_nodes)>1:
    c2 = layers.concatenate(in_prep_nodes, name='MergeSources')
  else:
    c2 = in_prep_nodes[0]
  stack_in = c2
  
  for j in range(stack_count):
    stack_in = dropout_layer(DROPOUT_RATE)(stack_in)
    mid_layers = {0: stack_in}
    in_filt = stack_in
    for i in range(dil_rates):
      c_filt = layers.Conv1D(base_depth*2**i, (3,), 
                             padding='same', 
                                activation='relu',
                                name='C1D_DN_S{}_L{}'.format(j, i))(in_filt)
      c_filt = layers.BatchNormalization(name='BN_DN_S{}_L{}'.format(j, i))(c_filt)
      c_filt = layers.Activation('relu', name='RELU_DN_S{}_L{}'.format(j, i))(c_filt)
      c_mp = layers.MaxPool1D(2, name='MX_S{}_L{}'.format(j, i))(c_filt)
      mid_layers[i+1] = c_mp
      in_filt = c_mp
      
    last_filt = c_filt
    
    for i in reversed(range(dil_rates)):
      if i in mid_layers:
        c_cat = layers.concatenate([last_filt, mid_layers[i]])
      else:
        c_cat = last_filt
      c_filt = layers.Conv1D(base_depth*2**(i+1), (3,), 
                             padding='same', 
                                activation='linear',
                                name='C1D_UP_S{}_L{}'.format(j, i))(c_cat)
      
      c_cat = layers.BatchNormalization(name='BN_S{}_L{}'.format(j, i))(c_filt)
      c_cat = layers.Activation('relu', name='Relu_S{}_L{}'.format(j, i))(c_cat)
      last_filt = layers.UpSampling1D(2, name='US_S{}_L{}'.format(j, i))(c_cat)
      
    stack_in = layers.concatenate([c_cat, c2])
  
  def _make_foot(in_feat, name):
    c_feat = layers.Conv1D(base_depth+4, (1,), 
                  padding='same', 
                  activation='relu', 
                  name='{}_features'.format(name))(in_feat)
    last_conv_name = '{}_output'.format(name) if crop_dist>0 else 'foot_{}'.format(name)
    foot_out = layers.Conv1D(3, (1,), 
                  padding='same', 
                  activation='tanh', 
                  name=last_conv_name)(c_feat)
    if crop_dist>0:
      c_crop = layers.Cropping1D((crop_dist))(foot_out)
      c_pad = layers.ZeroPadding1D((crop_dist), name='foot_{}'.format(name))(c_crop)
    else:
      c_pad = foot_out
    return c_pad, c_feat
  
  c_left, c_left_feat = _make_foot(c_cat, 'left')
  c_right, c_right_feat = _make_foot(c_cat, 'right')
  
  
  if MERGE_FEET_FEATURES:
    # use the step output
    step_cat = layers.concatenate([c_cat, c_left_feat, c_right_feat])
  else:
    step_cat = c_cat
    
  step_cat = layers.AvgPool1D(pool_size=AVG_POOL_SIZE,
                              strides=1,
                              padding='same',
                              name='SmoothingStep')(step_cat)
  
  if crop_dist>0:
    c_crop = layers.Cropping1D((0, crop_dist))(step_cat)
    c_pad = layers.ZeroPadding1D((0, crop_dist))(c_crop)
  else:
    c_pad = step_cat
      
  
  dense_step_feat_a = layers.Conv1D(2*base_depth, (1,), 
                             padding='same', 
                             activation='relu')(c_pad)
  
  which_foot = layers.Conv1D(1, (1,), 
                  padding='same', 
                  activation='sigmoid', 
                  name='foot_right_down')(dense_step_feat_a)
  
  dense_step_feat_b = layers.Conv1D(2*base_depth, (1,), 
                             padding='same', 
                             activation='relu')(c_pad)
  
  step_event = layers.Conv1D(1, (1,), 
                  padding='same', 
                  activation='sigmoid', 
                  name='step_event')(dense_step_feat_b)
  
  out_nodes = [which_foot, step_event]
  if PREDICT_FEET_SEP:
    feet_sep = _feet_sep_module(step_cat, crop_dist=crop_dist)
    stride_len = stride_length_summarizer(feet_sep, crop_dist=crop_dist)
    out_nodes += [feet_sep, stride_len]
  out_nodes += [step_count_summarizer(step_event)]
  
  return models.Model(inputs=in_nodes, 
                      outputs=out_nodes,
                     name='HourGlass_Stacked{}'.format(suffix))


# ### LSTM Model

# In[ ]:


def lstm_gait_mod(
    in_node_names,
    n_lstms, 
    base_depth=16, 
    crop_dist=CROP_DIST, 
    seq_len=tX['headpose'].shape[1], 
    suffix=''):
  in_nodes = []
  in_prep_nodes = []
  
  for c_node in in_node_names:
    in_mod = layers.Input((seq_len, 3), name=c_node) 
    in_nodes += [in_mod]
    if DIFF_HEADPOSE and (c_node=='headpose'):
      in_mod = diff_filter_layer(depth=3)(in_mod)
    sd_in = dropout_layer(DROPOUT_RATE)(in_mod)
    bn_in = layers.BatchNormalization()(sd_in)
    c1 = layers.Conv1D(base_depth, (3,), 
                       padding='same', 
                       activation='relu')(bn_in)
    c2 = layers.Conv1D(base_depth, (3,), 
                       padding='same', 
                       activation='relu')(c1)
    in_prep_nodes += [c2]
  
  if len(in_prep_nodes)>1:
    c2 = layers.concatenate(in_prep_nodes, name='MergeSources')
  else:
    c2 = in_prep_nodes[0]
  
  last_layer = c2
  for k in range(n_lstms):
    c_lstm = LSTM(base_depth*2, return_sequences=True, name='LSTM_{}'.format(k))
    last_layer = layers.Bidirectional(c_lstm, name='BD_LSTM_{}'.format(k))(last_layer)
    last_layer = dropout_layer(DROPOUT_RATE)(last_layer)
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
    # a fixed scaling of the output (x2)
    scale_layer = layers.Conv1D(3, (1,), weights= [np.expand_dims(2*np.eye(3), 0)],
                                use_bias=False, 
                                activation='linear',
                             name='{}_scaling'.format(name) if crop_dist>0 else 'foot_{}'.format(name)
                            )
    
    scale_layer.trainable = False
    foot_out = scale_layer(foot_out)
    
    if crop_dist>0:
      c_crop = layers.Cropping1D((0, crop_dist))(foot_out)
      c_pad = layers.ZeroPadding1D((0, crop_dist), name='foot_{}'.format(name))(c_crop)
    else:
      c_pad = foot_out
    
    return c_pad, c_feat
  c_left, c_left_feat = _make_foot(c_cat, 'left')
  c_right, c_right_feat = _make_foot(c_cat, 'right')
  if MERGE_FEET_FEATURES:
    # use the step output
    step_cat = layers.concatenate([c_cat, c_left_feat, c_right_feat])
  else:
    step_cat = c_cat
  
  if crop_dist>0:
    c_crop = layers.Cropping1D((0, crop_dist))(step_cat)
    c_pad = layers.ZeroPadding1D((0, crop_dist))(c_crop)
  else:
    c_pad = step_cat
    
  dense_step_feat_a = layers.Conv1D(2*base_depth, (1,), 
                             padding='same', 
                             activation='relu')(c_pad)
  
  which_foot = layers.Conv1D(1, (1,), 
                  padding='same', 
                  activation='sigmoid', 
                  name='foot_right_down')(dense_step_feat_a)
  
  dense_step_feat_b = layers.Conv1D(2*base_depth, (1,), 
                             padding='same', 
                             activation='relu')(c_pad)
  
  step_event = layers.Conv1D(1, (1,), 
                  padding='same', 
                  activation='sigmoid', 
                  name='step_event')(dense_step_feat_b)
  out_nodes = [which_foot, step_event]
  
  if PREDICT_FEET_SEP:
    feet_sep = _feet_sep_module(step_cat, crop_dist=crop_dist)
    stride_len = stride_length_summarizer(feet_sep, crop_dist=crop_dist)
    out_nodes += [feet_sep, stride_len]
  
  out_nodes += [step_count_summarizer(step_event)]
  
  return models.Model(inputs=in_nodes, 
                      outputs=out_nodes,
                     name='LSTM_Stacked{}'.format(suffix))


# ### Assemble Model
# Here we assemble the model together

# In[ ]:


# so we can build it again later with different settings
input_names = ['headpose']+IMU_VARS

if USE_LSTM:
  build_model = lambda names=input_names, **kwargs: lstm_gait_mod(
      names, 
      STACK_UNITS, 
      base_depth=BASE_DEPTH,
      **kwargs)
elif USE_HOURGLASS:
  build_model = lambda names=input_names, **kwargs: hourglass_gait_mod(
      names,
      stack_count=STACK_UNITS, 
      dil_rates=DILATION_COUNT,
      base_depth=BASE_DEPTH, 
      **kwargs)
else:
  build_model = lambda names=input_names, **kwargs: simple_gait_mod(
      names,
      stack_count=STACK_UNITS,
      dil_rates=DILATION_COUNT,
      base_depth=BASE_DEPTH, 
      **kwargs)


# In[ ]:


if SEPERATE_MODELS:
  seq_len = tX['headpose'].shape[1]
  in_list = []
  out_list = []
  for c_input in ['headpose']+IMU_VARS:
    c_in_layer = layers.Input((seq_len, 3), name=c_input) 
    c_model = build_model(names=[c_input], suffix='_{}'.format(c_input))
    c_outputs = c_model([c_in_layer])
    in_list += [c_in_layer]
    out_list += [c_outputs]
  out_list = [layers.average(list(x), 
                             name=x[0].name.split('/')[-2]) # cheeky hack to get name
              for x in zip(*out_list)]
  basic_gait = models.Model(inputs=in_list, outputs=out_list, name='MegaModel')
      
else:
  basic_gait = build_model()
basic_gait.summary()


# In[ ]:


from keras.utils import vis_utils
from IPython.display import SVG
SVG(vis_utils.model_to_dot(basic_gait, show_shapes=True).create_svg())


# In[ ]:


vis_utils.model_to_dot(basic_gait, show_shapes=True).write_png('model.png')
FileLink('model.png')


# ## Setup Loss and Metrics

# In[ ]:


from keras import optimizers, losses, metrics
loss_dict = {'foot_right_down': losses.binary_crossentropy,
             'step_event': losses.binary_crossentropy
                       }
metric_dict = { 'foot_right_down': metrics.binary_accuracy,
                'step_event': metrics.binary_accuracy
                          }
loss_weight_dict = {'foot_right_down': STEP_LOSS_WEIGHT,
                    'step_event': STEP_LOSS_WEIGHT
                               }
def signless_mae(a, b):
  import tensorflow as tf
  return losses.mean_absolute_error(tf.abs(a), tf.abs(b))
if PREDICT_FEET_SEP:
  
  loss_dict['feet_dist'] = losses.mean_squared_error
  metric_dict['feet_dist'] = [losses.mean_absolute_error, signless_mae]
  
  loss_weight_dict['feet_dist'] = FEET_SEP_WEIGHT
  
basic_gait.compile(optimizer=optimizers.Adam(lr=LEARNING_RATE), 
                  loss=loss_dict,
                  metrics=metric_dict,
                  loss_weights=loss_weight_dict)


# ## Setup Training and Validation
# Use different people for training and validation to make the results more meaningful

# In[ ]:


from sklearn.model_selection import train_test_split

full_train_df = valid_meta_df.copy()
if not USE_NON_WALKING:
  full_train_df = full_train_df[full_train_df['walking_like']]
keep_data_dirs = []

train_df, valid_df = train_test_split(full_train_df, 
                                      test_size=0.25, 
                                      random_state=RANDOM_SEED,
                 stratify=full_train_df[['data_dir', 'walking_like']].values)
print('Train:', train_df.shape[0])
print('Valid:', valid_df.shape[0])   


# In[ ]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('footpose')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', 
                             save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, 
                                   verbose=1, mode='auto', epsilon=0.0001, 
                                   cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=10)
callbacks_list = [checkpoint, early, reduceLROnPlat]


# ### Data Augmentation

# In[ ]:


@autotest
def closing_cumsum(x, axis, rate):
  """Cumsum that resets to zero.
  
  >>> np.random.seed(2019)
  >>> closing_cumsum([0, 1, 2, 3, 4, 1], axis=0, rate=0.2)
  array([ 0,  1,  3,  0,  4, -6])
  """
  raw_cumsum = np.cumsum(x, axis=axis)
  if rate>0:
    p_rate = np.random.poisson(lam=rate, size=np.shape(x))>0
    return np.cumsum(x-p_rate*raw_cumsum, axis=axis)
  else:
    return raw_cumsum


# In[ ]:


from scipy.ndimage import zoom
@autotest
def zoom_but_pad(in_tensor, scale, order=0, prefilter=False):
  """Zoom a tensor but pad it so its size is fixed
  
  >>> zoom_but_pad(np.eye(3), 2)
  array([[1., 0., 0.],
         [0., 1., 1.],
         [0., 1., 1.]])
  >>> zoom_but_pad(np.eye(6), (2, 0.5))
  array([[0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.]])
  >>> zoom_but_pad(np.eye(3), 2, order=1)
  array([[0.52, 0.44, 0.32],
         [0.44, 0.68, 0.64],
         [0.32, 0.64, 0.68]])
  """
  out_t = np.zeros_like(in_tensor)
  rs_t = zoom(in_tensor, scale, order=order, prefilter=prefilter)
  out_slices = []
  rs_slices = []
  #print(out_t.shape, rs_t.shape)
  #print(rs_t)
  for k, (rs_dim, out_dim) in enumerate(zip(rs_t.shape, out_t.shape)):
    if rs_dim==out_dim:
      out_slices += [slice(0, out_dim)]
      rs_slices += [slice(0, out_dim)]
    elif rs_dim<out_dim:
      # put it in the middle
      offset = (out_dim-rs_dim)//2
      out_slices += [slice(offset, rs_dim+offset)]
      rs_slices += [slice(0, rs_dim)]
    else:
      offset = (rs_dim-out_dim)//2
      out_slices += [slice(0, out_dim)]
      rs_slices += [slice(offset, out_dim+offset)]
  out_t.__setitem__(out_slices, rs_t.__getitem__(rs_slices))
  return out_t
    


# In[ ]:


from keras.utils import Sequence

class keras_walk_gen(Sequence):
  def __init__(self, in_df, steps, batch_size, augment=AUGMENT_WALKS):
    self._in_df = in_df
    self._steps = steps
    self._batch_size = batch_size
    self._augment = augment
    self.on_epoch_end() # initalize generator
    
  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(self._steps / self._batch_size))
  
  def __getitem__(self, index):
    'Generate one batch of data'
    x_batch, y_batch = next(self.__igen)
    c_vec_items = {k: v for k,v in x_batch.items() if (len(v.shape)==3) and (v.shape[2]==3)}
    # keep random seeds lined up
    rand_offset={}
    rand_drift={}
    rand_theta={}
    bobbing_scalar = np.random.uniform(1/HEADPOSE_BOBBING, HEADPOSE_BOBBING)
    for k, v in c_vec_items.items():
      rand_theta = np.tile(
        np.random.uniform(-np.pi, np.pi, size=v.shape[0:1]+(1,)),
        (1,)+v.shape[1:2]
      )
      rand_offset[k] = np.random.normal(0, np.std(np.abs(v))*GAUSS_NOISE, size=v.shape)
      drift_vec = np.random.normal(0, np.std(np.abs(v))*DRIFT_NOISE, size=v.shape)
      rand_drift[k] = closing_cumsum(drift_vec, 1, rate=LOOP_CLOSURE)
      frequency_scale_factor = np.random.uniform(-AUGMENT_FREQUENCY, 
                                                   AUGMENT_FREQUENCY)
    if self._augment:
      # add a random rotation
      # for every input that looks like an xyz vec (currently just headpose)
      for k, raw_head in c_vec_items.items():
        if HEADPOSE_BOBBING>0:
          x_head = scale_bobble(raw_head, bobbing_scalar, axis=1)
        else:
          x_head = raw_head.copy()
        new_x = x_head[:, :, 0]*np.cos(rand_theta)-x_head[:, :, 2]*np.sin(rand_theta)
        new_z = x_head[:, :, 0]*np.sin(rand_theta)+x_head[:, :, 2]*np.cos(rand_theta)
        x_head[:, :, 0] = new_x
        x_head[:, :, 2] = new_z
        # add noise to headpose
        x_head += rand_drift[k]+rand_offset[k]
        x_batch[k] = x_head
      if AUGMENT_FREQUENCY>0:
        
        fsf = np.power(2, frequency_scale_factor)
        for k,v in x_batch.items():
          if len(v.shape)==3:
            x_batch[k] = zoom_but_pad(v, [1, fsf, 1], order=1)
        for k,v in y_batch.items():
          if len(v.shape)==3:
            y_batch[k] = zoom_but_pad(v, [1, fsf, 1], order=1)
            
    return x_batch, y_batch
        
  def on_epoch_end(self):
    self.__igen = batch_it(
          generate_batch(
              self._in_df,
              seq_len=TRAIN_SEQ_LEN, 
              debug=False), 
          batch_size=self._batch_size
    )


# In[ ]:


fig, m_axs = plt.subplots(2, 10, figsize=(20, 10))
for n_axs, augment in zip(m_axs, [False, True]):
  np.random.seed(RANDOM_SEED)
  t_gen = keras_walk_gen(train_df, 
                         steps=TRAIN_STEPS, 
                         batch_size=BATCH_SIZE,
                        augment=augment)
  c_pose = t_gen[0][0]['headpose']
  for i, ax1 in enumerate(n_axs):
    ax1.plot(c_pose[i, :, :])
    ax1.set_title('Augmented' if augment else 'Normal')


# In[ ]:


# prepare validation data
def _get_and_batch(in_df, batch_size):
  return next(
      batch_it(
          generate_batch(
              in_df,
              seq_len=TRAIN_SEQ_LEN, debug=False), 
          batch_size=batch_size)
  )
np.random.seed(RANDOM_SEED)
valid_X, valid_y = _get_and_batch(valid_df, VALID_STEPS)
print(valid_X['headpose'].shape)
# make training tool
train_gen = keras_walk_gen(train_df, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)


# In[ ]:


# sanity check
pred_dict = dict(zip(basic_gait.output_names,
     basic_gait.predict({k: v[:16] for k,v in valid_X.items()})))
for k, v in pred_dict.items():
  print(k, v.shape, v.min(),'-', v.max())


# ### Training Loop

# In[ ]:


from IPython.display import clear_output
model_results = basic_gait.fit_generator(train_gen,
                                         validation_data=(valid_X, valid_y), 
                                         callbacks=callbacks_list,
                                         workers=4,
                                         use_multiprocessing=True,
                                         epochs=EPOCHS)
clear_output()


# In[ ]:


def show_results(model_hist):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
  ax1.semilogy(model_hist.history['loss'], label='Training')
  ax1.semilogy(model_hist.history['val_loss'], label='Validation')
  ax1.set_title('Loss')
  ax1.legend()
  ax2.plot(model_hist.history['foot_left_mean_absolute_error'], label='Training Left')
  ax2.plot(model_hist.history['foot_right_mean_absolute_error'], label='Training Right')
  ax2.plot(model_hist.history['val_foot_left_mean_absolute_error'], label='Validation Left')
  ax2.plot(model_hist.history['val_foot_right_mean_absolute_error'], label='Validation Right')
  ax2.set_title('MAE')
  ax2.legend()
  
  ax3.plot(model_hist.history['foot_right_down_binary_accuracy'], label='Training Step')
  ax3.plot(model_hist.history['val_foot_right_down_binary_accuracy'], label='Validation Step')
  ax3.legend()
  ax3.set_title('Step Accuracy')
  
  ax4.plot(model_hist.history['step_event_binary_accuracy'], label='Training Step')
  ax4.plot(model_hist.history['val_step_event_binary_accuracy'], label='Validation Step')
  ax4.legend()
  ax4.set_title('Step Detection Accuracy')
show_results(model_results)


# ### Showing Outputs

# In[ ]:


basic_gait.load_weights(weight_path)


# In[ ]:


get_ipython().run_cell_magic('time', '', '_ = basic_gait.predict({k: v[:BATCH_SIZE*8] for k, v in valid_X.items()}, \n                       batch_size=BATCH_SIZE, \n                       verbose=True);')


# In[ ]:


v_idx = np.random.choice(range(valid_X['headpose'].shape[0]))
X = {k: np.expand_dims(v[v_idx], 0) for k, v in valid_X.items()}
y = {k: np.expand_dims(v[v_idx], 0) for k, v in valid_y.items()}
y_pred = dict(zip(basic_gait.output_names, basic_gait.predict(X)))


# In[ ]:


np.random.seed(RANDOM_SEED-1)
v_idx = np.random.choice(range(valid_X['headpose'].shape[0]))
X = {k: np.expand_dims(v[v_idx], 0) for k, v in valid_X.items()}
y = {k: np.expand_dims(v[v_idx], 0) for k, v in valid_y.items()}


# In[ ]:


fig = plt.figure(figsize=(20, 8))
body_height_estimate = 50 # person is 50au tall
body_stance_estimate = 20 # legs start 20au apart
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133)

show_head_vec = X['headpose'][0].copy()
show_head_vec[:, 1]+=body_height_estimate # above the fray

fl_pos = np.cumsum(y['foot_left'][0, :, :], 0)
fr_pos = np.cumsum(y['foot_right'][0, :, :], 0)

fd_status = y['foot_right_down'][0, :, 0]
fd_cutoff = 0.5

if FEET_RELATIVE_TO_HEAD:
  fl_pos+=X['headpose'][0]
  fr_pos+=X['headpose'][0]

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
fd_settings = dict(ms=15, alpha=0.05)
# draw foot down
ax1.plot(fl_pos[fd_status<fd_cutoff, 0], 
         fl_pos[fd_status<fd_cutoff, 2], 
         fl_pos[fd_status<fd_cutoff, 1], 
         'gs', **fd_settings)

ax1.plot(fr_pos[fd_status>=fd_cutoff, 0], 
         fr_pos[fd_status>=fd_cutoff, 2], 
         fr_pos[fd_status>=fd_cutoff, 1], 
         'rs', **fd_settings)

y_pred = dict(zip(basic_gait.output_names, basic_gait.predict(X)))
# we don't predict left and right anymore
fl_pred = np.cumsum(y['foot_left'][0, :, :], 0)
fr_pred = np.cumsum(y['foot_right'][0, :, :], 0)
fd_pred = y_pred['foot_right_down'][0, :, 0]

if FEET_RELATIVE_TO_HEAD:
  fl_pred+=X['headpose'][0]
  fr_pred+=X['headpose'][0]
  
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

# draw foot down
ax2.plot(fl_pred[fd_pred<fd_cutoff, 0], 
         fl_pred[fd_pred<fd_cutoff, 2], 
         fl_pred[fd_pred<fd_cutoff, 1], 
         'gs', **fd_settings)

ax2.plot(fr_pred[fd_pred>=fd_cutoff, 0], 
         fr_pred[fd_pred>=fd_cutoff, 2], 
         fr_pred[fd_pred>=fd_cutoff, 1], 
         'rs', **fd_settings)


ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())

ax1.view_init(45, 30)
ax2.view_init(45, 30)


#ax1.view_init(90, 0)
#ax2.view_init(90, 0)

ax3.plot(fd_status, label='Foot Status')
ax3.plot(fd_pred, label='Predicted')
ax3.legend(bbox_to_anchor=(0.5, 1.05), 
          fancybox=True, shadow=True)


# In[ ]:


vec_names = ['head', 'leftfoot_pos', 'rightfoot_pos', 'leftfoot_pred', 'rightfoot_pred']
demo_df = pd.DataFrame([{'{}_{}'.format(v_name, c_x): c_v 
  for v_name, v_pos in zip(vec_names, vec_pos) 
  for c_x, c_v in zip('xyz', v_pos)} 
 for vec_pos in zip(show_head_vec, fl_pos, fr_pos, fl_pred, fr_pred)])
demo_df['rightfoot_down'] = fd_status
demo_df['rightfoot_down_pred'] = fd_status
demo_df.to_csv('demo.csv', index=False)
FileLink('demo.csv')


# In[ ]:


from matplotlib.animation import FuncAnimation
def _show_frame(i, ms=10, add_body=True):
  c_head = show_head_vec[i:i+1]
  [c_line.remove() 
   for c_ax in [ax1, ax2, ax3] 
   for c_line in c_ax.get_lines() 
   if c_line.get_label().startswith('_')];
  ax3.axvline(i, color='red')
  for c_ax, [c_fl, c_fr] in zip([ax1, ax2], 
                  [(fl_pos[i:i+1], fr_pos[i:i+1]), 
                   (fl_pred[i:i+1], fr_pred[i:i+1])]):
    
    
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
    
if SAVE_VIDEO:
  out_anim = FuncAnimation(fig, _show_frame, range(0, show_head_vec.shape[0], 1))
  out_anim.save('walk_recording.mp4', bitrate=8000, fps=8)
  FileLink('walk_recording.mp4')


# In[ ]:


trained_model_path = 'model_weights.h5'
basic_gait.save_weights(trained_model_path)


# In[ ]:


import time
from keras.layers import LSTM # CuDNNLSTM chokes up most model converters
fancy_gait = build_model(seq_len=None)
fancy_gait.load_weights(weight_path)
time_suffix = ''.join(['{:02d}'.format(x) for x in time.gmtime()[:6]])
model_path = str(RESULTS_DIR / 'trained_models' / '{}_{}.h5'.format(basic_gait.name, time_suffix))
fancy_gait.save(model_path)
fancy_gait.summary()


# # Saving Experiments Infrastructure
# Save all the results into a Google Sheet (via Google Forms) so we can have nice(ish) dashboards showing progress and results on training. 
# - Sheet is [here](https://docs.google.com/spreadsheets/d/1B-xmr8tsC3g5hwSuCqAHPCxJMeCfmfYEwz3G37JCmfg/edit?usp=sharing)

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
    if k in q_dict:
      answers[k] = v
  # add kw arguments
  for k, v in kw_args.items():
    if (k in q_dict) and (k not in answers):
      answers[k] = v
  
  # add global/capital variables    
  bonus_args = fetch_json_globals()
  
  for k, v in bonus_args.items():
    if (k in q_dict) and (k not in answers):
      answers[k] = v
  answers['experiment_name'] = experiment_name
  answers['parameters_json'] = json.dumps(bonus_args)
  answers['metrics_json'] = json.dumps(metrics)
  answers['extra_args_json'] = json.dumps(kw_args)
  answers['user_name'] = user_name
  answers['model_name'] = model_name
  answers['model_json'] = json.dumps(model_dict)
  return submit_response(gform_url, q_dict, **answers)                


# In[ ]:


q_dict = get_questions(results_form_url)
q_dict


# # Apply for a full sequence
# We can take the model and apply it to a full sequence

# In[ ]:


single_scan = valid_df.sample(1, random_state=RANDOM_SEED)
big_X, big_y = next(generate_batch(single_scan, 
                                         seq_len=None, 
                                         debug=True, 
                                         include_offset=True
                                        ))


# In[ ]:


if USE_HOURGLASS:
  new_dim = np.floor(big_X['headpose'].shape[0]/64).astype(int)*64
else:
  new_dim = big_X['headpose'].shape[0]

X = {k: np.expand_dims(v[:new_dim], 0) for k, v in big_X.items()}
y = {k: np.expand_dims(v[:new_dim], 0) for k, v in big_y.items()}
y_pred = dict(zip(fancy_gait.output_names, fancy_gait.predict(X)))


# In[ ]:


y_pred


# In[ ]:


# evaluate on bigger dataset for AUC
from sklearn.metrics import roc_auc_score, roc_curve
cut_off = 0.5
big_auc = roc_auc_score(y['step_event'][0, :, 0]>cut_off, 
                        y_pred['step_event'][0, :, 0])


# In[ ]:


import inspect
metric_dict = dict(zip(basic_gait.metrics_names, 
                       basic_gait.evaluate(valid_X, valid_y, verbose=False)))

save_experiment_gforms(
    gform_url=results_form_url,
    experiment_name=EXPERIMENT_NAME,
    model_name=basic_gait.name,
    user_name='Kevin',
    metrics=metric_dict,
    model_path=model_path,
    auc = big_auc,
    model_dict={}, #json.loads(basic_gait.to_json())
)


# In[ ]:


fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133)
show_head_vec = X['headpose'][0].copy()
show_head_vec[:, 1]+=body_height_estimate # above the fray

fl_pos = np.cumsum(y['foot_left'][0, :, :], 0)
fr_pos = np.cumsum(y['foot_right'][0, :, :], 0)
fd_status = y['foot_right_down'][0, :, 0]
fd_cutoff = 0.5
if FEET_RELATIVE_TO_HEAD:
  fl_pos+=X['headpose'][0]
  fr_pos+=X['headpose'][0]

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
fd_settings = dict(ms=15, alpha=0.05)
# draw foot down
ax1.plot(fl_pos[fd_status<fd_cutoff, 0], 
         fl_pos[fd_status<fd_cutoff, 2], 
         fl_pos[fd_status<fd_cutoff, 1], 
         'gs', **fd_settings)

ax1.plot(fr_pos[fd_status>=fd_cutoff, 0], 
         fr_pos[fd_status>=fd_cutoff, 2], 
         fr_pos[fd_status>=fd_cutoff, 1], 
         'rs', **fd_settings)

fl_pred = np.cumsum(y['foot_left'][0, :, :], 0)
fr_pred = np.cumsum(y['foot_right'][0, :, :], 0)
fd_pred = y_pred['foot_right_down'][0, :, 0]

if FEET_RELATIVE_TO_HEAD:
  fl_pred+=X['headpose'][0]
  fr_pred+=X['headpose'][0]
  
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

# draw foot down
ax2.plot(fl_pred[fd_pred<fd_cutoff, 0], 
         fl_pred[fd_pred<fd_cutoff, 2], 
         fl_pred[fd_pred<fd_cutoff, 1], 
         'gs', **fd_settings)

ax2.plot(fr_pred[fd_pred>=fd_cutoff, 0], 
         fr_pred[fd_pred>=fd_cutoff, 2], 
         fr_pred[fd_pred>=fd_cutoff, 1], 
         'rs', **fd_settings)



ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
ax2.set_zlim(ax1.get_zlim())
ax1.view_init(30, 15)
ax2.view_init(30, 15)

ax3.plot(fd_status, label='Foot Status')
ax3.plot(fd_pred, label='Predicted')
ax3.legend(bbox_to_anchor=(0.5, 1.05), 
          fancybox=True, shadow=True)


# In[ ]:


from matplotlib.animation import FuncAnimation
def _show_frame(i, ms=10, add_body=True):
  c_head = show_head_vec[i:i+1]
  [c_line.remove() 
   for c_ax in [ax1, ax2, ax3] 
   for c_line in c_ax.get_lines() 
   if c_line.get_label().startswith('_')];
  ax3.axvline(i, color='red')
  for c_ax, [c_fl, c_fr] in zip([ax1, ax2], 
                  [(fl_pos[i:i+1], fr_pos[i:i+1]), 
                   (fl_pred[i:i+1], fr_pred[i:i+1])]):
    
    
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
    
if SAVE_VIDEO:
  out_anim = FuncAnimation(fig, _show_frame, range(0, show_head_vec.shape[0], 40))
  out_anim.save('big_walk_recording.mp4', bitrate=8000, fps=8)
  FileLink('big_walk_recording.mp4')


# # Summary Results

# ## Step Detection

# In[ ]:


fig, ax3 = plt.subplots(1, 1, figsize=(20, 5))
ax3.plot(y['step_event'][0, :, 0], label='Step')
ax3.plot(y_pred['step_event'][0, :, 0], label='Predicted')

ax3.legend(bbox_to_anchor=(0.5, 1.05), 
          fancybox=True, shadow=True)
ax3.set_xlim(0, len(fd_status))
plt.tight_layout()


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
fpr, tpr, th_val = roc_curve(y['step_event'][0, :, 0]>cut_off, 
                        y_pred['step_event'][0, :, 0])
auc = roc_auc_score(y['step_event'][0, :, 0]>cut_off, 
                        y_pred['step_event'][0, :, 0])
ax1.plot(fpr, tpr, '.-', label='Model: {:2.1%}'.format(auc))
ax1.legend()
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
# find the 'max' point on the curve to balance true positives and false positives
max_pt = np.argmax(tpr*(1-fpr))
print('Ideal Cutoff: {:2.2f} (FPR: {:2.1%}, TPR: {:2.1%})'.format(th_val[max_pt], fpr[max_pt], tpr[max_pt]))


# In[ ]:


fig, ax3 = plt.subplots(1, 1, figsize=(5, 5))
ax3.plot(y['step_event'][0, :, 0]+np.random.uniform(-0.025, 0.025, 
                                                    size=y_pred['step_event'].shape[1]), 
         y_pred['step_event'][0, :, 0], '.', label='Step', alpha=0.5)
ax3.plot(y['step_event'][0, :, 0], y['step_event'][0, :, 0], '-')
ax3.set_xlabel('Actual Step')
ax3.set_ylabel('Predicted')
ax3.axis('equal')
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)


# In[ ]:


import seaborn as sns
fig, ax3 = plt.subplots(1, 1, figsize=(10, 10))
sns.violinplot(x=fd_status, y=fd_pred, ax=ax3)
#sns.swarmplot(x=fd_status, y=fd_pred, ax=ax3)
#ax3.plot(fd_status, fd_pred, '.', label='Prediction')
#ax3.plot(fd_status, fd_status, '-', label='Ground Truth')
ax3.axis('equal')
plt.tight_layout()


# ## Foot Weight

# In[ ]:


fig, ax3 = plt.subplots(1, 1, figsize=(20, 5))
ax3.plot(fd_status, label='Foot Status')
ax3.plot(fd_pred, label='Predicted')
ax3.plot(fd_pred-fd_status, 'r-', label='Error', alpha=0.2, lw=2)
ax3.legend(bbox_to_anchor=(0.5, 1.05), 
          fancybox=True, shadow=True)
ax3.set_xlim(0, len(fd_status))
plt.tight_layout()


# ## Foot Seperation 

# In[ ]:


fig, (ax3, ax4) = plt.subplots(2, 1, figsize=(20, 10))
ax3.plot(y['feet_dist'][0, :, 0], label='Foot Distance')
ax3.plot(y_pred['feet_dist'][0, :, 0], label='Predicted-Foot Distance')
ax3.legend(bbox_to_anchor=(0.5, 1.05), 
          fancybox=True, shadow=True)

ax4.plot(np.abs(y['feet_dist'][0, :, 0]), label='|Foot Distance|')
ax4.plot(np.abs(y_pred['feet_dist'][0, :, 0]), label='|Predicted-Foot Distance|')

ax3.set_xlim(0, len(fd_status))
plt.tight_layout()


# In[ ]:


fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))
ax3.plot(y['feet_dist'][0, :, 0], 
         y_pred['feet_dist'][0, :, 0], '.', label='Foot Seperation', alpha=0.5)
ax3.plot(y['feet_dist'][0, :, 0], y['feet_dist'][0, :, 0], '-')
ax3.set_xlabel('Actual Step')
ax3.set_ylabel('Predicted')
ax3.axis('equal')

ax4.plot(np.abs(y['feet_dist'][0, :, 0]), 
         np.abs(y_pred['feet_dist'][0, :, 0]), '.', label='Foot Seperation', alpha=0.5)
ax4.plot(np.abs(y['feet_dist'][0, :, 0]), 
         np.abs(y['feet_dist'][0, :, 0]), '-')
ax4.set_xlabel('Actual Step')
ax4.set_ylabel('Predicted')
ax4.axis('equal')


# ## Simple TFLite Validation
# Make sure the model can be exported as TFLite

# In[ ]:


import tensorflow as tf
import keras
def convert_model_to_tflite(c_model, i):
  print(c_model, 'to load')
  keras.backend.clear_session()
  import tensorflow as tf
  walk_model = keras.models.load_model(str(c_model))
  fixed_in = keras.layers.Input((180, 3), name='fixed_headpose')
  fixed_out = walk_model(fixed_in)
  wrapper_model = keras.models.Model(inputs=[fixed_in],
                                  outputs=fixed_out)
  temp_path = 'fixed_model_180_{:04d}.h5'.format(i)
  wrapper_model.save(temp_path)
  converter = tf.lite.TFLiteConverter.from_keras_model_file(temp_path)
  tflite_path = 'model.tflite'
  tflite_model = converter.convert()
  tflite_path.open("wb").write(tflite_model)


# In[ ]:


basic_gait.save('junk.h5')
convert_model_to_tflite('junk.h5', 0)
FileLink('model.tflite')

