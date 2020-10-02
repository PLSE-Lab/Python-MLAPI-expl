#!/usr/bin/env python
# coding: utf-8

# # Overview
# A simple notebook to load and display the IMU data for different experiments

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def from_path_to_df(in_path):
    """Read experiment txt files."""
    with in_path.open('r') as f:
        cur_df = pd.read_csv(f, sep='\s+', header=None, skiprows=1)
        if len(cur_df.columns)==4:
            cur_df.columns = ['timestamp_ns', 'x', 'y', 'z']
        else:
            old_cols = cur_df.columns.tolist()
            old_cols[0] = 'timestamp_ns'
            if len(cur_df.columns)==8:
                old_cols[1] = 'x'
                old_cols[2] = 'y'
                old_cols[3] = 'z'
            cur_df.columns = old_cols
        cur_df['timestamp_s'] = cur_df['timestamp_ns']/1.0e9
        return cur_df.drop('timestamp_ns', axis=1).sort_values('timestamp_s')


# In[ ]:


BASE_DIR = Path('..') / 'input' / 'data_publish_v2' / 'data_publish_v2'


# In[ ]:


all_files_df = pd.DataFrame({'path': list(BASE_DIR.glob('*/*.txt'))})
all_files_df['exp_code'] = all_files_df['path'].map(lambda x: x.parent.stem)
all_files_df['activity'] = all_files_df['exp_code'].map(lambda x: '_'.join(x.split('_')[1:]))
all_files_df['person'] = all_files_df['exp_code'].map(lambda x: x.split('_')[0])
all_files_df['data_src'] = all_files_df['path'].map(lambda x: x.stem)
all_files_df.sample(5)


# In[ ]:


data_df = all_files_df.pivot_table(values='path', 
                         columns='data_src', 
                         index=['activity', 'person'],
                        aggfunc='first').\
    reset_index().\
    dropna(axis=1) # remove mostly empty columns
data_df.head(5)


# # Take a single experiment
# We can take a single experiment to process in more detail

# In[ ]:


cur_exp = data_df.iloc[0]
print(cur_exp.iloc[0:2])


# In[ ]:


dict_df = {k: from_path_to_df(v) 
           for k, v in cur_exp.iloc[2:].items()}
for k, v in dict_df.items():
    print(k, v.shape, 'Framerate:{:2.1f}'.format(1/(np.mean(v['timestamp_s'].diff()))))


# In[ ]:


dict_df['pose'].plot('timestamp_s')
dict_df['pose'].sample(3)


# In[ ]:


dict_df['linacce'].plot('timestamp_s')


# In[ ]:


dict_df['gravity'].plot('timestamp_s')


# ## Compare Simple Double Integration
# Here we compare simple double integration to the actual pose vector
# $$ \vec{x} = \int\int \vec{a} $$

# In[ ]:


la_df = dict_df['linacce'].copy()
pose_start = dict_df['pose'].iloc[0] # for the initial conditions
for c_x in 'xyz':
    la_df['vel_{}'.format(c_x)] = cumtrapz(la_df['{}'.format(c_x)].values, 
                                           x=la_df['timestamp_s'].values, 
                                           initial=0)
    la_df['pos_{}'.format(c_x)] = cumtrapz(la_df['vel_{}'.format(c_x)].values, 
                                           x=la_df['timestamp_s'], 
                                           initial=pose_start[c_x])


# ### Show curves

# In[ ]:


fig, m_axs = plt.subplots(3, 1, figsize=(20, 12))
c_la_df = la_df
c_pos_df = dict_df['pose']
c_pos_valid = c_pos_df['timestamp_s']>=c_la_df['timestamp_s'].iloc[0]
c_pos_valid &= c_pos_df['timestamp_s']<=c_la_df['timestamp_s'].iloc[-1]
c_pos_df = c_pos_df[c_pos_valid]
for c_x, c_ax in zip('xyz', m_axs):
    c_ax.plot(c_la_df['timestamp_s'], c_la_df['{}'.format(c_x)], '.', label='Linear Acceleration')
    c_ax.plot(c_la_df['timestamp_s'], c_la_df['vel_{}'.format(c_x)], label='Integrated Velocity')
    c_ax.plot(c_la_df['timestamp_s'], c_la_df['pos_{}'.format(c_x)], label='Integrated Position')
    c_ax.plot(c_pos_df['timestamp_s'], c_pos_df[c_x], '-', label='Actual Pose')
    c_ax.legend()
    c_ax.set_title(c_x)
    


# ## Integration Errors
# As we can see the errors accumulate quickly and we end up with a very different path

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.plot(la_df['pos_x'], la_df['pos_y'], '.-', label='Integrated Position')
ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], '+-', label='Actual Pose')
ax1.legend()
ax1.axis('equal');


# In[ ]:


fig = plt.figure(figsize=(10, 10), dpi=300)
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot(la_df['pos_x'], la_df['pos_y'], la_df['pos_z'], '.-', label='Integrated Position')
ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], dict_df['pose']['z'], '+-', label='Actual Pose')
ax1.legend()
ax1.axis('equal');
fig.savefig('hr_img.png')


# ## Small Time Window
# We can look at a small time window and see how the errors start

# In[ ]:


fig, m_axs = plt.subplots(3, 1, figsize=(15, 12))
c_la_df = la_df[la_df['timestamp_s']<la_df['timestamp_s'].iloc[1000]]
c_pos_df = dict_df['pose']
c_pos_valid = c_pos_df['timestamp_s']>=c_la_df['timestamp_s'].iloc[0]
c_pos_valid &= c_pos_df['timestamp_s']<=c_la_df['timestamp_s'].iloc[-1]
c_pos_df = c_pos_df[c_pos_valid]
for c_x, c_ax in zip('xyz', m_axs):
    c_ax.plot(c_la_df['timestamp_s'], c_la_df['{}'.format(c_x)], '.', label='Linear Acceleration')
    c_ax.plot(c_la_df['timestamp_s'], c_la_df['vel_{}'.format(c_x)], label='Integrated Velocity')
    c_ax.plot(c_la_df['timestamp_s'], c_la_df['pos_{}'.format(c_x)], label='Integrated Position')
    c_ax.plot(c_pos_df['timestamp_s'], c_pos_df[c_x], '-', label='Actual Pose')
    c_ax.legend()
    c_ax.set_title(c_x)


# ## Correcting Drift
# We can fake drift correction by periodically setting the velocity to 0

# In[ ]:


la_df = dict_df['linacce'].sort_values('timestamp_s').copy()
pose_start = dict_df['pose'].iloc[0] # for the initial conditions
for c_x in 'xyz':
    vel_vec = cumtrapz(la_df['{}'.format(c_x)].values, 
                                           x=la_df['timestamp_s'].values, 
                                           initial=0)
    for i in range(0, la_df.shape[0], 1000):
        vel_vec[(i+1):] -= vel_vec[i]
    la_df['vel_{}'.format(c_x)] = vel_vec
    la_df['pos_{}'.format(c_x)] = cumtrapz(la_df['vel_{}'.format(c_x)].values, 
                                           x=la_df['timestamp_s'], 
                                           initial=pose_start[c_x])


# In[ ]:


fig, m_axs = plt.subplots(3, 1, figsize=(20, 12))
c_la_df = la_df
c_pos_df = dict_df['pose']
c_pos_valid = c_pos_df['timestamp_s']>=c_la_df['timestamp_s'].iloc[0]
c_pos_valid &= c_pos_df['timestamp_s']<=c_la_df['timestamp_s'].iloc[-1]
c_pos_df = c_pos_df[c_pos_valid]
for c_x, c_ax in zip('xyz', m_axs):
    c_ax.plot(c_la_df['timestamp_s'], 
              c_la_df['vel_{}'.format(c_x)], 
              label='Integrated Velocity')
    c_ax.plot(c_la_df['timestamp_s'], 
              c_la_df['pos_{}'.format(c_x)], 
              label='Integrated Position')
    c_ax.plot(c_pos_df['timestamp_s'], c_pos_df[c_x], '-', label='Actual Pose')
    c_ax.legend()
    c_ax.set_title(c_x)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.plot(la_df['pos_x'], la_df['pos_y'], '.-', label='Integrated Position')
ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], '+-', label='Actual Pose')
ax1.legend()
ax1.axis('equal');


# ## Detrending to correct drift

# In[ ]:


from scipy.signal import detrend
la_df = dict_df['linacce'].sort_values('timestamp_s').copy()
pose_start = dict_df['pose'].iloc[0] # for the initial conditions
for c_x in 'xyz':
    vel_vec = cumtrapz(la_df['{}'.format(c_x)].values, 
                                           x=la_df['timestamp_s'].values, 
                                           initial=0)
    ij_idx = range(0, la_df.shape[0], 5000)
    for i, j in zip(ij_idx, ij_idx[1:]):
        vel_vec[i:j] = detrend(vel_vec[i:j])
    la_df['vel_{}'.format(c_x)] = vel_vec
    la_df['pos_{}'.format(c_x)] = cumtrapz(la_df['vel_{}'.format(c_x)].values, 
                                           x=la_df['timestamp_s'], 
                                           initial=pose_start[c_x])


# In[ ]:


fig, m_axs = plt.subplots(3, 1, figsize=(20, 12))
c_la_df = la_df
c_pos_df = dict_df['pose']
c_pos_valid = c_pos_df['timestamp_s']>=c_la_df['timestamp_s'].iloc[0]
c_pos_valid &= c_pos_df['timestamp_s']<=c_la_df['timestamp_s'].iloc[-1]
c_pos_df = c_pos_df[c_pos_valid]
for c_x, c_ax in zip('xyz', m_axs):
    c_ax.plot(c_la_df['timestamp_s'], 
              c_la_df['vel_{}'.format(c_x)], 
              label='Integrated Velocity')
    c_ax.plot(c_la_df['timestamp_s'], 
              c_la_df['pos_{}'.format(c_x)], 
              label='Integrated Position')
    c_ax.plot(c_pos_df['timestamp_s'], c_pos_df[c_x], '-', label='Actual Pose')
    c_ax.legend()
    c_ax.set_title(c_x)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
ax1.plot(la_df['pos_x'], la_df['pos_y'], '.-', label='Integrated Position')
ax1.plot(dict_df['pose']['x'], dict_df['pose']['y'], '+-', label='Actual Pose')
ax1.legend()
ax1.axis('equal');


# In[ ]:




