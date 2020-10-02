#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.io import imread
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = (8, 8)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})


# # Load and Organize Measurements

# In[2]:


sem_dir = Path('..') / 'input' / '3dsem'
sem_files_df = pd.DataFrame({'path': list(sem_dir.glob('*.*'))})
sem_files_df['id'] = sem_files_df['path'].map(lambda x: x.stem.lower())
sem_files_df['suffix'] = sem_files_df['path'].map(
    lambda x: x.suffix.lower()[1:])
sem_files_df['index'] = sem_files_df['id'].map(lambda x: x[-2:])
sem_files_df['prefix'] = sem_files_df['id'].map(lambda x: x[:5])
sem_files_df['simple_name'] = sem_files_df.apply(
    lambda x: '{prefix}-{suffix}'.format(**x), 1)
img_sem_df = sem_files_df[sem_files_df['suffix'].map(
    lambda x: x in ['jpg', 'tif'])]
img_sem_df.sample(3)


# In[3]:


img_summary_df = img_sem_df.pivot_table(index='index',
                                        columns='prefix',
                                        values='path',
                                        aggfunc='first').\
    reset_index(drop=True)
img_summary_df


# In[4]:


exp_list = img_summary_df.columns.tolist()
print(exp_list)


# In[20]:


c_exp = exp_list[-1]
c_img_df = img_summary_df[[c_exp]].dropna().copy()
c_img_df.columns = ['path']
c_img_df['base'] = c_img_df['path'].map(lambda c_path: c_path.stem)
c_img_df['img'] = c_img_df['path'].map(
    lambda c_path: imread(c_path, as_gray=True)[:-100])
c_img_df.sample(2)


# In[21]:


fig, m_axs = plt.subplots(1, len(c_img_df), figsize=(5*len(c_img_df), 5))
for c_ax, (_, c_row) in zip(m_axs, c_img_df.iterrows()):
    c_ax.imshow(c_row['img'])
    c_ax.set_title(c_row['base'])


# # Calculate Descriptors

# In[22]:


from skimage.feature import (corner_harris,
                             corner_peaks, ORB)


# In[ ]:


descriptor_extractor = ORB(n_keypoints=300)


def extract_points(in_img):
    descriptor_extractor.detect_and_extract(in_img)
    return {'keypoints': descriptor_extractor.keypoints,
            'descriptors': descriptor_extractor.descriptors}


c_img_df['descriptor'] = c_img_df['img'].map(extract_points)
c_img_df['keypoints'] = c_img_df['descriptor'].map(lambda x: x['keypoints'])
c_img_df['descriptors'] = c_img_df['descriptor'].map(
    lambda x: x['descriptors'])
c_img_df.sample(1)


# ## Match Points
# 
# Here we just match points on the descriptors

# In[ ]:


from skimage.feature import match_descriptors, plot_matches


# In[ ]:


fig, m_axs = plt.subplots(len(c_img_df)-1, 2, figsize=(15, 5*len(c_img_df)))
for (ax3, ax2), (_, c_row), (_, n_row) in zip(m_axs, c_img_df.iterrows(),
                                              c_img_df.shift(-1).dropna().iterrows()):
    c_matches = match_descriptors(c_row['descriptors'],
                                  n_row['descriptors'], cross_check=True)

    plot_matches(ax3,
                 c_row['img'], n_row['img'],
                 c_row['keypoints'], n_row['keypoints'],
                 c_matches)

    ax2.plot(c_row['keypoints'][:, 0],
             c_row['keypoints'][:, 1],
             '.',
             label=c_row['base'])

    ax2.plot(n_row['keypoints'][:, 0],
             n_row['keypoints'][:, 1],
             '.',
             label=n_row['base'])

    for i, (c_idx, n_idx) in enumerate(c_matches):
        x_vec = [c_row['keypoints'][c_idx, 0], n_row['keypoints'][n_idx, 0]]
        y_vec = [c_row['keypoints'][c_idx, 1], n_row['keypoints'][n_idx, 1]]
        dist = np.sqrt(np.square(np.diff(x_vec))+np.square(np.diff(y_vec)))
        alpha = np.clip(50/dist, 0, 1)

        ax2.plot(
            x_vec,
            y_vec,
            'k-',
            alpha=alpha,
            label='Match' if i == 0 else ''
        )

    ax2.legend()

    ax3.set_title(r'{} $\rightarrow$ {}'.format(c_row['base'], n_row['base']))


# # Filtering Matches
# We can filter the matches by excluding the matches which are too far away (we expect mostly small changes)

# In[11]:


last_idx = {}
track_list = []
idx_offset = c_img_df['keypoints'].map(len).sum()+1
for frame_idx, ((_, c_row), (_, n_row)) in enumerate(
    zip(c_img_df.iterrows(),
        c_img_df.shift(-1).dropna().iterrows())):

    c_matches = match_descriptors(c_row['descriptors'],
                                  n_row['descriptors'], cross_check=True)
    next_idx = {}

    for i, (c_idx, n_idx) in enumerate(c_matches):
        x_vec = [c_row['keypoints'][c_idx, 0], n_row['keypoints'][n_idx, 0]]
        y_vec = [c_row['keypoints'][c_idx, 1], n_row['keypoints'][n_idx, 1]]
        dist = np.sqrt(np.square(np.diff(x_vec))+np.square(np.diff(y_vec)))[0]
        tracked = c_idx in last_idx
        if c_idx not in last_idx:
            idx_offset += 1
            cur_idx = idx_offset
        else:
            cur_idx = last_idx[c_idx]
        next_idx[n_idx] = cur_idx
        track_list += [{
            'frame_idx': frame_idx,
            'frame': c_row['base'],
            'idx': cur_idx,
            'tracked': tracked,
            'x': c_row['keypoints'][c_idx, 0],
            'y': c_row['keypoints'][c_idx, 1],
            'xy_coord': c_row['keypoints'][c_idx],
            'dist': dist
        }]
    last_idx = next_idx


# In[12]:


track_df = pd.DataFrame(track_list)
track_df['dist'].hist(figsize=(5, 5))


# In[13]:


cut_dist = np.quantile(track_df['dist'], 0.80)
print(cut_dist)
well_tracked_df = track_df[track_df['dist'] <= cut_dist]
pivot_track_df = well_tracked_df.pivot_table(
    index=['idx', 'frame', 'frame_idx'],
    values=['x', 'y'],
    aggfunc='first'
).\
    reset_index()
pivot_track_df.sample(3)


# In[14]:


full_tracked = pivot_track_df.    pivot(index='idx',
          columns='frame_idx',
          values=['x', 'y']).\
    dropna()
full_tracked.sample(3)


# # Change to Angle Coordinates
# 
# ```
# This dataset contains four 2D images from a biological sample called 
# "pollen grain from Brassica rapa" and its 3D point cloud (.ply format) 
# which could be easily converted to a surface model by MeshLab. 
# The set of 2D images were obtained by tilting the specimen stage 3 
# degrees from one to the next in the image sequence.
# 
# http://selibcv.org/3dsem/
# ```

# # Fit Quadratic to the Values

# In[15]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[16]:


quad_reg = make_pipeline(PolynomialFeatures(2),
                         LinearRegression())


# # Fit Quadratic by Region

# In[17]:


new_tracked = full_tracked.copy()
new_tracked['x_0'] = new_tracked['x'][0]
new_tracked['y_0'] = new_tracked['y'][0]
new_tracked['y_vec'] = new_tracked['y'].apply(lambda x: x.tolist(), 1)
new_tracked['x_vec'] = new_tracked['x'].apply(lambda x: x.tolist(), 1)
new_tracked['x_bin'] = pd.qcut(new_tracked['x_0'], 4).cat.codes
new_tracked['y_bin'] = pd.qcut(new_tracked['y_0'], 4).cat.codes
new_tracked.sample(3)


# In[18]:


x_points = np.concatenate(c_img_df['keypoints'].map(lambda x: x[:, 0]).tolist(), 0)
y_points = np.concatenate(c_img_df['keypoints'].map(lambda x: x[:, 1]).tolist(), 0)
x_grid = np.linspace(x_points.min(), x_points.max(), 100)
grid_space = x_grid[1]-x_grid[0]
y_grid = np.arange(y_points.min(), y_points.max(), grid_space)


# In[19]:


from scipy.interpolate import interp2d
fig, m_axs = plt.subplots(4, 2, figsize=(15, 15))
(ax0, ax1) = m_axs[0]
ax0.imshow(c_img_df['img'].iloc[0][::-1])
theta = 3*np.arange(4)
pos_dict = {}
for bin_dex, c_rows in new_tracked.groupby(['x_bin', 'y_bin']):
    ax1.plot(c_rows['y_0'], c_rows['x_0'], '.', label='{}'.format(bin_dex))
    x_vec = np.stack(c_rows['x_vec'].values)
    y_vec = np.stack(c_rows['y_vec'].values)
    quad_reg = make_pipeline(PolynomialFeatures(2),
                         LinearRegression())
    
    xy_vec = np.concatenate([
        x_vec[:, :].reshape((-1, 1)),
        y_vec[:, :].reshape((-1, 1))], 1)
    theta_x = np.repeat(theta.reshape((1, -1)), 
                        x_vec.shape[0], axis=0)
    quad_reg.fit(theta_x.reshape((-1, 1)), xy_vec)
    
    x_coef = quad_reg.steps[-1][1].coef_[0, :]
    y_coef = quad_reg.steps[-1][1].coef_[1, :]
    pos_dict[(c_rows['x_0'].mean(), c_rows['y_0'].mean())]  = (x_coef, y_coef)
    
ax1.axis('equal')
p_keys = list(pos_dict.keys())
for (ax2, ax3), i in zip(m_axs[1:], range(3)):
    x_lin_func = interp2d(
        [y for x,y in p_keys],
        [x for x,y in p_keys],
        [pos_dict[k][0][i] for k in p_keys])

    ax2.imshow(x_lin_func(y_grid, x_grid))
    ax2.set_title('$x^{}$ Term'.format(i))
    ax2.axis('off')
    
    y_lin_func = interp2d(
        [y for x,y in p_keys],
        [x for x,y in p_keys],
        [pos_dict[k][1][i] for k in p_keys])

    ax3.imshow(y_lin_func(y_grid, x_grid))
    ax3.set_title('$y^{}$ Term'.format(i))
    ax3.axis('off')


# In[ ]:




