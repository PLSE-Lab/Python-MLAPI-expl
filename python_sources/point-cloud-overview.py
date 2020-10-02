#!/usr/bin/env python
# coding: utf-8

# # Overview
# The notebook shows users the training and labeled data from the Semantic3D challenge and renders it in 3D. The commands can serve as a starting point for further analyses and training baseline models.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
all_paths = [os.path.join(path, file) for path, _, files in os.walk(top = os.path.join('..', 'input')) 
             for file in files if ('.labels' in file) or ('.txt' in file)]
label_names = {0: 'unlabeled', 1: 'man-made terrain', 2: 'natural terrain', 3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', 6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}


# In[2]:


all_files_df = pd.DataFrame({'path': all_paths})
all_files_df['basename'] = all_files_df['path'].map(os.path.basename)
all_files_df['id'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[0])
all_files_df['ext'] = all_files_df['basename'].map(lambda x: os.path.splitext(x)[1][1:])
all_files_df.sample(3)


# In[3]:


all_training_pairs = all_files_df.pivot_table(values = 'path', columns = 'ext', index = ['id'], aggfunc = 'first').reset_index()
all_training_pairs


# # Reading Functions
# These are terribly inefficient / slow / memory-intensity, but work fairly reliably and are easy to modify

# In[7]:


_, test_row = next(all_training_pairs.dropna().tail(1).iterrows())
print(test_row)
read_label_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['class'], index_col = False)
read_xyz_data = lambda path, rows: pd.read_table(path, sep = ' ', nrows = rows, names = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b'], header = None) #x, y, z, intensity, r, g, b
read_joint_data = lambda c_row, rows: pd.concat([read_xyz_data(c_row['txt'], rows), read_label_data(c_row['labels'], rows)], axis = 1)
read_joint_data(test_row, 10)


# In[8]:


get_ipython().run_cell_magic('time', '', 'full_df = read_joint_data(test_row, None)')


# # 2D Rendering

# In[ ]:


test_df = full_df[(full_df.index % 10)==0]
print(full_df.shape[0], 'rows', test_df.shape[0], 'number of filtered rows')


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig, m_axs = plt.subplots(1, 3, figsize = (20, 5))\nax_names = 'xyz'\nfor i, c_ax in enumerate(m_axs.flatten()):\n    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]\n    c_ax.scatter(test_df[plot_axes[0]],\n                test_df[plot_axes[1]],\n                c=test_df[['r', 'g', 'b']].values/255, \n                 s=1\n                )\n    c_ax.set_xlabel(plot_axes[0])\n    c_ax.set_ylabel(plot_axes[1])")


# In[ ]:


get_ipython().run_cell_magic('time', '', "max_keys = max(label_names.keys())\nfig, m_axs = plt.subplots(max_keys+1, 3, figsize = (20, 30))\nfor i, c_axs in enumerate(m_axs.T):\n    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]\n    for c_ax, (c_key, c_value) in zip(c_axs, label_names.items()):\n        c_df = test_df[test_df['class']==c_key]\n        c_ax.plot(c_df[plot_axes[0]],\n                  c_df[plot_axes[1]],\n                  '.', \n                  label = c_value\n                 )\n        c_ax.set_title('{}: {}'.format(c_value, ''.join(plot_axes)))\n        c_ax.set_xlabel(plot_axes[0])\n        c_ax.set_ylabel(plot_axes[1])\n        c_ax.axis('off')")


# In[ ]:


fig, m_axs = plt.subplots(1, 3, figsize = (30, 10))
for i, c_ax in enumerate(m_axs.T):
    plot_axes = [x for j, x in enumerate(ax_names) if j!=i]
    for (c_key, c_value) in label_names.items():
        c_df = test_df[test_df['class']==c_key]
        c_ax.plot(c_df[plot_axes[0]],
                    c_df[plot_axes[1]],
                '.', 
                  label = c_value
                    )
    c_ax.legend()
    c_ax.set_xlabel(plot_axes[0])
    c_ax.set_ylabel(plot_axes[1])


# # 3D Renderings

# In[ ]:


test_df = full_df[(full_df.index % 50)==0]
print(full_df.shape[0], 'rows', test_df.shape[0], 'number of filtered rows')


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig = plt.figure(figsize=(15,10))\nax = plt.axes(projection='3d')\nax.scatter(\n            test_df['x'], test_df['y'], test_df['z'],\n            c=test_df[['r', 'g', 'b']].values/255, s=3)  \nax.view_init(15, 165)")


# In[ ]:


ax.view_init(45, 220)
fig.savefig('3D_rendering.png', dpi = 300)
fig


# # 3D Labels

# In[ ]:


get_ipython().run_cell_magic('time', '', "fig = plt.figure(figsize=(15,10))\nax = plt.axes(projection='3d')  \nfor (c_key, c_value) in label_names.items():\n    c_df = test_df[test_df['class']==c_key]\n    ax.plot(c_df['x'], c_df['y'], c_df['z'], '.', label = c_value, alpha = 0.5)  \nax.legend()\nax.view_init(15, 165)\nfig.savefig('3d_labels.png', dpi = 300)")


# In[ ]:


ax.view_init(45, 220)
fig


# In[ ]:


get_ipython().run_cell_magic('time', '', "fig = plt.figure(figsize=(15,10))\nax = plt.axes(projection='3d')\nax.scatter(test_df['x'], test_df['y'], test_df['z'],\n           c=test_df[['r', 'g', 'b']].values/255, s=3)  \nfor (c_key, c_value) in label_names.items():\n    if c_key>0:\n        c_df = test_df[test_df['class']==c_key]\n        ax.plot(c_df['x'], c_df['y'], c_df['z'], '.', label = c_value, alpha = 0.25, markersize = 1.0)  \nax.legend()\nax.view_init(45, 220)\nfig.savefig('3d_labels_overlay.png', dpi = 600)")


# In[ ]:




