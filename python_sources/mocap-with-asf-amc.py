#!/usr/bin/env python
# coding: utf-8

# # Overview
# A notebook showing how to read the AMC/ASF data in the CMU Mocap and make simple visualizations. The notebook uses the code from https://github.com/CalciferZh/AMCParser/ to read and show the data

# In[ ]:


get_ipython().system('pip install -qq transforms3d')


# In[ ]:


import os, sys
get_ipython().system('git clone -q https://github.com/CalciferZh/AMCParser')


# In[ ]:


sys.path.append('AMCParser')
import amc_parser as amc


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


BASE_DIR = Path('..') / 'input' / 'allasfamc' / 'all_asfamc'
datasets_df = pd.DataFrame({'path': list(BASE_DIR.glob('subjects/*/*.amc'))})
datasets_df['Subject'] = datasets_df['path'].map(lambda x: x.parent.stem)
datasets_df['Activity'] = datasets_df['path'].map(lambda x: x.stem.split('_')[-1].lower())
datasets_df['asf_path'] = datasets_df['path'].map(lambda x: x.parent / (x.parent.stem + '.asf'))

datasets_df.sample(3)


# In[ ]:


datasets_df[['Subject', 'Activity']].describe()


# In[ ]:


test_rec = datasets_df.iloc[0]
print(test_rec)


# ## Parse the data
# We read the joint information from the asf (one per subject) and then apply the motions (one per movement and hundreds of frames recorded)

# In[ ]:


joints = amc.parse_asf(test_rec['asf_path'])
motions = amc.parse_amc(test_rec['path'])


# Draw a random frame

# In[ ]:


frame_idx = np.random.choice(range(len(motions)))
joints['root'].set_motion(motions[frame_idx])
joints['root'].draw()


# ## Make a nice animation

# In[ ]:


from IPython.display import FileLink
from matplotlib.animation import FuncAnimation
fig = plt.figure()
ax = Axes3D(fig)

def draw_frame(i):
    ax.cla()
    ax.set_xlim3d(-50, 10)
    ax.set_ylim3d(-20, 40)
    ax.set_zlim3d(-20, 40)
    joints['root'].set_motion(motions[i])
    c_joints = joints['root'].to_dict()
    xs, ys, zs = [], [], []
    for joint in c_joints.values():
      xs.append(joint.coordinate[0, 0])
      ys.append(joint.coordinate[1, 0])
      zs.append(joint.coordinate[2, 0])
    ax.plot(zs, xs, ys, 'b.')

    for joint in c_joints.values():
      child = joint
      if child.parent is not None:
        parent = child.parent
        xs = [child.coordinate[0, 0], parent.coordinate[0, 0]]
        ys = [child.coordinate[1, 0], parent.coordinate[1, 0]]
        zs = [child.coordinate[2, 0], parent.coordinate[2, 0]]
        ax.plot(zs, xs, ys, 'r')
out_path = 'simple_animation.gif'
FuncAnimation(fig, draw_frame, range(0, len(motions), 10)).save(out_path, 
                                                  bitrate=8000,
                                                  fps=8)
plt.close('all')
FileLink(out_path)


# In[ ]:


def get_joint_pos_dict(c_joints, c_motion):
    c_joints['root'].set_motion(c_motion)
    out_dict = {}
    for k1, v1 in c_joints['root'].to_dict().items():
        for k2, v2 in zip('xyz', v1.coordinate[:, 0]):
            out_dict['{}_{}'.format(k1, k2)] = v2
    return out_dict
motion_df = pd.DataFrame([get_joint_pos_dict(joints, c_motion) for c_motion in motions])
motion_df.to_csv('motion.csv', index=False)
motion_df.sample(3)


# In[ ]:


fig, m_axs = plt.subplots(3, 1, figsize=(20, 10))
for c_x, c_ax in zip('xyz', m_axs):
    for joint_name in joints.keys():
        if ('foot' in joint_name) or ('toes' in joint_name):
            c_ax.plot(motion_df['{}_{}'.format(joint_name, c_x)], label=joint_name)
    c_ax.set_title(c_x);
m_axs[0].legend()


# ### Clean up bumpf
# Kaggle gets upset if there are too many directories in the output

# In[ ]:


get_ipython().system('rm -rf AMCParser')


# In[ ]:




