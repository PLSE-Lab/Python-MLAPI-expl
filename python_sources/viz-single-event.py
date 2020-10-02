#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[30]:


import trackml


# In[31]:


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# In[32]:



def plot_volumes(title, df):
    """ Plot each volume in a subplot, each layer with different color.
    """
    figsize = (36, 5)
    fig = plt.figure(figsize=figsize)    
    
    volume_ids = [7, 8,  9, 12, 13, 14, 16, 17, 18]
    for volume_idx, volume_id in enumerate(volume_ids):
        ax = fig.add_subplot('1{}{}'.format(len(volume_ids), volume_idx+1), projection='3d')
        df_volume = df[df.volume_id==volume_id]
        ax.scatter(df_volume.x, df_volume.y, df_volume.z, c=df_volume.layer_id)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(-3000, 3000)
        ax.set_title('event {} - volume {}'.format(title, volume_id))
        #ax.view_init(0, 0)
    return ax


# In[33]:


def plot_xz(df, color_by='layer_id'):
    figsize = (10, 10)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    axes[0].scatter(df.x, df.z, c=df[color_by])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    
    axes[1].scatter(df.x, df.y, c=df[color_by])
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    


# In[34]:


from trackml.dataset import load_dataset

idx = 0
for event_id, hits, cells, particles, truth in load_dataset('../input/train_1'):
    break


# In[20]:


plot_volumes(event_id, hits)


# In[21]:


plot_xz(hits[hits.volume_id.isin([7,8,9])])


# In[22]:


plot_xz(hits[hits.volume_id.isin([12,13,14])])


# In[24]:


plot_xz(hits[hits.volume_id.isin([16,17,18])])


# In[35]:


# join hits with truth so we can color by particle :-)
df = hits.merge(truth, on='hit_id', how='inner')


# In[36]:


# Plot on the left is xz projection
# Plot on the right is xy projection

particle_ids = list(set(df.particle_id))
sample = df[df.particle_id.isin(particle_ids[1:50])]
plot_xz(sample, 'particle_id')


# In[37]:


figsize = (10, 10)
fig, axes = plt.subplots(nrows=1, ncols=2)
    
for track_id in range(100):
    track = df[df.particle_id==particle_ids[1+track_id]]
    axes[0].scatter(track.x, track.z)
    axes[1].plot(track.x, track.y)
    
axes[0].set_xlabel("x")
axes[0].set_ylabel("z")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")


# In[ ]:




