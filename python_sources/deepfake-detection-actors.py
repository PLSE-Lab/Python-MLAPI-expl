#!/usr/bin/env python
# coding: utf-8

# This notebook contains a dataset of the real videos and actors. The dataset was manually labelled.
# The notebook will show you a sample of each actor in the dataset (386 actors). There are a few more actors that this, because for each video I labelled the main actor.
# Some videos contain more than one actor. The other actors in the video are usually appearing in multiple folders, but the main actors are in a single folder as far as I could see.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# Any results you write to the current directory are saved as output.

from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt


# Load the actor database. 
# The actor identifier is a tuple, whose first element is the folder number and the second element is actor rank in that folder.

# In[ ]:


ROOT = '/kaggle/input/deepfake-detection-challenge-actors/actors/'

videos_actor_df = pd.read_hdf(os.path.join(ROOT, 'videos_actor.h5'))
videos_actor_df


# Let's see a sample of each actor:

# In[ ]:


actors = videos_actor_df['actor'].unique()
num_actors = len(actors)
nc = 5
nr = int(np.ceil(num_actors / nc))

fig, ax = plt.subplots(nrows = nr, ncols=nc, figsize = (nc * 3, nr * 3))
fig.tight_layout()
r_idx = 0
c_idx = 0
for actor in actors:
    sample_path = os.path.join(ROOT, 'samples', f'{actor[0]}_{actor[1]}.jpg')
    img = Image.open(sample_path)
    ax[r_idx][c_idx].imshow(img)
    ax[r_idx][c_idx].set_title(actor)
    
    c_idx += 1
    if c_idx >= nc:
        c_idx = 0
        r_idx += 1


# From what I can tell, face swap actors are mixed from many folders, which made it very challenging to generate a validation dataset that didn't have an actor as either the source or target face.
# Have a look at the following examples of what I'm talking about. For each example, the first image is the real video, the second is the fake and the third is a real video of the actor which I think the swap came from. The second row, I mask out the outer head to make it easier to see the actor identities, and I show the actor IDs above the photos.

# In[ ]:


import glob
ex = sorted(glob.glob(os.path.join(ROOT, 'face_swaps', 's*.jpg')))
ex_imgs = [Image.open(v) for v in ex]

ex_imgs[0]


# In[ ]:


ex_imgs[1]


# In[ ]:


ex_imgs[2]


# In[ ]:


ex_imgs[3]


# In[ ]:


ex_imgs[4]


# In[ ]:


ex_imgs[5]


# In[ ]:


ex_imgs[6]

