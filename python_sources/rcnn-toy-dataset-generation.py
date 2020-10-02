#!/usr/bin/env python
# coding: utf-8

# # Generating synthetic temporal data for testing RCNNs
# Inorder to show the application of recurrent convolutional neural networks (RCNNs) we need data that is only correctly predicted when a memory of past events is present.
# 
# We will build data were predicting the next frame is ambigious given a single instance, but is predictable when multiple chronological instances are used.  
# 
# ## The data
# The data will of a moving box:
# ![box data](https://raw.githubusercontent.com/ZackAkil/understanding-recurrent-convolutional-neural-networks/master/images/box.gif)
# 
# ## The Task
# To predict the next position of the box:
# ![box predict](https://raw.githubusercontent.com/ZackAkil/understanding-recurrent-convolutional-neural-networks/master/images/box_predict.png)
# 
# ## The need for temporal memory in predicting the box
# As you can probebly see, the next position of the box is abigious based on a single frame, i.e you have no gauranteed way of correctly predicting the direction of the box based on a single frame:
# ![box predict](https://raw.githubusercontent.com/ZackAkil/understanding-recurrent-convolutional-neural-networks/master/images/abig_predict.png)
# 
# **However** if your model maintains a memory of the previous input, it can use that information to predict correctly:
# ![box memory](https://raw.githubusercontent.com/ZackAkil/understanding-recurrent-convolutional-neural-networks/master/images/temp_box.png)
# 
# This is what an RNN does.
# 
# Lets create this synthetic data:

# ### Inport libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## Functions to draw box in matrix

# In[50]:


# %%writefile box_gen.py

import numpy as np
from PIL import Image, ImageDraw

FRAME_SIZE = [5,50]
BOX_WIDTH = 3


def get_rect(x, y, width, height):
    rect = np.array([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1), (0, 0)])
    offset = np.array([x, y])
    transformed_rect = rect + offset
    return transformed_rect

def get_array_with_box_at_pos(x):
    data = np.zeros(FRAME_SIZE)
    img = Image.fromarray(data)
    draw = ImageDraw.Draw(img)
    rect = get_rect(x=x, y=1, width=BOX_WIDTH, height=BOX_WIDTH)
    draw.polygon([tuple(p) for p in rect], fill=1)
    new_data = np.asarray(img)
    return new_data


# ## Use functions to generate data sequence

# In[41]:


sway_offset = 1
sway_start = sway_offset
sway_end = (FRAME_SIZE[1]-1) - BOX_WIDTH
sway_range = sway_end - sway_offset
sway_start, sway_end, sway_range


# ### Create movement pattern

# In[42]:


DATA_POINTS = 100


# In[58]:


base = (np.arange(DATA_POINTS)/DATA_POINTS)* 6 *np.pi
sined = (np.sin(base) + 1 )/2
plt.scatter(base, sined)
plt.show()


# In[59]:


def sin_to_pos(sin_val):
    return (sin_val*sway_range)+sway_offset


# In[60]:


frames = []

print_every_n_frames = DATA_POINTS//10
for i,t in enumerate(sined):
    frame = get_array_with_box_at_pos(sin_to_pos(t))
    if(i % print_every_n_frames)==0:
        plt.imshow(frame, interpolation='nearest')
        plt.show()
    frames.append(frame)


# ## Export data

# In[61]:


y = sin_to_pos(sined[1:])
X = frames[:-1]

len(X), len(y)


# In[62]:


data = {'X':X, 'y':y}


# In[63]:


from sklearn.externals import joblib


# In[64]:


joblib.dump(data, 'sythetic_data.pkl')


# In[ ]:




