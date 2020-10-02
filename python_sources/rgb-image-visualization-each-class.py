#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# The hypothesis: if we visualize a few examples for each class, we will get the visual patterns for each class.
# 
# Also, we will get the method for merge channels into one image

# ## Imports and constants:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image


data_train_dir = r'../input/train'
answer_file_path = r'../input/train.csv'

channels = ['_yellow', '_red', '_green', '_blue']

index_class_dict = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}


# ## Load data:

# In[ ]:


train_df = pd.read_csv(answer_file_path)
train_df.head()


# ## Create cols for each class:

# In[ ]:


train_df[f'target_vec'] = train_df['Target'].map(lambda x: list(map(int, x.strip().split())))
for i in range(28):
    train_df[f'{index_class_dict[i]}'] = train_df['Target'].map(
             lambda x: 1 if str(i) in x.strip().split() else 0)
train_df.head()


# ## Get part of data with specific class:

# In[ ]:


class_index = 1  # Nuclear membrane
current_part = train_df[train_df[index_class_dict[class_index]] == 1]
print(f'shape before {train_df.shape}, shape after {current_part.shape}')
current_part.head()


# ## Functions for visualization:

# In[ ]:


def make_rgb_image_from_four_channels(channels: list, image_width=512, image_height=512) -> np.ndarray:
    """
    It makes literally RGB image from source four channels, 
    where yellow image will be yellow color, red will be red and so on  
    """
    rgb_image = np.zeros(shape=(image_height, image_width, 3), dtype=np.float)
    yellow = np.array(Image.open(channels[0]))
    # yellow is red + green
    rgb_image[:, :, 0] += yellow/2   
    rgb_image[:, :, 1] += yellow/2
    # loop for R,G and B channels
    for index, channel in enumerate(channels[1:]):
        current_image = Image.open(channel)
        rgb_image[:, :, index] += current_image
    # Normalize image
    rgb_image = rgb_image / rgb_image.max() * 255
    return rgb_image.astype(np.uint8)


# In[ ]:


def visualize_part(start_class_index=0, nrows=4, ncols=3):
    """
    Visualize the part of classes, started from class with index start_class_index,
    make nrows classes, ncols examples for each one
    """
    fig, ax = plt.subplots(nrows = nrows, ncols=ncols, figsize=(15, 25))
    for class_index in range(nrows):
        current_index = class_index + start_class_index
        for sample in range(ncols):
            current_part = train_df[train_df[index_class_dict[current_index]] == 1] 
            # 0 index is id
            random_index = np.random.choice(current_part.values.shape[0], 1, replace=False)
            # random line from data with selected class
            current_line = current_part.values[random_index][0]
            image_names = [os.path.join(data_train_dir, current_line[0]) 
                           + x + '.png' for x in channels]
            rgb_image = make_rgb_image_from_four_channels(image_names)
            # text annotations, main title and subclasses (may be empty in case one label)
            main_class = index_class_dict[current_index]+'\n'
            # 2 index is vector with classes, split version of Target col
            other_classes = [index_class_dict[x] for x in current_line[2] 
                             if x != (current_index)]
            subtitle = ', '.join(other_classes)
            # show image
            ax[class_index, sample].set_title(main_class, fontsize=18)
            ax[class_index, sample].text(250, -10, subtitle, 
                                         fontsize=14, horizontalalignment='center')
            ax[class_index, sample].imshow(rgb_image)
            ax[class_index, sample].set_xticklabels([])
            ax[class_index, sample].set_yticklabels([])
            ax[class_index, sample].tick_params(left=False, bottom=False)


# ## Let's visualize 3 examples for each class:

# In[ ]:


visualize_part(0)


# In[ ]:


visualize_part(4)


# In[ ]:


visualize_part(8)


# In[ ]:


visualize_part(12)


# In[ ]:


visualize_part(16)


# In[ ]:


visualize_part(20)


# In[ ]:


visualize_part(24)


# ## Conclusion
# In fact, I don't see any special features for each class and don't understand what differences really are. Provably, addition of yellow channel is a bad idea (yellow color is red + green) or we have to analyse only green one, but I did't get good results in my experiments yet. I guess I will allow to network choose the correct features.
