#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings, hope you all are having an amazing day. I've hand-annotated a bunch of keypoints on 1000 images of humpback whale flukes. I hope you can use this data to build some cool models! Please see the dataset for a more detailed description.

# In[ ]:


from PIL import Image # Use PIL's image reading to always read the image into RGB format (for matplotlib to display colors correctly)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle # we will be drawing circles to indicate the keypoints

import pandas as pd # reading/parsing cvs file

img_dir = '../input/humpback_fluke_keypoints/'


# In[ ]:


def plot_keypoints(filename, points):
    """Given a filename and a list of keypoints, display an image
    with keypoints highlighted
    
    Args:
        filename -- name of jpg
        points -- list of tuples where each tuple is: (x_coord, y_coord)
    """
    img = Image.open(filename)

    fig, ax = plt.subplots(1, figsize=(20, 20))
    #fig.set_size_inches(10)
    ax.set_aspect('equal')

    for coords in points:
        ax.add_patch(Circle(coords, 10, linewidth='2', edgecolor='yellow', facecolor='red'))
    ax.imshow(img)
    plt.show()


# In[ ]:


df = pd.read_csv(img_dir + 'keypoints.csv')


# In[ ]:


for entry in df.sample(5).values:
    filename = entry[0]
    print(filename)
    coords = entry[1:]
    keypoints = []
    for i in range(len(coords) // 2):
        keypoints.append((coords[2*i], coords[(2*i)+1]))
    plot_keypoints(img_dir+filename, keypoints)


# ## Conclusion
# Happy Kaggling!
