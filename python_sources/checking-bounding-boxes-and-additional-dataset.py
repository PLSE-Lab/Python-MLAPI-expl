#!/usr/bin/env python
# coding: utf-8

# ## First part: display bounding boxes
# This kernel is linked to the discussion thread: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/31565#174995 where I give three files with bounding boxes.
# 
# I merged all files into one called annot.tsv
# 
# I said that there could be a need to account for window resizing, so I show here what is working.
# Imports

# In[ ]:


import pandas as pd
import PIL
from PIL import Image


# In[ ]:


DATA_HOME_DIR = '../input/'
get_ipython().run_line_magic('matplotlib', 'inline')
data_path = DATA_HOME_DIR + '/' 
train_path = data_path + 'train/'
nb_full_train_samples = 1481
bb_json = {}

### dict with boxes: use this for your local verification
#j = pd.read_table('https://kaggle2.blob.core.windows.net/forum-message-attachments/174995/6330/Type_1_bbox.tsv',sep = " ",
#                header = None,
#                usecols = range(6),
#                names = ['filename','nbox','x','y','width','height'])
#j['y']=j['ymin']+j['height']
#filenames=[]
#for index, l in j.iterrows():
#     filenames.append(l['filename'])
#     bb_json[l['filename'].split('/')[-1]] = sorted(
#           [l[['height', 'width', 'x', 'y']].to_dict()],
#         key = lambda var: var['width']*var['height']
#         )
#print(l[['x','y','width','height']].to_dict())


# In[ ]:


from matplotlib import pyplot as plt
def to_plot(img):
    return np.rollaxis(img, 0, 3).astype(np.uint8)
def plot(img):
    plt.imshow(img)


# In[ ]:


def show_bb(i):
    img = PIL.Image.open(train_path+filenames[i])
    bb = bb_json[filenames[i].split('/')[-1]][0]
    plt.figure(figsize=(6,6))
    s = img.size
    plot(img)
    ax=plt.gca()
    ax.add_patch(create_rect([bb['x'],bb['y'],bb['width'],bb['height']], 'yellow'))
def create_rect(bb, color='red'):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], color=color, fill=False, lw=3)


# Since I cannot read the file on Kaggle kernel, I am manually creating the dict here.
# 
#           filename  nbox    x     y  width  height
#           Type_1\0.jpg     2  882   961   1042    1106
#           Type_1\10.jpg     1  972  2349   1052     715
#           Type_1\1013.jpg     1  606  1437    774     825
#           Type_1\1014.jpg     1  930  1090   1310    1384
#           Type_1\1019.jpg     1  620  1304    982    1168
#           Type_1\102.jpg     1  722  1486    546     495
# 

# In[ ]:


filenames = [ "Type_1/0.jpg",
      "Type_1/10.jpg",
    "Type_1/1013.jpg"]
bb_json = {}

bb_json["0.jpg"] = sorted(
           [{'x': 882,
           'y':972,
           'width':1042,
           'height': 1106
            }],
    key = lambda var: var['width']*var['height']
         )
bb_json["10.jpg"] = sorted(
           [{'x': 972,
           'y':2349,
           'width':1022,
           'height':725}]
         ,key = lambda var: var['width']*var['height'])
print(bb_json['0.jpg'][0])


# In[ ]:


show_bb(0)
show_bb(1)


# ## Second part: showing abnormal images in additional dataset

# In[ ]:


add_path = '../input/additional/'
def plot_from_path(path):
    img = PIL.Image.open(add_path+path)
    plt.figure(figsize=(6,6))
    plot(img)


# ### Timeseries of treatment

# In[ ]:


plot_from_path('Type_3/5684.jpg')
plot_from_path('Type_3/5683.jpg')
plot_from_path('Type_3/5685.jpg')
plot_from_path('Type_3/5688.jpg')


# ###Other examples of timeseries:

# In[ ]:


plot_from_path('Type_2/1816.jpg')
plot_from_path('Type_2/2946.jpg')
plot_from_path('Type_2/3803.jpg')
plot_from_path('Type_2/2971.jpg')
plot_from_path('Type_2/6893.jpg')
plot_from_path('Type_2/6894.jpg')
plot_from_path('Type_2/6892.jpg')
plot_from_path('Type_2/6891.jpg')


# ### Unrelated images: 
# Found a hand, a bag,...

# In[ ]:


plot_from_path('Type_2/1813.jpg')
plot_from_path('Type_1/746.jpg')
plot_from_path('Type_1/2030.jpg')


# ### Conclusions:
# 
# Additional dataset can easily be purged of 2/3 of images either because of duplicates (triplicates, n-plicates), unrelated images, or blurry images.

# ### Bonus
# Some advertisement for a cell phone company.

# In[ ]:


plot_from_path('Type_1/4065.jpg')

