#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from subprocess import check_output
print(check_output(["ls", "../input/train"]).decode("utf8"))
from glob import glob
basepath = '../input/train/'


# In[ ]:


def getCervixImagesTraining():
   


   all_cervix_images = []

   for path in sorted(glob(basepath + "*")):
       #print(path)
       cervix_type = path.split("/")[-1]
       # print(cervix_type)
       cervix_images = sorted(glob(basepath + cervix_type + "/*"))
       all_cervix_images = all_cervix_images + cervix_images
   all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
   all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
   all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
   return all_cervix_images
all_cervix_images=getCervixImagesTraining()
all_cervix_images.head()


# In[ ]:


def plotbasicGraph():
    print('We have a total of {} images in the whole dataset'.format(all_cervix_images.shape[0]))
    type_aggregation = all_cervix_images.groupby(['type', 'filetype']).agg('count')
    print(type_aggregation)
    type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images.shape[0], axis=1)
    print(type_aggregation_p)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    type_aggregation.plot.barh(ax=axes[0])
    axes[0].set_xlabel("image count")
    type_aggregation_p.plot.barh(ax=axes[1])
    axes[1].set_xlabel("training size fraction") 

plotbasicGraph()


# In[ ]:


fig = plt.figure(figsize=(12,8))

i = 1
for t in all_cervix_images['type'].unique():
    ax = fig.add_subplot(1,3,i)
    i+=1
    f = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[0]
    plt.imshow(plt.imread(f))
    plt.title('sample for cervix {}'.format(t))


# In[ ]:


from collections import defaultdict

images = defaultdict(list)

for t in all_cervix_images['type'].unique():
    sample_counter = 0
    for _, row in all_cervix_images[all_cervix_images['type'] == t].iterrows():
        #print('reading image {}'.format(row.imagepath))
        try:
            img = imread(row.imagepath)
            sample_counter +=1
            images[t].append(img)
        except:
            print('image read failed for {}'.format(row.imagepath))
        if sample_counter > 35:
            break


# In[ ]:


dfs = []
for t in all_cervix_images['type'].unique():
    t_ = pd.DataFrame(
        {
            'nrows': list(map(lambda i: i.shape[0], images[t])),
            'ncols': list(map(lambda i: i.shape[1], images[t])),
            'nchans': list(map(lambda i: i.shape[2], images[t])),
            'type': t
        }
    )
    print(t_)
    dfs.append(t_)

shapes_df = pd.concat(dfs, axis=0)
shapes_df_grouped = shapes_df.groupby(by=['nchans', 'ncols', 'nrows', 'type']).size().reset_index().sort_values(['type', 0], ascending=False)
shapes_df_grouped


# In[ ]:


shapes_df_grouped['size_with_type'] = shapes_df_grouped.apply(lambda row: '{}-{}-{}'.format(row.ncols, row.nrows, row.type), axis=1)
shapes_df_grouped = shapes_df_grouped.set_index(shapes_df_grouped['size_with_type'].values)
shapes_df_grouped['count'] = shapes_df_grouped[[0]]

plt.figure(figsize=(10,8))
#shapes_df_grouped['count'].plot.barh(figsize=(10,8))
sns.barplot(x="count", y="size_with_type", data=shapes_df_grouped)


# In[ ]:


def transform_image(img, rescaled_dim, to_gray=False):
    resized = cv2.resize(img, (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR)
    #print(resized)
    if to_gray:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype('float')
    else:
        resized = resized.astype('float')

    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    #print(normalized)
    normalized.shape
    timg = normalized.reshape(1, np.prod(normalized.shape))
    #print(timg)
    return timg/np.linalg.norm(timg)


# In[ ]:


rescaled_dim = 100
all_images = []
all_image_types = []

for t in all_cervix_images['type'].unique():
    all_images = all_images + images[t]
    all_image_types = all_image_types + len(images[t])*[t]
    #print(len(images[t])*[t])

#print(all_images  )
#print(all_image_types )
transform_image(all_images[0], rescaled_dim)

    
 


# In[ ]:





# In[ ]:





# In[ ]:




