#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
from ast import literal_eval
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go


# ## Read data

# In[ ]:


get_ipython().system('ls ../input/global-wheat-detection/')


# In[ ]:


root_path = "../input/global-wheat-detection/"
train_folder = os.path.join(root_path, "train")
test_folder = os.path.join(root_path, "test")
train_csv_path = os.path.join(root_path, "train.csv")
sample_submission = os.path.join(root_path, "sample_submission.csv")


# In[ ]:


train_df = pd.read_csv(train_csv_path)


# In[ ]:


train_df.head()


# ## Compute Area of bbox

# In[ ]:


import ast
train_df['bbox_area'] = train_df.bbox.apply(lambda x: ast.literal_eval(x)[2] * ast.literal_eval(x)[3])


# # Sources
# **1. How many sources are there and what's the distribution wheat heads looks like w.r.t sources?**  
# There are 7 sources, **ethz_1** and **arvalid_1** being the top post ones  
#   
# **2. What's the distribution of images w.r.t sources?**  
# - **ethz_1** has 747 images with 51k wheat heads, that's ~69 wheat heads per image. That's a lot!  
# - It looks like **ethz_1** has the smallest wheat heads too, with median bbox area of 4050, which is lowest in all sources.
# - It makes sense, most likely either **ethz_1** consists of baby wheat heads and a lot of them together in the image or the angle at which the are taken makes image with many wheat heads.
#   
# **3. Are there images linked to same source? Ideally that should not be the case**  
# That's not the case.

# In[ ]:


source_vc = train_df.source.value_counts()

print(f"There are {len(source_vc)} sources.\n")
print(source_vc)


# In[ ]:


summary_df = train_df.groupby(['source']).agg(unq_images=('image_id','nunique'), wheat_heads=('image_id','size'), mean_bbox_area=('bbox_area','mean'), median_bbox_area=('bbox_area','median'))
summary_df.reset_index(inplace=True,drop=False)
summary_df['mean_bbox_area'] = np.round(summary_df.mean_bbox_area, 2)
summary_df['median_bbox_area'] = np.round(summary_df.median_bbox_area, 2)
summary_df['mean_wheat_heads_per_img'] = np.round(summary_df.wheat_heads / summary_df.unq_images, 2)

summary_df


# In[ ]:


summary_df = train_df.groupby(['image_id']).agg(unq_source=('source','nunique'))
summary_df.reset_index(inplace=True,drop=False)
print(f"Images with >1 sources : {summary_df.loc[summary_df.unq_source > 1].shape[0]}")


# # Images
# 
# 1- Check if Width and Height differ for images, or are they same for all images?  
# Yes, all images are 1024 by 1024  
# 
# 2- 

# In[ ]:


print(f"Unique Widths: {pd.unique(train_df.width)}")
print(f"Unique HeightsA: {pd.unique(train_df.height)}")


# In[ ]:



summary_df = train_df.groupby(['source','image_id']).agg(wheat_heads=('image_id','size'), mean_bbox_area=('bbox_area','mean'))
summary_df.reset_index(inplace=True,drop=False)
summary_df




colors = ['rgba(93, 164, 214, 1.0)', 'rgba(255, 144, 14, 1.0)', 'rgba(44, 160, 101, 1.0)',
          'rgba(255, 65, 54, 1.0)', 'rgba(207, 114, 255, 1.0)', 'rgba(127, 96, 0, 1.0)', 'rgba(200, 200, 0, 1.0)']

fig = go.Figure()
index = 0
for cls in pd.unique(summary_df.source):
    fig.add_trace(go.Box(y=summary_df.loc[summary_df.source==cls,'wheat_heads'], name=cls,
                marker_color = colors[index]))
    index = index + 1

fig.update_yaxes(range=[0, 100])
fig.update_layout(title='Box-Plot Wheatheads w.r.t source',xaxis_title="Source",yaxis_title="No. Wheatheads per image")
fig.show()



fig = go.Figure()
index = 0
for cls in pd.unique(summary_df.source):
    fig.add_trace(go.Box(y=summary_df.loc[summary_df.source==cls,'mean_bbox_area'], name=cls,
                marker_color = colors[index]))
    index = index + 1

fig.update_yaxes(range=[0, 40000])
fig.update_layout(title='Box-Plot BBox Area w.r.t source',xaxis_title="Source",yaxis_title="Mean BBox Area")
fig.show()


# # Visualizing images
# 
# Copied code from https://www.kaggle.com/devvindan/wheat-detection-eda
# 

# In[ ]:


def show_images(images, num = 5):
    
    images_to_show = np.random.choice(images, num)

    for image_id in images_to_show:

        image_path = os.path.join(train_folder, image_id + ".jpg")
        image = Image.open(image_path)

        # get all bboxes for given image in [xmin, ymin, width, height]
        bboxes = [literal_eval(box) for box in train_df[train_df['image_id'] == image_id]['bbox']]

        # visualize them
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:    
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)

        plt.figure(figsize = (15,15))
        plt.imshow(image)
        plt.show()


# In[ ]:


for source in pd.unique(train_df.source):
    print(f"Showing images for {source}:")
    show_images(train_df[train_df['source'] == source]['image_id'].unique(), num = 1)


# In[ ]:




