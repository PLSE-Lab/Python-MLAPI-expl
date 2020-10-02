#!/usr/bin/env python
# coding: utf-8

# # Making Dough
# Let's get our hands dirty and make some dough. 
# 
# 
# <p align="center">
#   <img width="460" height="300" src="https://cdn-a.william-reed.com/var/wrbm_gb_food_pharma/storage/images/7/0/8/1/491807-1-eng-GB/Freezing-dough-Understand-impact-on-glutenin-protein-say-researchers_wrbm_large.jpg">
# </p>
# 
# - Aim: Detect wheat heads (in form of multiple bounding boxes) of wheat plant images, all of size 1024x1024. 
# - Data source: The [Global WHEAT dataset](http://www.global-wheat.com/2020-challenge/).
# - Metric: Mean average precision at different intersection over union (IoU) thresholds. Thresholds vary from 0.5 to 0.75 with a step size of 0.05.
# - Time: We have 3 months for now.

# In[ ]:


import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set(style="darkgrid")


# In[ ]:


DIR = "../input/global-wheat-detection/"
TRAIN = "train.csv"
SUBMISSION = "sample_submission.csv"
TRAIN_IMAGES = "train"
TEST_IMAGES = "test"
WIDTH = 1024
HEIGHT = 1024

TRAIN_IMAGES = [os.path.join(DIR, "train", fname) for fname in os.listdir(os.path.join(DIR, "train"))]
TEST_IMAGES = [os.path.join(DIR, "test", fname) for fname in os.listdir(os.path.join(DIR, "test"))]

train_df = pd.read_csv(os.path.join(DIR, TRAIN))
submission_df = pd.read_csv(os.path.join(DIR, TRAIN))


# In[ ]:


print(train_df.shape)
print(f"Total training images: {len(TRAIN_IMAGES)}")
print(f"Total test images: {len(TEST_IMAGES)}")
train_df.head()


# Only 3422 train and 10 test images.

# # DataSource

# In[ ]:


plt.figure(figsize=(12, 6))
sns.countplot(train_df.source)
plt.show()


# In[ ]:


train_df.source.value_counts()


# ## Number of bounding boxes in images provided by each source

# In[ ]:


bbox_counts_by_source = train_df.groupby(["source"]).apply(lambda x:x["image_id"].value_counts().mean())


# In[ ]:


plt.figure(figsize=(10, 6))
bbox_counts_by_source.plot(kind='bar')
plt.show()


# In[ ]:


bbox_counts_by_source


# Wow, not only ethz_1 have given the maximum number of test images, but they also provide more number of bounding boxes per image. I expect their data to be of very high quality. Maybe I should fine tune my model only on their data.

# # Number of boxes per image

# In[ ]:


box_count = train_df["image_id"].value_counts()
print(f"Min boxes: {box_count.min()}")
print(f"Max boxes: {box_count.max()}")
print(f"Mean boxes: {box_count.mean()}")
print(f"Std boxes: {box_count.std()}")


# In[ ]:


plt.figure(figsize=(10, 6))
sns.distplot(box_count.values)
plt.show()


# # Area/Location of boxes

# In[ ]:


bbox = lambda bbox: [float(x) for x in bbox[1:-1].split(",")]
train_df.bbox = train_df.bbox.apply(bbox)
train_df['xmin'] = train_df.bbox.apply(lambda x: x[0])
train_df['ymin'] = train_df.bbox.apply(lambda x: x[1])
train_df['width'] = train_df.bbox.apply(lambda x: x[2])
train_df['height'] = train_df.bbox.apply(lambda x: x[3])


# In[ ]:


area_percent = train_df['width']*train_df['height'] / (WIDTH*HEIGHT)
plt.figure(figsize=(10, 6))
plt.title("Area % for whole dataset.")
print(f"Min area: {area_percent.min()}%")
print(f"Max area: {area_percent.max()}%")
print(f"Mean area: {area_percent.mean()}%")
print(f"Std area: {area_percent.std()}%")
sns.distplot(area_percent)
plt.show()


# In[ ]:


area_per_image = train_df.groupby("image_id").apply(lambda x: (x["width"]*x["height"]).sum()/(WIDTH*HEIGHT))
plt.figure(figsize=(10, 6))
plt.title("Area % for each image.")
print(f"Min area per image: {area_per_image.min()}%")
print(f"Max area per image: {area_per_image.max()}%")
print(f"Mean area per image: {area_per_image.mean()}%")
print(f"Std area per image: {area_per_image.std()}%")
sns.distplot(area_per_image)
plt.show()


# # Height and width distribution of bounding boxes

# In[ ]:


plt.figure(figsize=(10, 6))
sns.distplot(train_df.height)
plt.show()

plt.figure(figsize=(10, 6))
sns.distplot(train_df.width)
plt.show()


# # Visualize images and their bounding boxes

# In[ ]:


def get_boxes(df):
    xmins, ymins, widths, heights = df['xmin'],df['ymin'], df['width'], df['height']
    ps = []
    for i in range(len(xmins)):
        p = patches.Rectangle((xmins.iloc[i], ymins.iloc[i]),widths.iloc[i], heights.iloc[i], linewidth=2, edgecolor='c', facecolor='none')
        ps.append(p)
    return ps  

def show_img_bbox(rows=2, columns=2, source=None):
    """
    source: Random selection only from images from `source`. 
    Thanks to https://www.kaggle.com/devvindan/wheat-detection-eda for the idea of this.
    """
    fig = plt.figure(figsize=(int(8*columns), int(8*rows)))
    if source is not None:
        image_names = np.random.choice(train_df[train_df.source==source].image_id.unique(), columns*rows)
    image_names = np.random.choice(train_df.image_id.unique(), columns*rows)
    image_paths = [os.path.join(DIR, 'train', img+".jpg") for img in image_names]
    for i in range(1, columns*rows +1):
        img = Image.open(image_paths[i-1])
        ax = fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        df = train_df[train_df.image_id==image_names[i-1]]
        bboxes = get_boxes(df)
        [ax.add_patch(bbox) for bbox in bboxes]
        plt.axis('off')

    plt.show()


# In[ ]:


show_img_bbox()


# In[ ]:


sources = train_df.source.unique()
for source in sources:
    print(f"Images from: {source}")
    show_img_bbox(source=source)


# # Some thoughts:
# - bbox is used for bounding box 
# - I am making dough here, these images shall be the wheat, pytorch dataloader shall be the grinder, convolutions shall be the water, Adam shall be the weather and public kernel tricks/discussions shall be my magic ingredient.
# - I am so glad the data for this competition is under 1GB.

# In[ ]:


get_ipython().system('du ../input/global-wheat-detection/ -h')


# 621M of total data only <3
# 
# Baseline model coming soon. 
