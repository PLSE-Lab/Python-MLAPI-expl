#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import json
from PIL import Image

BASE_DIR = "../input"

IMAGES_TRAIN_DIR = f"{BASE_DIR}/train"
IMAGES_TEST_DIR = f"{BASE_DIR}/test"
TRAIN_CSV = f"{BASE_DIR}/train.csv"
LABEL_DESCRIPTIONS = f"{BASE_DIR}/label_descriptions.json"
# Any results you write to the current directory are saved as output.


# In[ ]:


#print(os.listdir("../input"))
train_df = pd.read_csv(TRAIN_CSV)

with open(LABEL_DESCRIPTIONS) as f:
    image_info = json.load(f)
categories = pd.DataFrame(image_info['categories'])
attributes = pd.DataFrame(image_info['attributes'])
print("There are descriptions for", categories.shape[0],"categories and", attributes.shape[0], "attributes")

train_df['hasAttributes'] = train_df.ClassId.apply(lambda x: x.find("_") > 0)
train_df['CategoryId'] = train_df.ClassId.apply(lambda x: x.split("_")[0]).astype(int)
train_df = train_df.merge(categories, left_on="CategoryId", right_on="id")


# In[ ]:


categories.head()


# In[ ]:


attributes.head()


# Both attributes and categories contain additional `supercategory` column, which might be the source of insights related to our data. Cloths from the same supercategory are similar in some sense. Attribute's supercategory denotes that it describes some specific property (I.e., length, style).

# # Supercategories

# Supercategories might be the key to answering following questions:
# 1. How many mask annotations have any associated attributes?
# 2. How often specific attributes (or attribute groups) appear within category's supercategory?
# 3. Is there a way to filter train data somehow?
# 4. etc.

# In[ ]:


print("Fraction of mask annotations with any attributes within train data:", train_df.hasAttributes.mean())


# In[ ]:


subset = train_df[~train_df.hasAttributes]
supercategory_names = np.unique(subset.supercategory)
plt.figure(figsize=(10, 10))
g = sns.countplot(x = 'supercategory', data=subset, order=supercategory_names)
ax = g.axes
tl = [x.get_text() for x in ax.get_xticklabels()]    
ax.set_xticklabels(tl, rotation=90)
for p, label in zip(ax.patches, supercategory_names):
    c = subset[(subset['supercategory'] == label)].shape[0]
    ax.annotate(str(c), (p.get_x(), p.get_height() + 1000))
plt.title("Supercategories with no attributes")
plt.show()


# In[ ]:


subset = train_df[train_df.hasAttributes]
supercategory_names = np.unique(subset.supercategory)
g = sns.countplot(x = 'supercategory', data=subset, order=supercategory_names)
ax = g.axes
tl = [x.get_text() for x in ax.get_xticklabels()]    
ax.set_xticklabels(tl, rotation=90)
for p, label in zip(ax.patches, supercategory_names):
    c = subset[(subset['supercategory'] == label)].shape[0]
    ax.annotate(str(c), (p.get_x()+0.3, p.get_height() + 50))
plt.title("Supercategories with any attributes")
plt.show()


# As you can see from previous 2 plots, number of mask annotations with any attribute is relatively small. Only masks related to 4 supercategories (well, 3, if we neglect 3 mask annotations related to `garment parts`) have any associated attribute. For every other supercategory, we might ignore them.

# Now we will take a closer look at train dataset and check how many categories are presented there. What supercategories are varied the most?

# In[ ]:


supercategory_names = train_df[['supercategory', 'name']].groupby('supercategory').agg(
    lambda x: x.unique().shape[0]).reset_index().sort_values("name", ascending=False).set_index('name')
supercategory_names


# Relative counts for 3 supercategories (`decorations`, `garment parts`, `upperbody`) are shown below:

# In[ ]:


def buildPlot(**kwargs):
    data = kwargs['data']
    g = sns.countplot(y="name", data=data)
    g.set_yticklabels(data['name'].unique())#, rotation=90)
    
idx = train_df.supercategory.isin(['decorations', 'garment parts', 'upperbody'])
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)


# In[ ]:


idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[4].values)
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)


# In the dataset the most common `wholebody` cloth type is `dress`. `Shoe` dominates in 2nd most common supercategory (`legs and feet`), that means that it might be worth to search for shoes on the photos :)

# In[ ]:


total = train_df.ImageId.unique().shape[0]
print(f"There are {total} images in train dataset.")
images_with_shoes = train_df[train_df.name=="shoe"].ImageId.unique().shape[0]
images_with_legs = train_df[train_df.supercategory=="legs and feet"].ImageId.unique().shape[0]
print(f"However, only {images_with_legs} images have associated legs and feet annotation, and only {images_with_shoes} have any shoes on it.")


# In[ ]:


idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[3].values)
g = sns.FacetGrid(data=train_df[idx], row="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)


# In[ ]:


idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[2].values)
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)


# In[ ]:


idx = train_df.supercategory.isin(supercategory_names.supercategory.loc[1].values)
g = sns.FacetGrid(data=train_df[idx], col="supercategory", sharey=False)
g = g.map_dataframe(buildPlot)


# # Connecting categories and attributes

# In[ ]:


#extract all available attributes and create separate table
cat_attributes = []
for i in train_df[train_df.hasAttributes].index:
    item = train_df.loc[i]
    xs = item.ClassId.split("_")
    for a in xs[1:]:
        cat_attributes.append({'ImageId': item.ImageId, 'category': int(xs[0]), 'attribute': int(a)})
cat_attributes = pd.DataFrame(cat_attributes)

cat_attributes = cat_attributes.merge(
    categories, left_on="category", right_on="id"
).merge(attributes, left_on="attribute", right_on="id", suffixes=("", "_attribute"))


# In[ ]:


# helper objects and methods
scat_x, count_x = np.unique(cat_attributes['supercategory'], return_counts=True)
categories_by_x = {
    x: dict(cat_attributes[cat_attributes['supercategory'] == x][['name', 'category']].drop_duplicates().values)
    for x in scat_x}
scat_y, count_y = np.unique(cat_attributes['supercategory_attribute'], return_counts=True)
categories_by_y = {
    y: dict(cat_attributes[cat_attributes['supercategory_attribute'] == y][['name_attribute', 'attribute']].drop_duplicates().values) 
    for y in scat_y}
vals = cat_attributes.groupby(['category', 'attribute']).count().reset_index(drop=True).values[:,0]
scale_min, scale_max = vals.min(), vals.max()

def get_scatter_data(x, y, cat, attr):
    ids_x = {cat[k]: i for i, k in enumerate(cat)}
    ids_y = {attr[k]: i for i, k in enumerate(attr)}
    data = np.zeros((len(cat), len(attr)), dtype=np.uint)
    for k, v in zip(x, y):
        data[ids_x[k], ids_y[v]]+=1
    ii, jj = np.where(data > 0)
    sizes = [data[i, j] for i, j in zip(ii, jj)]
    return ii, jj, sizes

def drawPunchcard(**kwargs):
    data = kwargs['data']
    x = data["category"]
    y = data["attribute"]
    supercategory_x = data["supercategory"].values[0]
    cat = categories_by_x[supercategory_x]
    supercategory_y = data["supercategory_attribute"].values[0]
    attr = categories_by_y[supercategory_y]
    ii, jj, sizes = get_scatter_data(
        x, y, 
        cat, 
        attr)
    g = sns.scatterplot(ii, jj, size=sizes, sizes=(20, 200), hue=np.log(sizes)+1)
    g.set_xticks(np.arange(len(cat)))
    g.set_xticklabels(list(cat), rotation=90)
    g.set_yticks(np.arange(len(attr)))
    g.set_yticklabels(list(attr))


# In[ ]:


sns.color_palette("bright")
sns.set(font_scale=1.0)
sns.set_style("white")
width_ratios=[len(categories_by_x[x]) for x in categories_by_x]
height_ratios=[len(categories_by_y[x]) for x in categories_by_y]
g = sns.FacetGrid(data=cat_attributes, col="supercategory",  row="supercategory_attribute", 
                  #margin_titles=True, 
                  gridspec_kws={'height_ratios': height_ratios, 'width_ratios': width_ratios},
                  sharex="col", sharey="row",
                  col_order=list(categories_by_x),
                  row_order=list(categories_by_y))#.set_titles('{col_name}', '{row_name}')
g = g.map_dataframe(drawPunchcard).set_titles('{col_name}', '{row_name}')
g.fig.set_size_inches(10, 20) 
for ax, cat_name in zip(g.axes, list(categories_by_y)):
    ax[-1].set_ylabel(cat_name, labelpad=10, rotation=-90)
    ax[-1].yaxis.set_label_position("right")


# # Closer look at train data

# In[ ]:


images = train_df[['ImageId', "Width", "Height"]].drop_duplicates()
print("Number of unique triplets (ImageId, Width, Height):", images.shape[0])
print("Unique image names: ", images['ImageId'].unique().shape[0])


# ## 1. Check Width and Height correctness
# There are no images with different width and height parameters in `train.csv` file. That doesn't mean that there are no errors. Just in case we'll check all dimensions for train images, and compare them with the ones provided in annotation file.

# In[ ]:


def read_image_dimensions(path):
    "returns real width and height"
    with Image.open(path) as image:
        dimensions = image.size
    return dimensions

images_with_incorrect_size = {}
for ImageId, width, height in images.values:
    image_path = os.path.join(IMAGES_TRAIN_DIR, ImageId)
    (real_width, real_height) = read_image_dimensions(image_path)
    if real_width != width or real_height!=height:
        images_with_incorrect_size[ImageId] = (real_width, real_height)


# In[ ]:


print("Number of images with incorrect dimensions:", len(images_with_incorrect_size))


# Next we fix annotation file.

# In[ ]:


for ImageId in images_with_incorrect_size:
    (width, height) = images_with_incorrect_size[ImageId]
    idx = train_df['ImageId'] == ImageId
    print(ImageId, train_df.loc[idx, "Width"].values[0], train_df.loc[idx, "Height"].values[0], "Real dimensions:", width, height)
    train_df.loc[idx, "Width"] = width
    train_df.loc[idx, "Height"] = height


# ## 2. Check if there are any duplicates
# We also check if there are any image masks with several accociated classes (we might want to ignore them while training our segmentation model).

# In[ ]:


df = train_df[["ImageId", "EncodedPixels", "ClassId"]].drop_duplicates()
grouped_df = df.groupby(["EncodedPixels", "ImageId"]).count().reset_index()
grouped_df = grouped_df[grouped_df.ClassId > 1]
print("Number of images with duplicated EncodedPixels:", grouped_df.shape[0])


# In[ ]:


duplicated_data = df[df.ImageId.isin(grouped_df.ImageId) & df.EncodedPixels.isin(grouped_df.EncodedPixels)].sort_values(["ImageId", "EncodedPixels"])
duplicated_data.to_csv("images_with_duplicated_masks.csv", index=None) # you can look at these images, if you want


# In[ ]:


duplicates = dict()
xlabels, ylabels = set(), set()

for (ImageId, EncodedPixels), x in duplicated_data.groupby(["ImageId", "EncodedPixels"]):
    pair = tuple(sorted(x.ClassId.values))
    s,e = pair
    xlabels.add(s)
    ylabels.add(e)
    if not pair in duplicates:
        duplicates[pair] = 0
    duplicates[pair] +=1

xlabels = {x: i for i, x in enumerate(sorted(xlabels))}
ylabels = {x: i for i, x in enumerate(sorted(ylabels))}
matrix = np.zeros((len(ylabels), len(xlabels)), dtype=np.int)
for (s, e) in duplicates:
    matrix[ylabels[e], xlabels[s]] = duplicates[(s, e)]


# In[ ]:


plt.figure(figsize=(10,10))
annot = np.array([[(str(x) if x >0 else "") for x in line]for line in matrix])
sns.heatmap(matrix, annot=annot, fmt="s",xticklabels=sorted(list(xlabels)), yticklabels=sorted(list(ylabels)), square=True, cbar=False)


# The most common duplicated mask is marked with image categories with ids 32 and 35.

# In[ ]:


categories[categories.id.isin((32, 35))]


# Pockets and zipper are usually located nearby, and they are relatively small. There also might be pockets with zipper. Maybe not a mistake after all?

# # Masking

# For every mask in train dataset, we convert number of masks pixels to a fraction of total image pixels:

# In[ ]:


def sum_mask_pixels(encoded_pixels):
    pixels = [np.int(x) for x in encoded_pixels.split(" ")]
    return np.sum(pixels[1::2])

def compute_mask_percentage(row):
    s = sum_mask_pixels(row['EncodedPixels'])
    return 1.0* s/row["Width"]/row["Height"]

train_df['mask_fraction'] = train_df.EncodedPixels.apply(sum_mask_pixels).astype(np.float)
train_df['mask_fraction'] = train_df['mask_fraction']/train_df["Width"]/train_df["Height"]


# We named this parameter `mask_fraction`. We use supercategories to compare how `mask_fraction` distribution differs from supercategory to supercategory:

# In[ ]:


plt.figure(figsize=(10, 8))
g = sns.stripplot(y="mask_fraction", data=train_df, x="supercategory")
labels = [x.get_text() for x in g.get_xticklabels()]
g = g.set_xticklabels(labels, rotation=90)


# The biggest masks (`mask_fraction` > 0.7) are associated only with 3 supercategories - `lowerbody`, `upperbody` and `wholebody`. Masks associated with `neck`, `arms and hands`, `closures` are almost always small (< 0.1 of total image pixels).
# 
# Let's take a look at some big masks (`mask_fraction` > 0.7):

# In[ ]:


def draw_images(data=None,**kwargs):
    plt.axis("off")
    path = os.path.join(IMAGES_TRAIN_DIR, data['ImageId'].values[0])
    with Image.open(path) as image:
        data = np.asarray(image)
    plt.imshow(data)

subset = train_df[train_df.mask_fraction > 0.7]
grid = sns.FacetGrid(subset, col="name", col_wrap=4)
grid.map_dataframe(draw_images)


# These sample images are all clipped. They have particular clothes shown on it, but there is no human on the photo.

# # Conclusions and other ideas
# 1. It is quite clear that for the draft segmentation model it is sufficient to use mask's category identifier and to ignore mask's attributes.
# 2. It is unclear (at least for me) if the train dataset attributes are given for all possible images, or are these attributes  provided for only a small subset of images. Do we have weakly-labeled data or fully-labeled data?
# 3. Train dataset contains a very diverse collection of images, with a lot of small details. There are also images which contain only 1 apparel, already clipped. 
# 4. Almost a half of the images have no shoes annotation. It means that human body pictured on them (if there is any) is probably clipped. 
# 5. We can split train dataset to several parts. One of these parts might contain clipped clothes, the other might contain full human body. And maybe it will improve segmentation results for both subsets. But maybe not.

# In[ ]:




