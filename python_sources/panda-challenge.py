#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install ../input/fastai017-whl/fastprogress-0.2.3-py3-none-any.whl')
get_ipython().system('pip install ../input/fastai017-whl/fastcore-0.1.18-py3-none-any.whl')
get_ipython().system('pip install ../input/fastai017-whl/fastai2-0.0.17-py3-none-any.whl')


# In[ ]:


#Load the dependancies
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *

import seaborn as sns
import numpy as np
import pandas as pd
import os
import cv2
import openslide

from tqdm.notebook import tqdm
import skimage.io
from skimage.transform import resize, rescale


# In[ ]:


sns.set(style="whitegrid")
sns.set_context("paper")


# In[ ]:


source = Path("../input/prostate-cancer-grade-assessment")
files = os.listdir(source)
files


# In[ ]:


train = source/'train_images'
mask = source/'train_label_masks'
train_labels = pd.read_csv(source/'train.csv')


# In[ ]:


train_labels.head()


# # Plotting isup_grade

# In[ ]:


def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(3*size,2*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set1')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()


# In[ ]:


plot_count(train_labels, 'isup_grade','ISUP grade - data count and percent', size=3)


# In[ ]:


isup_0 = train_labels[train_labels.isup_grade == 0]
isup_1 = train_labels[train_labels.isup_grade == 1]
isup_2 = train_labels[train_labels.isup_grade == 2]
isup_3 = train_labels[train_labels.isup_grade == 3]
isup_4 = train_labels[train_labels.isup_grade == 4]
isup_5 = train_labels[train_labels.isup_grade == 5]

print(f'isup_0: {len(isup_0)}, isup_1: {len(isup_1)}, isup_2: {len(isup_2)}, isup_3: {len(isup_3)}, isup_4: {len(isup_4)}, isup_5: {len(isup_5)}')


# #  balancing the data using sample so that each class has 1224 images each and then create a balanced dataset

# In[ ]:


isup_sam0 = isup_0.sample(n=1224)
isup_sam1 = isup_1.sample(n=1224)
isup_sam2 = isup_2.sample(n=1224)
isup_sam3 = isup_3.sample(n=1224)
isup_sam4 = isup_4.sample(n=1224)

frames = [isup_sam0, isup_sam1, isup_sam2, isup_sam3, isup_sam4, isup_5]
balanced_df = pd. concat(frames)
balanced_df.head()


# # Plotting Balanced isup_grade

# In[ ]:


plot_count(balanced_df, 'isup_grade','ISUP grade - data count and percent', size=3)


# # Splitting Data

# In[ ]:


df_copy = balanced_df.copy()

# 80/20 split or whatever you choose
train_set = df_copy.sample(frac=0.8, random_state=7)
test_set = df_copy.drop(train_set.index)
print(len(train_set), len(test_set))


# # Viewing images 
# Using Fastai Openslide to view images 

# In[ ]:


def view_image(folder, fn):
    filename = f'{folder}/{fn}.tiff'
    file = openslide.OpenSlide(str(filename))
    t = tensor(file.get_thumbnail(size=(255, 255)))
    pil = PILImage.create(t) 
    return pil


# In[ ]:


glee_35 = train_labels[train_labels.gleason_score == '3+5']
glee_35.head()


# In[ ]:


glee_35.info()


# In[ ]:


train


# In[ ]:


view_image(train, '05819281002c55258bb3086cc55e3b48')


# In[ ]:


view_image(train, '08134913a9aa1d541f719e9f356f9378')


# In[ ]:


view_image(train, '0f958c8bbbc828b2e043e49ea39e16e2')


# In[ ]:


view_image(train, '847db624a7a975df11caca3c97743359')


# creating function that can get the images from the folder, open it and change it into a tensor as Fastai2 needs batches to be in the form of tensors or arrays

# In[ ]:


import os

# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go

# Location of the training images
data_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images'
mask_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_label_masks'

# Location of training labels
train_labels1 = pd.read_csv('/kaggle/input/prostate-cancer-grade-assessment/train.csv').set_index('image_id')


# In[ ]:


# Open the image (does not yet read the image into memory)
image = openslide.OpenSlide(os.path.join(data_dir, '005e66f06bce9c2e49142536caf2f6ee.tiff'))

# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.
# At this point image data is read from the file and loaded into memory.
patch = image.read_region((17800,19500), 0, (256, 256))

# Display the image
display(patch)

# Close the opened slide after use
image.close()


# In[ ]:


def print_slide_details(slide, show_thumbnail=True, max_size=(600,400)):
    """Print some basic information about a slide"""
    # Generate a small image thumbnail
    if show_thumbnail:
        display(slide.get_thumbnail(size=max_size))

    # Here we compute the "pixel spacing": the physical size of a pixel in the image.
    # OpenSlide gives the resolution in centimeters so we convert this to microns.
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


# In[ ]:


example_slides = [
    '00951a7fad040bf7e90f32e81fc0746f',
    '00a26aaa82c959624d90dfb69fcf259c',
    '007433133235efc27a39f11df6940829',
    '024ed1244a6d817358cedaea3783bbde',
]

for case_id in example_slides:
    biopsy = openslide.OpenSlide(os.path.join(data_dir, f'{case_id}.tiff'))
    print_slide_details(biopsy)
    biopsy.close()
    
    # Print the case-level label
    print(f"ISUP grade: {train_labels.loc[case_id, 'isup_grade']}")
    print(f"Gleason score: {train_labels.loc[case_id, 'gleason_score']}\n\n")


# In[ ]:


biopsy = openslide.OpenSlide(os.path.join(data_dir, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))

x = 5150
y = 21000
level = 0
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)


# In[ ]:


x = 5140
y = 21000
level = 2
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)


# In[ ]:


def print_mask_details(slide, center='radboud', show_thumbnail=True, max_size=(400,400)):
    """Print some basic information about a slide"""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Generate a small image thumbnail
    if show_thumbnail:
        # Read in the mask data from the highest level
        # We cannot use thumbnail() here because we need to load the raw label data.
        mask_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
        # Mask data is present in the R channel
        mask_data = mask_data.split()[0]

        # To show the masks we map the raw label values to RGB values
        preview_palette = np.zeros(shape=768, dtype=int)
        if center == 'radboud':
            # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
            preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
        elif center == 'karolinska':
            # Mapping: {0: background, 1: benign, 2: cancer}
            preview_palette[0:9] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 1, 0, 0]) * 255).astype(int)
        mask_data.putpalette(data=preview_palette.tolist())
        mask_data = mask_data.convert(mode='RGB')
        mask_data.thumbnail(size=max_size, resample=0)
        display(mask_data)

    # Compute microns per pixel (openslide gives resolution in centimeters)
    spacing = 1 / (float(slide.properties['tiff.XResolution']) / 10000)
    
    print(f"File id: {slide}")
    print(f"Dimensions: {slide.dimensions}")
    print(f"Microns per pixel / pixel spacing: {spacing:.3f}")
    print(f"Number of levels in the image: {slide.level_count}")
    print(f"Downsample factor per level: {slide.level_downsamples}")
    print(f"Dimensions of levels: {slide.level_dimensions}")


# In[ ]:


mask = openslide.OpenSlide(os.path.join(mask_dir, '090a77c517a7a2caa23e443a77a78bc7_mask.tiff'))
print_mask_details(mask, center='karolinska')
mask.close()


# In[ ]:


mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

plt.figure(figsize = (8, 15) )
plt.title("Mask with default cmap")
plt.imshow(np.asarray(mask_data)[:,:,0], interpolation='nearest')
plt.show()

plt.figure(figsize = (8, 15) )
plt.title("Mask with custom cmap")
# Optional: create a custom color map
cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'olive', 'yellow', 'mediumslateblue', 'fuchsia'])
plt.imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
plt.show()

mask.close()


# In[ ]:


def overlay_mask_on_slide(slide, mask, center='radboud', alpha=0.8, max_size=(800, 800)):
    """Show a mask overlayed on a slide."""

    if center not in ['radboud', 'karolinska']:
        raise Exception("Unsupported palette, should be one of [radboud, karolinska].")

    # Load data from the highest level
    slide_data = slide.read_region((0,0), slide.level_count - 1, slide.level_dimensions[-1])
    mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])

    # Mask data is present in the R channel
    mask_data = mask_data.split()[0]

    # Create alpha mask
    alpha_int = int(round(255*alpha))
    if center == 'radboud':
        alpha_content = np.less(mask_data.split()[0], 2).astype('uint8') * alpha_int + (255 - alpha_int)
    elif center == 'karolinska':
        alpha_content = np.less(mask_data.split()[0], 1).astype('uint8') * alpha_int + (255 - alpha_int)
    
    alpha_content = PIL.Image.fromarray(alpha_content)
    preview_palette = np.zeros(shape=768, dtype=int)
    
    if center == 'radboud':
        # Mapping: {0: background, 1: stroma, 2: benign epithelium, 3: Gleason 3, 4: Gleason 4, 5: Gleason 5}
        preview_palette[0:18] = (np.array([0, 0, 0, 0.5, 0.5, 0.5, 0, 1, 0, 1, 1, 0.7, 1, 0.5, 0, 1, 0, 0]) * 255).astype(int)
    elif center == 'karolinska':
        # Mapping: {0: background, 1: benign, 2: cancer}
        preview_palette[0:9] = (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]) * 255).astype(int)
    
    mask_data.putpalette(data=preview_palette.tolist())
    mask_rgb = mask_data.convert(mode='RGB')

    overlayed_image = PIL.Image.composite(image1=slide_data, image2=mask_rgb, mask=alpha_content)
    overlayed_image.thumbnail(size=max_size, resample=0)

    display(overlayed_image)


# In[ ]:


slide = openslide.OpenSlide(os.path.join(data_dir, '08ab45297bfe652cc0397f4b37719ba1.tiff'))
mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
overlay_mask_on_slide(slide, mask, center='radboud')
slide.close()
mask.close()


# In[ ]:


biopsy = skimage.io.MultiImage(os.path.join(data_dir, '0b373388b189bee3ef6e320b841264dd.tiff'))


# In[ ]:


def get_i(fn):
    filename = f'{train}/{fn.image_id}.tiff'
    example2 = openslide.OpenSlide(str(filename))
    ee = example2.get_thumbnail(size=(255, 255))
    return tensor(ee)


# In[ ]:


blocks = (
          ImageBlock,
          CategoryBlock
          )
         
getters = [
           get_i,
           ColReader('isup_grade')

          ]

trends = DataBlock(blocks=blocks,
              getters=getters,
              item_tfms=Resize(194),
              n_inp=1
              )


# In[ ]:


blocks


# In[ ]:


dls = trends.dataloaders(train_set, bs=32)
dls.show_batch()


# In[ ]:


dls.c


# In[ ]:


set_seed(7)
model = xresnet34(n_out=dls.c, sa=True, act_cls=Mish)

learn = Learner(dls, model, 
                opt_func=ranger,
                loss_func=LabelSmoothingCrossEntropy(),
                metrics=[accuracy],
                cbs = ShowGraphCallback())


# In[ ]:


learn.lr_find()


# In[ ]:


learn.freeze()
learn.fit_flat_cos(1, 5e-2)


# In[ ]:


learn.save('test_one')
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(12)


# # Loading Test Data

# In[ ]:


test_set.head()


# In[ ]:


for i in test_set.image_id:
    filename = f'{train}/{i}.tiff'
    example2 = openslide.OpenSlide(str(filename))
    ee = example2.get_thumbnail(size=(255, 255))
    ten = tensor(ee)
    clas, tens, prob = learn.predict(ten) 
    
    


# In[ ]:


test_df = test_set.copy()
test_df


# In[ ]:


del test_df['data_provider']
del test_df['gleason_score']
test_df


# In[ ]:


test_df['isup_grade_pred'] = clas
test_df[['image_id', 'isup_grade', 'isup_grade_pred']]

test_df

