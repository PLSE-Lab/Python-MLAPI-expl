#!/usr/bin/env python
# coding: utf-8

# ### This Kernel is a starter EDA for Global Wheat Prediction. It aims to generate insight into Global Wheat Prediction dataset. 

# ## Import the required Libraries

# In[ ]:


import pandas as pd
import numpy as np
import cv2
import os
import re
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#plotly
get_ipython().system('pip install chart_studio')
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



INPUT_PATH = '/kaggle/input/global-wheat-detection'
TRAIN_DIR = '/kaggle/input/global-wheat-detection/train/'
TEST_DIR = '/kaggle/input/global-wheat-detection/test/'


# In[ ]:


train_df = pd.read_csv(INPUT_PATH + '/train.csv')
train_df.shape


# In[ ]:


#Splitting the bboxes into x, y, w and h
bboxs = np.stack(train_df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    train_df[column] = bboxs[:,i]

train_df['x1'] = train_df['x'] + train_df['w']
train_df['y1'] = train_df['y'] + train_df['h']


# In[ ]:


train_df.head()


# In[ ]:


# Glob the directories and get the lists of train and test images
train_dir = glob(INPUT_PATH + '/train/' + '*')
test_dir = glob(INPUT_PATH + '/test/'+'*')

print('Number of Train images :', len(train_dir))
print('Number of Test images :', len(test_dir))


# In[ ]:


print('Number of Training images in the labelled csv file:', len(train_df.groupby('image_id')))


# * Not all the input images in the train directory are part of the train.csv
# * Total count of input images is 3422 where as the labelled dataset has only 3373 image_ids 
# * Let's analyze those images not on training directory

# ### Percentage of images from various sources through PieChart

# In[ ]:


src_dtls = train_df.source.value_counts(normalize=True).sort_values()

new = pd.DataFrame({
    'source': src_dtls.index,
    'percentage': src_dtls.values 
})

fig = go.Figure(
    data=[go.Pie(
        labels=new['source'],
        values=new['percentage'])
    ])
fig.show()


# ### Arvalis_1 and rres_1 has more number of bounding boxes than other sources of the images

# In[ ]:


# Create a dataframe with all train images
all_train_images = pd.DataFrame([i.split('/')[-1][:-4] for i in train_dir])
all_train_images.columns=['image_id']


# Merge all train images with the bounding boxes dataframe
all_train_images = all_train_images.merge(train_df, on='image_id', how='left')

# replace nan values with zeros
all_train_images['bbox'] = all_train_images.bbox.fillna('[0,0,0,0]')

# split bbox column
bbox_items = all_train_images.bbox.str.split(',', expand=True)
all_train_images['bbox_xmin'] = bbox_items[0].str.strip('[ ').astype(float)
all_train_images['bbox_ymin'] = bbox_items[1].str.strip(' ').astype(float)
all_train_images['bbox_width'] = bbox_items[2].str.strip(' ').astype(float)
all_train_images['bbox_height'] = bbox_items[3].str.strip(' ]').astype(float)


# In[ ]:


nobboxes_images = all_train_images[~all_train_images.image_id.isin(train_df.image_id)]
print('Number of images with no bounding boxes:', len(nobboxes_images))


# #### Plotting the Histogram plot for Number of bounding boxes per image Ids

# In[ ]:


all_train_images['image_id'].value_counts().iplot(kind='hist',bins=30,color='blue',xTitle='No. of bboxes per Image Id',yTitle='No. of Images')


# #### In order to know the distribution of images from various sources, lets plot the distribution plot for various sources. Distribution plot is combination of histogram plot fitted with Kernel Density Estimate funciton (KDE)[](http://)

# In[ ]:


# Group data together
inrae_1 = all_train_images.loc[all_train_images['source'] =='inrae_1']['image_id'].value_counts()
arvalis_2 = all_train_images.loc[all_train_images['source'] =='arvalis_2']['image_id'].value_counts()
usask_1 = all_train_images.loc[all_train_images['source'] =='usask_1']['image_id'].value_counts()
arvalis_3 = all_train_images.loc[all_train_images['source'] =='arvalis_3']['image_id'].value_counts()
rres_1 = all_train_images.loc[all_train_images['source'] =='rres_1']['image_id'].value_counts()
ethz_1 = all_train_images.loc[all_train_images['source'] =='ethz_1']['image_id'].value_counts()

hist_data = [inrae_1, arvalis_2, usask_1, arvalis_3, rres_1, ethz_1]

labels = src_dtls.index

plt.figure(figsize=(12,8))

sns.distplot(inrae_1)
sns.distplot(arvalis_2)
sns.distplot(usask_1)
sns.distplot(arvalis_3)
sns.distplot(rres_1)
sns.distplot(ethz_1)
plt.figlegend(labels, loc='upper right')
plt.show()


# ### Common Utility function to plot various samples of input images.

# In[ ]:


def get_all_bboxes(df, image_id):
    image_bboxes = df[df.image_id == image_id]
    
    bboxes = []
    for _,row in image_bboxes.iterrows():
        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height))
        
    return bboxes

def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(10,10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            img_id = df.iloc[idx].image_id
            
            img = Image.open(TRAIN_DIR + img_id + '.jpg')
            axs[row, col].imshow(img)
            
            bboxes = get_all_bboxes(df, img_id)
            
            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
                axs[row, col].add_patch(rect)
            
            axs[row, col].axis('off')
            
    plt.suptitle(title)


# ### Images with No Bounding Boxes 

# In[ ]:


plot_image_examples(all_train_images[all_train_images.image_id.isin(nobboxes_images.image_id)], title='Images with no bounding boxes')


# In[ ]:


plot_image_examples(all_train_images.loc[all_train_images['source'] =='inrae_1'], title='Random samples for source INRAE_1')


# In[ ]:


plot_image_examples(all_train_images.loc[all_train_images['source'] =='arvalis_2'], title='Random samples for source arvalis_2')


# In[ ]:


plot_image_examples(all_train_images.loc[all_train_images['source'] =='usask_1'], title='Random samples for source usask_1')


# In[ ]:


plot_image_examples(all_train_images.loc[all_train_images['source'] =='ethz_1'], title='Random samples for source ethz_1')


# ### Test Image Samples to be predicted 

# In[ ]:


images = test_dir

# Extract 9 random images from it
random_images = [np.random.choice(images) for i in range(9)]

print('Display Test Images')

# Adjust the size of your images
plt.figure(figsize=(10,8))

# Iterate and plot random images
for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(TEST_DIR, random_images[i]))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
# Adjust subplot parameters to give specified padding
plt.tight_layout()   


# ### Data Visualization with albumentations for Wheat detection

# In[ ]:


# Select a randome image for albumentation 

idx = np.random.randint(len(all_train_images), size=1)[0]
img_id = all_train_images.iloc[idx].image_id 

img= cv2.imread(os.path.join(TRAIN_DIR, img_id + '.jpg'))[:,:,::-1]
plt.imshow(img)

bboxes = get_all_bboxes(all_train_images, img_id)
# All the bboxes are labelled to default 1 value as these belongs to single class
labels = np.ones((len(bboxes),))
# Original Annotations to apply Albumentations
orig_annotations = {'image': img, 'bboxes': bboxes, 'category_id': labels}


# In[ ]:


# Plots the images as per the annotations format and applied augmentations
def plot_image_list(annotations_list, subtitle_list, cols=2, title='Image Examples'):
    fig, axs = plt.subplots(nrows=1, ncols=cols, figsize=(16,12), squeeze=False)
    for i, (annotations, title) in enumerate(zip(annotations_list, subtitle_list)):
        axs[i // cols][i % cols].imshow(annotations['image'])
        axs[i // cols][i % cols].set_title(title, fontsize=14)
        for bbox in annotations['bboxes']:
            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')
            axs[i // cols][i % cols].add_patch(rect) 
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()


# In[ ]:


# Funtion to get the Albumentations
def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params=A.BboxParams(format='coco', min_area=min_area, 
                                               min_visibility=min_visibility, label_fields=['category_id']))


# In[ ]:


# Applying Vertical Flip on the Original Image
aug = get_aug([A.VerticalFlip(p=1)])
verticalFlip = aug(**orig_annotations)
annotations_list=[orig_annotations, verticalFlip]
subtitle_list=['Original Image', 'Vertically Flipped Image']
plot_image_list(annotations_list, subtitle_list, title='Vertical Flip' )


# In[ ]:


# Applying Horizontal Flip on the Original Image
aug = get_aug([A.HorizontalFlip(p=1)])
horizontalFlip = aug(**orig_annotations)
annotations_list=[orig_annotations, horizontalFlip]
subtitle_list=['Original Image', 'Horizontally Flipped Image']
plot_image_list(annotations_list, subtitle_list, title='Horizontal Flip' )


# In[ ]:


# Applying Center Cropping the Original Image
aug = get_aug([A.CenterCrop(p=1, height=512, width=512)])
Cropped = aug(**orig_annotations)
annotations_list=[orig_annotations, Cropped]
subtitle_list=['Original Image', 'Centre Cropped Image']
plot_image_list(annotations_list, subtitle_list, title='Image Cropping' )


# In[ ]:


# Changing the Brightness of the image randomly
aug = get_aug([A.RandomBrightness(p=0.4)])
brightness = aug(**orig_annotations)
annotations_list=[orig_annotations, brightness]
subtitle_list=['Original Image', 'Random Brightness']
plot_image_list(annotations_list, subtitle_list, title='Change in Brightness' )


# In[ ]:


# Applying Random SunFlare

aug = get_aug([A.RandomSunFlare(p=1)])
shadow = aug(**orig_annotations)
annotations_list=[orig_annotations, shadow]
subtitle_list=['Original Image', 'Random Sun Flare']
plot_image_list(annotations_list, subtitle_list, title='Random Sun Flare Results' )


# In[ ]:




