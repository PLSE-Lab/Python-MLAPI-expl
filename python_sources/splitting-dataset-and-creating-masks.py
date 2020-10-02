#!/usr/bin/env python
# coding: utf-8

# <a id="toc"></a>
# # Table of Contents
# 1. [Configure parameters](#configure_parameters)
# 1. [Import modules](#import_modules)
# 1. [Get annotations](#get_annotations)
# 1. [Draw some charts for input dataset](#draw_some_charts_for_input_dataset)
# 1. [Get encoded pixels of each class for each image](#get_encoded_pixels_of_each_class_for_each_image)
# 1. [Split dataset into training and validation sets](#split_dataset_into_training_and_validation_sets)
# 1. [Draw some charts for training and validation sets](#draw_some_charts_for_training_and_validation_sets)
# 1. [Copy images into right folders](#copy_images_into_right_folders)
# 1. [Create corresponding mask for each image](#create_corresponding_mask_for_each_image)
# 1. [Visualize some images and corresponding labels](#visualize_some_images_and_corresponding_labels)
# 1. [Zip training and validation sets](#zip_training_and_validation_sets)
# 1. [Post process annotations](#post_process_annotations)
# 1. [Save annotations](#save_annotations)

# <a id="configure_parameters"></a>
# # Configure parameters
# [Back to Table of Contents](#toc)

# In[ ]:


DATASET_DIR = '../input/severstal-steel-defect-detection/'
TEST_SIZE = 0.3
RANDOM_STATE = 123

NUM_TRAIN_SAMPLES = 20 # The number of train samples used for visualization
NUM_VAL_SAMPLES = 20 # The number of val samples used for visualization
COLORS = ['b', 'g', 'r', 'm'] # Color of each class


# <a id="import_modules"></a>
# # Import modules
# [Back to Table of Contents](#toc)

# In[ ]:


import pandas as pd
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from shutil import copyfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook


# <a id="get_annotations"></a>
# # Get annotations
# [Back to Table of Contents](#toc)

# In[ ]:


df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))


# In[ ]:


df.head()


# ##### Convert training data-frame to the legacy version

# In[ ]:


legacy_df = pd.DataFrame(columns=['ImageId_ClassId', 'EncodedPixels'])

for img_id, img_df in tqdm_notebook(df.groupby('ImageId')):
    for i in range(1, 5):
        avail_classes = list(img_df.ClassId)

        row = dict()
        row['ImageId_ClassId'] = img_id + '_' + str(i)

        if i in avail_classes:
            row['EncodedPixels'] = img_df.loc[img_df.ClassId == i].EncodedPixels.iloc[0]
        else:
            row['EncodedPixels'] = np.nan
        
        legacy_df = legacy_df.append(row, ignore_index=True)


# In[ ]:


legacy_df.head()


# In[ ]:


df = legacy_df


# ##### Continue the preprocessing process

# In[ ]:


df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
df['HavingDefection'] = df['EncodedPixels'].map(lambda x: 0 if x is np.nan else 1)


# In[ ]:


image_col = np.array(df['Image'])
image_files = image_col[::4]
all_labels = np.array(df['HavingDefection']).reshape(-1, 4)

num_img_class_1 = np.sum(all_labels[:, 0])
num_img_class_2 = np.sum(all_labels[:, 1])
num_img_class_3 = np.sum(all_labels[:, 2])
num_img_class_4 = np.sum(all_labels[:, 3])
print('Class 1: {} images'.format(num_img_class_1))
print('Class 2: {} images'.format(num_img_class_2))
print('Class 3: {} images'.format(num_img_class_3))
print('Class 4: {} images'.format(num_img_class_4))


# <a id="draw_some_charts_for_input_dataset"></a>
# # Draw some charts for input dataset
# [Back to Table of Contents](#toc)

# In[ ]:


def plot_figures(
    sizes,
    pie_title,
    bar_title,
    bar_ylabel,
    labels=('Class 1', 'Class 2', 'Class 3', 'Class 4'),
    colors=None,
    explode=(0, 0, 0, 0.1),
):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    y_pos = np.arange(len(labels))
    barlist = axes[0].bar(y_pos, sizes, align='center')
    axes[0].set_xticks(y_pos, labels)
    axes[0].set_ylabel(bar_ylabel)
    axes[0].set_title(bar_title)
    if colors is not None:
        for idx, item in enumerate(barlist):
            item.set_color(colors[idx])

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            axes[0].text(
                rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='bottom', fontweight='bold'
            )

    autolabel(barlist)
    
    pielist = axes[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, counterclock=False)
    axes[1].axis('equal')
    axes[1].set_title(pie_title)
    if colors is not None:
        for idx, item in enumerate(pielist[0]):
            item.set_color(colors[idx])

    plt.show()


# In[ ]:


print('[THE WHOLE DATASET]')

sizes = np.sum(all_labels, axis=0)
plot_figures(
    sizes,
    pie_title='The percentage of each class',
    bar_title='The number of images for each class',
    bar_ylabel='Images',
    colors=COLORS,
    explode=(0, 0, 0, 0.1)
)

sum_of_each_sample = np.sum(all_labels, axis=1)
unique, counts = np.unique(sum_of_each_sample, return_counts=True)

plot_figures(
    counts,
    pie_title='The percentage of the number of classes appears in an image',
    bar_title='The number of classes appears in an image',
    bar_ylabel='Images',
    labels=list(unique),
    explode=np.zeros(len(unique))
)


# <a id="get_encoded_pixels_of_each_class_for_each_image"></a>
# # Get encoded pixels of each class for each image
# [Back to Table of Contents](#toc)

# In[ ]:


annotations = np.array(df['EncodedPixels']).reshape(-1, 4)


# <a id="split_dataset_into_training_and_validation_sets"></a>
# # Split dataset into training and validation sets
# [Back to Table of Contents](#toc)

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(image_files, annotations, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# In[ ]:


print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_val:', X_val.shape)
print('y_val', y_val.shape)


# <a id="draw_some_charts_for_training_and_validation_sets"></a>
# # Draw some charts for training and validation sets
# [Back to Table of Contents](#toc)

# In[ ]:


print('[TRAINING SET]')

tmp = y_train.reshape(-1)
tmp = list(map(lambda x: 0 if x is np.nan else 1, tmp))
train_labels = np.array(tmp).reshape(-1, 4)

train_sizes = np.sum(train_labels, axis=0)
plot_figures(
    train_sizes,
    pie_title='The percentage of each class',
    bar_title='The number of images for each class',
    bar_ylabel='Images',
    colors=COLORS,
    explode=(0, 0, 0, 0.1)
)


sum_of_each_sample = np.sum(train_labels, axis=1)
unique, counts = np.unique(sum_of_each_sample, return_counts=True)

plot_figures(
    counts,
    pie_title='The percentage of the number of classes appears in an image',
    bar_title='The number of classes appears in an image',
    bar_ylabel='Images',
    labels=list(unique),
    explode=np.zeros(len(unique))
)


# In[ ]:


print('[VALIDATION SET]')

tmp = y_val.reshape(-1)
tmp = list(map(lambda x: 0 if x is np.nan else 1, tmp))
val_labels = np.array(tmp).reshape(-1, 4)

val_sizes = np.sum(val_labels, axis=0)
plot_figures(
    val_sizes,
    pie_title='The percentage of each class',
    bar_title='The number of images for each class',
    bar_ylabel='Images',
    colors=COLORS,
    explode=(0, 0, 0, 0.1)
)


sum_of_each_sample = np.sum(val_labels, axis=1)
unique, counts = np.unique(sum_of_each_sample, return_counts=True)

plot_figures(
    counts,
    pie_title='The percentage of the number of classes appears in an image',
    bar_title='The number of classes appears in an image',
    bar_ylabel='Images',
    labels=list(unique),
    explode=np.zeros(len(unique))
)


# <a id="copy_images_into_right_folders"></a>
# # Copy images into right folders
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('mkdir train_images')
get_ipython().system('mkdir val_images')


# In[ ]:


for image_file in tqdm_notebook(X_train):
    src = os.path.join(DATASET_DIR, 'train_images', image_file)
    dst = os.path.join('./train_images', image_file)
    copyfile(src, dst)

for image_file in tqdm_notebook(X_val):
    src = os.path.join(DATASET_DIR, 'train_images', image_file)
    dst = os.path.join('./val_images', image_file)
    copyfile(src, dst)


# <a id="create_corresponding_mask_for_each_image"></a>
# # Create corresponding mask for each image
# [Back to Table of Contents](#toc)

# In[ ]:


def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# In[ ]:


def generate_masks(pairs, mask_size=(256, 1600), out_dir='./'):
    for pair in tqdm_notebook(pairs):
        img_name = pair[0].split('.')[0]
        for idx, rle in enumerate(pair[1]):
            if rle is np.nan:
                mask = np.zeros(mask_size, dtype=np.uint8)
            else:
                mask = rle2mask(rle)

            # Save result
            cv2.imwrite(os.path.join(out_dir, '{}_{}.jpg'.format(img_name, idx+1)), mask)


# In[ ]:


get_ipython().system('mkdir train_masks')
get_ipython().system('mkdir val_masks')


# In[ ]:


train_pairs = np.array(list(zip(X_train, y_train)))
generate_masks(train_pairs, out_dir='./train_masks')


# In[ ]:


val_pairs = np.array(list(zip(X_val, y_val)))
generate_masks(val_pairs, out_dir='./val_masks')


# <a id="visualize_some_images_and_corresponding_labels"></a>
# # Visualize some images and corresponding labels
# [Back to Table of Contents](#toc)

# In[ ]:


def show_samples(samples):
    for sample in samples:
        fig, ax = plt.subplots(figsize=(16, 10))
        img_path = os.path.join(DATASET_DIR, 'train_images', sample[0])
        img = cv2.imread(img_path, 1)

        patches = []
        for idx, rle in enumerate(sample[1]):
            if rle is not np.nan:
                mask = rle2mask(rle)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=1, edgecolor=COLORS[idx], fill=False)
                    patches.append(poly_patch)
        p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet)

        ax.imshow(img/255)
        ax.set_title(sample[0])
        ax.add_collection(p)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()


# In[ ]:


def show_corresponding_masks(samples, mask_dir):
    for sample in samples:
        fig, axes = plt.subplots(2, 2, figsize=(16, 4))
        for i in range(4):
            mask_name = '{}_{}.jpg'.format(sample[0].split('.')[0], i+1)
            mask_file = os.path.join(mask_dir, mask_name)
            mask = cv2.imread(mask_file, 0)

            axes[i//2][i%2].imshow(mask*255)
            axes[i//2][i%2].set_title('{} - class {}'.format(sample[0], i+1))
            axes[i//2][i%2].set_xticklabels([])
            axes[i//2][i%2].set_yticklabels([])
        plt.show()


# In[ ]:


def show_images_and_corresponding_masks(samples, mask_dir):
    for sample in samples:
        show_samples([sample])
        show_corresponding_masks([sample], mask_dir)


# In[ ]:


train_samples = train_pairs[np.random.choice(train_pairs.shape[0], NUM_TRAIN_SAMPLES, replace=False), :]
show_images_and_corresponding_masks(train_samples, './train_masks')


# In[ ]:


val_samples = val_pairs[np.random.choice(val_pairs.shape[0], NUM_VAL_SAMPLES, replace=False), :]
show_images_and_corresponding_masks(val_samples, './val_masks')


# <a id="zip_training_and_validation_sets"></a>
# # Zip training and validation sets
# [Back to Table of Contents](#toc)

# In[ ]:


get_ipython().system('apt install zip')


# In[ ]:


get_ipython().system('zip -r -m -1 -q train_images.zip ./train_images')
get_ipython().system('zip -r -m -1 -q val_images.zip ./val_images')
get_ipython().system('zip -r -m -1 -q train_masks.zip ./train_masks')
get_ipython().system('zip -r -m -1 -q val_masks.zip ./val_masks')


# <a id="post_process_annotations"></a>
# # Post process annotations
# [Back to Table of Contents](#toc)

# In[ ]:


y_train = y_train.reshape(-1)
y_val = y_val.reshape(-1)


# In[ ]:


X_train = np.repeat(X_train, 4)
X_val = np.repeat(X_val, 4)

X_train = X_train.reshape(-1, 4)
X_val = X_val.reshape(-1, 4)

indices = np.array(['_1', '_2', '_3', '_4'])

X_train += indices
X_val += indices

X_train = X_train.reshape(-1)
X_val = X_val.reshape(-1)


# <a id="save_annotations"></a>
# # Save annotations
# [Back to Table of Contents](#toc)

# In[ ]:


train_set = {
    'ImageId_ClassId': X_train,
    'EncodedPixels': y_train
}

val_set = {
    'ImageId_ClassId': X_val,
    'EncodedPixels': y_val
}

train_df = pd.DataFrame(train_set)
val_df = pd.DataFrame(val_set)

train_df.to_csv('./train.csv', index=False)
val_df.to_csv('./val.csv', index=False)

