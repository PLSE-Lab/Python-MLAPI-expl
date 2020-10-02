#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# As an assignmentfor for the course Machine Learning we choose to create an Convolutional Neural Network where we are going to classify red blood cells. We choose the following dataset. https://www.kaggle.com/paultimothymooney/blood-cells
# 
# Part of the assignment is that we need to create our own augmentation data from the original samples. We wanted to almost re-create the original. The key is that we have an even distribution over the categories [EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL].
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import random
import scipy
import shutil

from pprint import pprint
from glob import glob
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Copy images to an own created directory for convinience

# In[ ]:


get_ipython().system('mkdir ~/blood-cells')
get_ipython().system('mkdir ~/blood-cells/dataset-master/')
get_ipython().system('cp -r ../input/dataset-master/dataset-master/labels.csv ~/blood-cells/dataset-master/')
get_ipython().system('cp -r ../input/dataset-master/dataset-master/JPEGImages/ ~/blood-cells/dataset-master/')


# In[ ]:


os.listdir('/tmp/blood-cells/dataset-master/')


# Root path. From here we can reach all the files.

# In[ ]:


PATH = Path('/tmp/blood-cells/dataset-master/') # Kaggle is '/tmp/blood-cells/dataset-master/'


# In[ ]:





# #### Reading CSV file for image information
# Change image name to the full name of the images, just as in the directory.
# We do this so that we can map the path to the image directly.

# In[ ]:


labels_df = pd.read_csv(PATH.joinpath('labels.csv'))
labels_df = labels_df.dropna(subset=['Image', 'Category']) # drop columns that we don't use
labels_df['Image'] = labels_df['Image'].apply(
    lambda x: 'BloodImage_0000' + str(x) + '.jpg' 
    if x < 10 
    else ('BloodImage_00' + str(x) + '.jpg' if x > 99 else 'BloodImage_000' + str(x) + '.jpg')
)
labels_df = labels_df[['Image', 'Category']]
labels_df.head(15)


# #### Finding all the test set images

# In[ ]:


all_image_paths = {
    os.path.basename(x): x for x in glob(
        os.path.join(PATH, '*', '*.jpg')
    )
}
# all_image_paths = [x for x in p.glob('**/*.jpg')]
print('Scans found:', len(all_image_paths), ', Total Headers', labels_df.shape[0])


# #### Mapping the image to the right image path

# In[ ]:


labels_df['image_path'] = labels_df['Image'].map(all_image_paths.get)
labels_df.head(10)


# ##### Even more cleaning of the data
# Dropping categories that have not enough data. Anything less then 10 we drop. Meaning that we only get to keep 4 categories.

# In[ ]:


count_of_labels_per_cat = labels_df.Category.value_counts()
to_remove_cat = count_of_labels_per_cat[count_of_labels_per_cat < 10].index 
df_next = labels_df.replace(to_remove_cat, np.nan)
df = df_next.dropna()
print(df.Category.value_counts())
df.head(12)


# In[ ]:


train_df, test_df = train_test_split(
    df, 
    test_size = 0.30,
    stratify = df['Category']
)
print('shape of data split: ', 'train:', f'{train_df.shape}', 'test:', f'{test_df.shape}')


# -

# In[ ]:


print(train_df.Category.value_counts(), '\n')
print(test_df.Category.value_counts())


# Creating directory and seperate the bulk of images to its own categories

# In[ ]:


os.mkdir(f'{PATH}/train_original')
os.mkdir(f'{PATH}/test_original')

for f in df.Category.unique():
    os.mkdir(f'{PATH}/train_original/{f}')
    os.mkdir(f'{PATH}/test_original/{f}')


# In[ ]:


for p in train_df.itertuples():
    file_path = f'{PATH}/JPEGImages/{p.Image}' 
    train_path = f'{PATH}/train_original/{p.Category}/{p.Image}'
    shutil.copyfile(f'{file_path}', f'{train_path}')


# In[ ]:


for p in test_df.itertuples():
    file_path = f'{PATH}/JPEGImages/{p.Image}' 
    test_path = f'{PATH}/test_original/{p.Category}/{p.Image}'
    shutil.copyfile(f'{file_path}', f'{test_path}')


# Creating an ImageDataGenerator. From this we get the augmented images.
# 
# Tweak these settings to your liking, to get the right kind of augmented images

# In[ ]:


augmented_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
#     zoom_range=0.1,
    channel_shift_range=50,
    rescale=1. / 255,
    shear_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='constant'
)


# #### Creating 
# Iterate over the test folder. Alter every image and create the altered images in the right folder, with respect to the category.
# 
# Tweak the 'batches' > 20 for wanting to create the amount of augmented images

# In[ ]:


os.mkdir(f'{PATH}/train')
 
for f in df.Category.unique():
    os.mkdir(f'{PATH}/train/{f}')

for f in os.listdir(f'{PATH}/train_original'):
    if (str(f) != '.DS_Store'):
        pics = os.listdir(f'{PATH}/train_original/{f}')
        
        for img in pics:
            if len(os.listdir(f'{PATH}/train/{f}')) >= 2500:
                break
            loaded_img = load_img(f'{PATH}/train_original/{f}/{img}')
            x_test = img_to_array(loaded_img)
            x_test = x_test.reshape((1,) + x_test.shape)
            augmented_datagen.fit(x_test)
            batches = 0
            for batch in augmented_datagen.flow(x_test, save_to_dir=f'{PATH}/train/{f}', save_prefix=f'{f}'):
                batches += 1
                if batches > 400 or len(os.listdir(f'{PATH}/train/{f}')) >= 2500 :
                    break


# In[ ]:


os.mkdir(f'{PATH}/test')
 
for f in df.Category.unique():
    os.mkdir(f'{PATH}/test/{f}')

for f in os.listdir(f'{PATH}/test_original'):
    if (str(f) != '.DS_Store'):
        pics = os.listdir(f'{PATH}/test_original/{f}')
        
        for img in pics:
            if len(os.listdir(f'{PATH}/test/{f}')) >= 500:
                break
            loaded_img = load_img(f'{PATH}/test_original/{f}/{img}')
            x_test = img_to_array(loaded_img)
            x_test = x_test.reshape((1,) + x_test.shape)
            augmented_datagen.fit(x_test)
            batches = 0
            for batch in augmented_datagen.flow(x_test, save_to_dir=f'{PATH}/test/{f}', save_prefix=f'{f}'):
                batches += 1
                if batches > 150 or len(os.listdir(f'{PATH}/test/{f}')) >= 500 :
                    break


# In[ ]:


# Example of just created images in the file system of Kaggle
print(os.listdir('/tmp/blood-cells/dataset-master/train/MONOCYTE')[:3], '\n')
print(os.listdir('/tmp/blood-cells/dataset-master/test/MONOCYTE')[:3], '\n')


# ### Augmented images
# 
# Find all the just created augmented images.

# In[ ]:


all_augmented_train_image_paths = {
    os.path.basename(x): x for x in glob(
        os.path.join(PATH / "train/", '*', '*.png')
        )   
}
all_augmented_test_image_paths = {
    os.path.basename(x): x for x in glob(
        os.path.join(PATH / "test/", '*', '*.png')
        )   
}
# all_image_paths = [x for x in p.glob('**/*.jpg')]
print('Augmented train scans found:', len(all_augmented_train_image_paths))
print('Augmented test scans found:', len(all_augmented_test_image_paths))


# In[ ]:





# Create new DataFrame, just for augmented images.

# In[ ]:


train_augmented_df = pd.DataFrame()
test_augmented_df = pd.DataFrame()

train_augmented_df['image'] = all_augmented_train_image_paths.keys()
test_augmented_df['image'] = all_augmented_test_image_paths.keys()

train_augmented_df['image_path'] = train_augmented_df['image'].map(all_augmented_train_image_paths.get)
test_augmented_df['image_path'] = test_augmented_df['image'].map(all_augmented_test_image_paths.get)

train_augmented_df['category'] = train_augmented_df['image'].apply(lambda x: x.split('_')[0])
test_augmented_df['category'] = test_augmented_df['image'].apply(lambda x: x.split('_')[0])

print(train_augmented_df.category.value_counts(), '\n')
print(test_augmented_df.category.value_counts())


# ### Normalise the distribution over the augmented data
# We want to have an equel set distribution over the categories

# In[ ]:





# In[ ]:


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
)
test_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator=train_datagen.flow_from_directory(
    f'{PATH}/train',
    target_size=(72,96), 
    batch_size=32,
)
test_generator=test_datagen.flow_from_directory(
    f'{PATH}/test', 
    class_mode="categorical", 
    target_size=(72,96), 
    batch_size=32
)


# In[ ]:


#augmented_datagen

x,y = train_generator.next()
for i in range(0,1):
    image = x[i]
    plt.imshow(image)
    plt.show()


# -

# In[ ]:


for i in [train_generator, test_generator]:
    x, y = next(i)
    fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))
    for (c_x, c_y, c_ax) in zip(x, y, m_axs.flatten()):
    #     print(c_x, '\n') # image array
    #     print(c_y, '\n') # category
    #     print(c_ax, '\n') # dunno?
        c_ax.imshow(c_x[:,:])
        c_ax.set_title(c_y)
        c_ax.axis('off')


# In[ ]:


# from sklearn.metrics import classification_report, confusion_matrix

# pred_Y = tb_model.predict(test_X, 
#                           batch_size = 32, 
#                           verbose = True)

# plt.matshow(confusion_matrix(test_Y, pred_Y>0.5))
# print(classification_report(test_Y, pred_Y>0.5, target_names = ['Healthy', 'Cardiomegaly']))


# In[ ]:





# In[ ]:




