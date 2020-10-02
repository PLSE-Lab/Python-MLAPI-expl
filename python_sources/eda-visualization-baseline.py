#!/usr/bin/env python
# coding: utf-8

# Hi everyone! This is my first kaggle notebook, but I hope that you'll find it usefull.
# 
# Most significant features:
# * useful train.csv unpacking procedure
# * EDA of full data, not "first car only"
# * proper masking and vizualization without alpha hacks
# * x, y, z mapping to image plane
# * baseline submission without ml

# In[ ]:


import pandas as pd
import os.path
import PIL
import PIL.ImageChops  
import PIL.ImageOps
import numpy as np

from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.figsize'] = [15, 5]


# # Unpack train.csv

# There is a one to many relationship: ImageId -> Cars. It's not very convenient to analyze it this way. So lets construct a table, that has a one to one relationship.

# In[ ]:


def get_path(path):
    return os.path.join('/kaggle/input/pku-autonomous-driving/', path)


# In[ ]:


train_df = pd.read_csv(get_path('train.csv'))


# In[ ]:


train_df.head()


# In[ ]:


def unpack(group):
    row = group.iloc[0]
    result = []
    data = row['PredictionString']
    while data:
        data = data.split(maxsplit=7)
        result.append(OrderedDict((
            ('image_id', row['ImageId']),
            ('model_type', int(data[0])),
            ('yaw', float(data[1])), 
            ('pitch', float(data[2])), 
            ('roll', float(data[3])), 
            ('x', float(data[4])), 
            ('y', float(data[5])),
            ('z', float(data[6]))
        )))
        data = data[7] if len(data) == 8 else ''
    return pd.DataFrame(result)


# In[ ]:


unpacked_train_df = train_df.groupby('ImageId', group_keys=False).apply(unpack).reset_index(drop=True)


# In[ ]:


unpacked_train_df.head()


# # 3D to 2D

# It's rather straightforward to convert x, y, z to image plane coordinates, but it took me some time to figure this out, so here it is.

# In[ ]:


get_ipython().system('cat /kaggle/input/pku-autonomous-driving/camera/camera_intrinsic.txt')


# In[ ]:


def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    return x * fx / z + cx, y * fy / z + cy


# # Number of cars per image distribution

# In[ ]:


unpacked_train_df.groupby('image_id').size().describe()


# In[ ]:


unpacked_train_df.groupby('image_id').size().hist(bins=44);


# In[ ]:


len(train_df), len(unpacked_train_df)


# # Visualization

# In[ ]:


def read_masked_image(image_id, partition='train'):
    i = PIL.Image.open(get_path('{}_images/{}.jpg'.format(partition, image_id)))
    try:
        m = PIL.Image.open(get_path('{}_masks/{}.jpg'.format(partition, image_id)))
        return PIL.Image.composite(m, i, m.convert(mode='L'))
    except FileNotFoundError:
        return i


# In[ ]:


def highlight(df, image_id):
    df = df[df['image_id'] == image_id]
    coords = df.apply(lambda row: pd.Series(convert_3d_to_2d(row['x'], row['y'], row['z'])), axis=1).values
    plt.figure(figsize=(10, 10))
    plt.imshow(read_masked_image(image_id))
    plt.plot(coords[:, 0], coords[:, 1], 'ro', alpha=0.8, markersize=10)


# ## Sanity check on one car images

# In[ ]:


s = unpacked_train_df.groupby('image_id').size()
for image_id in s[s == 1].index:
    highlight(unpacked_train_df, image_id)


# # Sanity check with multiple cars

# Notice the occlusion here.

# In[ ]:


highlight(unpacked_train_df, 'ID_001d6829a')


# # Car models distribution

# In[ ]:


unpacked_train_df['model_type'].value_counts().sort_index().plot(grid=True);


# In[ ]:


unpacked_train_df['model_type'].value_counts().describe()


# # Yaw, pitch, roll distributions

# Let's define a helper function that will allow us to find some specific examples in the dataset.

# In[ ]:


def find_example(df, column, func):
    row = df[df[column] == func(df[column])][:1]
    highlight(row, row['image_id'].iloc[0])
    return row


# In[ ]:


unpacked_train_df['yaw'].hist(bins=100);


# In[ ]:


unpacked_train_df['pitch'].hist(bins=100);


# In[ ]:


unpacked_train_df['roll'].hist(bins=100);


# According to this distributions "pitch" is actually yaw, "yaw" is pitch, "roll" is roll, but upside down for some reason. More over input is stated as 'model type, yaw, pitch, roll, x, y, z', but output should be 'pitch, yaw, roll, x, y, z and confidence'. Notice the yaw - pitch switch.
# 
# Let's take a look at some examples. Feal free to experiment on your own. You'll find out some really weird markup errors.

# In[ ]:


find_example(unpacked_train_df, 'pitch', lambda x: x.min())


# In[ ]:


find_example(unpacked_train_df, 'pitch', lambda x: -1.58834)


# In[ ]:


find_example(unpacked_train_df, 'pitch', lambda x: 1.58841)


# # Random submission 

# Lets simply submit the most likely (independently though) parameters over the train dataset.

# In[ ]:


submission_df = pd.read_csv(get_path('sample_submission.csv'))


# In[ ]:


df = unpacked_train_df.rename({'pitch': 'yaw', 'yaw': 'pitch'})


# In[ ]:


columns = ('pitch', 'yaw', 'roll', 'x', 'y', 'z')
best_guess = []
for c in columns:
    best_guess.append(df[c].median())
best_guess.append(1.0)  # confidence


# In[ ]:


submission_df['PredictionString'] = ' '.join(map(str, best_guess))


# In[ ]:


submission_df.to_csv('best_guess_submission.csv', index=False)


# In[ ]:


submission_df.head()

