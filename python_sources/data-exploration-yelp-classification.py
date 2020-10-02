#!/usr/bin/env python
# coding: utf-8

# # Data Exploration for Yelp Image Classification Challenge
# ---
# Exploring the image data and labels, visualizing random samples of images, and plotting image shape distributions.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import cv2
import random
import os
get_ipython().run_line_magic('matplotlib', 'inline')

# Init plotly for offline plotting
plotly.offline.init_notebook_mode(connected=True)

print('Pandas version:', pd.__version__)
print('Numpy version:', np.__version__)
print('OpenCV version:', cv2.__version__)
print('Plotly version:', plotly.__version__)
print(os.listdir("../input"))


# # Data Loading
# ---
# Load the data. The csv files that map image IDs to business IDs and business IDs to labels

# In[ ]:


# Load training data that maps business ID to labels
train = pd.read_csv('../input/train.csv')
display(train.head())
print('Shape of train data:', train.shape)
print('Number of unique businesses:', train.shape[0])


# In[ ]:


# Load training data that maps photos to business ID
train_photo_to_id = pd.read_csv('../input/train_photo_to_biz_ids.csv')
display(train_photo_to_id.head())
print('Shape of train_photo_to_id:', train_photo_to_id.shape)
print('Number of images in training set:', train_photo_to_id.shape[0])


# In[ ]:


train_dir = '../input/train_photos'
train_imgs = os.listdir(train_dir)

test_dir = '../input/test_photos'
test_imgs = os.listdir(test_dir)

print('Number of training images:', len(train_imgs))
print('Number of testing images:', len(test_imgs))


# ## Missing Values

# In[ ]:


# Business id to labels dataframe
print('Total number of missing labels:', train['labels'].isnull().sum())
display(train[train['labels'].isnull()])


# # Visualize some Images
# ---
# Take some random images and visualize them with their lables

# In[ ]:


# Randomly sample 8 images
imgs_samples = random.sample(train_imgs, 8)

# Plot random sample of 8 images
plt.figure(figsize=(15, 10))
for i in range(len(imgs_samples)):
    # OpenCV2 reads images in BGR format
    img = cv2.imread(os.path.join(train_dir, imgs_samples[i]))
    # Switch color channels to RGB to make compatible with matplotlib imshow func
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Grab image's business ID and labels
    business = train_photo_to_id.loc[train_photo_to_id['photo_id'] == int(imgs_samples[i][:-4]), 'business_id']
    labels = train.loc[train['business_id'] == business.values[0], 'labels']
    # Annotate each image with image ID, business ID, and labels
    title = "Image ID: " + imgs_samples[i] + ' Business: ' + str(business.values[0]) + '\nLabels: ' + ''.join(labels.values)
    # Plot the image
    plt.subplot(2, 4, i+1)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
plt.show()


# ## List of Labels
# 
# 0: good_for_lunch
# 
# 1: good_for_dinner
# 
# 2: takes_reservations
# 
# 3: outdoor_seating
# 
# 4: restaurant_is_expensive
# 
# 5: has_alcohol
# 
# 6: has_table_service
# 
# 7: ambience_is_classy
# 
# 8: good_for_kids

# # Plot Image Size Distribution
# ---
# Plot the distribution of image sizes for training and testing datasets. This is useful because if there are images with different pixel shapes then we have to resize them before feeding into a machine learning model.

# In[ ]:


def load_img_shapes(path_to_img):
    """ Return only the shape of an image (width, height, channels) """
    return cv2.imread(path_to_img).shape


# In[ ]:


# Initialize arrays to hold image sizes
train_shapes = []
test_shapes = []
# Load in training/testing image sizes
for i in range(len(train_imgs)):
    img_path = os.path.join(train_dir, train_imgs[i])
    train_shapes.append(load_img_shapes(img_path))
for i in range(len(test_imgs)):
    img_path = os.path.join(test_dir, test_imgs[i])
    test_shapes.append(load_img_shapes(img_path))

# Store training image sizes in dataframe
df_train = pd.DataFrame({'Shapes': train_shapes})
train_counts = df_train['Shapes'].value_counts()
# Store testing image sizes in dataframe
df_test = pd.DataFrame({'Shapes': test_shapes})
test_counts = df_test['Shapes'].value_counts()

print("Training Image Shapes: First 100")
for i in range(100):
    print("Shape %s counts: %d" % (train_counts.index[i], train_counts.values[i]))
print("*"*50)
print("Testing Image Shapes: First 100")
for i in range(100):
    print("Shape %s counts: %d" % (test_counts.index[i], test_counts.values[i]))


# In[ ]:


# Create barplot for image sizes distribution (training set)
x_train = train_counts.index[:100]
x_train = [str(x) for x in x_train]
y_train = train_counts.values[:100]

# Only plot first 100 value counts
x_test = test_counts.index[:100]
x_test = [str(x) for x in x_test]
y_test = test_counts.values[:100]

# Create traces
training_trace = go.Bar(x=x_train,
                        y=y_train,
                        marker=dict(
                            color='rgb(158,202,225)',
                            line=dict(
                                color='rgb(8,48,107)',
                                width=1.5),
                        ),
                        opacity=0.6,
                        name='training'
                       )
testing_trace = go.Bar(x=x_test,
                       y=y_test,
                       marker=dict(
                           color='rgb(58,102,245)',
                           line=dict(
                               color='rgb(8,48,107)',
                               width=1.5),
                       ),
                       opacity=0.6,
                       name='testing'
                      )
# Create layout
layout = go.Layout(font = dict(family = "Overpass"),
                   title = 'Image size distributions (Top is training set, bottom is testing set)',
#                    xaxis = dict(tickangle=-45, title = "Image shapes"),
#                    xaxis2 = dict(tickangle=-45, title = "Image shapes"),
                  )

fig = plotly.tools.make_subplots(rows=2, cols=1,
                                 vertical_spacing=0.4
                                )
fig.append_trace(training_trace, 1, 1)
fig.append_trace(testing_trace, 2, 1)
fig['layout'].update(font = dict(family = "Overpass"),
                     title='Image size distributions (Top is training set, bottom is testing set)',
                     xaxis = dict(tickangle=-45, title = "Image shapes"),
                     xaxis2 = dict(tickangle=-45, title = "Image shapes")
                    )
plotly.offline.iplot(fig, validate=False, show_link=False)


# Most common image shape is (357, 500, 3) and all images have 3 color channels (RGB). There are a lot of images with different shapes so there will definitely be a lot of resizing. Probably to have a shape of (357, 500, 3) since that is the most common shape for both training and testing sets.

# # Distribution of Classes
# ---
# Each image can have multiple classes assigned to it. Here we will look at the number of times each class occurs over the entire training set.

# In[ ]:


# Count all labels in training set
all_labels = ' '.join(list(train['labels'].fillna('nan').values)).split()
from collections import Counter
label_counts = Counter(all_labels)


# In[ ]:


for key in label_counts:
    print('Label {0} appears {1} times in training dataset'.format(key, label_counts[key]))


# # Duplicate Values in Dataset
# ---
# Find duplicate values in our training dataset if any.

# In[ ]:


# train_photo_to_id.groupby("photo_id")
print('Number of duplicate photo IDs:', len(train_photo_to_id[train_photo_to_id['photo_id'].duplicated()]))
print('Number of duplicate business IDS:', len(train[train['business_id'].duplicated()]))

