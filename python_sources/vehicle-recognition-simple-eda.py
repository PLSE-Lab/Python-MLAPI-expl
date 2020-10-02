#!/usr/bin/env python
# coding: utf-8

# # TAU Vehicle Type Recognition Competition

# In this competitions we have to classify images of different vehicle types, including cars, bicycles, vans, ambulances, etc. (total 17 categories).
# The data for the competition consists of training data together with the class labels and test data without the labels. The task is to predict the secret labels for the test data. So, this is a straight forward image classification task.
# The data has been collected from the [Open Images dataset](https://storage.googleapis.com/openimages/web/index.html); an annotated collection of over 9 million images. We have only subset of openimages, selected to contain only vehicle categories among the total of 600 object classes.
# 
# Few important points to note:
# * Any use of external data is not allowed
# * The evaluation metric for this competition is classification accuracy; 

# In[ ]:


import os
from collections import defaultdict
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm_notebook as tqdm


# In[ ]:


get_ipython().system('ls -lh ../input/vehicle/')


# In[ ]:


get_ipython().system('ls -lh ../input/vehicle/train/train')


# In[ ]:


get_ipython().system('ls -lh ../input/vehicle/test/testset | head -5')


# Folder description:
# 
# * `train/train` folder contains the training set: a set of images with true labels in the folder names. The the folder contains altogether 28045 files organized in many sub-folders. The sub-folder names are the true classes; i.e., "Boat" sub-folder has all boat images, "Car" sub-folder has all the car images and so on.
# 
# * `test/testset` - folder contains the test set: a set of images without labels. The folder contains altogether 7958 files in a single folder. The file name is the id for the solution's first column; i.e., the predicted class for file "000000.jpg" should appear on the first row of your submission.
# 
# * `sample_submission.csv` - a sample submission file in the correct format.
# 

# Let's get the training data in a dataframe

# In[ ]:


'''walks through the train directory, creates a dataframe with class and filepaths for all images present in the train directory'''

root = '../input/vehicle/train/train/'
data = []
for category in sorted(os.listdir(root)):
    for file in sorted(os.listdir(os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))

df = pd.DataFrame(data, columns=['class', 'file_path'])


# In[ ]:


df.head()


# In[ ]:


len_df = len(df)
print(f"There are {len_df} images")


# In[ ]:


df['class'].value_counts().plot(kind='bar');
plt.title('Class counts');


# In[ ]:


df['class'].value_counts()


# Looks like the dataset is highly imbalanced, we have 8695 `Boat` category images and only 51 `Cart` category images, let's plot few images of each category to see what the images look like.

# In[ ]:


fig = plt.figure(figsize=(25, 16))
for num, category in enumerate(sorted(df['class'].unique())):
    for i, (idx, row) in enumerate(df.loc[df['class'] == category].sample(4).iterrows()):
        ax = fig.add_subplot(17, 4, num * 4 + i + 1, xticks=[], yticks=[])
        im = Image.open(row['file_path'])
        plt.imshow(im)
        ax.set_title(f'Class: {category}')
fig.tight_layout()
plt.show()


# The images are quite distinguishable. Let's analyse the shape and size of the images

# In[ ]:


data = defaultdict(lambda: defaultdict(list))
for idx, row in tqdm(df.iterrows(), total=len(df)):
    image = Image.open(row[1])
    data[row[0]]['width'].append(image.size[0])
    data[row[0]]['height'].append(image.size[1])


# In[ ]:


def plot_dist(category):
    '''plot height/width dist curves for given category'''
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(data[category]['height'], color='darkorange', ax=ax).set_title(category, fontsize=16)
    sns.distplot(data[category]['width'], color='purple', ax=ax).set_title(category, fontsize=16)
    plt.xlabel('size', fontsize=15)
    plt.legend(['height', 'width'])
    plt.show()


# In[ ]:


for category in df['class'].unique():
    plot_dist(category)


# There is quite a variation in sizes for almost all categories

# In[ ]:


sample_submission = pd.read_csv('../input/vehicle/sample_submission.csv')
sample_submission.head()


# In[ ]:


sample_submission.shape


# We have to predict for 7958 test images

# In[ ]:


sample_submission.to_csv('submission.csv', index=False)


# ## All the best :)
