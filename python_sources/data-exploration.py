#!/usr/bin/env python
# coding: utf-8

# # OREGON WILDLIFE DATA SET

# > ## install GapCV

# In[2]:


get_ipython().run_cell_magic('capture', '', '!pip install -q gapcv')


# In[ ]:


get_ipython().system('pip show gapcv')


# ## import libraries

# In[3]:


import os
import cv2
import numpy as np

from gapcv.vision import Images

import warnings
warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("../input"))
print(os.listdir("./"))


# ## Utils function

# In[ ]:


def plot_sample(imgs_set, labels_set, img_size=(12,12), columns=4, rows=4, random=False):
    """
    Plot a sample of images
    """
    
    fig=plt.figure(figsize=img_size)
    
    for i in range(1, columns*rows + 1):
        if random:
            img_x = np.random.randint(0, len(imgs_set))
        else:
            img_x = i-1
        img = imgs_set[img_x]
        ax = fig.add_subplot(rows, columns, i)
        ax.set_title(str(labels_set[img_x]))
        plt.axis('off')
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# ## oregon_wildlife.zip data set content

# In[10]:


data_set = '../input/oregon_wildlife/oregon_wildlife'

labels = os.listdir(data_set)
print("Number of Labels:", len(labels))

total = 0
for lb in os.scandir(data_set):
    print('folder: {} images: {}'.format(lb.name, len(os.listdir(lb))))
    total += len(os.listdir(lb))
print('Total images:', total)


# ## GapCV image preprocessing
# In this step, I'm going to create two lists and use them as parameters in GapCV para to only load 5 classes of the entire data set.  
# If you want to preprocess the entire data set skip the loop and run:
# 
#     images = Images(
#         dataset_name,
#         dataset_content,
#         config=[
#             'resize=(128,128)',
#             'store', 'stream'
#         ]
#     )

# ### how to preprocess a new wildlife data set 

# In[ ]:


dataset_name = 'wildlife128'
wildlife_filter = ['black_bear', 'bald_eagle', 'cougar', 'elk', 'gray_wolf']

if not os.path.isfile('{}.h5'.format(dataset_name)):
    ## create two list to use as paramaters in GapCV
    print('{} preprocessing started'.format(dataset_name))
    images_list = []
    classes_list = []
    for folder in os.scandir('../input/oregon_wildlife/oregon_wildlife'):
        if folder.name in wildlife_filter:
            for image in os.scandir(folder.path):
                images_list.append(image.path)
                classes_list.append(image.path.split('/')[-2])

    ## GapCV
    images = Images(
        dataset_name,
        images_list,
        classes_list,
        config=[
            'resize=(128,128)',
            'store', 'stream'
        ]
    )


# ### wildlife128.h5 data set content**

# In[ ]:


print('content:', os.listdir("./"))
print('time to load data set:', images.elapsed)
print('number of images in data set:', images.count)
print('classes:', images.classes)
print('data type:', images.dtype)


# ## split data set between train and test data sets

# In[ ]:


images.split = 0.2
X_test, Y_test = images.test
images.minibatch = 128
gap_generator = images.minibatch
X_train, Y_train = next(gap_generator)


# ## train data set sample

# In[ ]:


plot_sample(X_train, Y_train, random=True)


# ## test data set sample

# In[ ]:


plot_sample(X_test, Y_test, random=True)


# ## GapCV config 'stream'
# 
# If we already have the data set we can re-use it in `stream` config. This option alow us to use the generator and only load in memory the choose minibatch. Ideal for limited CPU or GPU resources. Plus while is streaming the data the generator split the minibatch and provides half of non-augmented and augmented images when is fitting the model.

# In[ ]:


del images
images = Images(
    config=['stream'],
    augment=[
        'flip=horizontal',
        'edge',
        'zoom=0.3',
        'denoise'
    ]
)
images.load(dataset_name)
print('{} dataset ready for streaming'.format(dataset_name))


# In[ ]:


images.split = 0.2
X_test, Y_test = images.test
images.minibatch = 16
gap_generator = images.minibatch
X_train, Y_train = next(gap_generator)


# ### data set with image augmentation

# In[ ]:


plot_sample(X_train, Y_train)

