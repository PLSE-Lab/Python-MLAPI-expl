#!/usr/bin/env python
# coding: utf-8

# ## The segmentation is based on @chattob's reference to "Greenspan et al., Automatic detection of anatomical landmarks in uterine cervix images. IEEE transaction on medical imaging, 2009." in his/her kernel.
# 
# # Loading images (vfdev's kernel)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

#additional imports
import matplotlib.pylab as plt
import math
from sklearn import mixture
from sklearn.utils import shuffle

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
from glob import glob
TRAIN_DATA = "../input/train"
type_1_files = glob(os.path.join(TRAIN_DATA, "Type_1", "*.jpg"))
type_1_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_1"))+1:-4] for s in type_1_files])
type_2_files = glob(os.path.join(TRAIN_DATA, "Type_2", "*.jpg"))
type_2_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_2"))+1:-4] for s in type_2_files])
type_3_files = glob(os.path.join(TRAIN_DATA, "Type_3", "*.jpg"))
type_3_ids = np.array([s[len(os.path.join(TRAIN_DATA, "Type_3"))+1:-4] for s in type_3_files])

print(len(type_1_files), len(type_2_files), len(type_3_files))
print("Type 1", type_1_ids[:10])
print("Type 2", type_2_ids[:10])
print("Type 3", type_3_ids[:10])


# In[ ]:


TEST_DATA = "../input/test"
test_files = glob(os.path.join(TEST_DATA, "*.jpg"))
test_ids = np.array([s[len(TEST_DATA)+1:-4] for s in test_files])
print(len(test_ids))
print(test_ids[:10])


# In[ ]:


ADDITIONAL_DATA = "../input/additional"
additional_type_1_files = glob(os.path.join(ADDITIONAL_DATA, "Type_1", "*.jpg"))
additional_type_1_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_1"))+1:-4] for s in additional_type_1_files])
additional_type_2_files = glob(os.path.join(ADDITIONAL_DATA, "Type_2", "*.jpg"))
additional_type_2_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_2"))+1:-4] for s in additional_type_2_files])
additional_type_3_files = glob(os.path.join(ADDITIONAL_DATA, "Type_3", "*.jpg"))
additional_type_3_ids = np.array([s[len(os.path.join(ADDITIONAL_DATA, "Type_3"))+1:-4] for s in additional_type_3_files])

print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_2_files))
print("Type 1", additional_type_1_ids[:10])
print("Type 2", additional_type_2_ids[:10])
print("Type 3", additional_type_3_ids[:10])


# In[ ]:


def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type[1:])
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ## Creating an R, a array
# 
# # based on @chattob's "cervix segmentation GMM" kernel

# In[ ]:


# defining the color of pixels outside of the roi in RGB
mask_color = [0, 0, 0]


# In[ ]:


def Ra_space(img, Ra_ratio, a_threshold):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            Ra[i*h+j, 1] = min(imgLab[i][j][1], a_threshold)
            
    Ra[:,0] /= max(Ra[:,0])
    Ra[:,0] *= Ra_ratio
    Ra[:,1] /= max(Ra[:,1])

    return Ra


# ### Clustering pixels to 2 groups

# In[ ]:


# sample images
img_1 = get_image_data('7', 'Type_1')
img_2 = get_image_data('35', 'Type_1')
img_3 = get_image_data('77', 'Type_2')
img_4 = get_image_data('178', 'Type_2')
img_5 = get_image_data('11', 'Type_3')
img_6 = get_image_data('212', 'Type_3')

test_1 = get_image_data('81', 'Type_1')
test_2 = get_image_data('31', 'Type_2')
test_3 = get_image_data('62', 'Type_3')

sample_images = [img_1, img_2, img_3, img_4, img_5, img_6]
sample_labels = ['Type_1', 'Type_1', 'Type_2', 'Type_2', 'Type_3', 'Type_3']
sample_test_images = [test_1, test_2, test_3]
sample_test_labels = ['Type_1', 'Type_2', 'Type_3']

#image reduction for computational speed
tile_size = (256, 256)

small_sample_images = []
small_sample_test_images = []

for image in sample_images:
    small_sample_images.append(cv2.resize(image, dsize=tile_size))
    
for image in sample_test_images:
    small_sample_test_images.append(cv2.resize(image, dsize=tile_size))   


# SVM experimentation

# In[ ]:


def crop_roi(image):
    
    # a channel saturation threshold
    a_threshold = 300
    
    # creating the R-a feature for the image
    Ra_array = Ra_space(image, 1.0, a_threshold)
    
    # k-means gaussian mixture model
    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
    image_array_sample = shuffle(Ra_array, random_state=0)[:1000]
    g.fit(image_array_sample)
    labels = g.predict(Ra_array)
    
    # creating the mask array and assign the correct cluster label
    boolean_image_mask = np.array(labels).reshape(image.shape[0], image.shape[1])
    outer_cluster_label = boolean_image_mask[0,0]
    
    new_image = image.copy()
    
    for i in range(boolean_image_mask.shape[0]):
        for j in range(boolean_image_mask.shape[1]):
            if boolean_image_mask[i, j] == outer_cluster_label:
                new_image[i, j] = mask_color
    
    plt.figure(figsize=(10,10))
    
    plt.subplot(221)
    plt.title("Original image")    
    plt.imshow(image)
    
    plt.subplot(222)
    plt.title("Region of interest")
    plt.imshow(new_image)
    
    a_channel = np.reshape(Ra_array[:,1], (image.shape[0], image.shape[1]))
    plt.subplot(223)
    plt.title("a channel")
    plt.imshow(a_channel)
  
    plt.subplot(224)
    plt.title("Gaussiam mixture scatter plot")    
    plt.scatter(Ra_array[:,0], Ra_array[:,1], c=boolean_image_mask)
    plt.show()
    
    return new_image


# In[ ]:


def mean_redness(image):
    total = 0
    count = 0
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            #print(sum(image[i,j]))
            #print(sum(mask_color))
            if sum(image[i, j]) != sum(mask_color):
                total += image[i, j, 0]
                count += 1
    mean_red = total / count
    
    return mean_red


# In[ ]:


initial_roi_images = []

for image in small_sample_images:
    initial_roi_images.append(crop_roi(image)) 

mean_red_function = []

for image in initial_roi_images:
    mean_red_function.append(mean_redness(image)) 


# In[ ]:


initial_roi_test_images = []
test_mean_red_function = []

for image in small_sample_test_images:
    initial_roi_test_images.append(crop_roi(image)) 
    
for image in initial_roi_test_images:
    test_mean_red_function.append(mean_redness(image)) 


# In[ ]:


from sklearn import svm

clf = svm.SVC()
clf.fit(np.array(mean_red_function).reshape(-1,1), sample_labels)


# In[ ]:


print(clf.predict(np.array(test_mean_red_function[0]).reshape(-1,1)))
print(clf.predict(np.array(test_mean_red_function[1]).reshape(-1,1)))
print(clf.predict(np.array(test_mean_red_function[2]).reshape(-1,1)))


# Plotting the data

# to be continued

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




