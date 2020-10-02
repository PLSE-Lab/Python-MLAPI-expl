#!/usr/bin/env python
# coding: utf-8

# ## Hi everyone, this kernel is about applying different segmentation methods to the images of the cervix
# 
# Methods used:
# - a channel saturaion
# - Watershed
# - Edge detection
# - K-means
# 
# References can be found at the beginning of each method

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

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
from sklearn import svm
import pickle

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


print(len(additional_type_1_files), len(additional_type_2_files), len(additional_type_3_files))
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


def get_image_data(image_id, image_type, rsz_ratio=1):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    if rsz_ratio != 1:
        img = cv2.resize(img, dsize=(int(img.shape[1] * rsz_ratio), int(img.shape[0] * rsz_ratio)))
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# # Definitions

# In[ ]:


# defining the color of pixels outside of the roi in RGB
mask_color = [0, 0, 0]

# a channel saturation threshold
LOWER_A_SAT = 0
UPPER_A_SAT = 300

# resize ratio for computational speed
resize_ratio = 0.1

# figure size for plotting
figure_size = 7

# sample images
normal_ids = [['1414', 'Type_1'], ['60', 'Type_2'], ['491', 'Type_3']]
difficult_ids = [['212', 'Type_3'], ['1093', 'Type_1'], ['1473', 'Type_1'], ['267', 'Type_1'], ['446', 'Type_1']]
ids = [['1414', 'Type_1'], ['60', 'Type_2'], ['267', 'Type_1']]


# # a_channel and distance from center gaussian mixture
# 
# Taken from chattob's kernel: https://www.kaggle.com/chattob/cervix-segmentation-gmm
# 
# Ispired by:  https://www.researchgate.net/publication/24041301_Automatic_Detection_of_Anatomical_Landmarks_in_Uterine_Cervix_Images

# In[ ]:


def Ra_space(img, Ra_ratio=1, a_upper_threshold=UPPER_A_SAT, a_lower_threshold=LOWER_A_SAT):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w*h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w/2-i)*(w/2-i) + (h/2-j)*(h/2-j))
            Ra[i*h+j, 0] = R
            a = min(imgLab[i][j][1], a_upper_threshold)
            a = max(imgLab[i][j][1], a_lower_threshold)
            Ra[i*h+j, 1] = a
            
    if Ra_ratio != 1:
        Ra[:,0] /= max(Ra[:,0])
        Ra[:,0] *= Ra_ratio
        Ra[:,1] /= max(Ra[:,1])

    return Ra


# In[ ]:


def crop_roi(image, display_image=False):
    
    # creating the R-a feature for the image
    Ra_array = Ra_space(image)
    
    # k-means gaussian mixture model
    g = mixture.GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0, init_params = 'kmeans')
    g.fit(Ra_array)
    labels = g.predict(Ra_array)
    
    # creating the mask array and assign the correct cluster label
    boolean_image_mask = np.array(labels).reshape(image.shape[0], image.shape[1])
    
    if display_image==True:
        outer_cluster_label = boolean_image_mask[0,0]
    
        new_image = image.copy()
    
        for i in range(boolean_image_mask.shape[0]):
            for j in range(boolean_image_mask.shape[1]):
                if boolean_image_mask[i, j] == outer_cluster_label:
                    new_image[i, j] = mask_color
    
        plt.figure(figsize=(figure_size,figure_size))
    
        plt.subplot(221)
        plt.title("Original image")    
        plt.imshow(image), plt.xticks([]), plt.yticks([])
    
        plt.subplot(222)
        plt.title("Region of interest")
        plt.imshow(new_image), plt.xticks([]), plt.yticks([])
    
        a_channel = np.reshape(Ra_array[:,1], (image.shape[0], image.shape[1]))
        plt.subplot(223)
        plt.title("a channel")
        plt.imshow(a_channel, cmap='gist_heat'), plt.xticks([]), plt.yticks([])
  
        plt.subplot(224)
        plt.title("Gaussiam mixture scatter plot")    
        plt.scatter(Ra_array[:,0], Ra_array[:,1], c=boolean_image_mask)
        plt.show()
    
    return boolean_image_mask


# In[ ]:


#ids = difficult_ids
#ids = normal_ids

for i in range(len(ids[:])):
    print('Loading image %i out of %i' % (i+1, len(ids[:])))
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    # watershed algorithm
    crop_roi(image, True)


# # Watershed algorithm
# 
# http://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html

# In[ ]:


def watershed(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
     
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    plt.figure(figsize=(figure_size,figure_size))
    
    plt.subplot(221)
    plt.title("Original image")    
    plt.imshow(img), plt.xticks([]), plt.yticks([])
    
    plt.subplot(222)
    plt.title("thres")    
    plt.imshow(thresh, cmap="Greys_r"), plt.xticks([]), plt.yticks([])
    
    plt.subplot(223)
    plt.title("sure_fg")    
    plt.imshow(sure_fg, cmap="Greys_r"), plt.xticks([]), plt.yticks([])
    
    plt.subplot(224)
    plt.title("sure_bg")    
    plt.imshow(sure_bg, cmap="Greys_r"), plt.xticks([]), plt.yticks([])
    


# In[ ]:


#ids = difficult_ids
#ids = normal_ids

for i in range(len(ids[:])):
    print('Loading image %i out of %i' % (i+1, len(ids[:])))
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    # watershed algorithm
    watershed(image)


# # Edge detection
# 
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php

# In[ ]:


def edge_detection(img0):
    # converting to gray scale
    gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(gray,(3,3),0)

    # convolute with proper kernels
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

    plt.figure(figsize=(figure_size,figure_size))
    
    plt.subplot(2,2,1),plt.imshow(img0)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()


# In[ ]:


#ids = difficult_ids
#ids = normal_ids[:]

for i in range(len(ids[:])):
    print('Loading image %i out of %i' % (i+1, len(ids[:])))
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    edge_detection(image)


# # Canny Edge Detection
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Canny_Edge_Detection.php

# In[ ]:


def canny_edge(img):
    edges = cv2.Canny(img,100,200)

    plt.figure(figsize=(figure_size,figure_size))
    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


# In[ ]:


#ids = difficult_ids
#ids = normal_ids[:]

for i in range(len(ids[:])):
    print('Loading image %i out of %i' % (i+1, len(ids[:])))
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    canny_edge(image)


# # K-Means color clustering
# 
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html#kmeans-opencv

# In[ ]:


def kmeans_color(img, K=8):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    
    plt.figure(figsize=(figure_size,figure_size))
    
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(1,2,2),plt.imshow(res2)
    plt.title('K = %i' % K), plt.xticks([]), plt.yticks([])
        
    plt.show()
    
    return res2


# In[ ]:


#ids = difficult_ids[:]
#ids = normal_ids[:]

for i in range(len(ids[:])):
    print('Loading image %i out of %i' % (i+1, len(ids[:])))
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    kmeans_color(image, K=4)
    kmeans_color(image, K=8)


# # K-Means with canny edge

# In[ ]:


#ids = difficult_ids[:]
#ids = normal_ids[:]

for i in range(len(ids[:])):
    print('Loading image %i out of %i' % (i+1, len(ids[:])))
    image_id = ids[i]
    image = get_image_data(image_id[0], image_id[1], resize_ratio)
    
    image = kmeans_color(image)
    canny_edge(image)


# To be continued
