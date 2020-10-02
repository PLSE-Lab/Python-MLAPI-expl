#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The paper [Improving dermoscopy image classification using color constancy](https://ieeexplore.ieee.org/abstract/document/6866131/) shows that using a color compensation technique to reduce the influence of the acquisition setup on the color features extracted from the images provides a improvement on the performance for skin cancer classification. 
# 
# In ISIC 2019 challenge, the top three approaches in both tasks [[1]](https://isic-challenge-stade.s3.amazonaws.com/99bdfa5c-4b6b-4c3c-94c0-f614e6a05bc4/method_description.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=3myZOh3ZfEdZ5UFO8Z1DGmelRrk%3D&Expires=1593068545) [[2]](https://isic-challenge-stade.s3.amazonaws.com/9e2e7c9c-480c-48dc-a452-c1dd577cc2b2/ISIC2019-paper-0816.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=Up3vDSfqGwmf%2FS6nKDOlNSmKZug%3D&Expires=1593068545) [[3]](https://isic-challenge-stade.s3.amazonaws.com/f6d46ceb-bf66-42ff-8b22-49562aefd4b8/ISIC_2019.pdf?AWSAccessKeyId=AKIA2FPBP3II4S6KTWEU&Signature=3XwGMDlkwcusfCwZ1Nk%2Fw5IFwUY%3D&Expires=1593068545) applied the Shades of Gray algorithm [[4]](https://pdfs.semanticscholar.org/acf3/6cdadfec869f136602ea41cad8b07e3f8ddb.pdf) as their color constancy method to improve their performance.
# 
# The goal of this notebook is to apply this algorithm to the current dataset and rise some discussion about this method.

# In[ ]:


import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# The function below was originally designed by [LincolnZjx](https://github.com/LincolnZjx/ISIC_2018_Classification) for the ISIC 2018 challenge.
# 
# Edit: As [Andrew Anikin](https://www.kaggle.com/andrewanikin) pointed out in comments, we shoud include `img = np.clip(img, a_min=0, a_max=255)` to avoid values above 255 in the image, which results in red, yellow, purple etc colors.

# In[ ]:


def shade_of_gray_cc(img, power=6, gamma=None):
    """
    img (numpy array): the original image with format of (h, w, c)
    power (int): the degree of norm, 6 is used in reference paper
    gamma (float): the value of gamma correction, 2.2 is used in reference paper
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype('uint8')
        look_up_table = np.ones((256,1), dtype='uint8') * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i/255, 1/gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype('float32')
    img_power = np.power(img, power)
    rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
    rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec/rgb_norm
    rgb_vec = 1/(rgb_vec*np.sqrt(3))
    img = np.multiply(img, rgb_vec)

    # Andrew Anikin suggestion
    img = np.clip(img, a_min=0, a_max=255)
    
    return img.astype(img_dtype)


# In[ ]:


img_train_paths = glob("../input/siim-isic-melanoma-classification/jpeg/train/*.jpg")
img_test_paths = glob("../input/siim-isic-melanoma-classification/jpeg/test/*.jpg")


# Testing the method and displaying random images to compare the image with and without color constancy

# In[ ]:


_n_samples = 8

for path in img_train_paths[0:_n_samples]:
    _img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
    img_cc = shade_of_gray_cc (img)  
    _, (ax1,ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(img_cc)
    plt.show()


# ### Applying the color constacy method to the whole dataset

# In[ ]:


def apply_cc (img_paths, output_folder_path, resize=None):
    
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)    

    with tqdm(total=len(img_paths), ascii=True, ncols=100) as t:
        
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            img_ = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if resize is not None:
                img_ = cv2.resize(img_, resize, cv2.INTER_AREA)
            np_img = shade_of_gray_cc (img_)            
            cv2.imwrite(os.path.join(output_folder_path, img_name.split('.')[0] + '.jpg'), np_img)
            t.update()


# Applying the color constancy to the train folder

# In[ ]:


apply_cc (img_train_paths, 'cc_train/', (224,224))


# Applying the color constancy to the test folder

# In[ ]:


apply_cc (img_test_paths, 'cc_test/', (224,224))


# **That's all folks!**
# 
# I hope it was useful for you!
