#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

from random import randrange
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from skimage.segmentation import mark_boundaries
from skimage.morphology import skeletonize
from skimage import io
from skimage.feature import blob_doh
from skimage.filters import sato, threshold_otsu
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.morphology import disk

import lime
from lime import lime_image

from copy import deepcopy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
 


# In[ ]:


img_x = 224
img_y = 224
bat_siz = 32
num_epok = 32
# In[2]:


data_generator = ImageDataGenerator(
        zoom_range = 0.4,
        vertical_flip  = True,
        horizontal_flip = True,
        rescale=1.0/255.0
        )

model = load_model('../input/aptos-densenet-train-submit/densenet_plus_five')

test_data_labels = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_data_labels['id_code'] = test_data_labels['id_code'] + '.png'


test_generator = data_generator.flow_from_dataframe(dataframe = test_data_labels,
                                                     directory = os.path.join('..', 'input','aptos2019-blindness-detection','test_images'),
        target_size = (img_x, img_y), 
    
        x_col = 'id_code',
        class_mode = None,
        batch_size = bat_siz
        )


# In[ ]:


def lime_explanation(timage, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(timage, model.predict, hide_color = 0, 
                                             num_samples = 5000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                positive_only=True, 
                                                num_features=20, 
                                                hide_rest=True, 
                                                min_weight = 0.0251)
    return(temp, mask)

xx = []
for i in range(3): 
    xx.append(randrange(bat_siz))

for img in xx:
    print(img)
    temp, mask = lime_explanation(test_generator[0][img], model)
    plt.axis('off')
    plt.imshow(mark_boundaries(temp, mask))
    plt.savefig("markers" + str(img)+ ".png", bbox_inches = 0)


# In[ ]:


inx = 0
skeleton_array = np.asarray([])
for img_file in glob.iglob("*markers*png"):
    img = io.imread(img_file)
    gray_image = rgb2gray(img)
    gray_image[gray_image > .85] = 0
    vessel_image = sato(gray_image,sigmas = np.linspace(0.1,6.0,8))
    vessel_thresh = 0.0025
    print('The vessel threshold is: {:3f}.'.format(vessel_thresh))
    bin_vessels = deepcopy(vessel_image)

    bin_vessels[bin_vessels >= vessel_thresh] = 1
    bin_vessels[bin_vessels < vessel_thresh] = 0


    blobs = blob_doh(gray_image)
    blob_indices =  blobs[:,:2].astype(int)

    for idx in blob_indices: 
        bin_vessels[idx[0], idx[1]] = 1

    # perform skeletonization 
    skeleton = skeletonize(bin_vessels)
    skeleton_prearray = (skeleton.astype('int')).ravel()
    skeleton_npprearray = np.asarray(skeleton_prearray)
    to_pad = img_x*img_y - (skeleton_npprearray.shape[0])
    pad_skeleton = np.pad(array  = skeleton_npprearray, pad_width = ((0,0),(to_pad,0)), 
                      mode = 'constant')
    skeleton_array = np.append(skeleton_array, pad_skeleton, axis = 0)
    fig, axes = plt.subplots(nrows = 3, ncols = 2, figsize = (32, 32),
                             sharex = True, sharey = True)

    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('original', fontsize = 20)

    ax[1].imshow(gray_image, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('grayscale', fontsize = 20)

    ax[2].imshow(vessel_image, cmap=plt.cm.gray)
    ax[2].axis('off')
    ax[2].set_title('vessels', fontsize = 20)

    ax[3].imshow(bin_vessels, cmap=plt.cm.gray)
    ax[3].axis('off')
    ax[3].set_title('binarized vessels', fontsize = 20)

    ax[4].imshow(skeleton, cmap=plt.cm.gray)
    ax[4].axis('off')
    ax[4].set_title('skeleton', fontsize = 20)

    ax[5].axis('off')
    ax[5].imshow(skeleton.astype(int) + gray_image, cmap=plt.cm.gray)
    ax[5].set_title('Skeleton overlay', fontsize = 20)
    plt.savefig("skeletal_" + str(inx) + ".png", bbox_inches = 0)
    inx += 1 


# In[ ]:


skeleton_array.shape

