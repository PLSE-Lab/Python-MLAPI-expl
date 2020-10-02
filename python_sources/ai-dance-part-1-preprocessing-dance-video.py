#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/Dance_Robots_Comic.jpg")


# (This is part 1 of 3 of my How to Teach an AI to Dance. I originally made 3 separate notebooks for this task before compiling them into one later. The complete assembled notebook of all 3 parts can be found here: https://www.kaggle.com/valkling/how-to-teach-an-ai-to-dance)
# 
# # AI Dance Part 1: Video Preprocessing
# 
# We will be using the same video of training as in the youtube video. It is a >1 hour of go-go dancing female silhouettes. This is ideal as most other green screen dancing videos are too short, loops, and/or messy for easy preprocessing. While not greatly varied, this dancing is not looped and we do not have to cobble together multiple dance videos (which might also require different preprocessing steps for each).
# 
# Here is the original video on youtube: https://www.youtube.com/watch?v=NdSqAAT28v0

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb

import os

import zipfile
z = zipfile.ZipFile("Dancer_Images.zip", "w")


# In[ ]:


cap = cv2.VideoCapture('../input/Shadow Dancers 1 Hour.mp4')
print(cap.get(cv2.CAP_PROP_FPS))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntry:\n    if not os.path.exists(\'data\'):\n        os.makedirs(\'data\')\nexcept OSError:\n    print (\'Error: Creating directory of data\')\n\ncurrentFrame = 0\ncount = 0\nTRAIN_SIZE = 27000\nFRAME_SKIP = 2\nIMG_WIDTH = 96\nIMG_HEIGHT = 64\nIMG_CHANNELS = 1\n\nvideo = cv2.VideoWriter(\'Simple_Shadow_Dancer_Video.avi\',cv2.VideoWriter_fourcc(*"MJPG"), 30, (IMG_WIDTH, IMG_HEIGHT), False)\n\nwhile(count < TRAIN_SIZE):\n    try:\n        ret, frame = cap.read()\n\n        if currentFrame % FRAME_SKIP == 0:\n            count += 1\n            if count % int(TRAIN_SIZE/10) == 0:\n                print(str((count/TRAIN_SIZE)*100)+"% done")\n            # preprocess frames\n            img = frame\n            img = rgb2gray(img)\n            img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode=\'constant\', preserve_range=True)\n            img[img > 0.2] = 255\n            img[img <= 0.2] = 0\n            # save frame to zip and new video sample\n            name = \'./data/frame\' + str(count) + \'.jpg\'\n            cv2.imwrite(name, img)\n            video.write(img.astype(\'uint8\'))\n            z.write(name)\n            os.remove(name)\n    except:\n        print(\'Frame error\')\n        break\n    currentFrame += 1\n\nprint(str(count)+" Frames collected")\ncap.release()\nz.close()\nvideo.release()\n\ncap.release()\nz.close()\nvideo.release()')


# ## Part 1 Results
# The dancer comes out clearly but a bit blocky. There is a bit of dirt in the frames but not too much and less than in the youtube video. The arms occasionally clip during quick motions, but that happens a lot in the original video as well just from the normal green screen clipping.
# 
# ### Possible Improvements
# - The frames of black and dirt as the video transitions to new dancers could be controlled and cleaned for.
# 
# - Changing the binary threshold from 0.2. This can be a tradeoff between getting more of the dancer's pixels and picking up more dirt from the background.
# 
# - To that effect, careful use of Image thresholding packages might work to cut out the dancer from the background too (but also might not work due to the constantly changing background in the video).
# 
# - It is always an option to take larger and/or more frames of dancing for training, as long as we still got memory for it.
