#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('HTML', '', '\n<style type="text/css">\n     \n \ndiv.h2 {\n    background-color: #159957;\n    background-image: linear-gradient(120deg, #155799, #159957);\n    text-align: left;\n    color: white;              \n    padding:9px;\n    padding-right: 100px; \n    font-size: 20px; \n    max-width: 1500px; \n    margin: auto; \n    margin-top: 40px; \n}\n                                     \n                                      \nbody {\n  font-size: 11px;\n}    \n     \n                                    \n                                      \ndiv.h3 {\n    color: #159957; \n    font-size: 18px; \n    margin-top: 20px; \n    margin-bottom:4px;\n}\n   \n                                      \ndiv.h4 {\n    color: #159957;\n    font-size: 15px; \n    margin-top: 20px; \n    margin-bottom: 8px;\n}\n   \n                                      \nspan.note {\n    font-size: 7; \n    color: gray; \n    font-style: italic;\n}\n  \n                                      \nhr {\n    display: block; \n    color: gray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}\n  \n                                      \nhr.light {\n    display: block; \n    color: lightgray\n    height: 1px; \n    border: 0; \n    border-top: 1px solid;\n}   \n    \n                                      \ntable.dataframe th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n}\n    \n                                      \ntable.dataframe td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    text-align: center;\n} \n   \n            \n                                      \ntable.rules th \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 11px;\n    align: left;\n}\n       \n                                      \ntable.rules td \n{\n    border: 1px darkgray solid;\n    color: black;\n    background-color: white;\n    font-size: 13px;\n    text-align: center;\n} \n                                       \n                                      \ntable.rules tr.best\n{\n    color: green;\n}    \n                             \n.output { \n    align-items: left; \n}\n        \n                                      \n.output_png {\n    display: table-cell;\n    text-align: left;\n    margin:auto;\n}                                          \n                                                                    \n                                      \n                                      \n</style> \n                                     \n                                      ')


# In[ ]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Reference: 
#      - I really liked the way JohnM's punt kaggle submission had the headers, extremely aesthetically pleasing
#        and aids viewing - borrowing his div.h header concept (so much nicer looking than using conventional
#        ## headers etc), and adding a 'cayman' color theme to it, as a nod to R ...  
#        Isn't it nice looking ?  ->  https://jasonlong.github.io/cayman-theme/
#      - I would strongly suggest we follow JohnM's push into professoinal looking css-based headers, we can't 
#        keep using old-fashioned markdown for headers, its so limited... just my personal opinion
#
# -%%HTML
# <style type="text/css">
#
# div.h2 {
#     background-color: steelblue; 
#     color: white; 
#     padding: 8px; 
#     padding-right: 300px; 
#     font-size: 20px; 
#     max-width: 1500px; 
#     margin: auto; 
#     margin-top: 50px;
# }
# etc
# etc
# --- end reference ---

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNCOMMENT ALL OF THIS OUT:
# abc
# def
#
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
import matplotlib. pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patches as patches
import seaborn as sns
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#import sparklines
import colorcet as cc
plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##%config InlineBackend.figure_format = 'retina'   < - keep in case 
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from sklearn import preprocessing
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold
# from sklearn.feature_selection import SelectFromModel
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from IPython.display import Video
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2 as cv  # or import cv2 as cv
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import json
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from tqdm import tqdm_notebook
from tqdm import tqdm
#import gc, pickle, tqdm, os, datetime
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from skimage.measure import compare_ssim
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# <div class="h3"><i>Intro:</i></div>  
# * I pretty much know nothing about neural networks, face detection, OpenCV, etc, but let's give this a shot 
#   * This is my second Kaggle competition

# <div class="h3"><i>Approach:</i></div>
# * Start at the absolute bottom, and learn how neural networks actually work
# * Research CV 
# * Build on knowledge to get deeper and deeper into variations of CNN/RNN, etc
# * Learn Keras
# * Investigate autoencoders
# * Dive hard core into the mathematics
# * Determine precisely how Deepfakes are made/created/propagated
# * Determine conventional way of detecting Deepfakes
# * Create an unconventional approach to detecting Deepfakes
# * Determine path forward 

# <div class="h3"><i>Summary of our dataset:</i></div>
# <p style="margin-top: 50px">It is always important to look at our entire dataset and examine the descriptive statistics:</p>
# 
# &ensp; **Number of training videos (.mp4):** &ensp;  &nbsp;  400  
# &ensp; **Number of test videos (.mp4):** &ensp; &nbsp;  &ensp; &ensp; &nbsp;  401  

# In[ ]:


train_sample_metadata = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json').T
train_sample_metadata.head(20)


# <div class="h3"><i>Percentage of Fake vs Real within training video dataset:</i></div>
# * 323 Fakes
# * 77 Real 

# In[ ]:


pd.DataFrame(train_sample_metadata['label'].value_counts(normalize=True))


# In[ ]:


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#train_sample_metadata.groupby('label')['label'].count()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
train_dir = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
train_video_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.mp4')]
test_dir = '/kaggle/input/deepfake-detection-challenge/test_videos/'
test_video_files = [test_dir + x for x in os.listdir(test_dir)]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df_train = pd.read_json('/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json').transpose()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#df_train.head()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#df_train.shape 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# <div class="h3"><i>Examining a single video:</i></div>
# * Let's take a look at a random video: &nbsp;  drcyabprvt.mp4
# * We will freeze it's first frame and output as an image
# * Dimension:  (1080, 1920, 3)

# In[ ]:



# FREEZE-FRAME:
import cv2 as cv
import matplotlib.pyplot as plt
dp1 = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/drcyabprvt.mp4'
#dp2 = 'dzieklokdr.mp4'    
fig, ax = plt.subplots(1,1, 
                       figsize=(8,8))
# fake:  cap = cv.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dkrvorliqc.mp4') 
# cap = cv.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dzieklokdr.mp4')
mycap = cv.VideoCapture(dp1); mycap.set(1,2)
ret, image = mycap.read()
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
raw_image = image
#print(raw_image.shape)
mycap.release() 
cv.destroyAllWindows()
ax.set_xticks([]); ax.set_yticks([]); ax.imshow(image);


# * Let's take a look at another random video frame image
# * Here you can see that the image is 1080 pixels x 1920 pixels (tick marks)
# * This time we will show a deepfake, of the same person, while keeping the pixel tick marks/tags: 

# In[ ]:



fig, ax = plt.subplots(1,1, figsize=(8,8))
cap = cv.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dkrvorliqc.mp4') 
cap.set(1,2); ret, image = cap.read()
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cap.release()   
cv.destroyAllWindows()
file_name = 'dkrvorliqc.mp4'
ax.title.set_text(file_name)
ax.imshow(image); 


# <hr>

# <div class="h3"><i>The Problem:</i></div> 
# * Deepfake videos are becoming easier and easier to make
# * Deepfake video sophistication is increasing

# <div class="h3"><i>Potential Methods to Detect Deepfakes:</i></div> 
# * An AI-produced video (generative adversarial networks based) could show a world leader doing or saying something inflammatory, and worst-case scenario could lead to the population formulating a different opinion of the leader, or even triggering violence and chaos. 
# * Soft-biometric signatures such as blink rate ? 
# * I don't think face landmarks will necessarily solve this issue 
# * It would make sense to say that the longer the video under inspection is, the **greater** the probability of detecting that it is in fact a deepfake video (if it were a deepfake video), i.e. direct correlation (probably) between length of video and probability of detecting its status
#   * Ideally the deepfake under investigation was longer, as this would allow an algorithm to find 'signatures' that it was 'tainted'
# * **aayfryxljh.mp4:**  &nbsp; Possible correlation between turning head beyond 45 degrees and blink ?   Glasses reflection make it harder to modify video due to reflection background ? 
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/reflection.png" width="200px">
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/reflection2.png" width="400px">
# 

# <div class="h3"><i>The problem is not the face, it's the concentric circle:</i></div> 
# * Concentric Circles:  Two or more circles which have the same center point
# * Concentric Ovals:  Faces are closer to ovals than circles
# * Perhaps mathematically constructing the differences in the region between the inserted face and the original face, many times it appears this area has a signature that is easier to see
# <br>
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/tom2.jpg" width="800px">
# <br>
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/concentric.jpg" width="200px">
# 
# 

# <div class="h3"><i>Calendar 2020:</i></div> 
# * Why do we care ? 
#   * The United States of America has a presidential election November 3rd, 2020
#   * The period of time from July to November will be a period of time when U.S. citizens will be inundated (whether they like it or not) with media coverage
#   * Citizens may be particularly susceptible to misinformation/disinformation/deepfakes

# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/election3.png" width="700px">

# <div class="h3"><i>Categorizing Deepfakes:</i></div> 
# * It would seem that categorizing/quantifying the potential damage a deepfake could inflict may be necessary at some point
# * Some deepfakes are relatively harmless, while others may be more damaging
# * J-level assignment, from 1 to 10, i.e. J-level of 9 would be a deepfake associated with inflicting chaos during the run up to an election, misrepresenting what a candidate stated as their position, etc. Deepfake with J-level of 1 would be effectively harmless.  
# 

# <div class="h3"><i>Quantum Neural Network (QNN):</i></div>  
# * Conventional means of combating Deepfakes may not be possible
# * A new means of detecting Deepfakes may be more successful
# * Perhaps a new quantum-based approach ?  
#   * I don't think the term exists yet, but maybe what could be known as QNNs ? 
#   * Why ?  Because using a conventional approach to detect Deepfakes could potentially be used in the same algorithm that is used to create Deepfakes, nullifying the gain

# <div class="h3"><i>Reality Check:</i></div> 
# * There seems to be the belief that it is in fact possible to detect Deepfakes, when over the course of time as technology/algorithms advance, it **may not actually be possible** to detect them with high probability
# * In that event, potential paths forward:
# * Modification of the 5G standard to allow direct IPSEC-like connections from user elements (UEs) to secure video servers, which are considered to be 'the source of truth' from various publications / news agencies / government agencies

# <div class="h3"><i>References:</i></div>
# [1]  Deepfake, Wikipedia, https://en.wikipedia.org/wiki/Deepfake  

# In[ ]:


# overall notes, do not destroy:
#
# ![title](https://www.desipio.com/wp-content/uploads/2019/06/walter-payton-leap-2-ah.jpg)
# <br>&ensp; *Walter Payton (34) and the need for z-coordinate data ...*
#
#
#
#
#
#
#
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
#
#
#
#
# <img src="https://raw.githubusercontent.com/tombresee/Temp/master/ENTER/box2.png" width="400px">
#
#
#
#
#
#  https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py
#
#
#
#
# import numpy as np
# import cv2

# cap = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/train_sample_videos/dkrvorliqc.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# keep:
# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()


