#!/usr/bin/env python
# coding: utf-8

# This notebook shows how to extract char's from the captcha dataset images <br/>
# Additional code and notebooks are avaliable on my repository: https://github.com/Vykstorm/CaptchaDL

# ## Import statements

# In[61]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2 as cv
import pickle
import warnings
from itertools import product, repeat, permutations, combinations_with_replacement, chain
from math import floor, ceil

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Load dataset

# Load the captcha images and their labels (check the part 1 notebook)

# In[3]:


data = np.load('/kaggle/input/reading-captcha-dataset-part-1/preprocessed-data.npz')
X, y_labels, y = data['X'], data['y_labels'], data['y']


# # Extract chars from the captcha

# Now we are going to take a random image from the dataset and show how to extract each char's pixels.

# In[4]:


n = X.shape[0]
img = (X[np.random.randint(n)] * 255)[:, :, 0].astype(np.uint8)


# In[5]:


plt.imshow(img, cmap='gray');


# ## Image preprocessing

# We invert the image so that the background is black and foreground white

# In[6]:


inverted = 255 - img
plt.imshow(inverted, cmap='gray');


# Now apply binary thresholding.

# In[7]:


ret, thresholded = cv.threshold(inverted, 140, 255, cv.THRESH_BINARY)
plt.imshow(thresholded, cmap='gray');


# Reduce the noise with a filter

# In[8]:


blurred = cv.medianBlur(thresholded, 3)
plt.imshow(blurred, cmap='gray');


# Apply morphological transformations to eliminate tiny objects & horizontal lines

# In[9]:


kernel = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
]).astype(np.uint8)

ex = cv.morphologyEx(blurred, cv.MORPH_OPEN, kernel)
plt.imshow(ex, cmap='gray');


# In[10]:


kernel2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]).astype(np.uint8)
ex2 = cv.morphologyEx(ex, cv.MORPH_DILATE, kernel2)
plt.imshow(ex2, cmap='gray');


# Apply a bitwise AND operation between the last image and the blurred image (the last one is used as a "mask")

# In[11]:


mask = ex2
processed = cv.bitwise_and(mask, blurred)
plt.imshow(processed, cmap='gray');


# ## Finding contours

# The next step is finding contours in the image. We can use the OpenCV api method findContours()

# In[12]:


contours, hierachy = cv.findContours(processed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
contours = [contours[k] for k in range(0, len(contours)) if hierachy[0, k, 3] == -1]
contours.sort(key=lambda cnt: cv.boundingRect(cnt)[0])


# In[13]:


len(contours)


# We can plot the contours

# In[14]:


plt.imshow(cv.drawContours(cv.cvtColor(img, cv.COLOR_GRAY2RGB), contours, -1, (255, 0, 0), 1, cv.LINE_4));


# For each contour, we calculate its rectangle bounding box and draw them

# In[15]:


contour_bboxes = [cv.boundingRect(contour) for contour in contours]
img_bboxes = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
for bbox in contour_bboxes:
    left, top, width, height = bbox
    img_bboxes = cv.rectangle(img_bboxes,
                              (left, top), (left+width, top+height),
                              (0, 255, 0), 1)
plt.imshow(img_bboxes, cmap='gray');


# As we can see in the image, sometimes one contour can hold more than one character. <br/>
# We need to figure out how many chars are inside each one to split the image correctly <br/>
# Also, there are unwanted small contours that we need to remove

# ## Contour classification

# In this section we need to guess the number of chars inside each contour. <br/>
# To do so, we need to classify each contour in 6 classes (from 0 to 5) <br/>
# A contour of class k is one that contains k characters

# I created a Support vector machine (SVM) to do this task. It uses different contour properties as features such as bounding box width & height, perimeter, area, ... <br/>
# You can download the pre-trained SVM here: https://github.com/Vykstorm/CaptchaDL/blob/master/models/.contour-classifier

# In[16]:



with open('/kaggle/input/captchadlutils/repository/Vykstorm-CaptchaDL-utils-a8458b5/contour-classifier', 'rb') as file:
    contour_classifier = pickle.load(file)


# In[17]:


contour_classifier


# For each contour we extract the next features:
# * Bounding box width (Width of the bounding box that encloses the contour)
# * Bounding box height (Height of the bbox)
# * Area (Area of the contour)
# * Extent (Ratio between the area of the contour and its bounding box)
# * Perimeter (Perimiter of the contour)
# <br/>
# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

# In[18]:


contours_features = pd.DataFrame.from_dict({
    'bbox_width': [bbox[2] for bbox in contour_bboxes],
    'bbox_height': [bbox[3] for bbox in contour_bboxes],
    'area': [cv.contourArea(cnt) for cnt in contours],
    'extent': [cv.contourArea(cnt) / (bbox[2] * bbox[3]) for cnt, bbox in zip(contours, contour_bboxes)],
    'perimeter': [cv.arcLength(cnt, True) for cnt in contours]
})
contours_features


# Now we scale the features. <br/>
# You can download https://github.com/Vykstorm/CaptchaDL/blob/master/models/.contour-classifier-preprocessor
# which is an instance of sklearn.preprocessing.StdScaler already fit with the data

# In[19]:



with open('/kaggle/input/captchadlutils/repository/Vykstorm-CaptchaDL-utils-a8458b5/contour-classifier-preprocessor', 'rb') as file:
    contour_features_scaler = pickle.load(file)


# In[20]:


contour_features = contour_features_scaler.transform(contours_features[['bbox_width', 'bbox_height', 'area', 'extent', 'perimeter']])
contour_features


# Finally we classify each contour

# In[21]:


contour_num_chars = contour_classifier.predict(contour_features)
contour_num_chars


# In[22]:


n = len(contours)
cols = 2
rows = n // cols
if n % cols > 0:
    rows += 1
rows = max(rows, 2)

fig, ax = plt.subplots(rows, cols, figsize=(15, 2.5*rows))
for i, j in product(range(0,rows), range(0,cols)):
    k = i * cols + j
    if k < n:
        left, top, width, height = contour_bboxes[k]
        img_bbox = cv.rectangle(cv.cvtColor(img, cv.COLOR_GRAY2RGB),
                                (left, top), (left+width, top+height), (0, 255, 0), 1)
        
        plt.sca(ax[i,j])
        plt.title('Contour {}, Number of chars: {}'.format(k, contour_num_chars[k]))
        plt.imshow(img_bbox, cmap='gray')
        plt.xticks([])
        plt.yticks([])
    else:
        ax[i,j].set_visible(False)

plt.tight_layout()


# In[23]:


print('Total number of predicted characters in the image: {}'.format(contour_num_chars.sum()))


# But there is a problem: what if the total number of predicted chars in the image is not 5? <br/>
# For example, we can predict that a contour holds 2 chars but in reality it has 3

# To solve this problem, im going to define the next matrix of size num.contours (n) x 6:
# <p>
# $P = 
# \begin{bmatrix}
# p_0^0 & p_0^1 & ... & p_0^6 \\
# & & ... & & \\
# p_n^0 & p_n^1 & ... & p_n^6
# \end{bmatrix}$
# </p>
# Where $p_i^j$ is the probability of the ith contour of having j characters inside

# In[24]:


P = contour_classifier.predict_proba(contour_features)
np.round(P, 2)


# We are also going to define $\alpha$ as 1D vector with n (number of contours) elements:
# $\begin{bmatrix}
# \alpha_0 & \alpha_1 & ... & \alpha_n
# \end{bmatrix}$ <br/>
# $\alpha_i$ will be the number of characters that we predict are inside the ith contour

# Our goal is: <br>
# Find a configuration $\alpha$ such that $\sum_{i=1}^{n} \alpha_i = 5$ that maximize the next function: <br/>
# $$ score(\alpha) = \prod_{i=0}^{n} p_{i}^{\alpha_i} $$

# The code below creates a list of all possible valid configurations for $\alpha$[](http://)

# In[25]:


configs = filter(lambda x: np.sum(x) == 5, combinations_with_replacement(range(0, 6), n))
configs = list(frozenset(chain.from_iterable(map(lambda config: permutations(config, n), configs))))
configs = np.array(configs, dtype=np.uint8)
len(configs)


# Compute $score(\alpha)$ for each possible configuration

# In[26]:


scores = np.zeros([len(configs)]).astype(np.float32)
for i in range(0, len(configs)):
    scores[i] = np.prod(P[np.arange(0, n), configs[i]])


# Get the best configuration

# In[27]:


best_config = configs[scores.argmax()]
best_config


# ## Word pixels extraction

# Now we can extract the pixels inside the contours in which we predict the chars are in

# In[28]:


frames = []
for i in range(0, n):
    if best_config[i] > 0:
        left, top, width, height = contour_bboxes[i]
        right, bottom = left+width, top+height
        frame = img[top:bottom, left:right]
        frames.append(frame)
frame_num_chars = best_config[np.nonzero(best_config)[0]]
num_frames = len(frames)

cols = 3
rows = num_frames // cols
if num_frames % cols > 0:
    rows += 1
rows = max(rows,2)
fig, ax = plt.subplots(rows, cols, figsize=(15, 2.5*rows))
for i, j in product(range(0,rows), range(0,cols)):
    k = i * cols + j
    if k < num_frames:
        plt.sca(ax[i,j])
        plt.imshow(frames[k], cmap='gray')
        plt.title('Frame {}. Num chars: {}'.format(k, frame_num_chars[k]))
        plt.xticks([])
        plt.yticks([])
    else:
        ax[i,j].set_visible(False)


# On the frames with more than 1 char, we need to split them so that we can get the pixels of each char individually

# A decent approximation (far from perfect) to divide the frame is to draw n-1 vertical lines equally spaced (where n is the number of chars inside the frame)

# In[31]:


frame = [frames[i] for i in range(0, num_frames) if frame_num_chars[i] > 1][-1]
num_chars = frame_num_chars[frames.index(frame)]
plt.imshow(frame, cmap='gray');
separators = np.linspace(0, frame.shape[1], num_chars+1)[1:-1]
plt.vlines(separators, ymin=0, ymax=frame.shape[0]-1, color='red');


# In[58]:


def split_array(a, separators, axis=1):
    # This is a helper function to split a numpy array along the given axis using the separators
    # specified
    seperators = sorted(separators)
    n_sep = len(separators)
    if n_sep == 1:
        sep = separators[0]
        a = a.swapaxes(0, axis)
        return [a[0:sep].swapaxes(0, axis), a[sep:].swapaxes(0, axis)]

    head, body = split_array(a, [separators[0]], axis)
    splits = split_array(body, np.array(separators[1:]) - separators[0], axis)
    return [head] + splits

def find_separators(frame, n):
    # This method returns n-1 vertical lines equally spaced to split the frame indicated
    return np.floor(np.linspace(0, frame.shape[1], n+1)[1:-1]).astype(np.uint16)


chars = []
for frame, num_chars in zip(frames, frame_num_chars):
    if num_chars == 1:
        # No need to split frames with only 1 char
        chars.append(frame)
    else:
        # Divide the frame into num_chars splits
        splits = split_array(frame, find_separators(frame, num_chars), axis=1)
        chars.extend(splits)
        

fig, ax = plt.subplots(1, 5, figsize=(13, 3))
for i in range(0, 5):
    plt.sca(ax[i])
    plt.imshow(chars[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Char {}, shape: {}'.format(i+1, chars[i].shape))

plt.tight_layout()


# Our final step is to resize each char image so that all has the same fixed shape (40, 40) <br/>
# If the image is smaller than (40, 40), white borders will be added on each side to fill the gaps <br/>
# If its greater, its cropped about the center <br/>
# Pixels with intensities higher than ~70 are set to 255 to remove color differences between the generated borders and
# the background

# In[65]:


chars_processed = np.zeros([5, 40, 40, 1]).astype(np.float32)
for i, char in zip(range(0, 5), chars):
    img = char
    inverted = 255 - img
    ret, thresholded = cv.threshold(inverted, 70, 255, cv.THRESH_BINARY)
    img = 255 - np.multiply((thresholded > 0), inverted)
    
    dh, dw = 40, 40
    h, w = img.shape
    if w < dw:
        left = floor((dw - w) / 2)
        right = dw - w - left
        img = cv.copyMakeBorder(img, 0, 0, left, right, cv.BORDER_CONSTANT, value=(255, 255, 255))
    elif w > dw:
        left = floor((w - dw) / 2)
        img = img[:, left:left+dw]

    if h < dh:
        top = floor((dh - h) / 2)
        bottom = dh - h - top
        img = cv.copyMakeBorder(img, top, bottom, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
    elif h > dh:
        top = floor((h - dh) / 2)
        img = img[top:top+dh, :]
    
    chars_processed[i, :, :, 0] = img.astype(np.float32) / 255


# This is our final result...

# In[70]:


fig, ax = plt.subplots(1, 5, figsize=(13, 3))
for i in range(0, 5):
    plt.sca(ax[i])
    plt.imshow(chars_processed[i, :, :, 0], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Char {}'.format(i+1))

plt.tight_layout()


# In[ ]:




