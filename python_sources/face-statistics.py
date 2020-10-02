#!/usr/bin/env python
# coding: utf-8

# A few fun tests to look at the differences in (self-reported) male and female profile pictures. The idea is look at color, face count, and a few other features to see if there are any clear trends visible in the images and maybe possibilities for automatically grouping them by gender

# In[ ]:


import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
OPENCV_HAAR_DIR = '/usr/local/share/OpenCV/haarcascades'
get_haarcc = lambda x: cv2.CascadeClassifier(os.path.join(OPENCV_HAAR_DIR,
                                                  'haarcascade_{}.xml'.format(x)))


# In[ ]:


all_img_df = pd.DataFrame({'path': glob.glob('../input/*/*.jpg')})
all_img_df['name'] = all_img_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])
all_img_df['file_size'] = all_img_df['path'].map(os.path.getsize)
all_img_df['folder'] = all_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[1].replace('Photos_sample',''))
all_img_df.sample(5)


# # Color
# Here we calculate very simple color features like the mean intensity in red, green and blue channels. 

# In[ ]:


all_img_df['mean_bgr'] = all_img_df['path'].map(lambda x: np.mean(cv2.imread(x),(0,1)))
all_img_df['mean_blue'] = all_img_df['mean_bgr'].map(lambda x: x[0])
all_img_df['mean_green'] = all_img_df['mean_bgr'].map(lambda x: x[1])
all_img_df['mean_red'] = all_img_df['mean_bgr'].map(lambda x: x[1])


# In[ ]:


sns.pairplot(all_img_df,hue='folder')


# # Face Detection
# Here we use the Haar Cascade Classifier to look for faces in the images

# In[ ]:


face_cascade = get_haarcc('frontalface_default')
eye_cascade = get_haarcc('eye')
def find_features(im_path, cc_obj, def_args = (1.3, 5)):
    in_img = cv2.imread(im_path)
    gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    # taken from: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    return cc_obj.detectMultiScale(gray, *def_args)


# In[ ]:


get_ipython().run_cell_magic('time', '', "all_img_df['face_count']=all_img_df['path'].map(lambda x: len(find_features(x, cc_obj=face_cascade)))")


# In[ ]:


sns.pairplot(all_img_df,hue='folder')


# In[ ]:


# show the top face counts
top_n = 5
fig, m_axes = plt.subplots(top_n,2, figsize=(8,4*top_n))
[x.axis('off') for x in m_axes.flatten()] # hide all axes
sns.set_style("whitegrid", {'axes.grid' : False}) # hide gridelines
for (ax1, ax2),(_, i_row) in zip(m_axes, 
                                 all_img_df.sort_values('face_count', ascending=False).iterrows()):
    in_img = cv2.imread(i_row['path'])
    gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    # taken from: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces)>0:
        ax1.imshow(in_img[:,:,::-1])
        ax1.set_title('{folder}\n{name}'.format(**i_row))
        img=in_img.copy() # make a copy for editing
        for (x,y,w,h) in faces:
            # draw a rectangle for each face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        ax2.imshow(img[:,:,::-1])
        ax2.set_title('Found {} faces'.format(len(faces)))
    else:
        print(c_row['name'],'has no face!')
        


# ## Red
# Here we show the reddest images

# In[ ]:


# show the reddest images
top_n = 5
fig, m_axes = plt.subplots(top_n,2, figsize=(8,4*top_n))
[x.axis('off') for x in m_axes.flatten()] # hide all axes
sns.set_style("whitegrid", {'axes.grid' : False}) # hide gridelines
for (ax1, ax2),(_, i_row) in zip(m_axes, 
                                 all_img_df.sort_values('mean_red', ascending=False).iterrows()):
    in_img = cv2.imread(i_row['path'])
    gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    # taken from: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ax1.imshow(in_img[:,:,::-1])
    ax1.set_title('{folder}\n{name}'.format(**i_row))
    if len(faces)>0:
        img=in_img.copy() # make a copy for editing
        for (x,y,w,h) in faces:
            # draw a rectangle for each face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        ax2.imshow(img[:,:,::-1])
        ax2.set_title('Found {} faces'.format(len(faces)))
    else:
        print(i_row['name'],'has no face!')
        


# In[ ]:


glob.glob(os.path.join(OPENCV_HAAR_DIR,'*'))


# ## Red Difference
# Instead of showing just the higest red, we want to show the ones with the highest difference between red and the other colors
# $$ 2 * R - (G + B) $$

# In[ ]:


# show the reddest images
top_n = 5
fig, m_axes = plt.subplots(top_n,2, figsize=(8,4*top_n))
[x.axis('off') for x in m_axes.flatten()] # hide all axes
all_img_df['red_diff'] = all_img_df.apply(lambda c_row: 2*c_row['mean_red']-(c_row['mean_blue']+c_row['mean_green']),1)
for (ax1, ax2),(_, i_row) in zip(m_axes, 
                                 all_img_df.sort_values('red_diff', ascending=False).iterrows()):
    in_img = cv2.imread(i_row['path'])
    gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    # taken from: http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    ax1.imshow(in_img[:,:,::-1])
    ax1.set_title('{folder}\n{name}'.format(**i_row))
    if len(faces)>0:
        img=in_img.copy() # make a copy for editing
        for (x,y,w,h) in faces:
            # draw a rectangle for each face
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        ax2.imshow(img[:,:,::-1])
        ax2.set_title('Found {} faces'.format(len(faces)))
    else:
        print(i_row['name'],'has no face!')
        

