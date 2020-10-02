#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/train/")[:5])


# In[ ]:


ls_fam = os.listdir("../input/train/")


# In[ ]:


ls_fam[:5]


# In[ ]:


print(len(os.listdir("../input/train/")))


# In[ ]:


print(os.listdir("../input/test/")[:5])


# In[ ]:


df_train = pd.read_csv("../input/train_relationships.csv")


# In[ ]:


df_train.info()


# In[ ]:


df_train.head()


# In[ ]:


df_train['p1'].value_counts()[-5:]


# In[ ]:


len(df_train['p1'])


# In[ ]:


len(df_train['p1'].append(df_train['p2']))


# In[ ]:


for _, row in df_train.sample(2, random_state=2).iterrows():
    print(row.p1)
    print(row.p2)
    


# In[ ]:


_ = df_train['p1'].append(df_train['p2']).value_counts().hist()


# In[ ]:


df_train[df_train["p1"].str.contains('F0617')]


# In[ ]:


N_ROWS = 5


# In[ ]:


related_a_folder = "../input/train/F0617/MID1/"
related_a = np.array(os.listdir("../input/train/F0617/MID1/")[:N_ROWS])
related_b_folder = "../input/train/F0617/MID4/"
related_b = np.array(os.listdir("../input/train/F0617/MID4/")[:N_ROWS])
unrelated_folder = "../input/train/F0206/MID1/"
unrelated = np.array(os.listdir("../input/train/F0206/MID1")[:N_ROWS])


# In[ ]:


nrows = N_ROWS
ncols = 3
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

ctr=0
for i in range(N_ROWS):
    img_patha = related_a[i]
    img_pathb = related_b[i]
    img_pathc = unrelated[i]
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr, aspect='auto')
    img_patha = related_a_folder + img_patha
    img_pathb = related_b_folder + img_pathb
    img_pathc = unrelated_folder + img_pathc
    imga = np.array(Image.open(img_patha))
    imgb = np.array(Image.open(img_pathb))
    imgc = np.array(Image.open(img_pathc))
    plt.imshow(imga)
    plt.title(related_a_folder)
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgb)
    plt.title(related_b_folder)
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgc)
    plt.title(unrelated_folder)


# In[ ]:


#cv2.HISTCMP_CHISQR


# In[ ]:


#https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
#https://stackoverflow.com/questions/40451706/how-to-use-comparehist-function-opencv
import cv2
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]
            
nrows = N_ROWS
ncols = 3
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

ctr=0
for i in range(N_ROWS):
    img_patha = related_a[i]
    img_pathb = related_b[i]
    img_pathc = unrelated[i]
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr, aspect='auto')
    img_patha = related_a_folder + img_patha
    img_pathb = related_b_folder + img_pathb
    img_pathc = unrelated_folder + img_pathc
    imga = np.array(Image.open(img_patha))
    imgb = np.array(Image.open(img_pathb))
    imgc = np.array(Image.open(img_pathc))
    
    cv_a = cv2.imread(img_patha)
    cv_b = cv2.imread(img_pathb)
    cv_c = cv2.imread(img_pathc)
    
    hsv_a = cv2.cvtColor(cv_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(cv_b, cv2.COLOR_BGR2HSV)
    hsv_c = cv2.cvtColor(cv_c, cv2.COLOR_BGR2HSV)

    
    hist_a = cv2.calcHist([hsv_a], channels, None, histSize, ranges, accumulate=False)
    n_a = cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_b = cv2.calcHist([hsv_b], channels, None, histSize, ranges, accumulate=False)
    n_b = cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_c = cv2.calcHist([hsv_c], channels, None, histSize, ranges, accumulate=False)
    n_c = cv2.normalize(hist_c, hist_c, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


    a_a = cv2.compareHist(n_a, n_a, cv2.HISTCMP_CORREL)
    a_b = cv2.compareHist(n_a, n_b, cv2.HISTCMP_CORREL)
    a_c = cv2.compareHist(n_a, n_c, cv2.HISTCMP_CORREL)

    

    plt.imshow(imga)
    #plt.title(related_a_folder)
    plt.title(str(a_a))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgb)
    #plt.title(related_b_folder)
    plt.title(str(a_b))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgc)
    #plt.title(unrelated_folder)
    plt.title(str(a_c))
    
    


# In[ ]:


#https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
#https://stackoverflow.com/questions/40451706/how-to-use-comparehist-function-opencv
import cv2
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]
            
nrows = N_ROWS
ncols = 3
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

ctr=0
for i in range(N_ROWS):
    img_patha = related_a[i]
    img_pathb = related_b[i]
    img_pathc = unrelated[i]
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr, aspect='auto')
    img_patha = related_a_folder + img_patha
    img_pathb = related_b_folder + img_pathb
    img_pathc = unrelated_folder + img_pathc
    imga = np.array(Image.open(img_patha))
    imgb = np.array(Image.open(img_pathb))
    imgc = np.array(Image.open(img_pathc))
    
#     cv_a = cv2.imread(img_patha)
#     cv_b = cv2.imread(img_pathb)
#     cv_c = cv2.imread(img_pathc)
    
#     hsv_a = cv2.cvtColor(cv_a, cv2.COLOR_BGR2HSV)
#     hsv_b = cv2.cvtColor(cv_b, cv2.COLOR_BGR2HSV)
#     hsv_c = cv2.cvtColor(cv_c, cv2.COLOR_BGR2HSV)

    
#     hist_a = cv2.calcHist([hsv_a], channels, None, histSize, ranges, accumulate=False)
#     n_a = cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#     hist_b = cv2.calcHist([hsv_b], channels, None, histSize, ranges, accumulate=False)
#     n_b = cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
#     hist_c = cv2.calcHist([hsv_c], channels, None, histSize, ranges, accumulate=False)
#     n_c = cv2.normalize(hist_c, hist_c, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


#     a_a = cv2.compareHist(n_a, n_a, cv2.HISTCMP_CORREL)
#     a_b = cv2.compareHist(n_a, n_b, cv2.HISTCMP_CORREL)
#     a_c = cv2.compareHist(n_a, n_c, cv2.HISTCMP_CORREL)
    img_a = cv2.imread(img_patha,0)
    img_b = cv2.imread(img_pathb,0)
    img_c = cv2.imread(img_pathc,0)
    
    a_a = cv2.matchTemplate(img_a,img_a,cv2.TM_SQDIFF_NORMED)
    a_b = cv2.matchTemplate(img_b,img_a,cv2.TM_SQDIFF_NORMED)
    a_c = cv2.matchTemplate(img_c,img_a,cv2.TM_SQDIFF_NORMED)
    

    plt.imshow(imga)
    #plt.title(related_a_folder)
    plt.title(str(a_a))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgb)
    #plt.title(related_b_folder)
    plt.title(str(a_b))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgc)
    #plt.title(unrelated_folder)
    plt.title(str(a_c))
    


# https://www.tensorflow.org/beta/tutorials/generative/style_transfer

# In[ ]:


def high_pass_x_y(image):
  x_var = image[:,1:,:] - image[:,:-1,:]
  y_var = image[1:,:,:] - image[:-1,:,:]

  return x_var, y_var

def clip_0_1(image):
  return np.clip(image, a_min=0.0, a_max=1.0)


            
nrows = N_ROWS
ncols = 3
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

ctr=0
#Horizontal
for i in range(N_ROWS):
    img_patha = related_a[i]
    img_pathb = related_b[i]
    img_pathc = unrelated[i]
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr, aspect='auto')
    img_patha = related_a_folder + img_patha
    img_pathb = related_b_folder + img_pathb
    img_pathc = unrelated_folder + img_pathc
    imga = np.array(Image.open(img_patha))
    imgb = np.array(Image.open(img_pathb))
    imgc = np.array(Image.open(img_pathc))
    
    
 
   
    
    x_deltas, y_deltas = high_pass_x_y(imga)
    plt.imshow(clip_0_1(2*y_deltas+0.5))
    plt.title(related_a_folder)
    #plt.title(str(a_a))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    x_deltas, y_deltas = high_pass_x_y(imgb)
    plt.imshow(clip_0_1(2*y_deltas+0.5))
    plt.title(related_b_folder)
    #plt.title(str(a_b))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    x_deltas, y_deltas = high_pass_x_y(imgc)
    plt.imshow(clip_0_1(2*y_deltas+0.5))
    plt.title(unrelated_folder)
    #plt.title(str(a_c))


# In[ ]:


nrows = N_ROWS
ncols = 3
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

ctr=0
#Vertical
for i in range(N_ROWS):
    img_patha = related_a[i]
    img_pathb = related_b[i]
    img_pathc = unrelated[i]
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr, aspect='auto')
    img_patha = related_a_folder + img_patha
    img_pathb = related_b_folder + img_pathb
    img_pathc = unrelated_folder + img_pathc
    imga = np.array(Image.open(img_patha))
    imgb = np.array(Image.open(img_pathb))
    imgc = np.array(Image.open(img_pathc))
    
    
 
   
    
    x_deltas, y_deltas = high_pass_x_y(imga)
    plt.imshow(clip_0_1(2*x_deltas+0.5))
    plt.title(related_a_folder)
    #plt.title(str(a_a))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    x_deltas, y_deltas = high_pass_x_y(imgb)
    plt.imshow(clip_0_1(2*x_deltas+0.5))
    plt.title(related_b_folder)
    #plt.title(str(a_b))
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    x_deltas, y_deltas = high_pass_x_y(imgc)
    plt.imshow(clip_0_1(2*x_deltas+0.5))
    plt.title(unrelated_folder)
    #plt.title(str(a_c))


# In[ ]:


df_sample = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


df_sample.info()


# In[ ]:


df_sample.head()


# In[ ]:


#https://www.geeksforgeeks.org/python-pandas-split-strings-into-two-list-columns-using-str-split/
separate_col = df_sample["img_pair"].str.split("-", n=1, expand=True)


# In[ ]:


df_sample["imagea"] = separate_col[0]
df_sample["imageb"] = separate_col[1]


# In[ ]:


df_sample.head()


# In[ ]:


df_sample.info()


# In[ ]:


df_sample.sample(4)


# In[ ]:


nrows = 50
ncols = 2
fig = plt.gcf()
fig.set_size_inches(ncols*5, nrows*5)

ctr=0
for _, row in df_sample.sample(50, random_state=2).iterrows():
    img_patha = row.imagea
    img_pathb = row.imageb
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr, aspect='auto')
    img_patha = '../input/test/' + img_patha
    img_pathb = '../input/test/' + img_pathb
    imga = np.array(Image.open(img_patha))
    imgb = np.array(Image.open(img_pathb))
    plt.imshow(imga)
    ctr+=1
    sp = plt.subplot(nrows, ncols, ctr)
    plt.imshow(imgb)

