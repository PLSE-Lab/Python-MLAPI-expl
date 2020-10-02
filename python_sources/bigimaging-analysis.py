#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import os

def load_patient_data(pid):
    pid = str(pid)
    filename = "patient_%s.npz" % pid
    path = os.path.join("..", "input","bigimaging-preprocess", filename)
    np_data = np.load(path)
    slice_dist = np.float(np_data['slice_dist'])
    images = np_data['images']
    
    return images, slice_dist


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage as montage2d

montage3d = lambda x, **k: montage2d(np.stack([montage2d(y, **k) for y in x], 0))

def plot_patient_slices_3d(patient_slices, title=False, figsize=(20, 20)):
    '''Plots a 2D image per slice in series (3D in total)'''
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    image = montage2d(patient_slices)
    if title: ax.set_title(title)
    ax.imshow(image, cmap='bone')


def plot_patient_data_4d(patient_data, all_slices=False, num_slices=[0], figsize=(20, 20)):
    '''Plots a 3D image per time step in patient data (4D in total)'''
    if all_slices:
        # Number of slices is equal to the first dimension of the patient image array
        num_slices = range(patient_data.shape[0])
    for i in num_slices:
        plot_patient_slices_3d(patient_data[i],
                               title=('Showing slice %i' % i))


# ### Load a patient from input 

# In[ ]:


images, slice_dist = np_data = load_patient_data(13)


# ### Plot patient data

# In[ ]:


slice_idx = 4#int(len(images) / 2)
plot_patient_data_4d(images, num_slices=[slice_idx]) #all_slices=True


# ## Segmentation and watershed

# In[ ]:


from skimage.filters import try_all_threshold

fig, ax = try_all_threshold(images[6,0], figsize=(10, 8), verbose=True)


# In[ ]:


import cv2
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

def Segmentation(patient_img):
    """Returns matrix
    Segmententation of patient_img with k-means
    """
    """Z = np.float32(np.ravel(patient_img))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(Z, 2, None, criteria, 10, flags)
    return labels.reshape(patient_img.shape)
    """
    """
    xx, yy = np.meshgrid(np.arange(patient_img.shape[1]),np.arange(patient_img.shape[0]))
    pat_df = pd.DataFrame(dict(x=xx.ravel(),y=yy.ravel(),intensity=patient_img.ravel()))
    km = KMeans(n_clusters=2, random_state=2018)
    
    scale_pat_df = pat_df.copy()
    scale_pat_df.x = scale_pat_df.x/xy_div
    scale_pat_df.y = scale_pat_df.y/xy_div
    scale_pat_df['group'] = km.fit_predict(scale_pat_df[['intensity']].values)

    return scale_pat_df['group'].values.reshape(patient_img.shape)
    """
    
    thresh = threshold_otsu(patient_img)
    binary = patient_img > thresh
    return binary
    


# In[ ]:


# NOT USED

from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks

def hough_circle_transform_xy(image, min_radii = 30, max_radii = 35):
    """
    Returns two lists
    Circle Hough Transform on image and returns x and y coordinates for 10 prominent circles
    """    
    # Radii for hough transform circles
    hough_radii = np.arange(min_radii, max_radii, 2)
    
    # Detect edges for hough transform.
    edges = canny(image, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    
    # Detect circles with circle hough transform. https://en.wikipedia.org/wiki/Circle_Hough_Transform
    hough_res = hough_circle(edges, hough_radii)
            
    # Select the most prominent 10 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=10)
    
    return cx, cy    


# In[ ]:


from skimage.morphology import opening, disk
from scipy.ndimage import distance_transform_edt
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# Code from: https://nbviewer.jupyter.org/github/kmader/Quantitative-Big-Imaging-2019/blob/master/Lectures/07-ComplexObjects.ipynb
def watershed_img(image):
    # Distance map
    image_dmap = distance_transform_edt(image)
    # Distance peaks
    image_peaks = label(peak_local_max(image_dmap, indices=False, footprint=np.ones((40, 40)),labels=image, exclude_border=True))
    # Watershed first once
    ws_labels = watershed(-image_dmap, image_peaks, mask=image)
    
    # Reomve small segments
    label_area_dict = {i: np.sum(ws_labels == i)for i in np.unique(ws_labels[ws_labels > 0])}
    clean_label_maxi = image_peaks.copy()
    lab_areas = list(label_area_dict.values())
    # Remove 20 percentile
    area_cutoff = np.percentile(lab_areas, 20)
    for i, k in label_area_dict.items():
        if k <= area_cutoff:
            clean_label_maxi[clean_label_maxi == i] = 0
    # Watershed again
    ws_labels = watershed(-image_dmap, clean_label_maxi, mask=image)

    return ws_labels


# In[ ]:


from skimage.measure import label

def labeled_segmented_images(images):
    """
    Returns numpy array (4d)
    Segments image and used watershed for labeling.
    """
    
    num_slices, time, height, width = images.shape
    segmented_slices = np.zeros((num_slices, time, height, width))
    
    # Iterate over all slices and whole timeseries for images
    for i in range(num_slices):
        for j in range(time):
            # Segmentation
            seg_slice = Segmentation(images[i,j])
            
            # Makes all segmented images same, Only used for Kmeans. (Background = 0)
            #if seg_slice.sum() > seg_slice.size*0.5:
            #    seg_slice = 1 - seg_slice
            
            # Watershed
            labels = watershed_img(seg_slice)
            
            # Writes labeled segmented object to return images                     
            segmented_slices[i,j] = labels

    return segmented_slices


# In[ ]:


from skimage.measure import regionprops

def find_left_ventricle(images):
    """
    Returns numpy array (4d)
    Finds left ventricle from labeled segmented images
    """
    
    num_slices, time, height, width = images.shape
    segmented_slices = np.zeros((num_slices, time, height, width))
    
    all_labels = labeled_segmented_images(images)
    
    # Iterate over all slices and whole timeseries for images
    for i in range(num_slices):
        for j in range(time):
            
            labels = all_labels[i,j]
            min_dist = 50
            min_dist_label = 0
            segment_found =  False
            
            # Iterate over every label in watershed labels to predict which is the left ventricle.
            for label in np.unique(labels):
        
                # yx coordinates for labaled segmentation
                yx_coord_labels = np.where(labels == label)
                
                # Do not count small or big segmatations (removes dots and background)
                if len(yx_coord_labels[0]) > 8000 or len(yx_coord_labels[0]) < 100:
                    continue
                
                # Upper right middle coordinates
                cx = 3*(height/4)
                cy = width/4
                
                
                # Calculates euclidiean distance between mean coordinates for segmentated labels and middle of image
                euclidiean_dist = np.sqrt((int(cy)-np.mean(yx_coord_labels[0]))**2+(int(cx)-np.mean(yx_coord_labels[1]))**2)
                
                # Gets min distance
                if euclidiean_dist < min_dist:
                    
                    # Check if segment shape is round.
                    regions = regionprops((labels == label).astype(int))
                    props = regions[0]
                    y0, x0 = props.centroid
                    orientation = props.orientation
                    x1 = x0 + np.cos(orientation) * 0.5 * props.major_axis_length
                    y1 = y0 - np.sin(orientation) * 0.5 * props.major_axis_length
                    x2 = x0 - np.sin(orientation) * 0.5 * props.minor_axis_length
                    y2 = y0 - np.cos(orientation) * 0.5 * props.minor_axis_length
                
                    d1_dist = np.sqrt(abs(x0-x1)**2+abs(y0-y1)**2)
                    d2_dist = np.sqrt(abs(x0-x2)**2+abs(y0-y2)**2)
                    
                    # Checks if segment is round.
                    if abs(d1_dist-d2_dist) > 15:
                        continue
                    
                    min_dist_label = label
                    min_dist = euclidiean_dist
                    segment_found = True
            
            # Checks if we found a image or not
            if segment_found:
                # Writes segmented object to return images                     
                segmented_slices[i,j] = (labels == min_dist_label).astype(int)
            else:
                segmented_slices[i,j] = np.zeros(labels.shape)
                
    return segmented_slices, all_labels


# In[ ]:


segmented_left_ventricle, all_segments = find_left_ventricle(images)


# In[ ]:


fig, ax  = plt.subplots(1,3, figsize=(30,10))

i = 4
j = 15

fig.suptitle('Patient #150', fontsize=16)

ax[0].imshow(images[i,j])
ax[1].imshow(segmented_left_ventricle[i,j])
ax[2].imshow(all_segments[i,j], cmap='gray')

ax[0].set_title('Original Slice')
ax[1].set_title('Selected Segment')
ax[2].set_title('All Watershedded Segments')


# In[ ]:


#plot_patient_data_4d(segmented_left_ventricle, num_slices=[i])
plot_patient_data_4d(segmented_left_ventricle, all_slices= True) 


# In[ ]:


plot_patient_data_4d(all_segments, all_slices= True) 


# ## Analysis for Volume

# In[ ]:


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# In[ ]:


def volume_for_patient(patient_images, slice_dist):
    """
    Return numpy array
    Array of total volume at each time for segmented images
    """
    
    num_slices, time, height, width = patient_images.shape
    volume = np.zeros((time))
    
    if slice_dist == 0:
        print("WARNING! Slice ditance is: 0 \n Setting slice distance to 10.")
        slice_dist = 10
    
    for i in range(time):
        time_volume = 0
        for j in range(num_slices):
            xy_size = np.sum(patient_images[j,i])
            time_volume = time_volume + xy_size * slice_dist
            
        # Volume in ml instead of mm^3
        volume[i] = time_volume/1000
        
    # Smoothing volume with convolution and removes last elements
    smooth_volume = smooth(volume,4)[2:-2]
    return smooth_volume    


# In[ ]:


v1 = volume_for_patient(segmented_left_ventricle, slice_dist)
plt.plot(v1)
plt.xlabel('Time')
plt.ylabel('Volume')
plt.title('Total Volume')


# ### Analysis for multiple patients

# In[ ]:


# Analysis for nbr patients and saves their max/min volume

nbr = 49 #Nbr of patients to be loaded

min_max_volumes = np.zeros((nbr,2))

for i in range(nbr):
    print(i+1)
    
    # Special dumb patients...
    if i == 40:
        print("Patient 41 does not exist. Replaces with patient 40 data. Not a solution...")
        min_max_volumes[i,0] = min(volumes)
        min_max_volumes[i,1] = max(volumes)
        continue
    #if i == 2:
        #print("Patient 3 is bad outlier. Replaces with patient 40 data. Not a solution...")
        #min_max_volumes[i,0] = min(volumes)
        #min_max_volumes[i,1] = max(volumes)
        #continue    
    
    # Load patient
    images, slice_dist = np_data = load_patient_data(i+1)
    # Segment and load images of patient
    segmented_left_ventricle, _ = find_left_ventricle(images)
    # Calculate volume for patient
    volumes = volume_for_patient(segmented_left_ventricle, slice_dist)
    
    min_max_volumes[i,0] = min(volumes)
    min_max_volumes[i,1] = max(volumes)
    


# In[ ]:


# Import training data. (There is probably a easyer and faster way way to do this...)

import pandas as pd
filename = "train (1).csv"
path = os.path.join("..", "input","true-volume", filename)
true_min_max = pd.read_csv(path)


# In[ ]:


true_min_max = true_min_max[(true_min_max.Id < nbr+1) & (true_min_max.Id > 0)]
true_min_max_volumes = np.zeros((nbr,2))

for Id in range(nbr):
    true_min_max_volumes[Id,0] = true_min_max.Systole.iloc[Id]
    true_min_max_volumes[Id,1] = true_min_max.Diastole.iloc[Id]


# In[ ]:


plt.plot(true_min_max_volumes[:,0], '*', label="true")
plt.plot(min_max_volumes[:,0], '+', color='r', label="predicted")
plt.suptitle('Max Volume for each patient', fontsize=16)
plt.ylabel('Volume')
plt.xlabel('Patient')
plt.legend()


# In[ ]:


plt.plot(min_max_volumes[:,0],true_min_max_volumes[:,0], '+')
plt.suptitle('True vs Pred - min Volume', fontsize=16)
plt.ylabel('True min Volume')
plt.xlabel('Predicted min Volume')


# In[ ]:


plt.plot(true_min_max_volumes[:,1], '*', label="true")
plt.plot(min_max_volumes[:,1], '+', color='r', label="predicted")
plt.suptitle('Max Volume for each patient', fontsize=16)
plt.ylabel('Volume')
plt.xlabel('Patient')
plt.legend()


# In[ ]:


plt.plot(min_max_volumes[:,1],true_min_max_volumes[:,1], '+')
plt.suptitle('True vs Pred - max Volume', fontsize=16)
plt.ylabel('True max Volume')
plt.xlabel('Predicted max Volume')


# In[ ]:


# Correlation check
corr_max = np.corrcoef(min_max_volumes[:,0], true_min_max_volumes[:,0])
corr_min = np.corrcoef(min_max_volumes[:,1], true_min_max_volumes[:,1])

print(corr_max[0,1], "Correlation predicted and true max values")
print(corr_min[0,1], "Correlation predicted and true min values")


# ### Linear regression

# In[ ]:


min_max_volumes_re = np.zeros(min_max_volumes.shape)

min_slope,min_offset = np.polyfit(min_max_volumes[:,0], true_min_max_volumes[:,0], 1)
max_slope,max_offset = np.polyfit(min_max_volumes[:,1], true_min_max_volumes[:,1], 1)

min_max_volumes_re[:,0] = min_offset + min_slope*min_max_volumes[:,0]
min_max_volumes_re[:,1] = max_offset + max_slope*min_max_volumes[:,1]


# In[ ]:


def ejection_func(volume_diastole, volume_systole):
    """
    Returns int
    Calculates ejection
    """
    ejection = 100 * ((volume_systole-volume_diastole) / volume_diastole)
    return ejection


# In[ ]:


# Calculating ture and predicted ejection for each patient
true_ejection = np.zeros(len(true_min_max_volumes))
pred_ejection = np.zeros(len(true_min_max_volumes))

for i in range(len(true_min_max_volumes)):
    true_ejection[i] = ejection_func(true_min_max_volumes[i,0],true_min_max_volumes[i,1])
    pred_ejection[i] = ejection_func(min_max_volumes_re[i,0],min_max_volumes_re[i,1])


# In[ ]:


plt.plot(true_ejection, '*', label="true")
plt.plot(pred_ejection, '+', color='r', label="predicted")
plt.suptitle('Ejection for each patient', fontsize=16)
plt.ylabel('Ejection')
plt.xlabel('Patient')
plt.legend()


# In[ ]:


# Calculating MSE
mse_error = sum((true_ejection-pred_ejection)**2)/nbr


# In[ ]:


print(mse_error, "MSE")
print((np.sqrt(mse_error)/np.mean(true_ejection)), "sqrt(MSE) divided on mean true ejection")


# In[ ]:




