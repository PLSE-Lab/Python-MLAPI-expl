#!/usr/bin/env python
# coding: utf-8

# First need to see how data is distributed and make sense of the DataFiles

# In[ ]:


get_ipython().run_line_magic('pylab', '--no-import-all')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom #for importing medical images into numpy arrays
import os #to get the names of files in a directory or something 

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
images_path = '../input/sample_images/'
# Any results you write to the current directory are saved as output.


# In[ ]:


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices]) # put slices all together in a single new numpy array


# In[ ]:


patients = os.listdir(images_path)
patients.sort()

sample_image = get_3d_data(images_path + patients[0])
sample_image.shape # first element is the numbers of slices for a patients and the other two are the pixel sizes


# sample_image includes the numbers in **Hounsfield Unit** which is a measure of radio density. For more information you can visit either Guido's Tutorial or Wiki page at https://en.wikipedia.org/wiki/Hounsfield_scale.  The unknown pixels are depicted as -2000
# 
# ![HU examples][1]
# 
#   [1]: http://i.imgur.com/4rlyReh.png

# In[ ]:


#the images have the unavailable pixel set to -2000, changing them to 0 makes the picture clearer
sample_image[sample_image == -2000] = 0
np.savetxt('sample_image_mtx.csv', sample_image[1,...], fmt = '%.18e')
print('done!')


# In[ ]:


#same plane as the original data, cut at the Z axis (Chose a cut in the middle)
pylab.imshow(sample_image[100], cmap=pylab.cm.bone)
pylab.show()


# By opening the data in excel and color code the cells based on the HU codes, The way that data is shaped is getting more clear. i.e. the image below shows the data in excel in which gray color is assigned to zero values 

# In[ ]:


patients = os.listdir(images_path)
first_patient_path = '../input/sample_images/' +patients[0]+ '/'
# getting Slice details:
sample_slices = [dicom.read_file(first_patient_path + s) for s in os.listdir(first_patient_path)]
print(sample_slices[0])
#for Slices in sample_slices:
#    print(Slices.SliceLocation)


# ##Resampling
# The purpose of resampling here is to change the scan shapes (pixels) to be able to have a 1:1 comparisons between scans to train, test, and validate the data. To start with, I am using the same method that Guido is using in his method to resample. He map all the images to a new spacing (1,1,1) and then round the images. To improve the accuracy, I might change the new spacing features to reduce the rounding impacts.  

# In[ ]:


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)

    # The following codes will resize and map the shape to the closest space of [1,1,1] 
    # based on the current space that image is located
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

